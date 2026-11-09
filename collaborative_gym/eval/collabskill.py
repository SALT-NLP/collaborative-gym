"""CollabSkill: a TrueSkill-style Bayesian rating system for human-agent collaboration.

Unlike a single combined task score, CollabSkill **decouples** the contribution of 
the human and the agent within the same collaborative episode, producing an independent 
skill estimate for each agent and each human that ever participated in a session. 

CollabSkill Model
-----
A raw teamwork score reflects the *joint* contribution of the human and the
agent, so simply averaging scores per agent (as completion rate, average
session score, or win rate implicitly do) is not a trustworthy ranking:
humans differ substantially in AI literacy, and this inter-human variability
does not average out in expectation. CollabSkill **decouples** the contribution 
of  the human and the agent within the same collaborative episode, producing
an independent skill estimate for each agent and each human.

For every teamwork outcome ``(A, H, y)`` (agent ``A``, human ``H``, scalar
score ``y``), we posit a latent collaboration skill for each participant
and model the score as an additive decomposition of agent skill, human
skill, and observation noise::

    s_i ~ N(mu0, sigma0^2)                for every agent/human i (prior)
    y   = s_A + s_H + eps,  eps ~ N(0, beta^2)          for each episode

Rather than treating human variability as noise, the formulation explicitly 
allocates a latent skill component to each human. If agent ``A`` collaborates 
with both ``H1`` and ``H2``, their skills ``s_H1`` and ``s_H2`` are estimated 
separately from their respective sessions, and ``s_A`` is inferred by explaining 
away the portion of each outcome attributable to whichever human it was paired 
with. If ``H1`` consistently scores well across several different agents 
while ``H2`` does not, the model attributes that difference to ``H1``'s skill 
at collaborating with AI agents, not to whichever agent ``H1`` happened to work with.

In isolation, each new observation is a one-step Kalman filter measurement
update on the joint belief ``theta = [s_A, s_H] ~ N(m,
Sigma)``, with design vector ``x = [1, 1]``:

    r = y - x^T m                        prediction error
    K = (Sigma @ x) / (x^T @ Sigma @ x + beta^2)         Kalman gain
    m'     = m + K * r
    Sigma' = Sigma - K @ x^T @ Sigma

When an outcome exceeds expectation, both ``mu_A`` and ``mu_H`` increase,
with the larger share of the adjustment going to whichever entity is
currently more uncertain (larger ``sigma^2``); ``sigma`` shrinks
monotonically as observations accumulate.

Reliable rankings do not require many sessions per agent. What matters is 
that the human-agent interaction graph is *connected*, i.e. every agent 
shares at least one human with every other agent (directly or transitively).
A shared human lets the model attribute score differences to the agents 
rather than to the human they happened to be paired with; under random 
human-agent matching over a small, fixed set of agents 
(as in the CollabSkill study), this connectivity arises naturally.

The final CollabSkill rating is computed as a conservative
score ``mu - k * sigma`` (approximately the Gaussian's ``k``-sigma lower
quantile) rather than by ``mu`` alone, which penalizes high-uncertainty
entities and prevents an agent or human with only a few lucky observations
from out-ranking a well-established one. We set the default as ``k=3``.

- ``"exact"`` (default): ``sigma_i = sqrt((Lambda^-1)_ii)`` via a dense
  matrix solve. Exact, and fine for the hundreds-to-thousands of
  agents/humans a typical study produces.
- ``"hutchinson"``: a stochastic diagonal estimator (Hutchinson's method)
  that only needs ``hutch_probes`` solves against random +/-1 vectors,
  independent of ``n``. Approximate, but scales to internet-scale
  leaderboards where a full ``n x n`` solve is too slow.

Usage:
    >>> from collaborative_gym.eval.collabskill import CollabSkill
    >>> model = CollabSkill()
    >>> model.add_observation(agent_id="A1", human_id="H1", score=3.0)
    >>> model.add_observation(agent_id="A2", human_id="H1", score=2.0)
    >>> model.add_observation(agent_id="A1", human_id="H2", score=1.0)
    >>> model.rate()["agent:A1"].mu > model.rate()["agent:A2"].mu
    True
    >>> model.leaderboard("agent")[0]["entity_id"]
    'A1'

More details can be found in the CollabSkill paper: https://arxiv.org/abs/2606.09833
"""

from __future__ import annotations

import math
from typing import Dict, List, NamedTuple, Optional

import numpy as np

__all__ = ["Rating", "CollabSkill"]

DEFAULT_MU0 = 0.0
DEFAULT_SIGMA0 = 1.0
DEFAULT_BETA = 1.0
DEFAULT_K = 3.0

SIGMA_MODES = ("exact", "hutchinson")
DEFAULT_SIGMA_MODE = "exact"
DEFAULT_HUTCH_PROBES = 64
DEFAULT_HUTCH_SEED = 0


class Rating(NamedTuple):
    """A Gaussian belief over one entity's latent collaboration skill."""

    mu: float
    sigma: float


class CollabSkill:
    """Bayesian rating model that decouples agent skill from human skill.

    Parameters
    ----------
    mu0, sigma0:
        Mean and standard deviation of the prior placed on every new
        agent/human the first time it is observed.
    beta:
        Standard deviation of the observation noise ``eps`` in ``y = s_agent
        + s_human + eps``. Larger `beta` means a single episode's score is
        treated as less informative about either participant's skill.
    k:
        Conservative-score multiplier used by `leaderboard`, following the
        standard TrueSkill convention of ranking by ``mu - k * sigma`` so
        that entities with fewer observations (higher uncertainty) don't
        outrank well-established ones on a lucky score. Defaults to ``3``,
        matching the CollabSkill paper.
    sigma_mode:
        How `rate` computes each entity's posterior ``sigma``:

        - ``"exact"`` (default): exact dense solve, ``O(n^3)`` in the
          number of entities. Recommended unless ``n`` is in the
          thousands-plus.
        - ``"hutchinson"``: Hutchinson's stochastic diagonal estimator,
          ``O(hutch_probes)`` solves independent of ``n``. Approximate --
          accuracy improves with more probes -- but scales to much larger
          entity pools.
    hutch_probes:
        Number of random probe vectors used when ``sigma_mode="hutchinson"``.
        Ignored otherwise.
    hutch_seed:
        Seed for the probe vectors used when ``sigma_mode="hutchinson"``, so
        results are reproducible. Ignored otherwise.
    """

    def __init__(
        self,
        mu0: float = DEFAULT_MU0,
        sigma0: float = DEFAULT_SIGMA0,
        beta: float = DEFAULT_BETA,
        k: float = DEFAULT_K,
        sigma_mode: str = DEFAULT_SIGMA_MODE,
        hutch_probes: int = DEFAULT_HUTCH_PROBES,
        hutch_seed: int = DEFAULT_HUTCH_SEED,
    ):
        if sigma_mode not in SIGMA_MODES:
            raise ValueError(
                f"sigma_mode must be one of {SIGMA_MODES!r}, got {sigma_mode!r}"
            )

        self.mu0 = mu0
        self.sigma0 = sigma0
        self.beta = beta
        self.k = k
        self.sigma_mode = sigma_mode
        self.hutch_probes = hutch_probes
        self.hutch_seed = hutch_seed

        self._index: Dict[str, int] = {}
        # Raw sufficient statistics of the observations, kept separate from the
        # prior and from beta so the posterior can be re-solved (or the
        # hyperparameters re-fit) at any time: Z^T Z (pair counts), Z^T y, and
        # the count / sum / sum of squares of the scores.
        self._ztz: Dict[tuple, float] = {}
        self._zty: Dict[int, float] = {}
        self._n_scores = 0
        self._sum_y = 0.0
        self._sum_y2 = 0.0
        self._n_obs: Dict[str, int] = {}

    @staticmethod
    def _key(entity_type: str, entity_id: str) -> str:
        return f"{entity_type}:{entity_id}"

    def _ensure_entity(self, entity_type: str, entity_id: str) -> int:
        """Register an entity (if new). Its prior is added when the posterior is solved."""
        key = self._key(entity_type, entity_id)
        idx = self._index.get(key)
        if idx is not None:
            return idx

        idx = len(self._index)
        self._index[key] = idx
        return idx

    def add_observation(self, agent_id: str, human_id: str, score: float) -> None:
        """Record one collaborative episode.

        `agent_id` and `human_id` jointly produced the standardized outcome
        `score`. Call this once per completed session; observations can be
        added in any order without changing the final posterior.
        """
        idx_a = self._ensure_entity("agent", agent_id)
        idx_h = self._ensure_entity("human", human_id)

        score = float(score)
        lo, hi = min(idx_a, idx_h), max(idx_a, idx_h)

        self._ztz[(idx_a, idx_a)] = self._ztz.get((idx_a, idx_a), 0.0) + 1.0
        self._ztz[(idx_h, idx_h)] = self._ztz.get((idx_h, idx_h), 0.0) + 1.0
        self._ztz[(lo, hi)] = self._ztz.get((lo, hi), 0.0) + 1.0

        self._zty[idx_a] = self._zty.get(idx_a, 0.0) + score
        self._zty[idx_h] = self._zty.get(idx_h, 0.0) + score
        self._n_scores += 1
        self._sum_y += score
        self._sum_y2 += score * score

        agent_key, human_key = self._key("agent", agent_id), self._key("human", human_id)
        self._n_obs[agent_key] = self._n_obs.get(agent_key, 0) + 1
        self._n_obs[human_key] = self._n_obs.get(human_key, 0) + 1

    def rate(self, entity_type: Optional[str] = None) -> Dict[str, Rating]:
        """Solve the joint posterior and return a `Rating` per entity.

        Parameters
        ----------
        entity_type:
            Restrict the result to ``"agent"`` or ``"human"``. Defaults to
            both. Note that entities of the other type still participate in
            the solve -- they are only filtered out of the returned dict.

        Returns
        -------
        A mapping from ``"{entity_type}:{entity_id}"`` to its `Rating`.
        """
        n = len(self._index)
        if n == 0:
            return {}

        lambda_matrix, eta = self._natural_params(self.mu0, self.sigma0, self.beta)
        mu = np.linalg.solve(lambda_matrix, eta)
        sigma = self._solve_sigma(lambda_matrix, n)

        ratings = {}
        for key, idx in self._index.items():
            if entity_type is not None and not key.startswith(f"{entity_type}:"):
                continue
            ratings[key] = Rating(mu=float(mu[idx]), sigma=float(sigma[idx]))
        return ratings

    def _ztz_matrix(self) -> np.ndarray:
        n = len(self._index)
        ztz = np.zeros((n, n))
        for (i, j), value in self._ztz.items():
            ztz[i, j] += value
            if i != j:
                ztz[j, i] += value
        return ztz

    def _zty_vector(self) -> np.ndarray:
        zty = np.zeros(len(self._index))
        for i, value in self._zty.items():
            zty[i] = value
        return zty

    def _natural_params(self, mu0: float, sigma0: float, beta: float):
        """Posterior precision ``Lambda`` and information vector ``eta``."""
        n = len(self._index)
        lambda_matrix = self._ztz_matrix() / beta**2 + np.eye(n) / sigma0**2
        eta = self._zty_vector() / beta**2 + mu0 / sigma0**2
        return lambda_matrix, eta

    def _solve_sigma(self, lambda_matrix: np.ndarray, n: int) -> np.ndarray:
        """Compute ``sqrt(diag(Lambda^-1))`` per `self.sigma_mode`."""
        if self.sigma_mode == "exact":
            diag = np.linalg.inv(lambda_matrix).diagonal()
        else:  # "hutchinson"
            diag = self._hutchinson_diag(lambda_matrix, n)
        return np.sqrt(np.clip(diag, 0.0, None))

    def _hutchinson_diag(self, lambda_matrix: np.ndarray, n: int) -> np.ndarray:
        """Estimate ``diag(Lambda^-1)`` via Hutchinson's stochastic estimator.

        Draws `self.hutch_probes` Rademacher (+-1) probe vectors ``v`` and
        uses ``E[v * (Lambda^-1 v)] = diag(Lambda^-1)``. All probes are
        solved in a single ``np.linalg.solve`` call (one LU factorization,
        many right-hand sides), so this is much cheaper than a full
        ``O(n^3)`` inverse when only an approximate ``sigma`` is needed.
        """
        rng = np.random.default_rng(self.hutch_seed)
        probes = rng.integers(0, 2, size=(n, self.hutch_probes)).astype(np.float64)
        probes = probes * 2.0 - 1.0  # {0, 1} -> {-1, +1}
        solved = np.linalg.solve(lambda_matrix, probes)
        return np.mean(probes * solved, axis=1)

    def leaderboard(self, entity_type: Optional[str] = None) -> List[dict]:
        """Rank entities by conservative score ``mu - k * sigma``.

        Parameters
        ----------
        entity_type:
            Restrict the leaderboard to ``"agent"`` or ``"human"``. Defaults
            to both.

        Returns
        -------
        A list of dicts (highest-ranked first), each with keys ``rank``,
        ``entity_type``, ``entity_id``, ``mu``, ``sigma``, ``n``, and
        ``conservative``.
        """
        rows = []
        for key, rating in self.rate(entity_type).items():
            etype, eid = key.split(":", 1)
            rows.append(
                {
                    "entity_type": etype,
                    "entity_id": eid,
                    "mu": rating.mu,
                    "sigma": rating.sigma,
                    "n": self._n_obs.get(key, 0),
                    "conservative": rating.mu - self.k * rating.sigma,
                }
            )
        rows.sort(key=lambda row: row["conservative"], reverse=True)
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        return rows

    # ------------------------------------------------------------------
    # Comparing entities with the joint posterior
    # ------------------------------------------------------------------

    def _posterior(self, entity_type: Optional[str]):
        """Joint posterior mean and covariance over entities of `entity_type`."""
        if not self._index:
            return [], np.zeros(0), np.zeros((0, 0))
        lambda_matrix, eta = self._natural_params(self.mu0, self.sigma0, self.beta)
        mu = np.linalg.solve(lambda_matrix, eta)
        keys = [
            key
            for key in self._index
            if entity_type is None or key.startswith(f"{entity_type}:")
        ]
        idx = np.array([self._index[key] for key in keys], dtype=int)
        cov = np.linalg.inv(lambda_matrix)[np.ix_(idx, idx)]
        return keys, mu[idx], cov

    def win_probability(
        self, entity_a: str, entity_b: str, entity_type: str = "agent"
    ) -> float:
        """Posterior probability that ``s_a > s_b``.

        Uses the joint posterior, including the covariance between the two
        entities (agents that share humans are correlated), so it is not the
        same as comparing two marginal ``(mu, sigma)`` pairs.
        """
        keys, mu, cov = self._posterior(entity_type)
        pos = {key: i for i, key in enumerate(keys)}
        i = pos[self._key(entity_type, entity_a)]
        j = pos[self._key(entity_type, entity_b)]
        var = cov[i, i] + cov[j, j] - 2.0 * cov[i, j]
        diff = mu[i] - mu[j]
        if var <= 0.0:
            return 0.5 if diff == 0 else float(diff > 0)
        return 0.5 * (1.0 + math.erf(diff / math.sqrt(2.0 * var)))

    def rank_probabilities(
        self, entity_type: str = "agent", n_samples: int = 10_000, seed: int = 0
    ) -> Dict[str, dict]:
        """Monte Carlo rank distribution under the joint posterior.

        Draws `n_samples` skill vectors from the posterior over all entities of
        `entity_type` and ranks them by skill (1 = best). Returns, per entity
        id, ``{"mean_rank", "p_first", "p_rank"}`` where ``p_rank[r - 1]`` is
        the probability of rank ``r``. This is the posterior counterpart of a
        bootstrap rank-stability table, and it is only as calibrated as
        ``beta``/``sigma0`` are for the score scale (see
        `fit_hyperparameters`).
        """
        keys, mu, cov = self._posterior(entity_type)
        m = len(keys)
        if m == 0:
            return {}
        rng = np.random.default_rng(seed)
        chol = np.linalg.cholesky(cov + 1e-12 * np.eye(m))
        draws = mu + rng.standard_normal((n_samples, m)) @ chol.T
        ranks = np.argsort(np.argsort(-draws, axis=1), axis=1) + 1
        out = {}
        for col, key in enumerate(keys):
            counts = np.bincount(ranks[:, col], minlength=m + 1)[1:]
            out[key.split(":", 1)[1]] = {
                "mean_rank": float(ranks[:, col].mean()),
                "p_first": float(counts[0] / n_samples),
                "p_rank": (counts / n_samples).tolist(),
            }
        return out

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------

    def log_marginal_likelihood(
        self,
        mu0: Optional[float] = None,
        sigma0: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> float:
        """Log evidence ``log p(y | mu0, sigma0, beta)`` of the observed scores.

        Integrates the skills out of ``y = s_A + s_H + eps``. Defaults to the
        model's current hyperparameters. Computed from sufficient statistics
        with the matrix determinant lemma and Woodbury identity, so it costs
        one ``n x n`` factorization in the number of entities, not sessions.
        """
        mu0 = self.mu0 if mu0 is None else mu0
        sigma0 = self.sigma0 if sigma0 is None else sigma0
        beta = self.beta if beta is None else beta
        n_ent, n_y = len(self._index), self._n_scores
        if n_y == 0:
            return 0.0
        ztz = self._ztz_matrix()
        # residual r = y - Z m with prior mean m = mu0 * 1 (each row of Z has two 1s)
        r_r = self._sum_y2 - 4.0 * mu0 * self._sum_y + 4.0 * mu0 * mu0 * n_y
        zt_r = self._zty_vector() - mu0 * ztz.sum(axis=1)
        lam = ztz / beta**2 + np.eye(n_ent) / sigma0**2
        chol = np.linalg.cholesky(lam)
        half = np.linalg.solve(chol, zt_r)
        quad = (r_r - half @ half / beta**2) / beta**2
        logdet = (
            2.0 * n_y * math.log(beta)
            + 2.0 * n_ent * math.log(sigma0)
            + 2.0 * float(np.log(np.diag(chol)).sum())
        )
        return -0.5 * (quad + logdet + n_y * math.log(2.0 * math.pi))

    def fit_hyperparameters(self, update: bool = True) -> dict:
        """Fit ``mu0``, ``sigma0`` and ``beta`` by maximizing the marginal likelihood.

        The defaults (``mu0=0, sigma0=1, beta=1``) assume scores on a unit
        scale. On raw 0-100 grader scores they make ``sigma`` far too small
        and every ranking look certain. Fitting the three hyperparameters to
        the observations (type-II maximum likelihood) puts ``beta`` on the
        scale of the actual session noise. ``mu0`` has a closed form given the
        other two; ``log sigma0`` and ``log beta`` are found with Nelder-Mead.

        Returns ``{"mu0", "sigma0", "beta", "log_marginal_likelihood"}`` and,
        if `update` is true, sets them on the model.
        """
        if self._n_scores < 2:
            raise ValueError("need at least two observations to fit hyperparameters")

        def profile(log_s0: float, log_b: float):
            s0, b = math.exp(log_s0), math.exp(log_b)
            mu0 = self._best_mu0(s0, b)
            return -self.log_marginal_likelihood(mu0, s0, b), mu0

        mean_y = self._sum_y / self._n_scores
        spread = math.sqrt(max(self._sum_y2 / self._n_scores - mean_y**2, 1e-12))
        start = (math.log(max(spread / 2.0, 1e-6)), math.log(max(spread, 1e-6)))
        (log_s0, log_b), nll = _nelder_mead(lambda x: profile(*x)[0], start)
        s0, b = math.exp(log_s0), math.exp(log_b)
        mu0 = profile(log_s0, log_b)[1]
        fitted = {"mu0": mu0, "sigma0": s0, "beta": b, "log_marginal_likelihood": -nll}
        if update:
            self.mu0, self.sigma0, self.beta = mu0, s0, b
        return fitted

    def _best_mu0(self, sigma0: float, beta: float) -> float:
        """Closed-form ``mu0`` maximizing the evidence for fixed sigma0, beta.

        With ``u = Z 1`` (every entry 2), the GLS estimate is
        ``u^T K^-1 y / u^T K^-1 u`` for ``K = beta^2 I + sigma0^2 Z Z^T``,
        evaluated through the Woodbury identity.
        """
        ztz = self._ztz_matrix()
        zt_u = ztz.sum(axis=1)  # Z^T u, with u = Z 1
        lam = ztz / beta**2 + np.eye(len(self._index)) / sigma0**2
        u_u = 4.0 * self._n_scores
        u_y = 2.0 * self._sum_y
        zt_y = self._zty_vector()
        solved_u = np.linalg.solve(lam, zt_u)
        num = u_y - zt_y @ solved_u / beta**2
        den = u_u - zt_u @ solved_u / beta**2
        return float(num / den)


def _nelder_mead(f, x0, step=0.5, tol=1e-10, max_iter=2000):
    """Minimal Nelder-Mead for the 2-D hyperparameter search (numpy only)."""
    pts = [np.array(x0, dtype=float)]
    for k in range(len(x0)):
        p = np.array(x0, dtype=float)
        p[k] += step
        pts.append(p)
    vals = [f(p) for p in pts]
    for _ in range(max_iter):
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        width = max(np.abs(p - pts[0]).max() for p in pts)
        if abs(vals[-1] - vals[0]) < tol and width < 1e-8:
            break
        centroid = np.mean(pts[:-1], axis=0)
        refl = centroid + (centroid - pts[-1])
        f_refl = f(refl)
        if f_refl < vals[0]:
            exp = centroid + 2.0 * (centroid - pts[-1])
            f_exp = f(exp)
            pts[-1], vals[-1] = (exp, f_exp) if f_exp < f_refl else (refl, f_refl)
        elif f_refl < vals[-2]:
            pts[-1], vals[-1] = refl, f_refl
        else:
            contr = centroid + 0.5 * (pts[-1] - centroid)
            f_contr = f(contr)
            if f_contr < vals[-1]:
                pts[-1], vals[-1] = contr, f_contr
            else:
                for i in range(1, len(pts)):
                    pts[i] = pts[0] + 0.5 * (pts[i] - pts[0])
                    vals[i] = f(pts[i])
    best = int(np.argmin(vals))
    return tuple(pts[best]), float(vals[best])
