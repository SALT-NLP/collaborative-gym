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
        self._lambda: Dict[tuple, float] = {}
        self._eta: Dict[int, float] = {}
        self._n_obs: Dict[str, int] = {}

    @staticmethod
    def _key(entity_type: str, entity_id: str) -> str:
        return f"{entity_type}:{entity_id}"

    def _ensure_entity(self, entity_type: str, entity_id: str) -> int:
        """Register an entity (if new) and inject its prior exactly once."""
        key = self._key(entity_type, entity_id)
        idx = self._index.get(key)
        if idx is not None:
            return idx

        idx = len(self._index)
        self._index[key] = idx

        prior_precision = 1.0 / (self.sigma0**2)
        self._lambda[(idx, idx)] = self._lambda.get((idx, idx), 0.0) + prior_precision
        self._eta[idx] = self._eta.get(idx, 0.0) + self.mu0 * prior_precision
        return idx

    def add_observation(self, agent_id: str, human_id: str, score: float) -> None:
        """Record one collaborative episode.

        `agent_id` and `human_id` jointly produced the standardized outcome
        `score`. Call this once per completed session; observations can be
        added in any order without changing the final posterior.
        """
        idx_a = self._ensure_entity("agent", agent_id)
        idx_h = self._ensure_entity("human", human_id)

        w = 1.0 / (self.beta**2)
        lo, hi = min(idx_a, idx_h), max(idx_a, idx_h)

        self._lambda[(idx_a, idx_a)] = self._lambda.get((idx_a, idx_a), 0.0) + w
        self._lambda[(idx_h, idx_h)] = self._lambda.get((idx_h, idx_h), 0.0) + w
        self._lambda[(lo, hi)] = self._lambda.get((lo, hi), 0.0) + w

        self._eta[idx_a] = self._eta.get(idx_a, 0.0) + w * score
        self._eta[idx_h] = self._eta.get(idx_h, 0.0) + w * score

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

        lambda_matrix = np.zeros((n, n))
        for (i, j), value in self._lambda.items():
            lambda_matrix[i, j] += value
            if i != j:
                lambda_matrix[j, i] += value

        eta = np.zeros(n)
        for i, value in self._eta.items():
            eta[i] = value

        mu = np.linalg.solve(lambda_matrix, eta)
        sigma = self._solve_sigma(lambda_matrix, n)

        ratings = {}
        for key, idx in self._index.items():
            if entity_type is not None and not key.startswith(f"{entity_type}:"):
                continue
            ratings[key] = Rating(mu=float(mu[idx]), sigma=float(sigma[idx]))
        return ratings

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
