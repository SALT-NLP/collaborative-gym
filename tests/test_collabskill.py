"""Tests for `collaborative_gym.eval.collabskill`.

```
python -m pytest tests/test_collabskill.py -v
```
"""

import json
from pathlib import Path

import pytest

from collaborative_gym.eval.collabskill import CollabSkill

# Optional end-to-end check against the released CollabSkill trajectories
# dataset (https://huggingface.co/datasets/SALT-NLP/cogym-collabskill-trajectories).
# Download it with:
#   huggingface-cli download SALT-NLP/cogym-collabskill-trajectories \
#       --repo-type dataset --local-dir exp/cogym-collabskill-trajectories
DATASET_DIR = (
    Path(__file__).resolve().parents[1]
    / "exp"
    / "cogym-collabskill-trajectories"
    / "leaderboard_data"
)

EPISODES = [
    ("A1", "H1", 3.0),
    ("A2", "H1", 2.0),
    ("A1", "H2", 1.0),
]

EXPECTED_MU = {
    "agent:A1": 19 / 21,
    "agent:A2": 8 / 21,
    "human:H1": 26 / 21,
    "human:H2": 1 / 21,
}

EXPECTED_N = {
    "agent:A1": 2,
    "agent:A2": 1,
    "human:H1": 2,
    "human:H2": 1,
}


def _build_model(order, **kwargs):
    model = CollabSkill(**kwargs)
    for agent_id, human_id, score in order:
        model.add_observation(agent_id=agent_id, human_id=human_id, score=score)
    return model


def test_exact_posterior_mean():
    model = _build_model(EPISODES)
    ratings = model.rate()
    for key, expected in EXPECTED_MU.items():
        assert ratings[key].mu == pytest.approx(expected, abs=1e-9)


def test_observation_counts():
    model = _build_model(EPISODES)
    for key, expected in EXPECTED_N.items():
        assert model._n_obs[key] == expected


def test_add_observation_is_order_independent():
    forward = _build_model(EPISODES).rate()
    backward = _build_model(list(reversed(EPISODES))).rate()
    for key in EXPECTED_MU:
        assert forward[key].mu == pytest.approx(backward[key].mu, abs=1e-9)
        assert forward[key].sigma == pytest.approx(backward[key].sigma, abs=1e-9)


def test_leaderboard_ranks_by_conservative_score():
    model = _build_model(EPISODES)
    agent_board = model.leaderboard("agent")
    assert [row["entity_id"] for row in agent_board] == ["A1", "A2"]
    assert agent_board[0]["rank"] == 1
    assert agent_board[0]["conservative"] >= agent_board[1]["conservative"]

    human_board = model.leaderboard("human")
    assert [row["entity_id"] for row in human_board] == ["H1", "H2"]


def test_empty_model_returns_no_ratings():
    model = CollabSkill()
    assert model.rate() == {}
    assert model.leaderboard() == []


def test_default_k_matches_paper():
    assert CollabSkill().k == 3.0


def test_leaderboard_respects_custom_k():
    zero_penalty = _build_model(EPISODES, k=0.0)
    for row in zero_penalty.leaderboard():
        # With k=0, the conservative score collapses to mu.
        assert row["conservative"] == pytest.approx(row["mu"], abs=1e-9)

    strict = _build_model(EPISODES, k=3.0)
    for row in strict.leaderboard():
        assert row["conservative"] == pytest.approx(row["mu"] - 3.0 * row["sigma"], abs=1e-9)


def test_invalid_sigma_mode_raises():
    with pytest.raises(ValueError):
        CollabSkill(sigma_mode="bogus")


def test_hutchinson_sigma_mode_approximates_exact_sigma():
    exact = _build_model(EPISODES, sigma_mode="exact").rate()
    hutch = _build_model(
        EPISODES, sigma_mode="hutchinson", hutch_probes=2048, hutch_seed=42
    ).rate()

    for key in EXPECTED_MU:
        # mu is solved exactly regardless of sigma_mode.
        assert hutch[key].mu == pytest.approx(exact[key].mu, abs=1e-9)
        # sigma is only approximate, but should be close with enough probes.
        assert hutch[key].sigma == pytest.approx(exact[key].sigma, abs=0.05)


def test_hutchinson_sigma_mode_is_deterministic_given_seed():
    def _hutch_sigmas(seed):
        model = _build_model(
            EPISODES, sigma_mode="hutchinson", hutch_probes=32, hutch_seed=seed
        )
        return model.rate()

    first, second = _hutch_sigmas(seed=7), _hutch_sigmas(seed=7)
    for key in EXPECTED_MU:
        assert first[key].sigma == second[key].sigma


@pytest.mark.skipif(
    not (DATASET_DIR / "ratings.json").exists(),
    reason=f"CollabSkill trajectories dataset not found under {DATASET_DIR}",
)
def test_reproduces_released_collabskill_ratings():
    """Rebuild ratings from the released sessions and compare against the
    published `ratings.json` (computed by the production skill_sparse.py
    rating engine with mu0=0, sigma0=1, beta=1, k=3).
    """
    sessions = json.loads((DATASET_DIR / "sessions.json").read_text())
    expected = json.loads((DATASET_DIR / "ratings.json").read_text())

    model = CollabSkill(mu0=0.0, sigma0=1.0, beta=1.0, k=3.0)
    for session in sessions:
        if session.get("autograde_status") != "done" or session.get("skipped"):
            continue
        score = session.get("autograde_overall_score")
        agent_id = session.get("agent_id")
        human_id = session.get("user_id")
        if score is None or not agent_id or not human_id:
            continue
        model.add_observation(agent_id=agent_id, human_id=human_id, score=float(score))

    for entity_type, expected_rows in (
        ("agent", expected["agents"]),
        ("human", expected["humans"]),
    ):
        board = {row["entity_id"]: row for row in model.leaderboard(entity_type)}
        assert len(board) == len(expected_rows)

        for expected_row in expected_rows:
            row = board[expected_row["entity_id"]]
            assert row["n"] == expected_row["n"]
            assert row["mu"] == pytest.approx(expected_row["mu"], abs=1e-4)
            # Published sigma is a Hutchinson-approximate estimate (see the
            # production skill_sparse.py); our dense solve is exact, so allow
            # a looser tolerance than mu.
            assert row["sigma"] == pytest.approx(expected_row["sigma"], abs=1e-2)
            # Ranks can legitimately swap only among near-exact ties.
            if row["rank"] != expected_row["rank"]:
                assert row["conservative"] == pytest.approx(
                    expected_row["conservative"], abs=1e-3
                )


def _dense_log_evidence(episodes, mu0, sigma0, beta):
    """Brute-force log N(y | 2 mu0, beta^2 I + sigma0^2 Z Z^T) for checking."""
    import numpy as np

    keys = sorted({f"agent:{a}" for a, _, _ in episodes} | {f"human:{h}" for _, h, _ in episodes})
    col = {k: i for i, k in enumerate(keys)}
    z = np.zeros((len(episodes), len(keys)))
    for row, (a, h, _) in enumerate(episodes):
        z[row, col[f"agent:{a}"]] = 1.0
        z[row, col[f"human:{h}"]] = 1.0
    y = np.array([s for _, _, s in episodes])
    cov = beta**2 * np.eye(len(y)) + sigma0**2 * z @ z.T
    r = y - 2.0 * mu0
    _, logdet = np.linalg.slogdet(cov)
    return -0.5 * (r @ np.linalg.solve(cov, r) + logdet + len(y) * np.log(2 * np.pi))


def _simulated_episodes(n_agents=8, n_humans=150, per_human=4, mu0=30.0, sigma0=8.0, beta=15.0, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    agents = rng.normal(mu0, sigma0, n_agents)
    humans = rng.normal(mu0, sigma0, n_humans)
    episodes = []
    for h in range(n_humans):
        for a in rng.choice(n_agents, size=per_human, replace=False):
            y = agents[a] + humans[h] + rng.normal(0.0, beta)
            episodes.append((f"A{a}", f"H{h}", float(y)))
    return episodes


def test_log_marginal_likelihood_matches_dense_gaussian():
    episodes = _simulated_episodes(n_agents=4, n_humans=12, per_human=2, seed=1)
    model = _build_model(episodes)
    for mu0, sigma0, beta in [(0.0, 1.0, 1.0), (25.0, 6.0, 12.0), (-3.0, 0.5, 40.0)]:
        assert model.log_marginal_likelihood(mu0, sigma0, beta) == pytest.approx(
            _dense_log_evidence(episodes, mu0, sigma0, beta), rel=1e-9, abs=1e-6
        )
    # defaults come from the model's own hyperparameters
    assert model.log_marginal_likelihood() == pytest.approx(
        _dense_log_evidence(episodes, 0.0, 1.0, 1.0), rel=1e-9
    )


def test_fit_hyperparameters_recovers_simulated_scale():
    episodes = _simulated_episodes(seed=2)
    model = _build_model(episodes)
    before = model.log_marginal_likelihood()
    fitted = model.fit_hyperparameters()
    assert fitted["beta"] == pytest.approx(15.0, rel=0.15)
    assert fitted["sigma0"] == pytest.approx(8.0, rel=0.3)
    assert fitted["log_marginal_likelihood"] > before
    # update=True moved the model onto the fitted values
    assert (model.mu0, model.sigma0, model.beta) == (
        fitted["mu0"],
        fitted["sigma0"],
        fitted["beta"],
    )
    # the fit is a local optimum of the evidence in every coordinate
    best = fitted["log_marginal_likelihood"]
    for d_mu0, f_s0, f_b in [(0.5, 1, 1), (-0.5, 1, 1), (0, 1.05, 1), (0, 0.95, 1), (0, 1, 1.05), (0, 1, 0.95)]:
        assert model.log_marginal_likelihood(
            fitted["mu0"] + d_mu0, fitted["sigma0"] * f_s0, fitted["beta"] * f_b
        ) < best


def test_fit_hyperparameters_without_update_leaves_model_alone():
    model = _build_model(_simulated_episodes(seed=3))
    model.fit_hyperparameters(update=False)
    assert (model.mu0, model.sigma0, model.beta) == (0.0, 1.0, 1.0)


def test_fit_hyperparameters_needs_two_observations():
    model = _build_model(EPISODES[:1])
    with pytest.raises(ValueError):
        model.fit_hyperparameters()


def test_win_probability_uses_joint_posterior():
    import math

    import numpy as np

    model = _build_model(_simulated_episodes(n_agents=3, n_humans=20, per_human=2, seed=4))
    model.fit_hyperparameters()
    p = model.win_probability("A0", "A1")
    assert p + model.win_probability("A1", "A0") == pytest.approx(1.0)
    assert model.win_probability("A0", "A0") == 0.5
    # same number from an explicit dense posterior, including the covariance term
    lam, eta = model._natural_params(model.mu0, model.sigma0, model.beta)
    cov = np.linalg.inv(lam)
    mu = np.linalg.solve(lam, eta)
    i, j = model._index["agent:A0"], model._index["agent:A1"]
    var = cov[i, i] + cov[j, j] - 2 * cov[i, j]
    expected = 0.5 * (1 + math.erf((mu[i] - mu[j]) / math.sqrt(2 * var)))
    assert p == pytest.approx(expected, abs=1e-12)


def test_rank_probabilities_are_distributions_and_seeded():
    model = _build_model(_simulated_episodes(n_agents=5, n_humans=40, per_human=3, seed=5))
    model.fit_hyperparameters()
    first = model.rank_probabilities("agent", n_samples=4000, seed=11)
    again = model.rank_probabilities("agent", n_samples=4000, seed=11)
    assert first == again
    assert set(first) == {f"A{i}" for i in range(5)}
    assert sum(r["p_first"] for r in first.values()) == pytest.approx(1.0)
    for row in first.values():
        assert sum(row["p_rank"]) == pytest.approx(1.0)
        assert 1.0 <= row["mean_rank"] <= 5.0
    assert CollabSkill().rank_probabilities() == {}


def test_refit_does_not_change_default_posterior_after_reset():
    model = _build_model(EPISODES)
    model.fit_hyperparameters()
    model.mu0, model.sigma0, model.beta = 0.0, 1.0, 1.0
    ratings = model.rate()
    for key, expected in EXPECTED_MU.items():
        assert ratings[key].mu == pytest.approx(expected, abs=1e-12)


@pytest.mark.skipif(
    not (DATASET_DIR / "sessions.json").exists(),
    reason=f"CollabSkill trajectories dataset not found under {DATASET_DIR}",
)
def test_fitted_rank_probabilities_match_paper_bootstrap():
    """With hyperparameters fit to the released scores, the posterior rank
    distribution lines up with the paper's 10,000-round bootstrap (Table 3:
    Claude Cowork mean rank 1.36, P(rank=1) = 0.713)."""
    sessions = json.loads((DATASET_DIR / "sessions.json").read_text())
    model = CollabSkill()
    for session in sessions:
        if session.get("autograde_status") != "done" or session.get("skipped"):
            continue
        score = session.get("autograde_overall_score")
        if score is None or not session.get("agent_id") or not session.get("user_id"):
            continue
        model.add_observation(session["agent_id"], session["user_id"], float(score))

    # with the unit-scale defaults the ladder looks certain
    assert model.rank_probabilities("agent", seed=0)["cowork"]["p_first"] == 1.0

    fitted = model.fit_hyperparameters()
    assert fitted["beta"] == pytest.approx(20.34, abs=0.05)
    assert fitted["sigma0"] == pytest.approx(10.19, abs=0.05)
    ranks = model.rank_probabilities("agent", n_samples=20_000, seed=0)
    assert ranks["cowork"]["p_first"] == pytest.approx(0.713, abs=0.02)
    assert ranks["cowork"]["mean_rank"] == pytest.approx(1.36, abs=0.03)
