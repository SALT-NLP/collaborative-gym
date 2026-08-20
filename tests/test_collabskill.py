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
