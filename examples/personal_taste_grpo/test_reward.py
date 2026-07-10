from __future__ import annotations

import json
from pathlib import Path

from taste_reward import (
    clean_candidate_name,
    load_scorer,
    name_family,
    reward_group,
    score_candidate,
)
from train import load_briefs

HERE = Path(__file__).resolve().parent


def test_frozen_assets_are_ready() -> None:
    scorer = load_scorer(HERE / "data/scorer.json")
    briefs = load_briefs(HERE / "data/briefs.json")
    assert scorer["ready"] is True
    assert scorer["metrics"]["pairwise_accuracy"] > 0.75
    assert len(briefs) == 120


def test_candidate_cleaning_blocks_known_collapse_outputs() -> None:
    assert clean_candidate_name("None<|im_end|>") == ""
    assert clean_candidate_name("null") == ""
    assert clean_candidate_name("Vellum<|im_end|>") == "Vellum"
    assert name_family("Vellum") == name_family("Velvet") == "vel"


def test_frozen_scorer_and_family_guardrail() -> None:
    scorer = load_scorer(HERE / "data/scorer.json")
    brief = json.loads(json.loads((HERE / "data/briefs.json").read_text())[0]["brief"])
    candidates = [
        {"name": "Vellum", "brief": brief},
        {"name": "Velvet", "brief": brief},
        {"name": "Veloce", "brief": brief},
        {"name": "Moss", "brief": brief},
        {"name": "Rivet", "brief": brief},
        {"name": "Sable", "brief": brief},
        {"name": "Cinder", "brief": brief},
        {"name": "Keel", "brief": brief},
    ]
    config = {
        "taste_weight": 0.85,
        "aux_weight": 0.15,
        "family_duplicate_guardrail_threshold": 0.25,
        "literal_copy_guardrail_threshold": 0.50,
    }
    rewards = reward_group(candidates, scorer, config)
    assert rewards[0]["components"]["hard_guardrail"] == 1.0
    assert -1.0 <= score_candidate(candidates[-1], scorer)["score"] <= 1.0
