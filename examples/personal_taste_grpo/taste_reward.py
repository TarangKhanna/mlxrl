"""Frozen taste scorer, name validation, and anti-collapse reward shaping."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

GENERIC_TOKENS = {
    "ai",
    "app",
    "base",
    "boost",
    "cloud",
    "flow",
    "forge",
    "hub",
    "labs",
    "loop",
    "pilot",
    "stack",
    "studio",
}
GENERIC_NAME_STEMS = (
    "app",
    "cloud",
    "flow",
    "forge",
    "hub",
    "labs",
    "pilot",
    "stack",
    "studio",
    "tech",
)
INVALID_PREFIXES = (
    "additional",
    "answer",
    "area",
    "assistant",
    "backstory",
    "banned",
    "challenge",
    "constraints",
    "example",
    "for example",
    "goal",
    "language",
    "need",
    "okay",
    "option",
    "options",
    "output",
    "please",
    "rules",
    "targeted",
    "technology",
    "then",
    "think",
    "this is",
    "visual",
    "what",
    "and ",
    "empty",
)
INVALID_PHRASES = (
    "let's",
    "the domain",
    "the user",
    "generate",
    "create a",
    "what would",
    "return only",
    "naming brief",
    "banned words",
    "one sentence",
    "i need",
    "think in",
    "your answer",
    "format",
    "name song",
    "single-word",
    "packign",
)
INVALID_EXACT = {
    "affect",
    "alright",
    "another name",
    "assistant",
    "brief",
    "etc",
    "group",
    "hmm",
    "name",
    "none",
    "null",
    "only",
    "primary",
    "product",
    "string",
    "taste",
    "tasteful",
    "your name",
}
VOWELS = set("aeiou")
CONTROL_TOKEN_RE = re.compile(r"<\|[^|]+\|>|</?think>", re.IGNORECASE)


def load_scorer(path: str | Path) -> dict[str, Any]:
    model = json.loads(Path(path).read_text())
    if model.get("kind") != "taste-linear-v1":
        raise ValueError(f"Unsupported scorer kind: {model.get('kind')}")
    if not model.get("ready"):
        raise ValueError("The frozen taste scorer is not marked ready.")
    return model


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z0-9]+", text.lower())


def score_candidate(candidate: dict[str, Any], model: dict[str, Any]) -> dict[str, float]:
    features = candidate_features(candidate, model)
    raw = float(model.get("bias", 0.0)) + sum(
        float(model["weights"].get(key, 0.0)) * value for key, value in features.items()
    )
    return {"score": math.tanh(raw), "raw": raw}


def candidate_features(candidate: dict[str, Any], model: dict[str, Any]) -> dict[str, float]:
    features: dict[str, float] = {"bias": 1.0}
    tokens = tokenize(
        " ".join([str(candidate.get("name", "")), str(candidate.get("rationale", ""))])
    )
    counts = Counter(tokens)
    total = max(1, sum(counts.values()))
    for token in model.get("vocab", []):
        if counts.get(token):
            features[f"tok:{token}"] = (counts[token] / total) * float(
                model.get("idf", {}).get(token, 1.0)
            )

    name = str(candidate.get("name", ""))
    name_tokens = tokenize(name)
    lower_name = name.lower()
    brief = candidate.get("brief") or {}
    banned_tokens = {
        token for pattern in brief.get("banned_patterns", []) for token in tokenize(str(pattern))
    }
    brief_tokens = set(
        tokenize(" ".join([str(brief.get("domain", "")), str(brief.get("product", ""))]))
    )
    vowels = sum(1 for char in lower_name if char in "aeiou")
    letters = sum(1 for char in lower_name if char.isalpha())
    features.update(
        {
            "struct:char_len": min(len(name), 40) / 40.0,
            "struct:word_count": min(max(1, len(name_tokens)), 5) / 5.0,
            "struct:has_digit": float(any(char.isdigit() for char in name)),
            "struct:has_hyphen": float("-" in name),
            "struct:vowel_ratio": vowels / max(1, letters),
            "struct:generic_token_rate": overlap_rate(set(name_tokens), GENERIC_TOKENS),
            "struct:banned_token_rate": overlap_rate(set(name_tokens), banned_tokens),
            "brief:domain_overlap": overlap_rate(set(name_tokens), brief_tokens),
        }
    )
    return features


def overlap_rate(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left.intersection(right)) / len(left)


def clean_candidate_name(text: str) -> str:
    raw = strip_generation_artifacts(text).strip()
    if not raw:
        return ""
    if raw.lower().startswith("<think>") and "</think>" not in raw.lower():
        return ""
    raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
    if not raw:
        return ""

    candidates: list[str] = []
    label_match = re.search(r"(?:^|\n)[ \t]*(?:name|answer)[ \t]*:[ \t]*(.+)", raw, re.IGNORECASE)
    if label_match:
        candidates.extend(candidate_lines(label_match.group(1)))
    candidates.extend(
        re.findall(
            r"[\"'`\u201c\u201d\u2018\u2019]([A-Za-z][A-Za-z0-9 '&.-]{1,40})"
            r"[\"'`\u201c\u201d\u2018\u2019]",
            raw,
        )
    )
    candidates.extend(
        re.findall(r"(?:^|\n)\s*(?:[A-D][).]|\d+[).]|[-*])\s*([A-Za-z][^\n]{1,40})", raw)
    )
    candidates.extend(candidate_lines(raw))
    for candidate in candidates:
        normalized = normalize_candidate_name(candidate)
        if normalized and is_valid_candidate_name(normalized):
            return normalized
    return ""


def normalize_candidate_name(text: str) -> str:
    cleaned = strip_generation_artifacts(text).strip()
    cleaned = re.sub(r"^(?:name|answer|option)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:[A-D]|\d+)[).]\s*", "", cleaned)
    cleaned = cleaned.strip("_*`\"'\u201c\u201d\u2018\u2019 .,:;")
    cleaned = re.sub(r"\([^)]*\)", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return re.split(r"\s+-\s+|\s+\u2014\s+|\s+\u2013\s+", cleaned, maxsplit=1)[0].strip()[:48]


def candidate_lines(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            lines.extend(piece for piece in re.split(r"\s{2,}|\t", stripped) if piece.strip())
    return lines or ([text.strip()] if text.strip() else [])


def strip_generation_artifacts(text: str) -> str:
    return CONTROL_TOKEN_RE.sub("", text).strip()


def is_valid_candidate_name(name: str) -> bool:
    stripped = name.strip()
    lower = stripped.lower()
    compact_lower = re.sub(r"[^a-z0-9]", "", lower)
    words = re.findall(r"[A-Za-z][A-Za-z0-9']*", stripped)
    alpha_count = sum(char.isalpha() for char in stripped)
    if not stripped or len(stripped) > 36 or not stripped.isascii() or stripped[:1].isdigit():
        return False
    if is_low_information_name(stripped) or lower in INVALID_EXACT:
        return False
    if any(
        len(compact_lower) > len(stem) + 2
        and (compact_lower.startswith(stem) or compact_lower.endswith(stem))
        for stem in GENERIC_NAME_STEMS
    ):
        return False
    if stripped.isupper() and alpha_count >= 3:
        return False
    uppercase_letters = sum(char.isupper() for char in stripped if char.isalpha())
    if uppercase_letters >= 3 and not stripped.istitle():
        return False
    if not 1 <= len(words) <= 4:
        return False
    if len(words) > 1 and stripped == lower:
        return False
    if len(words) > 1 and any(
        word[:1].islower() and word.lower() not in {"and", "of"} for word in words
    ):
        return False
    if any(lower.startswith(prefix) for prefix in INVALID_PREFIXES):
        return False
    if any(phrase in lower for phrase in INVALID_PHRASES):
        return False
    if re.search(r"[^A-Za-z0-9 '&.-]", stripped) or re.search(r"[\[\]{}\/?:;|_*+<>]", stripped):
        return False
    if stripped.endswith((".", ",")):
        return False
    return alpha_count >= 3


def is_low_information_name(name: str) -> bool:
    compact = re.sub(r"[^A-Za-z]", "", name).lower()
    if len(compact) < 3:
        return True
    vowels = sum(1 for char in compact if char in VOWELS)
    consonants = sum(1 for char in compact if char.isalpha() and char not in VOWELS)
    vowel_ratio = vowels / max(1, len(compact))
    if len(compact) <= 5 and len(set(compact)) <= 2:
        return True
    if len(compact) <= 6 and vowel_ratio > 0.65:
        return True
    return bool(len(compact) <= 4 and (consonants == 0 or (vowel_ratio > 0.75 and consonants <= 1)))


def reward_group(
    candidates: list[dict[str, Any]], model: dict[str, Any], config: dict[str, Any]
) -> list[dict[str, Any]]:
    names = [str(candidate.get("name", "")) for candidate in candidates]
    return [reward_candidate(candidate, model, config, names) for candidate in candidates]


def reward_candidate(
    candidate: dict[str, Any],
    model: dict[str, Any],
    config: dict[str, Any],
    group_names: list[str] | None = None,
) -> dict[str, Any]:
    scored = score_candidate(candidate, model)
    uncapped_learned = float(scored["score"])
    aux = auxiliary_score(candidate, config, group_names)
    learned = uncapped_learned
    if aux["hard_guardrail"] > 0.0:
        learned = min(learned, float(config.get("guardrail_scorer_cap", -0.50)))
    reward = (
        float(config.get("taste_weight", 0.85)) * learned
        + float(config.get("aux_weight", 0.15)) * aux["aux_total"]
    )
    return {
        "reward": reward,
        "learned_taste": learned,
        "uncapped_learned_taste": uncapped_learned,
        "aux_total": aux["aux_total"],
        "components": {**aux, "scorer_raw": scored["raw"]},
    }


def auxiliary_score(
    candidate: dict[str, Any],
    config: dict[str, Any],
    group_names: list[str] | None = None,
) -> dict[str, float]:
    name = str(candidate.get("name", ""))
    tokens = set(tokenize(name))
    compact = re.sub(r"[^A-Za-z]", "", name)
    brief = candidate.get("brief") or {}
    banned_tokens = {
        token for pattern in brief.get("banned_patterns", []) for token in tokenize(str(pattern))
    }
    brief_tokens = set(
        tokenize(" ".join([str(brief.get("domain", "")), str(brief.get("product", ""))]))
    )
    low_information = is_low_information_name(name)
    unreadable = is_unreadable(name)
    valid_name = is_valid_candidate_name(name)
    exact_pressure = duplicate_rate(name, group_names) if group_names else 0.0
    family_pressure = family_duplicate_rate(name, group_names) if group_names else 0.0
    copy_pressure = literal_copy_pressure(name, brief)

    values = {
        "banned_penalty": -float(config.get("banned_penalty", 0.25))
        * bool(tokens.intersection(banned_tokens)),
        "unreadable_penalty": -float(config.get("unreadable_penalty", 0.20)) * unreadable,
        "low_information_penalty": -float(config.get("low_information_penalty", 0.75))
        * low_information,
        "short_name_penalty": -float(config.get("short_name_penalty", 0.20))
        * (0 < len(compact) < 4),
        "genericity_penalty": -float(config.get("genericity_penalty", 0.20))
        * min(1.0, len(tokens.intersection(GENERIC_TOKENS)) / max(1, len(tokens))),
        "invalid_name_penalty": -float(config.get("invalid_name_penalty", 0.35)) * (not valid_name),
        "literal_copy_penalty": -float(config.get("literal_copy_penalty", 0.65)) * copy_pressure,
        "brief_adherence_bonus": float(config.get("brief_adherence_bonus", 0.08))
        * min(1.0, len(tokens.intersection(brief_tokens)) / 2.0),
        "diversity_bonus": 0.0,
        "duplicate_penalty": 0.0,
        "family_duplicate_penalty": 0.0,
    }
    if group_names:
        values["diversity_bonus"] = float(config.get("diversity_bonus", 0.08)) * (
            1.0 - max(exact_pressure, family_pressure)
        )
        values["duplicate_penalty"] = -float(config.get("duplicate_penalty", 0.35)) * exact_pressure
        values["family_duplicate_penalty"] = (
            -float(config.get("family_duplicate_penalty", 0.65)) * family_pressure
        )
    hard_guardrail = float(
        (not valid_name)
        or low_information
        or unreadable
        or copy_pressure >= float(config.get("literal_copy_guardrail_threshold", 0.50))
        or family_pressure >= float(config.get("family_duplicate_guardrail_threshold", 0.50))
    )
    aux_total = max(-1.0, min(1.0, sum(values.values())))
    return {
        "aux_total": aux_total,
        **{key: float(value) for key, value in values.items()},
        "literal_copy_pressure": copy_pressure,
        "family_duplicate_pressure": family_pressure,
        "hard_guardrail": hard_guardrail,
    }


def is_unreadable(name: str) -> bool:
    compact = re.sub(r"[^A-Za-z]", "", name)
    if is_low_information_name(name) or len(compact) > 24:
        return True
    if re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", compact.lower()):
        return True
    return bool(re.search(r"(.)\1\1", compact.lower()))


def duplicate_rate(name: str, names: list[str]) -> float:
    normalized = name.strip().lower()
    if not normalized:
        return 1.0
    matches = sum(1 for item in names if item.strip().lower() == normalized)
    return max(0.0, (matches - 1) / max(1, len(names) - 1))


def family_duplicate_rate(name: str, names: list[str]) -> float:
    family = name_family(name)
    if not family:
        return 1.0
    matches = sum(1 for item in names if name_family(item) == family)
    return max(0.0, (matches - 1) / max(1, len(names) - 1))


def literal_copy_pressure(name: str, brief: dict[str, Any]) -> float:
    compact = re.sub(r"[^A-Za-z0-9]", "", name).lower()
    if not compact:
        return 1.0
    content_text = " ".join(
        [
            str(brief.get("domain", "")),
            str(brief.get("audience", "")),
            str(brief.get("product", "")),
        ]
    )
    banned_text = " ".join(str(item) for item in brief.get("banned_patterns", []))
    content_tokens = {
        token for token in tokenize(content_text) if len(token) >= 4 and token not in GENERIC_TOKENS
    }
    banned_tokens = {token for token in tokenize(banned_text) if len(token) >= 3}
    source_tokens = content_tokens | banned_tokens
    if not source_tokens:
        return 0.0
    name_tokens = tokenize(name)
    matched = 0
    for token in source_tokens:
        if token in compact or any(
            token in name_token or name_token in token for name_token in name_tokens
        ):
            matched += 1
    return min(1.0, matched / max(1, len(name_tokens) or 1))


def name_family(name: str) -> str:
    compact = re.sub(r"[^A-Za-z]", "", name).lower()
    if len(compact) < 4:
        return compact
    for suffix in ("line", "ly", "y", "a", "i", "o", "up"):
        if len(compact) > 5 and compact.endswith(suffix):
            compact = compact[: -len(suffix)]
            break
    return compact[:3]


def brief_to_grpo_prompt(brief: dict[str, Any]) -> str:
    source = " ".join(
        [
            str(brief.get("domain", "")),
            str(brief.get("audience", "")),
            str(brief.get("product", "")),
            " ".join(str(item) for item in brief.get("banned_patterns", [])),
        ]
    ).lower()
    source_tokens = {token for token in re.findall(r"[a-z][a-z0-9]{2,}", source) if len(token) >= 4}
    safe_vibes = []
    for vibe in brief.get("vibes", []):
        vibe_text = str(vibe).strip()
        vibe_tokens = re.findall(r"[a-z][a-z0-9]{2,}", vibe_text.lower())
        if vibe_tokens and not any(
            token in source
            or any(token in source_token or source_token in token for source_token in source_tokens)
            for token in vibe_tokens
        ):
            safe_vibes.append(vibe_text)
    safe_vibes = safe_vibes or ["memorable", "concrete", "concise"]
    return (
        "Invent exactly one tasteful project/domain name.\n"
        "Output only the name. No explanation. No list. No colon. No quotes.\n"
        "Use 1-3 words. Avoid generic startup words and literal category words.\n"
        "Prefer metaphor, texture, motion, or coined sound.\n\n"
        f"Style: {', '.join(safe_vibes)}\n"
        f"Constraints: {', '.join(brief.get('constraints', []))}\n"
        "Name:"
    )


def naming_prompt(prompt: str) -> str:
    return (
        "Invent exactly one tasteful project/domain name for the prompt below.\n"
        "Output a single word only. No explanation. No punctuation. No quotes.\n"
        "Do not copy words from the prompt. Prefer a concrete, memorable, non-generic name.\n\n"
        f"Prompt: {prompt.strip()}\n"
        "Name:"
    )


def render_no_think_prompt(tokenizer: Any, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None) is None:
        return prompt
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )


def copies_prompt_word(name: str, prompt: str) -> bool:
    compact_name = re.sub(r"[^A-Za-z0-9]", "", name).lower()
    if len(compact_name) < 3:
        return False
    stopwords = {
        "name",
        "project",
        "domain",
        "tool",
        "model",
        "first",
        "with",
        "from",
        "that",
        "this",
        "app",
        "used",
        "which",
        "small",
    }
    prompt_tokens = {
        token
        for token in (raw.lower() for raw in re.findall(r"[A-Za-z][A-Za-z0-9]{2,}", prompt))
        if token not in stopwords
    }
    return any(token in compact_name or compact_name in token for token in prompt_tokens)
