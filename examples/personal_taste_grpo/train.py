"""Run the frozen personal-taste naming reward through MLXRL GRPO."""

from __future__ import annotations

import argparse
import json
import tomllib
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from taste_reward import (
    brief_to_grpo_prompt,
    clean_candidate_name,
    literal_copy_pressure,
    load_scorer,
    name_family,
    render_no_think_prompt,
    reward_group,
)

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.toml")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def resolve_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_path.parent / path


def load_briefs(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text())
    briefs = []
    for row in rows:
        brief = row["brief"]
        if isinstance(brief, str):
            brief = json.loads(brief)
        briefs.append({"id": str(row["id"]), "brief": brief})
    if not briefs:
        raise ValueError(f"No briefs found in {path}")
    return briefs


def collect_rollout_group(
    *,
    model: Any,
    tokenizer: Any,
    generate_group_rollouts: Any,
    briefs: list[dict[str, Any]],
    step: int,
    run_config: dict[str, Any],
    sampling: Any,
    scorer: dict[str, Any],
    reward_config: dict[str, Any],
) -> dict[str, Any]:
    group_size = int(run_config["group_size"])
    max_rollout_attempts = int(run_config["rollout_attempts_per_step"])
    max_brief_attempts = min(len(briefs), int(run_config["brief_attempts_per_step"]))
    max_family_count = int(run_config["max_family_count"])
    literal_threshold = float(reward_config["literal_copy_guardrail_threshold"])
    seed = int(run_config["seed"])
    skipped = []

    for brief_offset in range(max_brief_attempts):
        item = briefs[(step - 1 + brief_offset) % len(briefs)]
        brief = item["brief"]
        prompt = render_no_think_prompt(tokenizer, brief_to_grpo_prompt(brief))
        completions = []
        candidates = []
        families: Counter[str] = Counter()
        rejects: Counter[str] = Counter()
        for attempt in range(max_rollout_attempts):
            batch = generate_group_rollouts(
                model=model,
                tokenizer=tokenizer,
                prompts=[prompt],
                group_size=group_size,
                config=sampling,
                seed=seed + step * 10_000 + brief_offset * 100 + attempt,
                use_chat_template=False,
            )
            for completion in batch:
                name = clean_candidate_name(completion.text)
                if not name:
                    rejects["empty_or_invalid"] += 1
                    continue
                if literal_copy_pressure(name, brief) >= literal_threshold:
                    rejects["literal_copy"] += 1
                    continue
                family = name_family(name)
                if family and families[family] >= max_family_count:
                    rejects["family_cap"] += 1
                    continue
                candidate = {
                    "id": f"grpo-{step}-{len(candidates)}",
                    "brief_id": item["id"],
                    "name": name,
                    "brief": brief,
                }
                diagnostic = reward_group([candidate], scorer, reward_config)[0]
                if diagnostic["components"]["hard_guardrail"] > 0.0:
                    rejects["hard_guardrail"] += 1
                    continue
                completions.append(completion)
                candidates.append(candidate)
                if family:
                    families[family] += 1
                if len(candidates) == group_size:
                    return {
                        "brief": item,
                        "completions": tuple(completions),
                        "candidates": candidates,
                        "rollout_attempts": attempt + 1,
                        "brief_attempts": brief_offset + 1,
                        "reject_counts": dict(rejects),
                    }
            rejects["rollout_batches"] += 1
        skipped.append(
            {"brief_id": item["id"], "accepted": len(candidates), "rejects": dict(rejects)}
        )
    raise RuntimeError(
        f"Could not collect a valid, diverse group at step {step}; skipped={skipped[:5]}"
    )


def save_adapter(mx: Any, tree_flatten: Any, model: Any, path: Path) -> None:
    arrays = {
        str(key).replace(".", "__"): value
        for key, value in tree_flatten(model.trainable_parameters())
    }
    mx.savez(str(path), **arrays)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    run_config = dict(config["run"])
    if args.steps is not None:
        run_config["steps"] = args.steps
    if args.model:
        run_config["model_id"] = args.model

    data_config = config["data"]
    briefs = load_briefs(resolve_path(config_path, data_config["briefs"]))
    scorer = load_scorer(resolve_path(config_path, data_config["scorer"]))
    reward_config = dict(config["reward"])

    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    from mlxrl.algo import GRPOAlgorithm
    from mlxrl.policy import LoRAConfig, load_policy_with_lora
    from mlxrl.policy.logprobs import pad_token_id_from_tokenizer
    from mlxrl.rollout.naive import SamplingConfig, generate_group_rollouts
    from mlxrl.train import batch_from_rollouts, optimizer_step

    model, tokenizer, report = load_policy_with_lora(
        model_id=str(run_config["model_id"]),
        config=LoRAConfig(
            rank=int(run_config["rank"]),
            scale=float(run_config["scale"]),
            dropout=float(run_config["dropout"]),
            grad_checkpoint=bool(run_config["gradient_checkpointing"]),
        ),
    )
    optimizer = optim.Adam(learning_rate=float(config["optimizer"]["learning_rate"]))
    algorithm = GRPOAlgorithm()
    sampling = SamplingConfig(
        max_tokens=int(run_config["max_completion_len"]),
        temperature=float(config["sampling"]["temperature"]),
        top_p=float(config["sampling"]["top_p"]),
        min_p=float(config["sampling"]["min_p"]),
        top_k=int(config["sampling"]["top_k"]),
    )
    pad_token_id = int(pad_token_id_from_tokenizer(tokenizer))
    steps = int(run_config["steps"])
    output_root = args.output_dir or Path(str(run_config["output_dir"]))
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root
    run_id = datetime.now(UTC).strftime("taste-%Y%m%d-%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = run_dir / "train.jsonl"
    rewards_history = []

    with metrics_path.open("w") as metrics_file:
        for step in range(1, steps + 1):
            rollout = collect_rollout_group(
                model=model,
                tokenizer=tokenizer,
                generate_group_rollouts=generate_group_rollouts,
                briefs=briefs,
                step=step,
                run_config=run_config,
                sampling=sampling,
                scorer=scorer,
                reward_config=reward_config,
            )
            candidates = rollout["candidates"]
            rewards = [row["reward"] for row in reward_group(candidates, scorer, reward_config)]
            family_counts = Counter(name_family(row["name"]) for row in candidates)
            family_counts.pop("", None)
            top_family, top_family_count = (
                family_counts.most_common(1)[0] if family_counts else ("", 0)
            )
            batch = batch_from_rollouts(
                model=model,
                completions=rollout["completions"],
                rewards=rewards,
                group_size=int(run_config["group_size"]),
                pad_token_id=pad_token_id,
                algorithm=algorithm,
                use_checkpoint=bool(run_config["gradient_checkpointing"]),
                compute_reference=float(run_config["kl_beta"]) != 0.0,
            )
            metrics = optimizer_step(
                model=model,
                optimizer=optimizer,
                batch=batch,
                beta=float(run_config["kl_beta"]),
                pad_token_id=pad_token_id,
                algorithm=algorithm,
                use_checkpoint=bool(run_config["gradient_checkpointing"]),
                micro_batch_size=int(run_config["micro_batch_size"]),
            )
            rewards_history.append(float(metrics.mean_reward))
            row = {
                "step": step,
                "mean_reward": float(metrics.mean_reward),
                "loss": float(metrics.loss),
                "kl": float(metrics.kl),
                "clip_fraction": float(metrics.clip_fraction),
                "samples": [candidate["name"] for candidate in candidates],
                "brief_id": rollout["brief"]["id"],
                "top_family": top_family,
                "top_family_count": top_family_count,
                "unique_family_count": len(family_counts),
                "rollout_attempts": rollout["rollout_attempts"],
                "brief_attempts": rollout["brief_attempts"],
                "reject_counts": rollout["reject_counts"],
            }
            line = json.dumps(row, sort_keys=True)
            metrics_file.write(line + "\n")
            metrics_file.flush()
            print(line, flush=True)
            mx.synchronize()
            if step % int(run_config["checkpoint_every"]) == 0:
                save_adapter(mx, tree_flatten, model, run_dir / f"adapter_step{step}.npz")

    adapter_path = run_dir / "adapter_final.npz"
    save_adapter(mx, tree_flatten, model, adapter_path)
    summary = {
        "run_id": run_id,
        "steps": steps,
        "mean_reward": sum(rewards_history) / len(rewards_history),
        "model_id": report.model_id,
        "trainable_params": report.trainable_params,
        "adapter": str(adapter_path),
        "metrics": str(metrics_path),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
