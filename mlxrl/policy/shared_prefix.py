"""Differentiable shared-prefix completion logprobs.

Rows that share a prompt run the prompt once. Every completion row attends to
the prompt's keys and values, and gradients flow back into the prompt pass.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxrl.policy.logprobs import CompletionLogprobs, target_logprobs_from_logits


class _PrefixKV:
    """Functional KV 'cache': no in-place writes, so autograd sees everything."""

    def __init__(self, prefix_keys: mx.array | None, prefix_values: mx.array | None) -> None:
        self.prefix_keys = prefix_keys
        self.prefix_values = prefix_values
        self.offset = 0 if prefix_keys is None else prefix_keys.shape[2]
        self.keys: mx.array | None = None
        self.values: mx.array | None = None

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        self.keys, self.values = keys, values
        if self.prefix_keys is None:
            return keys, values
        batch = keys.shape[0]
        prefix_keys = mx.broadcast_to(self.prefix_keys, (batch, *self.prefix_keys.shape[1:]))
        prefix_values = mx.broadcast_to(self.prefix_values, (batch, *self.prefix_values.shape[1:]))
        return (
            mx.concatenate([prefix_keys, keys], axis=2),
            mx.concatenate([prefix_values, values], axis=2),
        )


def _stock_model_classes() -> tuple[type, ...]:
    classes: list[type] = []
    for name in ("qwen2", "qwen3", "llama"):
        try:
            module = importlib.import_module(f"mlx_lm.models.{name}")
        except ImportError:
            continue
        classes.append(module.Model)
    return tuple(classes)


def supports_shared_prefix(model: nn.Module) -> bool:
    """True only for the exact stock mlx-lm Qwen2/Qwen3/Llama ``Model`` classes.

    The shared-prefix pass calls embeddings, layers, norm and head directly, so any
    subclass or architecture that changes ``__call__`` (e.g. embedding or logit
    scaling) would be silently bypassed.
    """

    from mlx_lm.models.cache import KVCache, make_prompt_cache

    if type(model) not in _stock_model_classes():
        return False
    inner = getattr(model, "model", None)
    if inner is None or not all(hasattr(inner, n) for n in ("embed_tokens", "layers", "norm")):
        return False
    return all(type(c) is KVCache for c in make_prompt_cache(model))


def _original_call(layer: nn.Module) -> Any:
    call = type(layer).__call__
    if call.__name__ == "checkpointed_fn":  # mlx_lm.tuner.trainer.grad_checkpoint patch
        for cell in call.__closure__ or ():
            if callable(cell.cell_contents) and cell.cell_contents.__name__ == "__call__":
                return cell.cell_contents
    return call


def _run_layer(
    layer: nn.Module,
    x: mx.array,
    prefix_keys: mx.array | None,
    prefix_values: mx.array | None,
    checkpoint: bool,
) -> tuple[mx.array, mx.array, mx.array]:
    call = _original_call(layer)
    has_prefix = prefix_keys is not None

    def fn(params: Any, x: mx.array, *prefix: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        if checkpoint:
            layer.update(params)
        cache = _PrefixKV(*prefix) if has_prefix else _PrefixKV(None, None)
        out = call(layer, x, "causal" if x.shape[1] > 1 else None, cache)
        return out, cache.keys, cache.values

    args = (prefix_keys, prefix_values) if has_prefix else ()
    if checkpoint:
        return mx.checkpoint(fn)(layer.trainable_parameters(), x, *args)
    return fn(None, x, *args)


def _logits(model: nn.Module, h: mx.array) -> mx.array:
    h = model.model.norm(h)
    if hasattr(model, "lm_head"):
        return model.lm_head(h)
    return model.model.embed_tokens.as_linear(h)


def _group_logprobs(
    model: nn.Module,
    prompt: Sequence[int],
    completions: Sequence[Sequence[int]],
    pad_token_id: int,
    checkpoint: bool,
) -> mx.array:
    """Logprobs [n, max_len] of each completion, prompt computed once."""

    layers = model.model.layers
    width = max(len(c) for c in completions)
    prefix_kv: list[tuple[mx.array | None, mx.array | None]] = [(None, None)] * len(layers)
    if len(prompt) > 1:
        h = model.model.embed_tokens(mx.array([list(prompt[:-1])], dtype=mx.int32))
        prefix_kv = []
        for layer in layers:
            h, k, v = _run_layer(layer, h, None, None, checkpoint)
            prefix_kv.append((k, v))
    chunk = mx.array(
        [[prompt[-1]] + list(c[:-1]) + [pad_token_id] * (width - len(c)) for c in completions],
        dtype=mx.int32,
    )
    targets = mx.array(
        [list(c) + [pad_token_id] * (width - len(c)) for c in completions],
        dtype=mx.int32,
    )
    h = model.model.embed_tokens(chunk)
    for layer, (pk, pv) in zip(layers, prefix_kv, strict=True):
        h, _, _ = _run_layer(layer, h, pk, pv, checkpoint)
    logprobs = target_logprobs_from_logits(_logits(model, h), targets)
    mask = mx.array(
        [[1.0] * len(c) + [0.0] * (width - len(c)) for c in completions], dtype=logprobs.dtype
    )
    return logprobs * mask


def shared_prefix_completion_logprobs(
    model: nn.Module,
    prompt_token_ids: Sequence[Sequence[int]],
    completion_token_ids: Sequence[Sequence[int]],
    pad_token_id: int = 0,
    use_checkpoint: bool = False,
) -> CompletionLogprobs:
    """Drop-in for ``completion_logprobs`` that computes each unique prompt once.

    Raises ``ValueError`` for models other than the stock mlx-lm Qwen2/Qwen3/Llama
    classes (see ``supports_shared_prefix``); callers opt in explicitly, so an
    unsupported model is an error rather than a silent fallback.
    """

    if not supports_shared_prefix(model):
        raise ValueError(
            "shared_prefix=True supports only stock mlx-lm Qwen2/Qwen3/Llama models; "
            f"got {type(model).__name__}."
        )
    if len(prompt_token_ids) != len(completion_token_ids):
        raise ValueError("prompt_token_ids and completion_token_ids must have the same length.")
    if not prompt_token_ids:
        raise ValueError("At least one sequence is required.")
    if any(len(p) < 1 for p in prompt_token_ids) or any(len(c) < 1 for c in completion_token_ids):
        raise ValueError("Each row needs a non-empty prompt and completion.")
    groups: dict[tuple[int, ...], list[int]] = {}
    for index, prompt in enumerate(prompt_token_ids):
        groups.setdefault(tuple(prompt), []).append(index)
    width = max(len(c) for c in completion_token_ids)
    rows: list[mx.array | None] = [None] * len(prompt_token_ids)
    for prompt, indices in groups.items():
        out = _group_logprobs(
            model, prompt, [completion_token_ids[i] for i in indices], pad_token_id, use_checkpoint
        )
        if out.shape[1] < width:
            out = mx.pad(out, [(0, 0), (0, width - out.shape[1])])
        for row, index in enumerate(indices):
            rows[index] = out[row]
    mask = mx.array(
        [[1.0] * len(c) + [0.0] * (width - len(c)) for c in completion_token_ids],
        dtype=mx.float32,
    )
    return CompletionLogprobs(logprobs=mx.stack(rows, axis=0), mask=mask)  # type: ignore[arg-type]
