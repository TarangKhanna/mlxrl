"""Equivalence of the projected completion head against the full-logit forward.

Rule under test: the log-probability of a token is a function of the logits at
its predicting position only, so applying the output head to just the gathered
completion positions must give the same logprobs, loss, LoRA gradients and
optimizer update as applying it to every position and gathering afterwards.
The reference ("full") path is the unmodified code path, selected by making
``_completion_head`` report that the model is not eligible.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest
from mlx.utils import tree_flatten

import mlxrl.policy.logprobs as logprobs_module
from mlxrl.algo.grpo import GRPOAlgorithm
from mlxrl.policy.logprobs import completion_logprobs_from_inputs, prepare_completion_logprob_inputs
from mlxrl.policy.model import LoRAConfig, inject_lora_adapters
from mlxrl.rollout.naive import Completion
from mlxrl.train.grpo import batch_from_rollouts, grpo_metrics_from_batch, optimizer_step

pytestmark = pytest.mark.metal

VOCAB = 64
# Mixed prompt lengths and mixed completion lengths: exercises right padding and masking.
PROMPTS = ((3, 4, 5, 6, 7, 8, 9), (10, 11, 12), (13, 14, 15, 16, 17))
COMPLETIONS = ((20, 21, 22, 23), (24,), (25, 26))


def _tiny_model(arch: str, tie: bool, dtype: mx.Dtype, quantize: bool) -> nn.Module:
    mx.random.seed(1234)
    if arch == "qwen3":
        from mlx_lm.models.qwen3 import Model, ModelArgs

        args: Any = ModelArgs(
            model_type="qwen3",
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=VOCAB,
            num_key_value_heads=2,
            max_position_embeddings=128,
            rope_theta=10000.0,
            head_dim=8,
            tie_word_embeddings=tie,
        )
    else:
        from mlx_lm.models.llama import Model, ModelArgs

        args = ModelArgs(
            model_type="llama",
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=VOCAB,
            num_key_value_heads=2,
            max_position_embeddings=128,
            tie_word_embeddings=tie,
        )
    model = Model(args)
    model.set_dtype(dtype)
    if quantize:
        nn.quantize(model, group_size=32, bits=4)
    inject_lora_adapters(model, LoRAConfig(rank=4, scale=2.0))
    # LoRA B starts at zero, which makes every A-gradient zero; randomize so all grads are live.
    for name, param in tree_flatten(model.trainable_parameters()):
        if name.endswith("lora_b"):
            _set(model, name, (mx.random.normal(param.shape) * 0.05).astype(param.dtype))
    mx.eval(model.parameters())  # Test sync: materialize weights.
    return model


def _set(model: nn.Module, dotted: str, value: mx.array) -> None:
    obj: Any = model
    parts = dotted.split(".")
    for part in parts[:-1]:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    setattr(obj, parts[-1], value)


def _completions() -> list[Completion]:
    return [
        Completion(
            prompt_index=0,
            group_index=i,
            prompt_tokens=p,
            completion_tokens=c,
            old_policy_logprobs=(),
            text="",
        )
        for i, (p, c) in enumerate(zip(PROMPTS, COMPLETIONS, strict=True))
    ]


class _Full:
    """Context manager forcing the unmodified full-logit path."""

    def __enter__(self) -> None:
        self.orig = logprobs_module._completion_head
        logprobs_module._completion_head = lambda model: None  # type: ignore[assignment]

    def __exit__(self, *exc: object) -> None:
        logprobs_module._completion_head = self.orig  # type: ignore[assignment]


def _max_diff(a: mx.array, b: mx.array) -> float:
    d = mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)))
    mx.eval(d)  # Test sync.
    return float(d.item())


def _snapshot(model: nn.Module) -> dict[str, mx.array]:
    out = {k: mx.array(v) for k, v in tree_flatten(model.trainable_parameters())}
    mx.eval(*out.values())  # Test sync.
    return out


def _restore(model: nn.Module, snap: dict[str, mx.array]) -> None:
    for k, v in snap.items():
        _set(model, k, mx.array(v))
    mx.eval(model.parameters())  # Test sync.


def _run(model: nn.Module, full: bool) -> dict[str, Any]:
    """logprobs, loss, all LoRA grads, and params after one Adam / one SGD step; one path."""
    snap = _snapshot(model)
    algo = GRPOAlgorithm()
    rewards = [1.0, 0.0, 0.5]
    ctx = _Full() if full else None
    if ctx:
        ctx.__enter__()
    try:
        inputs = prepare_completion_logprob_inputs(PROMPTS, COMPLETIONS)
        lp = completion_logprobs_from_inputs(model, inputs).logprobs
        mx.eval(lp)  # Test sync.
        batch = batch_from_rollouts(model, _completions(), rewards, 3, 0, algo)

        def loss_fn(m: nn.Module) -> mx.array:
            return grpo_metrics_from_batch(m, batch, 0.05, 0, algo).loss

        model.train()
        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        flat = dict(tree_flatten(grads))
        mx.eval(loss, *flat.values())  # Test sync.
        model.eval()
        after = {}
        for name, make_opt in (
            ("adam", lambda: optim.Adam(learning_rate=1e-2)),
            ("sgd", lambda: optim.SGD(learning_rate=1e-2)),
        ):
            _restore(model, snap)
            optimizer_step(model, make_opt(), batch, 0.05, 0, algo)
            after[name] = _snapshot(model)
    finally:
        if ctx:
            ctx.__exit__()
    _restore(model, snap)
    return {"lp": lp, "old": batch.old_policy_logprobs, "loss": loss, "grads": flat, "after": after}


CASES = [
    (arch, tie, dtype, quant)
    for arch in ("qwen3", "llama")
    for tie in (True, False)
    for dtype in (mx.float32, mx.bfloat16)
    for quant in (False, True)
]


@pytest.mark.parametrize(("arch", "tie", "dtype", "quantize"), CASES)
def test_projected_head_matches_full_logits(
    arch: str, tie: bool, dtype: mx.Dtype, quantize: bool
) -> None:
    model = _tiny_model(arch, tie, dtype, quantize)
    assert logprobs_module._completion_head(model) is not None  # projection actually engaged
    proj = _run(model, full=False)
    full = _run(model, full=True)

    d_lp = _max_diff(proj["lp"], full["lp"])
    d_old = _max_diff(proj["old"], full["old"])
    d_loss = _max_diff(proj["loss"], full["loss"])
    d_grad = max(_max_diff(proj["grads"][k], full["grads"][k]) for k in full["grads"])
    d_adam = max(
        _max_diff(proj["after"]["adam"][k], full["after"]["adam"][k]) for k in full["grads"]
    )
    d_sgd = max(_max_diff(proj["after"]["sgd"][k], full["after"]["sgd"][k]) for k in full["grads"])
    print(
        f"[{arch} tie={tie} {dtype} q={quantize}] lp={d_lp:.3g} old={d_old:.3g} "
        f"loss={d_loss:.3g} grad={d_grad:.3g} adam={d_adam:.3g} sgd={d_sgd:.3g}"
    )

    # Every LoRA gradient is live (a check that cannot fail on all-zero grads is not a check).
    assert len(full["grads"]) > 0
    assert all(float(mx.max(mx.abs(g)).item()) > 0.0 for g in full["grads"].values())

    if dtype == mx.float32:
        tol = 1e-6  # zero difference expected; only reduction-order noise would show here
        grad_tol = 1e-6
    else:
        # bf16 noise floor: one bf16 ulp at logprob magnitude ~4 is 2^-5 = 0.03125.
        tol = 0.0625
        grad_tol = 0.05
    assert d_lp <= tol
    assert d_old <= tol
    assert d_loss <= tol
    assert d_grad <= grad_tol
    if dtype == mx.float32:
        # SGD update is linear in the gradient: diff <= lr * grad diff. Adam normalizes by
        # sqrt(v)+eps, so a ~1e-9 gradient difference on an entry with |g| ~ 1e-7 moves that
        # entry by up to ~1% of lr; bound it at 1e-4 (= 1% of lr = 1e-2).
        assert d_sgd <= 1e-7
        assert d_adam <= 1e-4


def test_unlisted_model_type_uses_full_forward() -> None:
    """Models not known to be 'backbone + plain head' must not take the projection path."""
    model = _tiny_model("qwen3", True, mx.float32, False)
    model.args.model_type = "gemma3"
    assert logprobs_module._completion_head(model) is None
    inputs = prepare_completion_logprob_inputs(PROMPTS, COMPLETIONS)
    got = completion_logprobs_from_inputs(model, inputs).logprobs
    logits = model(inputs.input_ids)
    ref = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    ref = mx.squeeze(mx.take_along_axis(ref, inputs.target_ids[..., None], axis=-1), axis=-1)
    ref = mx.take_along_axis(ref, inputs.gather_indices, axis=1) * inputs.mask
    assert _max_diff(got, ref) <= 1e-6


def test_projection_matches_independent_log_softmax() -> None:
    """Independent oracle: logprob = log_softmax(model(tokens))[pos, next token], per sequence."""
    model = _tiny_model("llama", False, mx.float32, False)
    inputs = prepare_completion_logprob_inputs(PROMPTS, COMPLETIONS)
    got = completion_logprobs_from_inputs(model, inputs).logprobs
    for i, (p, c) in enumerate(zip(PROMPTS, COMPLETIONS, strict=True)):
        seq = mx.array([list(p + c)])  # unpadded, single sequence
        logits = model(seq[:, :-1])
        lsm = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        for j, tok in enumerate(c):
            expect = lsm[0, len(p) - 1 + j, tok]
            assert abs(float(got[i, j].item()) - float(expect.item())) <= 1e-5
        assert float(mx.sum(mx.abs(got[i, len(c) :])).item()) == 0.0  # padded slots are masked
