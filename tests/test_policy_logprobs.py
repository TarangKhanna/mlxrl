from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

import mlxrl.policy.logprobs as logprobs_module
from mlxrl.algo.grpo import GRPOAlgorithm
from mlxrl.policy.logprobs import (
    CompletionLogprobInputs,
    completion_logprobs,
    completion_logprobs_from_inputs,
    dual_logprobs,
    prefix_cached_completion_logprobs,
    prepare_completion_logprob_inputs,
    target_logprobs_from_logits,
)
from mlxrl.rollout.naive import Completion, SamplingConfig
from mlxrl.rollout.optimized import generate_from_prefix_cache, prefill_prompt_once
from mlxrl.train.grpo import batch_from_rollouts, old_policy_logprobs_from_rollouts

pytestmark = pytest.mark.metal


class ToyCache:
    def __init__(self, sums: mx.array | None = None) -> None:
        self.sums = sums if sums is not None else mx.zeros((1,), dtype=mx.int32)

    @property
    def state(self) -> tuple[mx.array]:
        return (self.sums,)

    @state.setter
    def state(self, state: Sequence[mx.array]) -> None:
        self.sums = state[0]

    @property
    def meta_state(self) -> tuple[()]:
        return ()

    @meta_state.setter
    def meta_state(self, meta_state: Sequence[str]) -> None:
        del meta_state

    @classmethod
    def from_state(
        cls,
        state: Sequence[mx.array],
        meta_state: Sequence[str],
    ) -> ToyCache:
        del meta_state
        return cls(state[0])

    @classmethod
    def merge(cls, caches: Sequence[ToyCache]) -> ToyCache:
        return cls(mx.concatenate([cache.sums for cache in caches], axis=0))


class ToyCachedModel(nn.Module):
    vocab_size = 16

    def __init__(self) -> None:
        super().__init__()
        self.cached_call_shapes: list[tuple[int, int]] = []

    def make_cache(self) -> list[ToyCache]:
        return [ToyCache()]

    def __call__(
        self,
        tokens: mx.array,
        cache: Sequence[ToyCache] | None = None,
    ) -> mx.array:
        batch_size, sequence_length = tokens.shape
        token_values = tokens.astype(mx.float32)
        base = (
            cache[0].sums.astype(mx.float32)[:, None]
            if cache is not None
            else mx.zeros((batch_size, 1), dtype=mx.float32)
        )
        cumulative = base + mx.cumsum(token_values, axis=1)
        if cache is not None:
            self.cached_call_shapes.append((batch_size, sequence_length))
            cache[0].sums = (base[:, 0] + mx.sum(tokens, axis=1)).astype(mx.int32)
        vocab = mx.arange(self.vocab_size, dtype=mx.float32)
        return cumulative[:, :, None] * (vocab[None, None, :] + 1.0) / 100.0


class ToyTokenizer:
    def decode(self, token_ids: Sequence[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


def test_target_logprobs_from_logits_matches_explicit_log_softmax_gather() -> None:
    logits = mx.array(
        [
            [[0.0, 1.0, -2.0], [2.0, -1.0, 0.5]],
            [[-0.5, 0.25, 1.25], [1.5, 0.0, -0.25]],
        ],
        dtype=mx.float32,
    )
    targets = mx.array([[1, 0], [2, 1]], dtype=mx.int32)

    explicit_logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    expected = mx.squeeze(
        mx.take_along_axis(explicit_logprobs, targets[..., None], axis=-1),
        axis=-1,
    )
    actual = target_logprobs_from_logits(logits, targets)
    error = mx.max(mx.abs(actual - expected))
    mx.eval(  # Test sync: materialize helper/reference logprob comparison.
        error,
    )

    assert float(error.item()) < 1e-6


def test_prefix_cached_completion_logprobs_match_full_forward_on_toy_model() -> None:
    model = ToyCachedModel()
    prompts = ((2, 3, 4), (2, 3, 4))
    completions = ((5, 6), (7,))

    full = completion_logprobs(model, prompts, completions, pad_token_id=0)
    cached = prefix_cached_completion_logprobs(model, prompts, completions, pad_token_id=0)
    max_error = mx.max(mx.abs(full.logprobs - cached.logprobs))
    mask_error = mx.max(mx.abs(full.mask - cached.mask))
    mx.eval(  # Test sync: materialize logprob/mask comparisons for assertions.
        max_error,
        mask_error,
    )

    assert float(max_error.item()) < 1e-6
    assert float(mask_error.item()) == 0.0
    assert model.cached_call_shapes == [(1, 2), (2, 2)]


def test_completion_logprobs_from_prepared_inputs_matches_public_helper() -> None:
    model = ToyCachedModel()
    prompts = ((2, 3, 4), (9, 8))
    completions = ((5, 6), (7,))

    prepared = prepare_completion_logprob_inputs(prompts, completions, pad_token_id=0)
    from_prepared = completion_logprobs_from_inputs(model, prepared)
    direct = completion_logprobs(model, prompts, completions, pad_token_id=0)
    max_error = mx.max(mx.abs(from_prepared.logprobs - direct.logprobs))
    mask_error = mx.max(mx.abs(from_prepared.mask - direct.mask))
    mx.eval(  # Test sync: materialize prepared/direct logprob comparison.
        max_error,
        mask_error,
    )

    assert float(max_error.item()) < 1e-6
    assert float(mask_error.item()) == 0.0


def test_rollout_captured_logprobs_match_completion_forward_on_toy_model() -> None:
    model = ToyCachedModel()
    tokenizer = ToyTokenizer()
    prompt = (2, 3, 4)
    prefix = prefill_prompt_once(model, prompt)

    completion_tokens, captured_logprobs, _ = generate_from_prefix_cache(
        model=model,
        tokenizer=tokenizer,
        prefix=prefix,
        config=SamplingConfig(max_tokens=2, temperature=0.0),
        eos_token_ids=frozenset(),
        compile_decode_step=False,
    )
    full = completion_logprobs(model, (prompt,), (completion_tokens,), pad_token_id=0)
    captured = mx.array([captured_logprobs], dtype=full.logprobs.dtype)
    max_error = mx.max(mx.abs(full.logprobs - captured))
    mx.eval(  # Test sync: materialize captured/full logprob comparison.
        max_error,
    )

    assert float(max_error.item()) < 1e-6


def test_dual_logprobs_can_skip_reference_forward_for_zero_beta() -> None:
    model = ToyCachedModel()
    prompts = ((2, 3, 4), (2, 3, 4))
    completions = ((5, 6), (7,))

    dual = dual_logprobs(
        model,
        prompts,
        completions,
        pad_token_id=0,
        compute_reference=False,
    )
    max_error = mx.max(mx.abs(dual.policy - dual.reference))
    mx.eval(  # Test sync: materialize policy/reference comparison.
        max_error,
        dual.mask,
    )

    assert float(max_error.item()) == 0.0
    assert dual.mask.tolist() == [[1.0, 1.0], [1.0, 0.0]]
    assert model.cached_call_shapes == []


def test_old_policy_logprobs_from_rollouts_pad_to_completion_mask() -> None:
    completions = (
        Completion(
            prompt_index=0,
            group_index=0,
            prompt_tokens=(2, 3, 4),
            completion_tokens=(5, 6),
            old_policy_logprobs=(-1.0, -2.0),
            text="5 6",
        ),
        Completion(
            prompt_index=0,
            group_index=1,
            prompt_tokens=(2, 3, 4),
            completion_tokens=(7,),
            old_policy_logprobs=(-3.0,),
            text="7",
        ),
    )

    old_policy = old_policy_logprobs_from_rollouts(completions)
    mx.eval(  # Test sync: materialize padded rollout logprobs and mask.
        old_policy.logprobs,
        old_policy.mask,
    )

    assert old_policy.logprobs.tolist() == [[-1.0, -2.0], [-3.0, 0.0]]
    assert old_policy.mask.tolist() == [[1.0, 1.0], [1.0, 0.0]]


def test_batch_from_rollouts_recomputes_old_policy_logprobs() -> None:
    model = ToyCachedModel()
    completions = (
        Completion(
            prompt_index=0,
            group_index=0,
            prompt_tokens=(2, 3, 4),
            completion_tokens=(5, 6),
            old_policy_logprobs=(-999.0, -999.0),
            text="5 6",
        ),
        Completion(
            prompt_index=0,
            group_index=1,
            prompt_tokens=(2, 3, 4),
            completion_tokens=(7,),
            old_policy_logprobs=(-999.0,),
            text="7",
        ),
    )
    expected = completion_logprobs(
        model,
        tuple(completion.prompt_tokens for completion in completions),
        tuple(completion.completion_tokens for completion in completions),
        pad_token_id=0,
    )

    batch = batch_from_rollouts(
        model=model,
        completions=completions,
        rewards=(1.0, 0.0),
        group_size=2,
        pad_token_id=0,
        algorithm=GRPOAlgorithm(),
    )
    mx.eval(  # Test sync: materialize batch logprobs/mask for assertions.
        batch.old_policy_logprobs,
        batch.mask,
        expected.logprobs,
    )

    max_error = mx.max(mx.abs(batch.old_policy_logprobs - expected.logprobs))
    mx.eval(max_error)  # Test sync: materialize recomputed old-policy comparison.
    assert float(max_error.item()) < 1e-6
    assert batch.mask.tolist() == [[1.0, 1.0], [1.0, 0.0]]
    assert batch.logprob_inputs is not None
    assert model.cached_call_shapes == [(1, 2), (2, 2)]


PROMPTS = [[1, 2, 3, 4, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15]]
COMPLETIONS = [[20, 21, 22, 23], [24], [25, 26]]


def _tiny_model(arch: str, tie: bool, dtype: mx.Dtype = mx.float32, quantize: bool = False) -> nn.Module:
    if arch == "qwen3":
        from mlx_lm.models.qwen3 import Model, ModelArgs

        extra = {"rope_theta": 10000.0, "head_dim": 8}
    else:
        from mlx_lm.models.llama import Model, ModelArgs

        extra = {}
    mx.random.seed(0)
    args = ModelArgs(
        model_type=arch,
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=50,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=tie,
        **extra,
    )
    model = Model(args)
    model.set_dtype(dtype)
    if quantize:
        nn.quantize(model, group_size=32, bits=4)
    mx.eval(model.parameters())  # Test sync: materialize random weights.
    return model


def _logprobs_and_grads(
    model: nn.Module, inputs: CompletionLogprobInputs, weights: mx.array
) -> tuple[mx.array, dict[str, mx.array]]:
    def loss(m: nn.Module) -> mx.array:
        out = completion_logprobs_from_inputs(m, inputs).logprobs.astype(mx.float32)
        return mx.sum(out * weights)

    value, grads = nn.value_and_grad(model, loss)(model)
    logprobs = completion_logprobs_from_inputs(model, inputs).logprobs
    flat = dict(tree_flatten(grads))
    mx.eval(value, logprobs, *flat.values())  # Test sync: materialize comparisons.
    return logprobs, flat


def _max_diff(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


@pytest.mark.parametrize(
    ("arch", "tie", "dtype", "quantize", "tol"),
    [
        ("qwen3", True, mx.float32, False, 1e-5),
        ("qwen3", False, mx.float32, False, 1e-5),
        ("llama", True, mx.float32, False, 1e-5),
        ("llama", False, mx.float32, False, 1e-5),
        ("qwen3", True, mx.float32, True, 1e-5),
        # bf16 ulp near |logprob|=4 is 2^-5; fp32 diffs measured <= 5e-7 (logprobs), 5e-6 (grads)
        ("qwen3", True, mx.bfloat16, False, 0.0625),
    ],
)
def test_completion_head_matches_full_logits(
    monkeypatch: pytest.MonkeyPatch,
    arch: str,
    tie: bool,
    dtype: mx.Dtype,
    quantize: bool,
    tol: float,
) -> None:
    """Applying the head at completion rows only equals applying it everywhere."""

    model = _tiny_model(arch, tie, dtype, quantize)
    assert logprobs_module._completion_head(model) is not None
    inputs = prepare_completion_logprob_inputs(PROMPTS, COMPLETIONS)
    weights = mx.arange(1, 5, dtype=mx.float32)[None, :]

    got, got_grads = _logprobs_and_grads(model, inputs, weights)
    monkeypatch.setattr(logprobs_module, "_completion_head", lambda m: None)
    ref, ref_grads = _logprobs_and_grads(model, inputs, weights)

    assert _max_diff(got, ref) <= tol
    assert got.shape == ref.shape
    assert max(float(mx.max(mx.abs(g)).item()) for g in ref_grads.values()) > 0.0
    assert max(_max_diff(got_grads[k], ref_grads[k]) for k in ref_grads) <= max(tol, 1e-4)


def test_completion_head_comparison_detects_misaligned_gather() -> None:
    model = _tiny_model("qwen3", True)
    inputs = prepare_completion_logprob_inputs(PROMPTS, COMPLETIONS)
    shifted = dataclasses.replace(inputs, gather_indices=inputs.gather_indices + 1)
    good = completion_logprobs_from_inputs(model, inputs).logprobs
    bad = completion_logprobs_from_inputs(model, shifted).logprobs
    mx.eval(good, bad)  # Test sync.
    assert _max_diff(good, bad) > 1e-2


def test_completion_head_sees_only_completion_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _tiny_model("qwen3", True)
    embed_cls = type(model.model.embed_tokens)
    original = embed_cls.as_linear
    seen: list[tuple[int, ...]] = []

    def spy(self: nn.Module, x: mx.array) -> mx.array:
        seen.append(tuple(x.shape))
        return original(self, x)

    monkeypatch.setattr(embed_cls, "as_linear", spy)
    inputs = prepare_completion_logprob_inputs(PROMPTS, COMPLETIONS)
    mx.eval(completion_logprobs_from_inputs(model, inputs).logprobs)  # Test sync.
    assert seen == [(3, 4, 32)]


def test_unlisted_model_type_uses_full_forward() -> None:
    model = _tiny_model("qwen3", True)
    model.args.model_type = "gemma3"
    assert logprobs_module._completion_head(model) is None
