from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlxrl.algo.grpo import GRPOAlgorithm
from mlxrl.policy.logprobs import (
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


def _tiny_qwen3(tie_word_embeddings: bool) -> nn.Module:
    from mlx_lm.models.qwen3 import Model, ModelArgs

    mx.random.seed(0)
    args = ModelArgs(
        model_type="qwen3",
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=50,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rope_theta=10000.0,
        head_dim=8,
        tie_word_embeddings=tie_word_embeddings,
    )
    model = Model(args)
    mx.eval(model.parameters())  # Test sync: materialize random weights.
    return model


@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_completion_logprobs_match_log_softmax_of_full_logits_with_gradients(
    tie_word_embeddings: bool,
) -> None:
    """Completion logprobs are log_softmax(model(tokens)) at the predicting positions."""

    model = _tiny_qwen3(tie_word_embeddings)
    prompts = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
    completions = [[13, 14, 15], [16]]
    inputs = prepare_completion_logprob_inputs(prompts, completions)

    def explicit(params_model: nn.Module) -> mx.array:
        logits = params_model(inputs.input_ids)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        picked = mx.squeeze(
            mx.take_along_axis(logprobs, inputs.target_ids[..., None], axis=-1), axis=-1
        )
        return mx.take_along_axis(picked, inputs.gather_indices, axis=1) * inputs.mask

    def actual(params_model: nn.Module) -> mx.array:
        return completion_logprobs_from_inputs(params_model, inputs).logprobs

    def loss(fn: object) -> object:
        return lambda m: mx.sum(fn(m) * mx.arange(1, 4, dtype=mx.float32)[None, :])  # type: ignore[operator]

    value, grads = nn.value_and_grad(model, loss(actual))(model)
    ref_value, ref_grads = nn.value_and_grad(model, loss(explicit))(model)
    from mlx.utils import tree_flatten

    flat = dict(tree_flatten(grads))
    ref_flat = dict(tree_flatten(ref_grads))
    errors = [mx.max(mx.abs(flat[k] - ref_flat[k])) for k in ref_flat]
    mx.eval(value, ref_value, *errors)  # Test sync: materialize comparisons.

    assert abs(float(value.item()) - float(ref_value.item())) < 1e-5
    assert max(float(e.item()) for e in errors) < 1e-5
    assert max(float(mx.max(mx.abs(g)).item()) for g in flat.values()) > 0.0


def test_completion_logprobs_project_only_completion_positions() -> None:
    """The vocab projection must see [batch, max_completion_len] rows, not the whole sequence."""

    model = _tiny_qwen3(tie_word_embeddings=True)
    seen: list[tuple[int, ...]] = []
    embed = model.model.embed_tokens
    original = type(embed).as_linear

    def spy(self: nn.Module, x: mx.array) -> mx.array:
        seen.append(tuple(x.shape))
        return original(self, x)

    type(embed).as_linear = spy  # type: ignore[method-assign]
    try:
        inputs = prepare_completion_logprob_inputs([[1, 2, 3, 4, 5, 6]] * 2, [[7, 8, 9], [10]])
        mx.eval(completion_logprobs_from_inputs(model, inputs).logprobs)  # Test sync.
    finally:
        type(embed).as_linear = original  # type: ignore[method-assign]

    assert seen == [(2, 3, 32)]
