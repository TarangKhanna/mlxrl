"""Shared-prefix policy forward must match the full-forward path, gradients included.

Tolerances (fixed before measuring, float32 tiny model):
  logprobs, loss:         max abs diff <= 1e-5
  every LoRA gradient:    max abs diff <= 1e-4 * max(1e-6, max abs of reference tensor)
  params after one SGD(lr=1e-2) step: max abs diff <= 1e-6
  params after one Adam(lr=1e-2) step: max abs diff <= 1e-5 on elements whose
      reference gradient is >= 1e-3 of its tensor's max (Adam normalizes, so a
      float-level gradient error on a ~1e-7 element moves it by up to lr;
      those elements are covered by the gradient check instead).
The dtype noise floor is measured in test_noise_floor and is reported, and the
tolerances must sit well above it. The detached-prefix control must exceed the
gradient tolerance by >= 100x on at least one tensor.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3

from mlxrl.algo.grpo import GRPOAlgorithm
from mlxrl.policy import shared_prefix as sp
from mlxrl.policy.logprobs import prepare_completion_logprob_inputs
from mlxrl.policy.model import LoRAConfig, enable_grad_checkpointing, inject_lora_adapters
from mlxrl.train.grpo import GRPOBatch, grpo_metrics_from_batch

LP_TOL = 1e-5
GRAD_REL_TOL = 1e-4
PARAM_TOL = 1e-5
VOCAB = 97


def tiny_model(seed: int = 0) -> nn.Module:
    mx.random.seed(seed)
    args = qwen3.ModelArgs(
        model_type="qwen3",
        hidden_size=64,
        num_hidden_layers=3,
        intermediate_size=128,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=VOCAB,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rope_theta=10000.0,
        head_dim=16,
        tie_word_embeddings=True,
    )
    model = qwen3.Model(args)
    inject_lora_adapters(model, LoRAConfig(rank=4, scale=8.0))
    model.freeze()
    model.unfreeze(keys=["lora_a", "lora_b"])
    # LoRA B starts at zero; make every adapter tensor matter.
    leaves = tree_flatten(model.trainable_parameters())
    model.update(
        __import__("mlx.utils", fromlist=["tree_unflatten"]).tree_unflatten(
            [(n, mx.random.normal(p.shape) * 0.05) for n, p in leaves]
        )
    )
    mx.eval(model.parameters())
    return model


def make_batch(model, prompt_lens, G, comp_lens, seed=1) -> GRPOBatch:
    rng = mx.random.key(seed)
    prompts, comps = [], []
    for i, pl in enumerate(prompt_lens):
        prompt = [
            int(t) for t in mx.random.randint(1, VOCAB, (pl,), key=mx.random.key(seed + 10 * i))
        ]
        for g in range(G):
            n = comp_lens[(i * G + g) % len(comp_lens)]
            prompts.append(tuple(prompt))
            comps.append(
                tuple(
                    int(t)
                    for t in mx.random.randint(
                        1, VOCAB, (n,), key=mx.random.key(seed + 100 * i + g)
                    )
                )
            )
    inputs = prepare_completion_logprob_inputs(prompts, comps, pad_token_id=0)
    from mlxrl.policy.logprobs import completion_logprobs_from_inputs

    cur = completion_logprobs_from_inputs(model, inputs)
    mx.eval(cur.logprobs)
    n = len(prompts)
    noise = mx.random.normal(cur.logprobs.shape, key=rng) * 0.1
    return GRPOBatch(
        prompt_token_ids=tuple(prompts),
        completion_token_ids=tuple(comps),
        rewards=mx.random.normal((n,), key=rng),
        advantages=mx.random.normal((n,), key=mx.random.key(seed + 7)),
        old_policy_logprobs=mx.stop_gradient(cur.logprobs + noise * cur.mask),
        reference_logprobs=mx.stop_gradient(
            cur.logprobs
            + 0.05 * mx.random.normal(cur.logprobs.shape, key=mx.random.key(5)) * cur.mask
        ),
        mask=cur.mask,
        logprob_inputs=inputs,
    )


def loss_grads(model, batch, shared, ckpt=False, beta=0.04):
    algo = GRPOAlgorithm()

    def f(m):
        met = grpo_metrics_from_batch(
            m, batch, beta, 0, algo, use_checkpoint=ckpt, shared_prefix=shared
        )
        return met.loss

    loss, grads = nn.value_and_grad(model, f)(model)
    mx.eval(loss, grads)
    return loss, dict(tree_flatten(grads))


def maxdiff(a, b):
    return float(mx.max(mx.abs(a - b)).item())


def compare_grads(ref, got):
    worst_rel = 0.0
    for k in ref:
        for t in (ref[k], got[k]):
            assert bool(mx.all(mx.isfinite(t)).item()), f"non-finite gradient in {k}"
        d = maxdiff(ref[k], got[k])
        scale = max(1e-6, float(mx.max(mx.abs(ref[k])).item()))
        worst_rel = max(worst_rel, d / scale)
    return worst_rel


CASES = [
    (G, pl, cl)
    for G in (1, 2, 4)
    for pl, cl in (((9, 17), (5, 3, 8, 1)), ((1, 6), (4, 7)), ((23,), (6, 2, 9)))
]


@pytest.mark.parametrize("G,prompt_lens,comp_lens", CASES)
def test_matches_full_forward(G, prompt_lens, comp_lens):
    model = tiny_model()
    batch = make_batch(model, prompt_lens, G, comp_lens)
    ref_loss, ref_g = loss_grads(model, batch, shared=False)
    loss, g = loss_grads(model, batch, shared=True)
    assert set(ref_g) == set(g) and len(ref_g) > 0
    assert bool(mx.isfinite(ref_loss).item()) and bool(mx.isfinite(loss).item())
    assert maxdiff(ref_loss, loss) <= LP_TOL
    assert compare_grads(ref_g, g) <= GRAD_REL_TOL

    out = sp.shared_prefix_completion_logprobs(
        model, batch.prompt_token_ids, batch.completion_token_ids
    )
    from mlxrl.policy.logprobs import completion_logprobs_from_inputs

    ref = completion_logprobs_from_inputs(model, batch.logprob_inputs)
    assert bool(mx.all(mx.isfinite(out.logprobs)).item())
    assert bool(mx.all(mx.isfinite(ref.logprobs)).item())
    assert maxdiff(out.logprobs, ref.logprobs) <= LP_TOL
    assert maxdiff(out.mask, ref.mask) == 0.0

    # one optimizer update lands in the same place
    from mlx.utils import tree_unflatten

    for make_opt in (lambda: optim.SGD(learning_rate=1e-2), lambda: optim.Adam(learning_rate=1e-2)):
        params = {}
        for shared, grads in ((False, ref_g), (True, g)):
            m = tiny_model()
            make_opt().update(m, tree_unflatten(list(grads.items())))
            mx.eval(m.parameters())
            params[shared] = dict(tree_flatten(m.trainable_parameters()))
        sgd = isinstance(make_opt(), optim.SGD)
        worst = 0.0
        for k in params[False]:
            for t in (params[False][k], params[True][k], ref_g[k]):
                assert bool(mx.all(mx.isfinite(t)).item()), f"non-finite value in {k}"
            d = mx.abs(params[False][k] - params[True][k])
            if not sgd:
                a = mx.abs(ref_g[k])
                d = mx.where(a >= 1e-3 * mx.max(a), d, 0.0)
            worst = max(worst, float(mx.max(d).item()))
        assert worst <= (1e-6 if sgd else PARAM_TOL), (sgd, worst)


def test_matches_with_grad_checkpointing():
    model = tiny_model()
    batch = make_batch(model, (9, 17), 3, (5, 3, 8))
    ref_loss, ref_g = loss_grads(model, batch, shared=False)
    layer_type = type(model.model.layers[0])
    saved = layer_type.__call__
    try:
        enable_grad_checkpointing(model)
        loss, g = loss_grads(model, batch, shared=True, ckpt=True)
        _ck_loss, ck_g = loss_grads(model, batch, shared=False, ckpt=True)
    finally:
        layer_type.__call__ = saved
        del layer_type._mlxrl_grad_checkpoint_enabled
    assert maxdiff(ref_loss, loss) <= LP_TOL
    assert compare_grads(ref_g, g) <= GRAD_REL_TOL
    assert compare_grads(ref_g, ck_g) <= GRAD_REL_TOL


def test_detached_prefix_is_caught(monkeypatch):
    """Negative control: stop-gradient on the prefix K/V must fail the gradient check."""
    model = tiny_model()
    batch = make_batch(model, (9, 17), 4, (5, 3, 8, 1))
    _, ref_g = loss_grads(model, batch, shared=False)
    orig = sp._PrefixKV.update_and_fetch

    def detached(self, keys, values):
        if self.prefix_keys is not None:
            self.prefix_keys = mx.stop_gradient(self.prefix_keys)
            self.prefix_values = mx.stop_gradient(self.prefix_values)
        return orig(self, keys, values)

    monkeypatch.setattr(sp._PrefixKV, "update_and_fetch", detached)
    _, g = loss_grads(model, batch, shared=True)
    assert compare_grads(ref_g, g) > 100 * GRAD_REL_TOL


def test_noise_floor():
    """Reference scale: same path rerun (bitwise-deterministic => 0), and the same
    loss with rows permuted (different reduction order). Tolerances must sit above."""
    model = tiny_model()
    batch = make_batch(model, (9, 17), 4, (5, 3, 8, 1))
    l1, g1 = loss_grads(model, batch, shared=False)
    _l2, g2 = loss_grads(model, batch, shared=False)
    order = list(reversed(range(len(batch.prompt_token_ids))))
    idx = mx.array(order)
    prompts = tuple(batch.prompt_token_ids[i] for i in order)
    comps = tuple(batch.completion_token_ids[i] for i in order)
    perm = GRPOBatch(
        prompt_token_ids=prompts,
        completion_token_ids=comps,
        rewards=batch.rewards[idx],
        advantages=batch.advantages[idx],
        old_policy_logprobs=batch.old_policy_logprobs[idx],
        reference_logprobs=batch.reference_logprobs[idx],
        mask=batch.mask[idx],
        logprob_inputs=prepare_completion_logprob_inputs(prompts, comps, pad_token_id=0),
    )
    l3, g3 = loss_grads(model, perm, shared=False)
    rerun, permuted = compare_grads(g1, g2), compare_grads(g1, g3)
    print(
        f"\nnoise floor: rerun grad_rel {rerun:.3e}; row-permuted grad_rel {permuted:.3e}, "
        f"loss {maxdiff(l1, l3):.3e}"
    )
    assert rerun == 0.0 and permuted < GRAD_REL_TOL / 10


def test_compare_grads_rejects_nan():
    good = {"w": mx.ones((3,))}
    bad = {"w": mx.array([1.0, float("nan"), 1.0])}
    with pytest.raises(AssertionError):
        compare_grads(good, bad)
    with pytest.raises(AssertionError):
        compare_grads(bad, good)


def test_unsupported_model_is_rejected():
    """A stock subclass may change __call__ (e.g. scale logits); the direct
    embeddings/layers/head path would bypass it, so it must be refused."""

    class Scaled(qwen3.Model):
        def __call__(self, *a, **k):
            return super().__call__(*a, **k) * 2.0

    base = tiny_model()
    scaled = Scaled(base.args)
    assert sp.supports_shared_prefix(base)
    assert not sp.supports_shared_prefix(scaled)
    with pytest.raises(ValueError, match="stock mlx-lm"):
        sp.shared_prefix_completion_logprobs(scaled, [(1, 2, 3)], [(4, 5)])
