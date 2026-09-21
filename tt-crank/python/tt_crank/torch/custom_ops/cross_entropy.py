# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Fused cross entropy on the ttml cross_entropy_fw / cross_entropy_bw kernels.

aten's cross_entropy is `_log_softmax` + `nll_loss_forward`, its backward `nll_loss_backward` +
`_log_softmax_backward_data`: four passes over the [rows x C] logits, two of them f32 in torch's
decompositions. ttml has one kernel each way: cross_entropy_fw (per-row loss from logits and integer
targets) and cross_entropy_bw ((softmax - onehot) * grad). The custom ops are their graph form.
`aten.cross_entropy_loss` is CompositeImplicitAutograd, so it is already the four aten ops by the time
aot traces; the dynamo graph still has the single `F.cross_entropy` call, and rewrite_cross_entropy
swaps that node onto TTCrossEntropy (an autograd.Function around the two ops) before aot. The aten
lowerings in `_compile` stay as the fallback for whatever the kernels do not take.

Exports: the two OpOverloads `cross_entropy_fw` / `cross_entropy_bw` (lowered in `_compile`, sharded in
`_sharding`), `TTCrossEntropy`, and `rewrite_cross_entropy` for the dynamo backend.
"""

from __future__ import annotations

import inspect

import torch
import torch.fx

# aten's Reduction enum, as `cross_entropy_loss` takes it.
_REDUCTION_MEAN, _REDUCTION_SUM = 1, 2


@torch.library.custom_op("tt_crank::cross_entropy_fw", mutates_args=())
def _cross_entropy_fw(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return -log_probs.gather(1, target.long().unsqueeze(1)).squeeze(1).to(logits.dtype)


@_cross_entropy_fw.register_fake
def _(logits, target):
    return logits.new_empty(logits.shape[:-1])


@torch.library.custom_op("tt_crank::cross_entropy_bw", mutates_args=())
def _cross_entropy_bw(
    grad: torch.Tensor, logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    onehot = torch.nn.functional.one_hot(target.long(), logits.shape[-1])
    return ((torch.softmax(logits.float(), dim=-1) - onehot) * grad.float()).to(
        logits.dtype
    )


@_cross_entropy_bw.register_fake
def _(grad, logits, target):
    return torch.empty_like(logits)


cross_entropy_fw = torch.ops.tt_crank.cross_entropy_fw.default
cross_entropy_bw = torch.ops.tt_crank.cross_entropy_bw.default


class TTCrossEntropy(torch.autograd.Function):
    """Mean/sum cross entropy over [rows x C] logits on the ttml pair. The kernels know no ignore_index
    and the backward one takes a single grad for all rows, so ignored rows are masked around them:
    their target is clamped to 0 for the kernels and their loss / grad selected out afterwards."""

    @staticmethod
    def forward(ctx, logits, target, ignore_index, mean):
        valid = target != ignore_index
        clamped = target * valid.to(target.dtype)
        per_row = cross_entropy_fw(logits, clamped)
        # Select, not multiply: a non-finite loss in an ignored row must not reach the sum.
        rows = torch.where(valid, per_row, torch.zeros_like(per_row))
        total_weight = valid.to(logits.dtype).sum(0)
        loss = rows.sum(0)
        if mean:
            loss = loss / total_weight
        ctx.mean = mean
        ctx.save_for_backward(logits, clamped, valid, total_weight)
        return loss

    @staticmethod
    def backward(ctx, grad):
        logits, clamped, valid, total_weight = ctx.saved_tensors
        if ctx.mean:
            # If no row is kept, torch's gradient is 0, not NaN; the clamp keeps the division finite.
            grad = grad / total_weight.clamp_min(1.0)
        d_rows = cross_entropy_bw(grad, logits, clamped)
        keep = valid.unsqueeze(1).expand(logits.shape)
        return torch.where(keep, d_rows, torch.zeros_like(d_rows)), None, None, None


# The two spellings dynamo records for cross entropy, with their argument lists. `F.cross_entropy` is
# the Python wrapper (string reduction, legacy size_average / reduce); `torch._C._nn.cross_entropy_loss`
# is the builtin it calls (int reduction), which shows up when a model calls it directly.
_CROSS_ENTROPY_SIGNATURES = {
    torch.nn.functional.cross_entropy: inspect.signature(
        torch.nn.functional.cross_entropy
    ),
    torch._C._nn.cross_entropy_loss: inspect.Signature(
        [
            inspect.Parameter(
                name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default
            )
            for name, default in (
                ("input", inspect.Parameter.empty),
                ("target", inspect.Parameter.empty),
                ("weight", None),
                ("reduction", _REDUCTION_MEAN),
                ("ignore_index", -100),
                ("label_smoothing", 0.0),
            )
        ]
    ),
}
_CROSS_ENTROPY_MEAN = {
    "mean": True,
    "sum": False,
    _REDUCTION_MEAN: True,
    _REDUCTION_SUM: False,
}


def rewrite_cross_entropy(gm: torch.fx.GraphModule) -> bool:
    """Swap each `F.cross_entropy` / `cross_entropy_loss` node of the dynamo graph onto TTCrossEntropy,
    when the kernels take it: bf16 [rows x C] logits, integer [rows] targets, no class weights, no
    label smoothing, mean or sum reduction. Anything else is left for aten to decompose."""
    changed = False
    for node in list(gm.graph.nodes):
        signature = (
            _CROSS_ENTROPY_SIGNATURES.get(node.target)
            if node.op == "call_function"
            else None
        )
        if signature is None:
            continue
        try:
            bound = signature.bind(*node.args, **node.kwargs)
        except TypeError:
            continue
        bound.apply_defaults()
        a = bound.arguments
        logits, target = a["input"], a["target"]
        if not (
            isinstance(logits, torch.fx.Node) and isinstance(target, torch.fx.Node)
        ):
            continue
        logits_val, target_val = logits.meta.get("example_value"), target.meta.get(
            "example_value"
        )
        if (
            a["weight"] is not None
            or a["label_smoothing"] != 0.0
            or a.get("size_average") is not None
            or a.get("reduce") is not None
            or a["reduction"] not in _CROSS_ENTROPY_MEAN
            or not isinstance(a["ignore_index"], int)
            or not isinstance(logits_val, torch.Tensor)
            or not isinstance(target_val, torch.Tensor)
            or logits_val.dtype is not torch.bfloat16
            or logits_val.dim() != 2
            or target_val.dim() != 1
            or target_val.is_floating_point()
        ):
            continue
        with gm.graph.inserting_before(node):
            fused = gm.graph.call_function(
                TTCrossEntropy.apply,
                (
                    logits,
                    target,
                    a["ignore_index"],
                    _CROSS_ENTROPY_MEAN[a["reduction"]],
                ),
            )
        fused.meta = dict(node.meta)
        node.replace_all_uses_with(fused)
        gm.graph.erase_node(node)
        changed = True
    if changed:
        gm.graph.lint()
        gm.recompile()
    return changed
