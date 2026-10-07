# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Fused cross entropy on the ttml cross_entropy_fw / cross_entropy_bw kernels.

aten's cross_entropy is `_log_softmax` + `nll_loss_forward`, its backward `nll_loss_backward` +
`_log_softmax_backward_data`: four passes over the [rows x C] logits, two of them f32 in torch's
decompositions. ttml has one kernel each way; the custom ops below are their graph form (lowered in
`_compile`, sharded in `_sharding`), with the backward one as the forward one's autograd.

`aten::cross_entropy_loss` is CompositeImplicitAutograd. The `AutogradPrivateUse1` kernel registered here
runs ahead of that decomposition for tt tensors and takes the call onto the custom ops when the kernels
can run it; anything else goes through aten's own decomposition and the aten lowerings in `_compile`.
"""

from __future__ import annotations

import torch

# aten's Reduction enum, as `cross_entropy_loss` takes it.
_REDUCTION_NONE, _REDUCTION_MEAN, _REDUCTION_SUM = 0, 1, 2


@torch.library.custom_op("tt_crank::cross_entropy_fw", mutates_args=())
def _cross_entropy_fw(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-row loss of [rows x C] logits against [rows] class indices."""
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return -log_probs.gather(1, target.long().unsqueeze(1)).squeeze(1).to(logits.dtype)


@_cross_entropy_fw.register_fake
def _(logits, target):
    return logits.new_empty(logits.shape[:-1])


@torch.library.custom_op("tt_crank::cross_entropy_bw", mutates_args=())
def _cross_entropy_bw(
    grad: torch.Tensor, logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """`(softmax(logits) - onehot(target)) * grad[:, None]` for the [rows] grad of the per-row losses."""
    onehot = torch.nn.functional.one_hot(target.long(), logits.shape[-1])
    return (
        (torch.softmax(logits.float(), dim=-1) - onehot) * grad.float().unsqueeze(1)
    ).to(logits.dtype)


@_cross_entropy_bw.register_fake
def _(grad, logits, target):
    return torch.empty_like(logits)


def _setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _backward(ctx, grad):
    return _cross_entropy_bw(grad, *ctx.saved_tensors), None


_cross_entropy_fw.register_autograd(_backward, setup_context=_setup_context)

cross_entropy_fw = torch.ops.tt_crank.cross_entropy_fw.default
cross_entropy_bw = torch.ops.tt_crank.cross_entropy_bw.default

_aten_cross_entropy_loss = torch.ops.aten.cross_entropy_loss.default


def _tt_cross_entropy_loss(
    input,
    target,
    weight=None,
    reduction=_REDUCTION_MEAN,
    ignore_index=-100,
    label_smoothing=0.0,
):
    """The kernels take bf16 [rows x C] logits with integer [rows] targets, no class weights, no label
    smoothing. They know no ignore_index: ignored rows get target 0 and their loss selected out, which
    through autograd zeroes their grad too."""
    if (
        weight is not None
        or label_smoothing != 0.0
        or input.dtype is not torch.bfloat16
        or input.dim() != 2
        or target.dim() != 1
        or target.is_floating_point()
    ):
        return _aten_cross_entropy_loss.decompose(
            input, target, weight, reduction, ignore_index, label_smoothing
        )
    valid = target != ignore_index
    per_row = cross_entropy_fw(input, torch.where(valid, target, 0))
    # Select, not multiply: a non-finite loss in an ignored row must not reach the sum.
    rows = torch.where(valid, per_row, 0.0)
    if reduction == _REDUCTION_NONE:
        return rows
    loss = rows.sum()
    return loss / valid.to(input.dtype).sum() if reduction == _REDUCTION_MEAN else loss


_aten_lib = torch.library.Library("aten", "IMPL")
_aten_lib.impl("cross_entropy_loss", _tt_cross_entropy_loss, "AutogradPrivateUse1")
