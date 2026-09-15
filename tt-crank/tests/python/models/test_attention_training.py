# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""SGD training of a causal self-attention block on tt, eager and compile, against a CPU reference."""

import contextlib

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tt_crank.torch.testing import ExecutionMode, strict_no_fallback

_BATCH, _SEQ, _MODEL_DIM, _HEADS = 2, 64, 128, 4
_NUM_STEPS = 8
_LR = 10.0  # large enough that each step's loss change is representable in bf16
_DTYPE = torch.bfloat16


class CausalSelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, model_dim = x.shape
        q, k, v = (
            t.view(batch, seq, self.num_heads, -1).transpose(1, 2)
            for t in self.qkv(x).chunk(3, dim=-1)
        )
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(attn.transpose(1, 2).reshape(batch, seq, model_dim))


def _train(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> list[float]:
    opt = torch.optim.SGD(model.parameters(), lr=_LR)
    losses: list[float] = []
    for _ in range(_NUM_STEPS):
        opt.zero_grad()
        loss = F.mse_loss(model(x), y)
        losses.append(float(loss.detach().cpu()))
        loss.backward()
        opt.step()
    return losses


@pytest.mark.parametrize(
    "mode", [ExecutionMode.EAGER, ExecutionMode.COMPILE], ids=lambda m: m.value
)
def test_attention_block_training(mode: ExecutionMode, tt_device: torch.device) -> None:
    inputs = torch.randn(_BATCH, _SEQ, _MODEL_DIM, dtype=_DTYPE)
    cpu_model = CausalSelfAttention(_MODEL_DIM, _HEADS).to(_DTYPE)
    tt_model = CausalSelfAttention(_MODEL_DIM, _HEADS).to(_DTYPE).to(tt_device)
    tt_model.load_state_dict(cpu_model.state_dict())
    if mode is ExecutionMode.COMPILE:
        tt_model = torch.compile(tt_model, backend="tt")

    guard = (
        strict_no_fallback()
        if mode is ExecutionMode.EAGER
        else contextlib.nullcontext()
    )
    with guard:
        tt_losses = _train(tt_model, inputs.to(tt_device), inputs.to(tt_device))
    cpu_losses = _train(cpu_model, inputs, inputs)

    assert (
        tt_losses[-1] < tt_losses[0]
    ), f"loss did not drop: {tt_losses[0]:.4f} -> {tt_losses[-1]:.4f}"
    for step, (tt_loss, cpu_loss) in enumerate(zip(tt_losses, cpu_losses)):
        assert tt_loss == pytest.approx(
            cpu_loss, rel=0.05
        ), f"step {step}: tt {tt_loss:.4f} vs cpu {cpu_loss:.4f}"
