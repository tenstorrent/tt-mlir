# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end SGD training of the small MNIST linear classifier on tt.

Eager mode runs under strict_no_fallback: the whole loop must run on device with no CPU fallback.
Compile mode wraps the model with torch.compile(backend="tt") so its forward and
backward lower to TTIR. Checked against the same loop run on CPU.
"""

import contextlib

import pytest
import torch
import torch.nn.functional as F

from tt_kurbla.torch.testing import (
    ExecutionMode,
    get_supported_dtypes,
    strict_no_fallback,
)

from _models import MNISTLinear


_BATCH = 32
_FEAT = 28 * 28
_HIDDEN = 32
_CLASSES = 10
_NUM_STEPS = 5
_LR = 0.5


def _train(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> list[float]:
    """Run _NUM_STEPS of SGD over MSE loss; return the per-step loss (on CPU)."""
    opt = torch.optim.SGD(model.parameters(), lr=_LR)
    losses: list[float] = []
    for _ in range(_NUM_STEPS):
        opt.zero_grad()
        logits = model(x)
        loss = F.mse_loss(logits, y)
        losses.append(float(loss.detach().cpu()))
        loss.backward()
        opt.step()
    return losses


@pytest.mark.parametrize("dtype", get_supported_dtypes(), ids=str)
@pytest.mark.parametrize(
    "mode", [ExecutionMode.EAGER, ExecutionMode.COMPILE], ids=lambda m: m.value
)
def test_mnist_linear_training(
    mode: ExecutionMode, dtype: torch.dtype, tt_device: torch.device
) -> None:
    inputs = torch.randn(_BATCH, _FEAT, dtype=dtype)
    targets = torch.randn(_BATCH, _CLASSES, dtype=dtype)

    # Same initial weights so the loss curves are comparable; CPU is the reference.
    cpu_model = MNISTLinear(_FEAT, _HIDDEN, _CLASSES).to(dtype)
    tt_model = MNISTLinear(_FEAT, _HIDDEN, _CLASSES).to(dtype).to(tt_device)
    tt_model.load_state_dict(cpu_model.state_dict())

    if mode is ExecutionMode.COMPILE:
        tt_model = torch.compile(tt_model, backend="tt")

    # Eager must run entirely on device; compile lowers the graph to its own
    # module, so the eager fallback guard doesn't apply there.
    guard = (
        strict_no_fallback()
        if mode is ExecutionMode.EAGER
        else contextlib.nullcontext()
    )
    with guard:
        tt_losses = _train(tt_model, inputs.to(tt_device), targets.to(tt_device))
    cpu_losses = _train(cpu_model, inputs, targets)

    assert (
        tt_losses[-1] < tt_losses[0] * 0.5
    ), f"on-device training did not reduce loss: {tt_losses[0]:.4f} -> {tt_losses[-1]:.4f}"
    for step_idx, (tt_loss, cpu_loss) in enumerate(zip(tt_losses, cpu_losses)):
        assert tt_loss == pytest.approx(
            cpu_loss, abs=0.1
        ), f"step {step_idx}: tt loss {tt_loss:.4f} diverged from cpu {cpu_loss:.4f}"


def _train_step(step, params, x: torch.Tensor, y: torch.Tensor) -> list[float]:
    opt = torch.optim.SGD(params, lr=_LR)
    losses: list[float] = []
    for _ in range(_NUM_STEPS):
        opt.zero_grad()
        loss = step(x, y)
        losses.append(float(loss.detach().cpu()))
        loss.backward()
        opt.step()
    return losses


@pytest.mark.parametrize("dtype", get_supported_dtypes(), ids=str)
def test_mnist_linear_fwd_loss_compiled(
    dtype: torch.dtype, tt_device: torch.device
) -> None:
    # Compile model forward + loss as a single function: mse_loss lowers into the
    # compiled forward, mse_loss_backward into the compiled backward. Optimizer
    # stays eager. Checked against the same loop run eagerly on CPU.
    inputs = torch.randn(_BATCH, _FEAT, dtype=dtype)
    targets = torch.randn(_BATCH, _CLASSES, dtype=dtype)

    cpu_model = MNISTLinear(_FEAT, _HIDDEN, _CLASSES).to(dtype)
    tt_model = MNISTLinear(_FEAT, _HIDDEN, _CLASSES).to(dtype).to(tt_device)
    tt_model.load_state_dict(cpu_model.state_dict())

    def cpu_step(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(cpu_model(x), y)

    def tt_step(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(tt_model(x), y)

    tt_step = torch.compile(tt_step, backend="tt")

    tt_losses = _train_step(
        tt_step, tt_model.parameters(), inputs.to(tt_device), targets.to(tt_device)
    )
    cpu_losses = _train_step(cpu_step, cpu_model.parameters(), inputs, targets)

    assert (
        tt_losses[-1] < tt_losses[0] * 0.5
    ), f"on-device training did not reduce loss: {tt_losses[0]:.4f} -> {tt_losses[-1]:.4f}"
    for step_idx, (tt_loss, cpu_loss) in enumerate(zip(tt_losses, cpu_losses)):
        assert tt_loss == pytest.approx(
            cpu_loss, abs=0.1
        ), f"step {step_idx}: tt loss {tt_loss:.4f} diverged from cpu {cpu_loss:.4f}"
