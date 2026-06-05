"""End-to-end SGD training of the small MNIST linear classifier on tt.

Eager mode runs under strict_no_fallback: the whole loop must run on device with no CPU fallback.
Compile mode is xfail until the backward graph lowers. Checked against the same loop run on CPU.
"""

import contextlib

import pytest
import torch
import torch.nn.functional as F

from tt_kurbla.torch.testing import ExecutionMode, get_supported_dtypes, strict_no_fallback

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
    "mode",
    [
        ExecutionMode.EAGER,
        pytest.param(
            ExecutionMode.COMPILE,
            marks=pytest.mark.xfail(reason="compile backend does not yet lower the backward graph"),
        ),
    ],
    ids=lambda m: m.value,
)
def test_mnist_linear_training(mode: ExecutionMode, dtype: torch.dtype, tt_device: torch.device) -> None:
    inputs = torch.randn(_BATCH, _FEAT, dtype=dtype)
    targets = torch.randn(_BATCH, _CLASSES, dtype=dtype)

    # Same initial weights so the loss curves are comparable; CPU is the reference.
    cpu_model = MNISTLinear(_FEAT, _HIDDEN, _CLASSES).to(dtype)
    tt_model = MNISTLinear(_FEAT, _HIDDEN, _CLASSES).to(dtype).to(tt_device)
    tt_model.load_state_dict(cpu_model.state_dict())

    if mode is ExecutionMode.COMPILE:
        tt_model = torch.compile(tt_model, backend="tt", dynamic=False)

    # Eager must run entirely on device; compile lowers the graph to its own
    # module, so the eager fallback guard doesn't apply there.
    guard = strict_no_fallback() if mode is ExecutionMode.EAGER else contextlib.nullcontext()
    with guard:
        tt_losses = _train(tt_model, inputs.to(tt_device), targets.to(tt_device))
    cpu_losses = _train(cpu_model, inputs, targets)

    assert tt_losses[-1] < tt_losses[0] * 0.5, (
        f"on-device training did not reduce loss: {tt_losses[0]:.4f} -> {tt_losses[-1]:.4f}"
    )
    for step_idx, (tt_loss, cpu_loss) in enumerate(zip(tt_losses, cpu_losses)):
        assert tt_loss == pytest.approx(cpu_loss, abs=0.1), (
            f"step {step_idx}: tt loss {tt_loss:.4f} diverged from cpu {cpu_loss:.4f}"
        )
