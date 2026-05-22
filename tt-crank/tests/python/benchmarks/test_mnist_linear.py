"""Benchmark: small MNIST linear classifier forward pass."""

import pytest
import torch

from _models import MNISTLinear

from ._runner import prepare_model, run_benchmark


_BATCH = 64
_FEAT = 28 * 28
_HIDDEN = 128
_CLASSES = 10
_DTYPE = torch.bfloat16


def _build_inputs(device: torch.device | str) -> tuple[torch.nn.Module, tuple[torch.Tensor]]:
    torch.manual_seed(0)
    model = MNISTLinear(_FEAT, _HIDDEN, _CLASSES).to(_DTYPE).eval().to(device)
    x = torch.randn(_BATCH, _FEAT, dtype=_DTYPE).to(device)
    return model, (x,)


@pytest.mark.benchmark
def test_mnist_linear(
    mode: str,
    warmup: int,
    iters: int,
    cpu_baseline: bool,
    accuracy: bool,
    profile_enabled: bool,
    profile_dir: str,
    record_bench,
    tt_device: torch.device,
) -> None:
    model, inputs = _build_inputs(tt_device)
    model = prepare_model(model, mode)

    ref_model, ref_inputs = (None, None)
    if accuracy:
        ref_model, ref_inputs = _build_inputs("cpu")

    record_bench(
        run_benchmark(
            model, inputs, warmup=warmup, iters=iters,
            label="mnist_linear", mode=mode, device="tt",
            reference_model=ref_model, reference_inputs=ref_inputs,
            profile_enabled=profile_enabled, profile_dir=profile_dir,
        )
    )

    if cpu_baseline:
        cpu_model, cpu_inputs = _build_inputs("cpu")
        record_bench(
            run_benchmark(
                cpu_model, cpu_inputs, warmup=warmup, iters=iters,
                label="mnist_linear", mode="eager", device="cpu",
            )
        )
