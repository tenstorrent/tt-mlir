"""Benchmark: ResNet50 forward pass (random weights, real shapes)."""

import pytest
import torch

from ._runner import prepare_model, run_benchmark

torchvision = pytest.importorskip("torchvision")
from torchvision.models import resnet50  # noqa: E402


_DTYPE = torch.bfloat16


def _build_inputs(device: torch.device | str) -> tuple[torch.nn.Module, tuple[torch.Tensor]]:
    torch.manual_seed(0)
    model = resnet50(weights=None).to(_DTYPE).eval().to(device)
    x = torch.randn(1, 3, 224, 224, dtype=_DTYPE).to(device)
    return model, (x,)


@pytest.mark.benchmark
def test_resnet50(
    mode: str,
    warmup: int,
    iters: int,
    cpu_baseline: bool,
    accuracy: bool,
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
            label="resnet50", mode=mode, device="tt",
            reference_model=ref_model, reference_inputs=ref_inputs,
        )
    )

    if cpu_baseline:
        cpu_model, cpu_inputs = _build_inputs("cpu")
        record_bench(
            run_benchmark(
                cpu_model, cpu_inputs, warmup=warmup, iters=iters,
                label="resnet50", mode="eager", device="cpu",
            )
        )
