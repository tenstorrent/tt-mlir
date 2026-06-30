"""Benchmark: ResNet50 forward pass (random weights, real shapes)."""

import copy

import pytest
import torch
from tt_kurbla.torch._compile import CompileOption

from ._runner import prepare_model, run_benchmark

torchvision = pytest.importorskip("torchvision")
from torchvision.models import resnet50  # noqa: E402


_DTYPE = torch.bfloat16


def _build() -> tuple[torch.nn.Module, tuple[torch.Tensor]]:
    """Build the model and inputs once, on CPU."""
    model = resnet50(weights=None).to(_DTYPE).eval()
    x = torch.randn(1, 3, 224, 224, dtype=_DTYPE)
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
    model, inputs = _build()

    # deepcopy because Module.to() is in-place; keep the CPU originals as the reference.
    device_model = prepare_model(copy.deepcopy(model).to(tt_device), mode, options={CompileOption.OPT_LEVEL: 1})
    device_inputs = tuple(t.to(tt_device) for t in inputs)

    record_bench(
        run_benchmark(
            device_model, device_inputs, warmup=warmup, iters=iters,
            label="resnet50", mode=mode, device="tt",
            reference_model=model if accuracy else None,
            reference_inputs=inputs if accuracy else None,
        )
    )

    if cpu_baseline:
        record_bench(
            run_benchmark(
                model, inputs, warmup=warmup, iters=iters,
                label="resnet50", mode="eager", device="cpu",
            )
        )
