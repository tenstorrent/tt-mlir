"""Benchmark: ResNet50 forward pass (HF microsoft/resnet-50, real shapes)."""

import copy

import pytest
import torch
from tt_kurbla.torch._compile import CompileOption

from ._runner import prepare_model, run_benchmark

transformers = pytest.importorskip("transformers")
from transformers import ResNetForImageClassification  # noqa: E402


_DTYPE = torch.bfloat16


def _build() -> tuple[torch.nn.Module, tuple[torch.Tensor]]:
    """Build the model and inputs once, on CPU.

    Matches tt-xla's vision benchmark: HF microsoft/resnet-50
    (ResNetForImageClassification), bfloat16, batch 8, 224x224.
    """
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50").to(_DTYPE).eval()
    x = torch.randn(8, 3, 224, 224, dtype=_DTYPE)
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
