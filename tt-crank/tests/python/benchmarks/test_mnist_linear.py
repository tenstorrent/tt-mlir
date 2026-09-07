# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark: small MNIST linear classifier forward pass."""

import copy

import pytest
import torch
from tt_crank.torch._compile import CompileOption

from _models import MNISTLinear

from ._runner import prepare_model, run_benchmark


_BATCH = 64
_FEAT = 28 * 28
_HIDDEN = 128
_CLASSES = 10
_DTYPE = torch.bfloat16


def _build() -> tuple[torch.nn.Module, tuple[torch.Tensor]]:
    """Build the model and inputs once, on CPU."""
    model = MNISTLinear(_FEAT, _HIDDEN, _CLASSES).to(_DTYPE).eval()
    x = torch.randn(_BATCH, _FEAT, dtype=_DTYPE)
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
    opt_level: int | None,
    record_bench,
    tt_device: torch.device,
) -> None:
    model, inputs = _build()

    # deepcopy because Module.to() is in-place; keep the CPU originals as the reference.
    device_model = prepare_model(
        copy.deepcopy(model).to(tt_device),
        mode,
        options={
            CompileOption.OPT_LEVEL: opt_level if opt_level is not None else 0,
        },
    )
    device_inputs = tuple(t.to(tt_device) for t in inputs)

    record_bench(
        run_benchmark(
            device_model,
            device_inputs,
            warmup=warmup,
            iters=iters,
            label="mnist_linear",
            mode=mode,
            device="tt",
            reference_model=model if accuracy else None,
            reference_inputs=inputs if accuracy else None,
            profile_enabled=profile_enabled,
            profile_dir=profile_dir,
        )
    )

    if cpu_baseline:
        record_bench(
            run_benchmark(
                model,
                inputs,
                warmup=warmup,
                iters=iters,
                label="mnist_linear",
                mode="eager",
                device="cpu",
            )
        )
