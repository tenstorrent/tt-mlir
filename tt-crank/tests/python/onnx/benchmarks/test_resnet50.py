# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark: ResNet50 through the ONNX Runtime TT EP (bf16, opt2, trace).

The ONNX analogue of torch/benchmarks/test_resnet50.py: HF microsoft/resnet-50,
batch 8, 224x224. The TT model is exported natively bf16 at opset 22 (ONNX Conv
accepts bf16 from opset 22); the CPU reference is an f32 export. The input is
uploaded once and bound via IOBinding and the output is drained once at the end,
so the timed loop carries no host<->device copies.
"""

import copy
import time

import ml_dtypes
import numpy as np
import onnxruntime as ort
import pytest
import torch

import tt_crank.onnx as tt_onnx
from _resnet_models import Bf16Logits, Logits, hf_resnet50
from bench import BenchmarkResult, Measurement

BATCH = 8
# bf16 against an f32 reference; the torch benchmark uses the same gate.
_PCC_TARGET = 0.9


def _build_models() -> tuple[bytes, bytes]:
    """(bf16 model for TT, f32 model for the CPU reference) from one download."""
    base = hf_resnet50()
    bf16_dummy = torch.randn(BATCH, 3, 224, 224, dtype=torch.bfloat16)
    bf16 = tt_onnx.export_torch(Bf16Logits(copy.deepcopy(base)), bf16_dummy)
    f32 = tt_onnx.export_torch(
        Logits(copy.deepcopy(base)), torch.randn(BATCH, 3, 224, 224)
    )
    return bf16, f32


def _bench_tt(
    session: ort.InferenceSession,
    x: np.ndarray,
    *,
    warmup: int,
    iters: int,
    label: str,
    want: np.ndarray | None,
) -> BenchmarkResult:
    """Time `iters` synchronous runs with the bf16 input pinned on device and the
    output drained once at the end."""
    io_binding, (y_dev,) = tt_onnx.bind(session, {"x": x.astype(ml_dtypes.bfloat16)})

    # Warmup: the first run compiles (opt2 layout analysis lands here).
    warmup_t0 = time.perf_counter_ns()
    for _ in range(warmup):
        session.run_with_iobinding(io_binding)
    warmup_ns = time.perf_counter_ns() - warmup_t0

    t0 = time.perf_counter_ns()
    for _ in range(iters):
        session.run_with_iobinding(io_binding)
    got = tt_onnx.to_host(y_dev)  # single drain: the sync point
    total_ns = time.perf_counter_ns() - t0

    measurements: list[Measurement] = []
    if want is not None:
        pcc = tt_onnx.pcc(got.astype(np.float32), want)
        measurements.append(Measurement("pcc", pcc, "pcc"))
        assert pcc >= _PCC_TARGET, f"{label}: pcc {pcc:.4f} < {_PCC_TARGET}"

    total_ms = total_ns / 1e6
    samples_per_sec = (iters * BATCH * 1e9 / total_ns) if total_ns else 0.0
    measurements += [
        Measurement("warmup_total_ms", warmup_ns / 1e6, "ms"),
        Measurement("total_ms", total_ms, "ms"),
        Measurement("iter_mean_ms", total_ms / iters if iters else 0.0, "ms"),
        Measurement("samples_per_sec", samples_per_sec, "samples/s"),
    ]
    return BenchmarkResult(
        label=label,
        mode="compile",
        device="tt",
        warmup=warmup,
        iters=iters,
        measurements=measurements,
    )


def _bench_cpu(
    session: ort.InferenceSession, x: np.ndarray, *, warmup: int, iters: int, label: str
) -> BenchmarkResult:
    """Plain synchronous CPU-EP loop for the --cpu-baseline comparison row."""
    for _ in range(warmup):
        session.run(None, {"x": x})
    t0 = time.perf_counter_ns()
    for _ in range(iters):
        session.run(None, {"x": x})
    total_ns = time.perf_counter_ns() - t0

    total_ms = total_ns / 1e6
    samples_per_sec = (iters * BATCH * 1e9 / total_ns) if total_ns else 0.0
    return BenchmarkResult(
        label=label,
        mode="eager",
        device="cpu",
        warmup=warmup,
        iters=iters,
        measurements=[
            Measurement("total_ms", total_ms, "ms"),
            Measurement("iter_mean_ms", total_ms / iters if iters else 0.0, "ms"),
            Measurement("samples_per_sec", samples_per_sec, "samples/s"),
        ],
    )


@pytest.mark.benchmark
def test_resnet50_onnx(
    warmup: int,
    iters: int,
    cpu_baseline: bool,
    accuracy: bool,
    opt_level: int | None,
    record_bench,
) -> None:
    model_bf16, model_f32 = _build_models()
    x = tt_onnx.randn(BATCH, 3, 224, 224)
    want = tt_onnx.cpu_golden(model_f32, {"x": x})[0] if accuracy else None

    # Same device config as the torch benchmark: bf16 model, opt2, trace.
    compile_options = {
        "optimization_level": str(opt_level if opt_level is not None else 2),
        "enable_trace": "1",
    }
    session = tt_onnx.session(model_bf16, compile_options=compile_options)
    record_bench(
        _bench_tt(
            session, x, warmup=warmup, iters=iters, label="resnet50-onnx", want=want
        )
    )

    if cpu_baseline:
        cpu = ort.InferenceSession(model_f32, providers=["CPUExecutionProvider"])
        record_bench(
            _bench_cpu(cpu, x, warmup=warmup, iters=iters, label="resnet50-onnx")
        )
