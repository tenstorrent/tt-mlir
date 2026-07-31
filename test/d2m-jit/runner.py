# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Co-located testing infrastructure for d2m-jit patterns.

A pattern file under ``test/d2m-jit/patterns/`` declares its own tests as
module-level data, so one file is the complete, self-contained unit:
kernel + rewrite + tests. The generic runner in ``test/d2m-jit`` discovers
these declarations and turns them into pytest cases — adding pattern #1001
is a zero-diff change to the harness.

Two declaration kinds:

* ``PATTERN_TESTS = [PatternTest(...)]`` — rewrite correctness. Each spec
  carries an input TTIR module and FileCheck directives; the runner applies
  the file's pattern(s) via ``apply_patterns_text`` and pipes the rewritten
  IR through the real ``FileCheck`` binary. Replaces the hand-written
  ``test/d2m-jit/lit/*_pattern.py`` files. No device needed.

* ``KERNEL_BENCHES = [KernelBench(...)]`` — on-device numerics, **in-process**.
  Each bench drives the ``@d2m.kernel`` entrypoint directly with an explicit
  ``(layout, block_shape, grid_shape)`` config and PCC-compares against a
  torch golden. Replaces ``test/d2m-jit/test_pattern_eltwise.py``.

* ``PatternTest(..., e2e=True)`` — **true e2e device execution**, IN-PROCESS.
  The rewritten module is compiled to a flatbuffer held *in memory* and run on
  device via the in-process tt-metal runtime (no ttrt subprocess, no flatbuffer
  or tensor files on disk); the device output is read straight back into a torch
  tensor and PCC-checked against a reference. Disk footprint is ~zero regardless
  of pattern count, and one device handle is reused per run. Inputs are
  generated deterministically from the ttir signature. The reference is the
  spec's ``golden`` if given, else the **ttnn device baseline** of the original
  (pre-pattern) TTIR — compiled via ``ttir -> ttnn`` and run on device, cached
  per (module, inputs). So a hand-written golden is optional. See
  ``compile_spec_to_fbb`` / ``compile_ttir_to_ttnn_fbb`` /
  ``execute_ttm_in_process`` / ``ttnn_baseline_outputs`` / ``run_e2e`` below
  (modelled on builder_runtime.py::execute_fb, plain-torch).

  Scalar kernel args are supported: in a rewrite scope they are always Python
  int constants, so the emitter bakes them into the kernel body as in-region
  constants (not host-scope ``additionalArgs``), leaving nothing for the
  flatbuffer translator to choke on. The in-process lazy builder takes the
  opposite route — there a scalar becomes an ``index`` function param so the
  binary stays parameterised and the runtime supplies its value per call.

Not implemented yet:

* **Autotuning** — ``KernelBench.space`` declares axes (block_shape /
  grid_shape / dtype) an autotuner would sweep, taking perf traces per config.
"""

from __future__ import annotations

import os
import math
import shutil
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch

from ttmlir import ir

# ----------------------------------------------------------------------
# Spec dataclasses (the data an agent emits / tweaks per pattern)
# ----------------------------------------------------------------------


@dataclass
class InputSpec:
    """How to materialise torch input tensors for a test.

    ``dist`` is either a named distribution string (``"uniform(-1,1)"``,
    ``"randn"``, ``"rand"``) or a callable ``(shape, torch_dtype, generator)
    -> tensor`` for full control. ``seed`` keeps generation deterministic.
    """

    dist: "str | Callable" = "uniform(-1,1)"
    seed: int = 0


@dataclass
class PatternTest:
    """Rewrite-correctness spec: one input module -> FileCheck.

    The ``ttir`` module's function signature is the single source of truth
    for input shapes/dtypes, so shapes are not duplicated in the e2e device
    runner. ``golden`` and ``inputs`` define deterministic device validation.
    """

    name: str
    ttir: str
    check: str = ""
    golden: Optional[Callable] = None
    inputs: InputSpec = field(default_factory=InputSpec)
    pcc: float = 0.99
    # Optional relative tolerance for the output standard-deviation ratio.
    # PCC alone cannot detect uniform amplitude errors.
    std_rtol: Optional[float] = None
    expect_match: bool = True
    # Opt in to true e2e device execution (rewrite -> compile -> in-process run).
    # `golden` is optional: when omitted, the runner cross-checks against the
    # TTNN device baseline of the original (pre-pattern) TTIR.
    e2e: bool = False
    use_tile_matmul: Optional[bool] = None
    num_stream_buffers: Optional[int] = None
    tags: tuple = ()
    source_file: str = ""  # set by discovery
    # Additional pattern files to apply together. Relative paths are resolved
    # beside source_file so composed tests remain relocatable.
    pattern_files: tuple[str, ...] = ()
    # Select which function result is compared in multi-output e2e tests.
    output_index: int = 0
    # Optional view applied to the selected runtime result before PCC. The
    # golden must return the corresponding selected value.
    output_selector: Optional[Callable] = None
    # Most goldens use FP32 inputs. Large-model specs can opt out and cast only
    # the operands their partial-output golden actually consumes.
    golden_inputs_as_float: bool = True


@dataclass
class TuneAxis:
    """One autotuning axis: a named config key and its candidate values."""

    name: str
    values: list


@dataclass
class KernelBench:
    """Direct-kernel device bench (numerics today, autotuning later).

    ``run(kernel, inputs, cfg) -> host tensor`` is the only pattern-specific
    glue: it maps a concrete ``cfg`` (block_shape / grid_shape / dtype) to a
    Layout and the kernel's call args. ``eltwise_block_run`` covers the
    common elementwise-block shape, so most patterns just set ``space``.
    """

    name: str
    kernel: Callable
    golden: Callable
    input_shapes: Sequence[tuple]
    run: Callable
    inputs: InputSpec = field(default_factory=InputSpec)
    default_cfg: dict = field(
        default_factory=lambda: dict(
            block_shape=[1, 1], grid_shape=[1, 1], dtype="float32"
        )
    )
    space: list = field(default_factory=list)
    pcc: float = 0.99
    source_file: str = ""


# ----------------------------------------------------------------------
# dtype helpers
# ----------------------------------------------------------------------

_TORCH_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

_MLIR_ELTY_TO_TORCH = {
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "i64": torch.int64,
    "si64": torch.int64,
    "i32": torch.int32,
    "si32": torch.int32,
    "ui32": torch.uint32,
}


def torch_dtype(name: str) -> torch.dtype:
    return _TORCH_DTYPES[name]


def d2m_dtype(name: str):
    import d2m_jit as d2m

    return {
        "float32": d2m.float32,
        "float16": d2m.float16,
        "bfloat16": d2m.bfloat16,
    }[name]


# ----------------------------------------------------------------------
# Input generation
# ----------------------------------------------------------------------


def _gen_tensor(shape, td, dist, gen):
    if callable(dist):
        return dist(shape, td, gen)
    spec = dist.strip()
    if spec.startswith("uniform"):
        lo, hi = (
            float(x) for x in spec[spec.index("(") + 1 : spec.index(")")].split(",")
        )
        return torch.rand(shape, generator=gen, dtype=td) * (hi - lo) + lo
    if spec == "randn":
        return torch.randn(shape, generator=gen, dtype=td)
    if spec == "rand":
        return torch.rand(shape, generator=gen, dtype=td)
    raise ValueError(f"unknown input distribution: {dist!r}")


def make_inputs(shapes, td, inspec: InputSpec):
    """Deterministically generate one torch tensor per shape."""
    gen = torch.Generator()
    gen.manual_seed(inspec.seed)
    return [_gen_tensor(tuple(s), td, inspec.dist, gen) for s in shapes]


def parse_func_io(ttir_text: str):
    """Return ``[(shape, torch_dtype), ...]`` for the first nested func's args."""

    def walk(operation):
        yield operation
        for region in operation.regions:
            for block in region.blocks:
                for child in block.operations:
                    yield from walk(child.operation)

    ctx = ir.Context()
    ctx.load_all_available_dialects()
    mod = ir.Module.parse(ttir_text, ctx)
    for operation in walk(mod.operation):
        if operation.name == "func.func":
            block = operation.regions[0].blocks[0]
            out = []
            for a in block.arguments:
                rt = ir.RankedTensorType(a.type)
                out.append((tuple(rt.shape), _MLIR_ELTY_TO_TORCH[str(rt.element_type)]))
            return out
    raise ValueError("no func.func found in module")


# ----------------------------------------------------------------------
# PCC
# ----------------------------------------------------------------------


def calculate_pcc(golden, actual, chunk_elements: int = 1 << 20) -> float:
    if chunk_elements <= 0:
        raise ValueError("PCC chunk_elements must be positive")
    if golden.shape != actual.shape:
        raise ValueError(
            f"PCC shape mismatch: golden {golden.shape}, actual {actual.shape}"
        )
    golden = golden.flatten().float()
    actual = actual.flatten().float()
    if golden.numel() == 0:
        raise ValueError("PCC requires non-empty tensors")

    golden_sum = 0.0
    actual_sum = 0.0
    for begin in range(0, golden.numel(), chunk_elements):
        end = min(begin + chunk_elements, golden.numel())
        golden_sum += golden[begin:end].double().sum().item()
        actual_sum += actual[begin:end].double().sum().item()

    golden_mean = golden_sum / golden.numel()
    actual_mean = actual_sum / actual.numel()
    covariance = 0.0
    golden_variance = 0.0
    actual_variance = 0.0
    for begin in range(0, golden.numel(), chunk_elements):
        end = min(begin + chunk_elements, golden.numel())
        golden_delta = golden[begin:end].double() - golden_mean
        actual_delta = actual[begin:end].double() - actual_mean
        covariance += torch.dot(golden_delta, actual_delta).item()
        golden_variance += torch.dot(golden_delta, golden_delta).item()
        actual_variance += torch.dot(actual_delta, actual_delta).item()

    if golden_variance == 0.0 or actual_variance == 0.0:
        return 1.0 if torch.equal(golden, actual) else 0.0
    return covariance / (golden_variance * actual_variance) ** 0.5


def calculate_std(value, chunk_elements: int = 1 << 20) -> float:
    if chunk_elements <= 0:
        raise ValueError("std chunk_elements must be positive")
    value = value.flatten().float()
    if value.numel() == 0:
        raise ValueError("std requires a non-empty tensor")

    total = 0.0
    square_total = 0.0
    for begin in range(0, value.numel(), chunk_elements):
        chunk = value[begin : begin + chunk_elements].double()
        total += chunk.sum().item()
        square_total += torch.dot(chunk, chunk).item()
    mean = total / value.numel()
    variance = square_total / value.numel() - mean * mean
    return math.sqrt(max(variance, 0.0))


def assert_pcc(golden, actual, threshold: float = 0.99):
    pcc = calculate_pcc(golden, actual)
    assert (
        pcc >= threshold
    ), f"Expected pcc {pcc} >= {threshold} for shape {golden.shape}"


# ----------------------------------------------------------------------
# Rewrite + FileCheck (no device)
# ----------------------------------------------------------------------


def run_rewrite(spec: PatternTest) -> str:
    """Apply just this pattern file's rewrites to the spec's TTIR module.

    Uses ``apply_patterns_text``, which snapshots/clears/restores the global
    pattern registry, so each spec runs in isolation even with thousands of
    pattern files imported into the process.
    """
    from d2m_jit._src.rewrite import apply_patterns_text

    if not spec.source_file:
        raise ValueError(
            f"PatternTest {spec.name!r} has no source_file (discovery sets it)"
        )
    pattern_files = spec.pattern_files or (spec.source_file,)
    source_dir = os.path.dirname(spec.source_file)
    resolved = [
        path if os.path.isabs(path) else os.path.join(source_dir, path)
        for path in pattern_files
    ]
    return apply_patterns_text(spec.ttir, resolved)


def _filecheck_bin() -> str:
    for cand in (
        shutil.which("FileCheck"),
        os.path.join(os.environ.get("TTMLIR_TOOLCHAIN_DIR", ""), "bin", "FileCheck"),
        "/opt/ttmlir-toolchain/bin/FileCheck",
    ):
        if cand and os.path.exists(cand):
            return cand
    raise RuntimeError("FileCheck binary not found (set TTMLIR_TOOLCHAIN_DIR or PATH)")


def filecheck(check_text: str, ir_text: str):
    """Run the real FileCheck binary: checks from ``check_text``, IR on stdin."""
    check_text = textwrap.dedent(check_text).strip() + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".check", delete=False) as f:
        f.write(check_text)
        checkfile = f.name
    try:
        proc = subprocess.run(
            [_filecheck_bin(), checkfile],
            input=ir_text,
            capture_output=True,
            text=True,
        )
    finally:
        os.unlink(checkfile)
    if proc.returncode != 0:
        raise AssertionError(
            f"FileCheck failed:\n{proc.stderr}\n--- checks ---\n{check_text}\n"
            f"--- rewritten IR ---\n{ir_text}"
        )


# ----------------------------------------------------------------------
# Device run helpers (direct-kernel path)
# ----------------------------------------------------------------------


def eltwise_block_run(kernel, inputs, cfg):
    """Stock ``run`` for elementwise-block kernels.

    Builds one tiled Layout from ``cfg`` shared by all inputs and the output,
    derives ``m_blocks``/``n_blocks`` from the shape and grid, and calls
    ``kernel(*inputs, out, m_blocks, n_blocks, grid=...)``.
    """
    import d2m_jit as d2m

    ref = inputs[0]
    gy, gx = cfg["grid_shape"]
    L = d2m.Layout(
        shape=tuple(ref.shape),
        dtype=d2m_dtype(cfg["dtype"]),
        block_shape=list(cfg["block_shape"]),
        grid_shape=[gy, gx],
        tiled=True,
    )
    ins = [d2m.to_layout(t, L) for t in inputs]
    out = d2m.empty(L)
    m_blocks = (ref.shape[-2] // 32) // gy
    n_blocks = (ref.shape[-1] // 32) // gx
    kernel(*ins, out, m_blocks, n_blocks, grid=(gy, gx))
    return out.to_host()


def run_bench(bench: KernelBench, cfg: Optional[dict] = None):
    """Execute one bench at ``cfg`` (default: ``bench.default_cfg``) and
    return ``(actual, expected)`` torch tensors for PCC comparison."""
    cfg = cfg or bench.default_cfg
    td = torch_dtype(cfg["dtype"])
    inputs = make_inputs(bench.input_shapes, td, bench.inputs)
    actual = bench.run(bench.kernel, inputs, cfg)
    expected = bench.golden(*inputs)
    return actual, expected


# ----------------------------------------------------------------------
# True e2e device backend: rewrite -> compile -> IN-PROCESS run -> PCC.
#
# The rewritten module is compiled to a flatbuffer held *in memory* and run on
# device via the in-process tt-metal runtime (``_ttmlir_runtime``) — no ttrt
# subprocess, no flatbuffer/tensor files on disk. The device output is read
# straight back into a torch tensor and PCC-compared against the golden. Disk
# footprint is ~zero regardless of pattern count, and one device handle is
# reused across every pattern (open once per session).
#
# This mirrors ``tools/builder/base/builder_runtime.py::execute_fb`` but with
# plain-torch I/O (no GoldenMapTensor dependency); the comparison is done here
# with ``assert_pcc``. Scalar kernel args are supported: in a rewrite scope they
# are baked into the kernel body as in-region constants, so the flatbuffer has
# no unserialisable scalar program args.
# ----------------------------------------------------------------------

# torch <-> runtime DataType (subset; mirrors builder_runtime).
_TORCH_TO_RT = {}
_RT_STR_TO_TORCH = {}


def _rt():
    from _ttmlir_runtime import runtime

    if not _TORCH_TO_RT:
        dt = runtime.DataType
        _TORCH_TO_RT.update(
            {
                torch.float32: dt.Float32,
                torch.float16: dt.Float16,
                torch.bfloat16: dt.BFloat16,
                torch.int32: dt.Int32,
                torch.uint32: dt.UInt32,
                torch.uint16: dt.UInt16,
                torch.uint8: dt.UInt8,
            }
        )
        _RT_STR_TO_TORCH.update(
            {
                "Float32": torch.float32,
                "Float16": torch.float16,
                "BFloat16": torch.bfloat16,
                "Int32": torch.int32,
                "UInt32": torch.uint32,
                "UInt16": torch.uint16,
                "UInt8": torch.uint8,
            }
        )
    return runtime


def compile_spec_to_fbb(spec: PatternTest):
    """Rewrite ``spec.ttir`` with its pattern, lower to ttmetal, and return the
    loaded flatbuffer Binary held *in memory* (no file written).

    Scalar kernel args are baked into the kernel body as in-region constants by
    the rewrite-scope emitter, so the flatbuffer has no scalar program args.
    """
    from ttmlir.passes import (
        ttir_to_ttmetal_backend_pipeline,
        ttmetal_to_flatbuffer_bin,
    )

    from _ttmlir_runtime import binary as _rt_binary
    from d2m_jit._src.builder import _get_system_desc_path

    # run_rewrite already applies *only* this file's pattern(s), in isolation.
    rewritten = run_rewrite(spec)
    ctx = ir.Context()
    ctx.load_all_available_dialects()
    module = ir.Module.parse(rewritten, ctx)

    sd = _get_system_desc_path()
    options = [
        "default-input-memspace=dram",
        "enable-form-expressions=false",
        "default-output-memspace=dram",
    ]
    if sd:
        options.append(f"system-desc-path={sd}")
    if spec.use_tile_matmul is not None:
        value = str(spec.use_tile_matmul).lower()
        options.append(f"use-tile-matmul={value}")
    if spec.num_stream_buffers is not None:
        options.append(f"num-stream-buffers={spec.num_stream_buffers}")
    ttir_to_ttmetal_backend_pipeline(module, " ".join(options))

    capsule = ttmetal_to_flatbuffer_bin(module)
    return _rt_binary.load_binary_from_capsule(capsule)


class E2EDevice:
    """Lazily opens a mesh device and reuses it across all e2e runs.

    The device is opened on first use (with the first flatbuffer's mesh shape)
    and closed at session teardown — one device-open amortized across every
    pattern, all in-process. Physical device IDs can be supplied explicitly or
    through ``TTMLIR_E2E_DEVICE_IDS`` as a comma-separated list. The dispatch
    core type can be selected with ``TTMLIR_E2E_DISPATCH_CORE_TYPE``."""

    def __init__(
        self,
        device_ids: Optional[Sequence[int]] = None,
        dispatch_core_type: Optional[str] = None,
    ):
        self.device = None
        if device_ids is None:
            env_device_ids = os.environ.get("TTMLIR_E2E_DEVICE_IDS")
            if env_device_ids is not None:
                try:
                    device_ids = [int(value) for value in env_device_ids.split(",")]
                except ValueError as error:
                    raise ValueError(
                        "TTMLIR_E2E_DEVICE_IDS must be a comma-separated list "
                        "of integers"
                    ) from error
        self.device_ids = None if device_ids is None else list(device_ids)
        if self.device_ids is not None and not self.device_ids:
            raise ValueError("device_ids must contain at least one physical device ID")
        if dispatch_core_type is None:
            dispatch_core_type = os.environ.get("TTMLIR_E2E_DISPATCH_CORE_TYPE")
        if dispatch_core_type is not None:
            dispatch_core_type = dispatch_core_type.upper()
            if dispatch_core_type not in ("WORKER", "ETH"):
                raise ValueError("dispatch_core_type must be either 'WORKER' or 'ETH'")
        self.dispatch_core_type = dispatch_core_type

    def options(self, fbb, program_index: int = 0):
        runtime = _rt()
        mesh_shape = list(fbb.get_program_mesh_shape(program_index))
        opts = runtime.MeshDeviceOptions()
        opts.mesh_shape = mesh_shape
        if self.device_ids is not None:
            mesh_volume = 1
            for dimension in mesh_shape:
                mesh_volume *= dimension
            if len(self.device_ids) != mesh_volume:
                raise ValueError(
                    f"program mesh {mesh_shape} requires {mesh_volume} devices, "
                    f"but device_ids contains {len(self.device_ids)}"
                )
            opts.device_ids = self.device_ids
        if self.dispatch_core_type is not None:
            opts.dispatch_core_type = getattr(
                runtime.DispatchCoreType, self.dispatch_core_type
            )
        return opts

    def get(self, fbb, program_index: int = 0):
        runtime = _rt()
        if self.device is None:
            runtime.set_compatible_device_runtime(fbb)
            self.device = runtime.open_mesh_device(self.options(fbb, program_index))
        return self.device

    def close(self):
        if self.device is not None:
            _rt().close_mesh_device(self.device)
            self.device = None


def execute_ttm_in_process(fbb, inputs, device, program_index: int = 0):
    """Submit ``fbb`` on ``device`` with torch ``inputs``; return torch outputs.

    No files, no subprocess. Inputs are marshalled to borrowed host tensors and
    converted to each program input's expected layout; outputs are copied back
    into freshly allocated torch tensors (shape/dtype from the program output
    descriptors). Mirrors the core of ``execute_fb`` for a single device.
    """
    import json
    import re

    runtime = _rt()
    in_json = fbb.get_program_inputs_as_json(program_index)
    input_descs = json.loads(
        re.sub(r"\binf\b", "Infinity", re.sub(r"\bnan\b", "NaN", in_json))
    )

    rt_inputs = []
    host_inputs = []
    for t in inputs:
        desc = input_descs[len(rt_inputs)]["desc"]
        expected_dtype = _RT_STR_TO_TORCH[desc["layout"]["memory_desc"]["data_type"]]
        if t.dtype != expected_dtype:
            t = t.to(expected_dtype)
        t = t.contiguous()
        # Borrowed host tensors do not own their backing storage. Keep any
        # converted/contiguous temporaries alive until queued transfers finish.
        host_inputs.append(t)
        rt_in = runtime.create_borrowed_host_tensor(
            t.data_ptr(),
            list(t.shape),
            list(t.stride()),
            t.element_size(),
            _TORCH_TO_RT[t.dtype],
        )
        layout = runtime.get_layout(fbb, program_index, len(rt_inputs))
        rt_inputs.append(runtime.to_layout(rt_in, device, layout, True))

    runtime.set_compatible_device_runtime(fbb)
    rt_outputs = runtime.submit(device, fbb, program_index, rt_inputs)
    runtime.wait(rt_outputs)

    out_json = fbb.get_program_outputs_as_json(program_index)
    out_descs = json.loads(
        re.sub(r"\binf\b", "Infinity", re.sub(r"\bnan\b", "NaN", out_json))
    )

    results = []
    for i, rt_out in enumerate(rt_outputs):
        desc = out_descs[i]["desc"]
        shape = desc["shape"]
        dtype = _RT_STR_TO_TORCH[desc["layout"]["memory_desc"]["data_type"]]
        t_out = torch.empty(shape, dtype=dtype)
        rt_host = runtime.create_borrowed_host_tensor(
            t_out.data_ptr(),
            list(t_out.shape),
            list(t_out.stride()),
            t_out.element_size(),
            _TORCH_TO_RT[dtype],
        )
        host_view = runtime.to_host(rt_out, untilize=True)[0]
        runtime.memcpy(rt_host, host_view)
        runtime.deallocate_tensor(rt_out, force=True)
        results.append(t_out)
    return results


# ----------------------------------------------------------------------
# TTNN reference baseline (golden-free cross-check)
#
# When a PatternTest has no hand-written ``golden``, the e2e runner falls back
# to a *device* reference: the ORIGINAL (pre-pattern) TTIR compiled straight
# through the standard ``ttir -> ttnn`` pipeline and run on device. The
# pattern's ttmetal output is PCC-checked against this ttnn output, so a golden
# is optional.
#
# A device handle is bound to the runtime it was opened under (ttnn and ttmetal
# cannot share one -- the runtime asserts on a cross-runtime cast), so the
# baseline runs in its own short-lived ttnn device session, with the shared
# ttmetal device closed first. Baseline outputs are cached (keyed by the TTIR
# text + the input bytes), so the cross-check runs ttnn on device only once per
# unique (module, inputs) -- repeat runs do no baseline device work at all.
# ----------------------------------------------------------------------

_BASELINE_CACHE = {}


def compile_ttir_to_ttnn_fbb(ttir_text: str):
    """Compile pre-pattern TTIR through the standard ``ttir -> ttnn`` pipeline
    and return the loaded ttnn flatbuffer Binary held *in memory* (no file).

    This is the reference path for the golden-free cross-check and for perf
    comparison against the pattern-lowered ttmetal result of the same TTIR."""
    from _ttmlir_runtime import binary as _rt_binary
    from ttmlir.passes import ttir_to_ttnn_runtime_pipeline, ttnn_to_flatbuffer_bin

    from d2m_jit._src.builder import _get_system_desc_path

    ctx = ir.Context()
    ctx.load_all_available_dialects()
    module = ir.Module.parse(ttir_text, ctx)
    sd = _get_system_desc_path()
    ttir_to_ttnn_runtime_pipeline(module, f"system-desc-path={sd}")
    return _rt_binary.load_binary_from_capsule(ttnn_to_flatbuffer_bin(module))


def _inputs_cache_key(inputs):
    """Stable key for a list of torch input tensors (shape/dtype/bytes)."""
    import hashlib

    h = hashlib.sha1()
    for t in inputs:
        tc = t.detach().contiguous().cpu()
        h.update(str(tuple(tc.shape)).encode())
        h.update(str(tc.dtype).encode())
        if tc.dtype == torch.bfloat16:
            tc = tc.to(torch.float32)
        h.update(tc.numpy().tobytes())
    return h.hexdigest()


def ttnn_baseline_outputs(ttir_text: str, inputs, e2e_device: "E2EDevice"):
    """Return the ttnn device reference outputs for ``ttir_text`` on ``inputs``.

    Cached by ``(ttir_text, input-bytes)``: on a miss, the shared ttmetal
    ``e2e_device`` is closed (the two runtimes can't hold a device open at
    once), the ttnn baseline is compiled and run in its own device session, and
    its torch outputs are cached. On a hit, no device is touched."""
    key = (ttir_text, _inputs_cache_key(inputs))
    if key in _BASELINE_CACHE:
        return _BASELINE_CACHE[key]

    runtime = _rt()
    # Free the shared ttmetal device so ttnn and ttmetal are never open at once.
    e2e_device.close()
    fbb = compile_ttir_to_ttnn_fbb(ttir_text)
    runtime.set_compatible_device_runtime(fbb)
    device = runtime.open_mesh_device(e2e_device.options(fbb))
    try:
        outs = execute_ttm_in_process(fbb, inputs, device)
    finally:
        runtime.close_mesh_device(device)

    _BASELINE_CACHE[key] = outs
    return outs


def run_e2e(spec: PatternTest, e2e_device: "E2EDevice"):
    """Compile ``spec`` to a flatbuffer, run it in-process, and return
    ``(pcc, expected, actual)`` for ``spec.output_index``. Inputs are generated
    deterministically from the spec's ttir signature.

    The reference (``expected``) is either the spec's ``golden`` evaluated on
    those inputs, or -- when no ``golden`` is given -- the cached ttnn device
    baseline of the original TTIR (see ``ttnn_baseline_outputs``). The baseline
    is computed before the ttmetal device is opened, so the two runtimes never
    contend for a device."""
    io = parse_func_io(spec.ttir)
    gen = torch.Generator()
    gen.manual_seed(spec.inputs.seed)
    inputs = [_gen_tensor(shape, td, spec.inputs.dist, gen) for shape, td in io]

    if spec.golden is not None:
        golden_inputs = inputs
        if spec.golden_inputs_as_float:
            golden_inputs = [tensor.float() for tensor in inputs]
        expected = spec.golden(*golden_inputs)
    else:
        # Golden-free cross-check: device reference via the ttnn baseline.
        baseline = ttnn_baseline_outputs(spec.ttir, inputs, e2e_device)
        expected = baseline[spec.output_index].float()

    fbb = compile_spec_to_fbb(spec)
    device = e2e_device.get(fbb)
    outputs = execute_ttm_in_process(fbb, inputs, device)

    actual = outputs[spec.output_index].float()
    if spec.output_selector is not None:
        actual = spec.output_selector(actual)
    pcc = calculate_pcc(expected, actual)
    return pcc, expected, actual


# ----------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------

_DISCOVERED = None


def discover(force: bool = False):
    """Import every pattern module from the co-located ``patterns/`` directory
    and collect declared specs.

    Returns ``(pattern_tests, kernel_benches)`` with ``source_file`` stamped
    on each. Cached after the first call.
    """
    global _DISCOVERED
    if _DISCOVERED is not None and not force:
        return _DISCOVERED

    import importlib.util

    pkg_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "kernels", "patterns"
    )
    pattern_tests, kernel_benches = [], []
    for fn in sorted(os.listdir(pkg_dir)):
        # Underscore-prefixed files are scaffolding (templates, shared
        # helpers), not discoverable patterns.
        if not fn.endswith(".py") or fn.startswith("_"):
            continue
        file_path = os.path.join(pkg_dir, fn)
        spec = importlib.util.spec_from_file_location(fn[:-3], file_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for t in getattr(mod, "PATTERN_TESTS", []):
            t.source_file = mod.__file__
            pattern_tests.append(t)
        for b in getattr(mod, "KERNEL_BENCHES", []):
            b.source_file = mod.__file__
            kernel_benches.append(b)

    _DISCOVERED = (pattern_tests, kernel_benches)
    return _DISCOVERED
