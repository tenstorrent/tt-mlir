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

* ``KERNEL_BENCHES = { "name": KernelBench(...) }`` — on-device numerics, **in-process**.
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

* **Autotuning** — perf traces per config to rank execution parameters.
"""

from __future__ import annotations

import os
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
    """How to materialise torch input tensors for a PatternTest.

    ``dist`` is either a named distribution string (``"uniform(-1,1)"``,
    ``"randn"``, ``"rand"``) or a callable ``(shape, torch_dtype, generator)
    -> tensor`` for full control. ``seed`` keeps generation deterministic.
    """

    dist: "str | Callable" = "uniform(-1,1)"
    seed: int = 0


@dataclass
class TensorSpec:
    """Shape and layout for one input tensor in a KernelBench.

    Fully describes a single kernel input: the logical shape, the tile-level
    block count per ``remote_load``, the element dtype, and the random
    distribution used to generate test data. All four are tensor-level
    properties — they can differ between inputs in the same kernel (e.g.
    lhs/rhs in a mixed-precision matmul).

    ``grid_shape`` is the only graph-level execution parameter and lives on
    ``KernelBench`` directly.
    """

    shape: tuple
    block_shape: list
    dtype: torch.dtype
    dist: "str | Callable" = "uniform(-1,1)"


@dataclass
class PatternTest:
    """Rewrite-correctness spec: one input module -> FileCheck.

    The ``ttir`` module's function signature is the single source of truth
    for input shapes/dtypes (used by the forthcoming e2e device runner, so
    shapes are not duplicated). ``golden``/``inputs`` are carried for that
    future e2e path and ignored by today's FileCheck-only runner.
    """

    name: str
    ttir: str
    check: str = ""
    golden: Optional[Callable] = None
    inputs: InputSpec = field(default_factory=InputSpec)
    pcc: float = 0.99
    expect_match: bool = True
    # Opt in to true e2e device execution (rewrite -> compile -> in-process run).
    # `golden` is optional: when omitted, the runner cross-checks against the
    # TTNN device baseline of the original (pre-pattern) TTIR.
    e2e: bool = False
    tags: tuple = ()
    source_file: str = ""  # set by discovery


@dataclass
class KernelBench:
    """Direct-kernel device bench: numerics and testing.

    ``tensors`` declares one ``TensorSpec`` per kernel INPUT, fully describing
    each tensor: shape, block_shape, dtype, and input distribution.
    ``grid_shape`` is the only graph-level parameter — the execution grid
    shared by all tensors in the kernel call.

    The materializer ``run(kernel, inputs, tensors, grid_shape) -> host_tensor``
    receives the generated torch inputs alongside the full ``TensorSpec`` list,
    so it can build per-tensor ``d2m.Layout`` objects directly.

    ``name`` is set by ``discover()`` from the key in ``KERNEL_BENCHES``; it
    is empty when the bench is used directly without discovery.
    """

    kernel: Callable
    golden: Callable
    run: Callable
    tensors: "list[TensorSpec]"
    grid_shape: "tuple | list"
    seed: int = 0
    pcc: float = 0.99
    name: str = ""  # stamped by discover() from the KERNEL_BENCHES dict key


# ----------------------------------------------------------------------
# dtype helpers
# ----------------------------------------------------------------------

_MLIR_ELTY_TO_TORCH = {
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
}


def d2m_dtype(dtype: torch.dtype):
    import d2m_jit as d2m

    return {
        torch.float32: d2m.float32,
        torch.float16: d2m.float16,
        torch.bfloat16: d2m.bfloat16,
    }[dtype]


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


def make_inputs(tensors: "list[TensorSpec]", seed: int = 0):
    """Deterministically generate one torch tensor per TensorSpec."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return [_gen_tensor(tuple(ts.shape), ts.dtype, ts.dist, gen) for ts in tensors]


def parse_func_io(ttir_text: str):
    """Return ``[(shape, torch_dtype), ...]`` for the first func's args."""
    ctx = ir.Context()
    ctx.load_all_available_dialects()
    mod = ir.Module.parse(ttir_text, ctx)
    for op in mod.body.operations:
        if op.operation.name == "func.func":
            block = op.regions[0].blocks[0]
            out = []
            for a in block.arguments:
                rt = ir.RankedTensorType(a.type)
                out.append((tuple(rt.shape), _MLIR_ELTY_TO_TORCH[str(rt.element_type)]))
            return out
    raise ValueError("no func.func found in module")


# ----------------------------------------------------------------------
# PCC
# ----------------------------------------------------------------------


def compute_pcc(golden, actual) -> float:
    """Compute Pearson correlation coefficient between two tensors."""
    combined = torch.stack([golden.flatten().float(), actual.flatten().float()])
    pcc = torch.corrcoef(combined)[0, 1].item()
    return pcc


def assert_pcc(golden, actual, threshold: float = 0.99):
    pcc = compute_pcc(golden, actual)
    assert (
        pcc >= threshold
    ), f"Expected pcc {pcc} >= {threshold}\ngolden:\n{golden}\nactual:\n{actual}"


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
    return apply_patterns_text(spec.ttir, [spec.source_file])


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


def eltwise_block_run(kernel, inputs, tensors, grid_shape):
    """Stock materializer for elementwise-block kernels.

    All input tensors share the same shape and block layout; builds one Layout
    from the first TensorSpec, wraps all inputs and the output in it, derives
    ``m_blocks``/``n_blocks`` from the shape and grid, and calls
    ``kernel(*inputs, out, m_blocks, n_blocks, grid=...)``.
    """
    import d2m_jit as d2m

    ts = tensors[0]
    gy, gx = grid_shape
    L = d2m.Layout(
        shape=tuple(ts.shape),
        dtype=d2m_dtype(ts.dtype),
        block_shape=list(ts.block_shape),
        grid_shape=[gy, gx],
        tiled=True,
    )
    ins = [d2m.to_layout(t, L) for t in inputs]
    out = d2m.empty(L)
    m_blocks = (ts.shape[-2] // 32) // gy
    n_blocks = (ts.shape[-1] // 32) // gx
    kernel(*ins, out, m_blocks, n_blocks, grid=(gy, gx))
    return out.to_host()


def run_bench(bench: KernelBench, *, tensors=None, grid_shape=None):
    """Execute one bench and return ``(actual, expected)`` torch tensors.

    Each keyword argument overrides the corresponding field of ``bench``;
    omitted arguments fall back to the bench's defaults.
    """
    tensors = tensors if tensors is not None else bench.tensors
    grid_shape = grid_shape if grid_shape is not None else bench.grid_shape
    inputs = make_inputs(tensors, bench.seed)
    actual = bench.run(bench.kernel, inputs, tensors, grid_shape)
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
    from ttmlir.passes import ttmetal_to_flatbuffer_bin
    from ttmlir.passmanager import PassManager

    from _ttmlir_runtime import binary as _rt_binary
    from d2m_jit._src.builder import _get_system_desc_path, _pipeline_passes

    # run_rewrite already applies *only* this file's pattern(s), in isolation.
    rewritten = run_rewrite(spec)
    ctx = ir.Context()
    ctx.load_all_available_dialects()
    module = ir.Module.parse(rewritten, ctx)

    sd = _get_system_desc_path()
    register = "ttcore-register-device"
    if sd:
        register += f"{{system-desc-path={sd}}}"
    pipeline_str = f"builtin.module({register},{','.join(_pipeline_passes())})"
    pm = PassManager.parse(pipeline_str, context=ctx)
    pm.enable_verifier(True)
    pm.run(module.operation)

    capsule = ttmetal_to_flatbuffer_bin(module)
    return _rt_binary.load_binary_from_capsule(capsule)


class E2EDevice:
    """Lazily opens a single mesh device and reuses it across all e2e runs.

    The device is opened on first use (with the first flatbuffer's mesh shape)
    and closed at session teardown — one device-open amortized across every
    pattern, all in-process."""

    def __init__(self):
        self.device = None

    def get(self, fbb, program_index: int = 0):
        runtime = _rt()
        if self.device is None:
            opts = runtime.MeshDeviceOptions()
            opts.mesh_shape = fbb.get_program_mesh_shape(program_index)
            runtime.set_compatible_device_runtime(fbb)
            self.device = runtime.open_mesh_device(opts)
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

    rt_inputs = []
    for t in inputs:
        t = t.contiguous()
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
    opts = runtime.MeshDeviceOptions()
    opts.mesh_shape = fbb.get_program_mesh_shape(0)
    device = runtime.open_mesh_device(opts)
    try:
        outs = execute_ttm_in_process(fbb, inputs, device)
    finally:
        runtime.close_mesh_device(device)

    _BASELINE_CACHE[key] = outs
    return outs


def run_e2e(spec: PatternTest, e2e_device: "E2EDevice"):
    """Compile ``spec`` to a flatbuffer, run it in-process, and return
    ``(pcc, expected, actual)`` for output 0. Inputs are generated
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
        expected = spec.golden(*[t.float() for t in inputs])
    else:
        # Golden-free cross-check: device reference via the ttnn baseline.
        baseline = ttnn_baseline_outputs(spec.ttir, inputs, e2e_device)
        expected = baseline[0].float()

    fbb = compile_spec_to_fbb(spec)
    device = e2e_device.get(fbb)
    outputs = execute_ttm_in_process(fbb, inputs, device)

    actual = outputs[0].float()
    combined = torch.stack([expected.flatten(), actual.flatten()])
    pcc = torch.corrcoef(combined)[0, 1].item()
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
        for key, bench in getattr(mod, "KERNEL_BENCHES", {}).items():
            bench.name = key
            kernel_benches.append(bench)

    _DISCOVERED = (pattern_tests, kernel_benches)
    return _DISCOVERED
