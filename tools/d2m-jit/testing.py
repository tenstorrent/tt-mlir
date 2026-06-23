# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Co-located testing infrastructure for d2m-jit patterns.

A pattern file under ``d2m_jit/patterns/`` declares its own tests as
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

* ``KERNEL_BENCHES = [KernelBench(...)]`` — on-device numerics. Each bench
  drives the ``@d2m.kernel`` entrypoint directly with an explicit
  ``(layout, block_shape, grid_shape)`` config and PCC-compares against a
  torch golden. Replaces ``test/d2m-jit/test_pattern_eltwise.py``.

The ``KernelBench`` shape is forward-compatible with two things we do *not*
implement yet:

* **Autotuning** — ``space`` declares axes (block_shape / grid_shape /
  dtype) an autotuner sweeps, taking perf traces per config.
* **True e2e device execution** — running the *rewritten* module on silicon
  (rewrite -> compile -> ttrt run). The d2m->ttmetal pipeline already lowers
  a rewritten module fully; the remaining blocker is flatbuffer
  serialization of scalar program args (see TTMetalToFlatbuffer.cpp). Once
  that lands, a new device backend slots in behind this same spec — no spec
  change.
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
    tags: tuple = ()
    source_file: str = ""  # set by discovery


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


def assert_pcc(golden, actual, threshold: float = 0.99):
    combined = torch.stack([golden.flatten().float(), actual.flatten().float()])
    pcc = torch.corrcoef(combined)[0, 1].item()
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
# Discovery
# ----------------------------------------------------------------------

_DISCOVERED = None


def discover(force: bool = False):
    """Import every ``d2m_jit.patterns`` submodule and collect declared specs.

    Returns ``(pattern_tests, kernel_benches)`` with ``source_file`` stamped
    on each. Cached after the first call.
    """
    global _DISCOVERED
    if _DISCOVERED is not None and not force:
        return _DISCOVERED

    import importlib

    import d2m_jit.patterns as patterns_pkg

    pkg_dir = os.path.dirname(patterns_pkg.__file__)
    pattern_tests, kernel_benches = [], []
    for fn in sorted(os.listdir(pkg_dir)):
        if not fn.endswith(".py") or fn == "__init__.py":
            continue
        mod = importlib.import_module(f"d2m_jit.patterns.{fn[:-3]}")
        for t in getattr(mod, "PATTERN_TESTS", []):
            t.source_file = mod.__file__
            pattern_tests.append(t)
        for b in getattr(mod, "KERNEL_BENCHES", []):
            b.source_file = mod.__file__
            kernel_benches.append(b)

    _DISCOVERED = (pattern_tests, kernel_benches)
    return _DISCOVERED
