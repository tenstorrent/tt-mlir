# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""d2m-jit kernel autotuner.

Sweeps ``grid_shape`` (per-kernel), ``block_shape`` (per-tensor), and
``mem_space`` (per-tensor) combinations for every ``KernelBench`` declared
in a kernel module, runs device perf traces in-process, and collects
``kernel_ns`` timing for comparison.

Performance metric
------------------
``kernel_ns`` is the *device kernel duration*: the span from the first
``*-KERNEL`` ZONE_START to the last ``*-KERNEL`` ZONE_END across all cores
and RISCs, as reported by the tt-metal device profiler.  This matches
tt-metal's ``device_kernel_duration`` definition and is the primary figure
used for ranking configurations.

Autotuning knobs
----------------
Every config is per-tensor: ``block_shape`` and ``mem_space`` carry one value
per kernel input, stamped onto each ``TensorSpec`` before the run.  A "uniform"
config is simply the case where every tensor shares a value.  The materializer
reads the values off the specs and builds the matching ``d2m.Layout`` — so a
knob only takes effect if the materializer honors it.  The autotuner verifies
this after each run (see ``_verify_config_applied``): a config whose swept knob
never reached a constructed ``Layout`` is reported as failed rather than ranked.

* ``grid_shape`` (per-kernel): logical 2-D core grid ``(gy, gx)``.
* ``block_shape`` (per-tensor): tile-block shape ``[by, bx]`` per input.
* ``mem_space`` (per-tensor): tensor memory placement (``"L1"`` or ``"DRAM"``)
  per input, passed straight to ``d2m.Layout`` by the materializer.

Heuristics for sweep narrowing
-------------------------------
When ``grid_shapes`` / ``block_shapes`` are ``None``, valid candidates are
auto-generated from the tile dimensions reduced (via GCD) across *all* the
bench's tensors:

* ``grid_shapes``: all ``(gy, gx)`` where ``gy`` divides
  ``gcd(total_tiles_y)`` and ``gx`` divides ``gcd(total_tiles_x)`` across
  all tensors, filtered to ``gy * gx ≤ max_cores``.  For non-elementwise
  kernels (e.g. matmul) where the grid maps only to output dimensions,
  supply explicit ``grid_shapes`` instead.
* ``block_shapes``: for each grid, all ``[by, bx]`` where ``by`` divides
  ``gcd(tiles_per_core_y)`` and ``bx`` divides ``gcd(tiles_per_core_x)``
  across all tensors in the bench, filtered to ``by ≤ max_block_tiles``
  and ``bx ≤ max_block_tiles``.

The number of configs scales roughly as (grid divisors) × (block divisors)
× (mem_spaces).  For large tensors, ``max_cores`` and ``max_block_tiles``
cap the explosion.

CLI usage
---------
::

    python test/d2m-jit/autotuner.py \\
        --kernel test/d2m-jit/kernels/prefill/rope.py \\
        --bench rope \\
        [--strategy sweep|hill-climb|default] \\
        [--grid-shapes 1x1,2x2,4x4] \\
        [--block-shapes 1x1,2x2] \\
        [--mem-spaces L1,DRAM] \\
        [--joint-mem-spaces all|L1-L1,L1-DRAM,…] \\
        [--joint-block-shapes 1x1_1x1,1x3_3x1] \\
        [--tensor-shapes 64x96,96x64] \\
        [--tensor-dtypes bf16,float32] \\
        [--output-dir autotune-artifacts] \\
        [--save-profiler-logs] \\
        [--check-pcc] \\
        [--no-sweep] \\
        [--max-cores 8] \\
        [--max-block 8] \\
        [--max-rounds 10] \\
        [--n-warmup 1] \\
        [--quiet]

Omitting ``--bench`` tunes every ``KernelBench`` declared in the kernel file.

Module usage
------------
The high-level ``autotune_kernel`` helper does everything in one call: it loads
the kernel file, tunes the requested benches, and writes ``results.json`` /
``summary.txt`` under ``output_dir``::

    from autotuner import autotune_kernel, AutotuneKnobs

    knobs = AutotuneKnobs(
        grid_shapes=[(1, 1), (2, 2)],
        block_shapes=[[1, 1], [2, 2]],
        mem_spaces=["L1"],
    )
    results = autotune_kernel(
        "test/d2m-jit/kernels/prefill/rope.py",
        bench_names=["rope"],          # None → tune every bench in the file
        knobs=knobs,                   # Omit to sweep all valid configs (full-sweep mode)
        check_pcc=True,
        strategy="hill-climb",         # or "sweep" (default) / "default"
        output_dir="autotune-artifacts",
    )
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import importlib.util
import itertools
import json
import math
import os
import pathlib
import shutil
import sys
import textwrap
import time
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Make runner.py importable when running from the repo root
# ---------------------------------------------------------------------------

_RUNNER_DIR = pathlib.Path(__file__).parent.parent
if str(_RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(_RUNNER_DIR))

from runner import KernelBench, TensorSpec, compute_pcc, make_inputs, run_bench

# ---------------------------------------------------------------------------
# Lazy-load perf-analyzer (hyphenated filename prevents normal import)
# ---------------------------------------------------------------------------

_perf_analyzer = None


def _load_perf_analyzer():
    global _perf_analyzer
    if _perf_analyzer is not None:
        return _perf_analyzer
    pa_path = (
        pathlib.Path(__file__).parents[3]
        / "tools"
        / "perf-analyzer"
        / "perf-analyzer.py"
    )
    spec = importlib.util.spec_from_file_location("perf_analyzer", pa_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _perf_analyzer = mod
    return mod


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AutotuneKnobs:
    """Controls which parameter values the autotuner will explore.

    **Full-sweep mode** (default): leave all knobs as ``None``.  The tuner
    auto-generates every valid ``grid_shape`` and ``block_shape`` from the tile
    dimensions reduced (via GCD) across all the bench's tensors, and tries both
    ``"L1"`` and ``"DRAM"`` mem spaces.

    **Focused mode**: set any one (or more) knob explicitly.  Unset knobs fall
    back to the bench's own value rather than being swept — the ``grid_shape``
    from the ``KernelBench`` definition, each tensor's own ``block_shape`` from
    its ``TensorSpec``, and ``"L1"`` for mem_space.  This avoids an exponential
    config explosion when you only care about one dimension.

    **``"all"`` shorthand**: any knob except ``joint_block_shapes`` may be
    set to the string ``"all"`` to request the full valid set for that axis,
    without having to enumerate the values manually:

    * ``grid_shapes="all"`` — auto-generate all valid grids.
    * ``block_shapes="all"`` — auto-generate all valid block shapes.
    * ``mem_spaces="all"`` — sweep ``["L1", "DRAM"]``.
    * ``joint_mem_spaces="all"`` — sweep every L1/DRAM combination across
      all tensors (``2 ** n_tensors`` combos per grid/block config).

    Examples
    --------
    Sweep only grid shapes, keep block and mem at bench defaults::

        AutotuneKnobs(grid_shapes=[(1, 1), (2, 2), (4, 4)])

    Sweep all mem spaces without listing them::

        AutotuneKnobs(mem_spaces="all")

    All per-tensor L1/DRAM combos for a 2-tensor kernel::

        AutotuneKnobs(joint_mem_spaces="all")  # 4 combos: L1-L1, L1-DRAM, …

    Attributes
    ----------
    grid_shapes:
        Explicit list of ``(gy, gx)`` tuples, or ``"all"`` to auto-generate.
        ``None`` → auto (full-sweep) or bench default (focused).
    block_shapes:
        Explicit list of ``[by, bx]`` shapes applied to every tensor, or
        ``"all"`` to auto-generate.  ``None`` → auto (full-sweep) or bench
        default (focused).
    mem_spaces:
        List of memory space strings (``"L1"``, ``"DRAM"``), or ``"all"``
        for both.  ``None`` → both (full-sweep) or ``"L1"`` only (focused).
    joint_block_shapes:
        Per-tensor block shapes dispatched as joint configs (not Cartesian).
        Each entry is ``[block_shape_t0, block_shape_t1, …]`` — one per tensor.
    joint_mem_spaces:
        Per-tensor mem spaces dispatched as joint configs (not Cartesian),
        or ``"all"`` to sweep every L1/DRAM combination across tensors.
        Each explicit entry is ``[mem_space_t0, mem_space_t1, …]``.
    max_cores:
        Maximum ``gy * gx`` when auto-generating grid shapes.
    max_block_tiles:
        Maximum per-dimension block size when auto-generating block shapes.
    """

    grid_shapes: Optional[list[tuple[int, int]]] = None
    block_shapes: Optional[list[list[int]]] = None
    mem_spaces: Optional[list[str]] = None
    # Per-tensor block shapes dispatched as joint configs (not Cartesian product).
    # Each entry is [block_shape_t0, block_shape_t1, ...] — one per tensor.
    # Example: [[[1, 1], [1, 1]], [[1, 3], [3, 1]]] to sweep k_block ∈ {1, 3}.
    joint_block_shapes: Optional[list[list[list[int]]]] = None
    # Per-tensor mem spaces dispatched as joint configs (not Cartesian product).
    # Each entry is [mem_space_t0, mem_space_t1, ...] — one per tensor.
    # Example: [["L1", "L1"], ["DRAM", "L1"], ["L1", "DRAM"], ["DRAM", "DRAM"]].
    joint_mem_spaces: Optional[list[list[str]]] = None
    max_cores: int = 8
    max_block_tiles: int = 8


@dataclass
class AutotuneConfig:
    """One complete set of parameters to evaluate.

    A config is *always* per-tensor: ``blocks`` and ``mems`` each hold one
    entry per kernel input (``len == len(bench.tensors)``), alongside a single
    graph-level ``grid_shape``.  A "uniform" config is just the case where
    every tensor shares a value — build one with :meth:`uniform`.  There is no
    separate scalar representation, so every downstream consumer (``run_config``,
    ``id``, ``as_dict``) reads a single shape without branching.
    """

    grid_shape: tuple[int, int]
    blocks: list[list[int]]  # one [by, bx] per tensor
    mems: list[str]  # one mem_space ("L1"/"DRAM") per tensor

    def __post_init__(self):
        self.grid_shape = tuple(self.grid_shape)
        self.blocks = [list(b) for b in self.blocks]
        self.mems = list(self.mems)

    @classmethod
    def uniform(
        cls,
        grid_shape: tuple[int, int],
        block_shape: list[int],
        mem_space: str,
        n_tensors: int,
    ) -> "AutotuneConfig":
        """Broadcast one block/mem value across *n_tensors* tensors."""
        return cls(
            grid_shape=grid_shape,
            blocks=[list(block_shape) for _ in range(n_tensors)],
            mems=[mem_space for _ in range(n_tensors)],
        )

    @property
    def is_uniform(self) -> bool:
        return all(b == self.blocks[0] for b in self.blocks) and all(
            m == self.mems[0] for m in self.mems
        )

    @property
    def id(self) -> str:
        # Collapse uniform axes so common-case ids stay short and stable
        # (e.g. ``g1x1_b1x1_mL1``); differing tensors are joined with ``-``.
        g = f"{self.grid_shape[0]}x{self.grid_shape[1]}"
        if all(b == self.blocks[0] for b in self.blocks):
            b = f"{self.blocks[0][0]}x{self.blocks[0][1]}"
        else:
            b = "-".join(f"{by}x{bx}" for by, bx in self.blocks)
        if all(m == self.mems[0] for m in self.mems):
            m = self.mems[0]
        else:
            m = "-".join(self.mems)
        return f"g{g}_b{b}_m{m}"

    def as_dict(self) -> dict:
        return {
            "grid_shape": list(self.grid_shape),
            "blocks": self.blocks,
            "mems": self.mems,
        }


@dataclass
class AutotuneResult:
    """Result from one autotuner config evaluation."""

    bench_name: str
    config: AutotuneConfig
    kernel_ns: Optional[float]
    wall_ns: Optional[float] = None
    pcc: Optional[float] = None
    error: Optional[str] = None
    profiler_log: Optional[str] = None
    elapsed_s: Optional[float] = None  # wall-clock time of the Python run call

    @property
    def config_id(self) -> str:
        return self.config.id

    def as_dict(self) -> dict:
        return {
            "bench_name": self.bench_name,
            "config": self.config.as_dict(),
            "config_id": self.config_id,
            "kernel_ns": self.kernel_ns,
            "wall_ns": self.wall_ns,
            "pcc": self.pcc,
            "error": self.error,
            "profiler_log": self.profiler_log,
            "elapsed_s": self.elapsed_s,
        }


# ---------------------------------------------------------------------------
# Config generation helpers
# ---------------------------------------------------------------------------


def _closest_block(target: list[int], candidates: list[list[int]]) -> list[int]:
    """Return the candidate whose total tile count is closest to *target*."""
    if not candidates:
        raise ValueError("candidates must be non-empty")
    target_tiles = target[0] * target[1]
    return min(candidates, key=lambda b: abs(b[0] * b[1] - target_tiles))


def _override_bench_tensors(
    bench: KernelBench,
    tensor_shapes: Optional[list[tuple[int, ...]]] = None,
    tensor_dtypes: Optional[list] = None,
) -> KernelBench:
    """Return a copy of *bench* with tensor shapes and/or dtypes overridden.

    Lengths of *tensor_shapes* / *tensor_dtypes* must match
    ``len(bench.tensors)`` when provided.  Returns *bench* unchanged if
    both arguments are ``None``.
    """
    if tensor_shapes is None and tensor_dtypes is None:
        return bench
    n = len(bench.tensors)
    if tensor_shapes is not None and len(tensor_shapes) != n:
        raise ValueError(
            f"tensor_shapes has {len(tensor_shapes)} entries but bench "
            f"has {n} tensor(s)"
        )
    if tensor_dtypes is not None and len(tensor_dtypes) != n:
        raise ValueError(
            f"tensor_dtypes has {len(tensor_dtypes)} entries but bench "
            f"has {n} tensor(s)"
        )
    new_tensors = []
    for i, ts in enumerate(bench.tensors):
        kwargs: dict = {}
        if tensor_shapes is not None:
            kwargs["shape"] = tuple(tensor_shapes[i])
        if tensor_dtypes is not None:
            kwargs["dtype"] = tensor_dtypes[i]
        new_tensors.append(dataclasses.replace(ts, **kwargs))
    return dataclasses.replace(bench, tensors=new_tensors)


def _divisors(n: int) -> list[int]:
    """Return all positive divisors of n in ascending order."""
    if n <= 0:
        return []
    divs = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)


def valid_grid_shapes(
    bench: KernelBench, knobs: AutotuneKnobs
) -> list[tuple[int, int]]:
    """Return all grid shapes valid for this bench under the given knobs.

    A grid ``(gy, gx)`` is valid when ``gy`` divides ``total_tiles_y`` and
    ``gx`` divides ``total_tiles_x`` for *every* tensor in the bench.
    Tile totals are reduced via GCD across all tensors so the resulting
    candidates are guaranteed divisors of each tensor's tile dimensions.

    For non-elementwise kernels (e.g. matmul) where the grid is only
    meaningful over a subset of tensor dimensions (e.g. output M×N), pass
    explicit ``knobs.grid_shapes`` rather than relying on auto-generation.
    """
    if knobs.grid_shapes is not None:
        return list(knobs.grid_shapes)

    y_totals: list[int] = []
    x_totals: list[int] = []
    for ts in bench.tensors:
        s = tuple(ts.shape)
        if len(s) >= 2:
            y_totals.append(max(1, s[-2] // 32))
        if len(s) >= 1:
            x_totals.append(max(1, s[-1] // 32))

    total_y = math.gcd(*y_totals) if y_totals else 1
    total_x = math.gcd(*x_totals) if x_totals else 1

    result = []
    for gy in _divisors(total_y):
        for gx in _divisors(total_x):
            if gy * gx <= knobs.max_cores:
                result.append((gy, gx))

    return sorted(result, key=lambda g: g[0] * g[1])


def valid_block_shapes(
    bench: KernelBench, grid_shape: tuple[int, int], knobs: AutotuneKnobs
) -> list[list[int]]:
    """Return all block shapes valid for this bench+grid under the given knobs.

    A block shape ``[by, bx]`` is valid when it divides the per-core tile
    count for *every* tensor in the bench.  Tile counts are computed per
    tensor, then reduced via GCD so that the resulting candidates are
    guaranteed divisors of all tensors' per-core dimensions simultaneously.

    ``tiles_per_core_y = gcd_over_tensors((shape[-2] // 32) // gy)``
    ``by`` must divide that GCD (and same for x).
    """
    gy, gx = grid_shape

    y_counts: list[int] = []
    x_counts: list[int] = []
    for ts in bench.tensors:
        s = tuple(ts.shape)
        if len(s) >= 2:
            y_counts.append(max(1, (s[-2] // 32) // gy))
        if len(s) >= 1:
            x_counts.append(max(1, (s[-1] // 32) // gx))

    tiles_per_core_y = math.gcd(*y_counts) if y_counts else 1
    tiles_per_core_x = math.gcd(*x_counts) if x_counts else 1

    if knobs.block_shapes is not None:
        # Filter to only shapes valid for this grid
        valid = []
        for bs in knobs.block_shapes:
            if (
                len(bs) >= 2
                and tiles_per_core_y % bs[0] == 0
                and tiles_per_core_x % bs[1] == 0
            ):
                valid.append(list(bs))
        return valid

    result = []
    for by in _divisors(tiles_per_core_y):
        if by > knobs.max_block_tiles:
            continue
        for bx in _divisors(tiles_per_core_x):
            if bx > knobs.max_block_tiles:
                continue
            result.append([by, bx])

    return result


# ---------------------------------------------------------------------------
# Profiling context manager and CSV parser
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _profiling_ctx(profiler_dir: str):
    """Enable in-process device profiling, directing output to *profiler_dir*.

    Sets the environment variable ``TT_METAL_PROFILER_DIR`` so the tt-metal
    runtime writes ``profile_log_device.csv`` into ``<profiler_dir>/.logs/``.
    Also enables ``config.enable_perf_trace``.  All mutations are reversed on
    exit.
    """
    from d2m_jit._src.config import config as _cfg

    # Env overrides
    old_profiler_dir = os.environ.get("TT_METAL_PROFILER_DIR")
    old_dispatch = os.environ.get("TT_METAL_DEVICE_PROFILER_DISPATCH")
    old_device_profiler = os.environ.get("TT_METAL_DEVICE_PROFILER")
    os.environ["TT_METAL_PROFILER_DIR"] = profiler_dir
    os.environ["TT_METAL_DEVICE_PROFILER_DISPATCH"] = "0"
    os.environ["TT_METAL_DEVICE_PROFILER"] = "1"

    # Config overrides
    old_perf = _cfg.enable_perf_trace
    old_traces = _cfg.insert_profiler_traces
    _cfg.enable_perf_trace = True
    # Do NOT set insert_profiler_traces: the inserted DeviceZoneScopedN scopes
    # register source-location hashes globally for the process lifetime.  With
    # many kernel compilations in one process the 32-bit hash space collides
    # (birthday problem) and tt-metal throws.  The firmware *-KERNEL zones
    # written by enable_perf_trace alone are sufficient for kernel_ns measurement.

    try:
        yield
    finally:
        _cfg.enable_perf_trace = old_perf
        _cfg.insert_profiler_traces = old_traces
        if old_profiler_dir is None:
            os.environ.pop("TT_METAL_PROFILER_DIR", None)
        else:
            os.environ["TT_METAL_PROFILER_DIR"] = old_profiler_dir

        if old_dispatch is None:
            os.environ.pop("TT_METAL_DEVICE_PROFILER_DISPATCH", None)
        else:
            os.environ["TT_METAL_DEVICE_PROFILER_DISPATCH"] = old_dispatch

        if old_device_profiler is None:
            os.environ.pop("TT_METAL_DEVICE_PROFILER", None)
        else:
            os.environ["TT_METAL_DEVICE_PROFILER"] = old_device_profiler


def _parse_profile_csv(
    csv_path: pathlib.Path, verbose: bool = False
) -> tuple[Optional[float], Optional[float]]:
    """Return ``(kernel_ns, wall_ns)`` from a ``profile_log_device.csv``.

    Uses ``collect_device_runtimes`` from perf-analyzer for kernel duration
    and ``read_chip_freq_mhz`` to convert cycles → ns.

    Returns ``(None, None)`` if the file does not exist or cannot be parsed
    (e.g. profiling not enabled in the build).  A *parse* failure (as opposed
    to a missing file) is warned about when ``verbose`` so it is not confused
    with "no device data".
    """
    if not csv_path.exists():
        return None, None
    try:
        pa = _load_perf_analyzer()
        freq_mhz = pa.read_chip_freq_mhz(csv_path)
        runtimes = pa.collect_device_runtimes(csv_path, freq_mhz)

        kernel_cycles = runtimes.get("kernel duration", 0)
        kernel_ns = pa.cycles_to_ns(kernel_cycles, freq_mhz) if kernel_cycles else None

        # Wall time from get_runtimes (requires timeline parse)
        try:
            rows, wall_cycles = pa.collect_device_timeline(csv_path)
            rt = pa.get_runtimes(rows, wall_cycles)
            wall_ns = pa.cycles_to_ns(rt["device wall time"], freq_mhz)
        except Exception:
            wall_ns = None

        return kernel_ns, wall_ns
    except Exception as exc:
        if verbose:
            print(f"  WARNING: failed to parse profiler CSV {csv_path}: {exc}")
        return None, None


@contextlib.contextmanager
def _silence_native_output(log_path: str):
    """Redirect fd-level stdout/stderr to *log_path* for the duration.

    tt-metal kernel JIT compilation emits profiler pragma notes through native
    stdout/stderr, so Python-level stdout redirection does not catch them.
    Redirecting file descriptors 1/2 keeps successful autotune runs quiet.
    """
    sys.stdout.flush()
    sys.stderr.flush()

    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    with open(log_path, "w") as log:
        try:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


# ---------------------------------------------------------------------------
# "config actually applied" guard
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _layout_probe(collected: list):
    """Record ``(block_shape, mem_space)`` for every ``Layout`` built inside.

    Wraps ``d2m_jit.Layout.__init__`` observationally (delegating via
    ``*args/**kwargs`` so it is robust to signature changes) and appends the
    constructed layout's resolved ``block_shape`` / ``mem_space`` to
    *collected*.  Used by ``_verify_config_applied`` to detect materializers
    that silently ignore a swept per-tensor knob.
    """
    from d2m_jit._src import tensor_layout as _tl

    prev_init = _tl.Layout.__init__

    def _probing_init(self_inner, *args, **kwargs):
        prev_init(self_inner, *args, **kwargs)
        collected.append((tuple(self_inner.block_shape), self_inner.mem_space))

    _tl.Layout.__init__ = _probing_init
    try:
        yield
    finally:
        _tl.Layout.__init__ = prev_init


def _verify_config_applied(config: "AutotuneConfig", collected: list) -> Optional[str]:
    """Return an error string if a requested knob never reached a Layout.

    A materializer that reads only ``tensors[0]`` (or never passes
    ``mem_space`` to ``Layout``) will silently drop per-tensor block/mem
    knobs.  Without this check the autotuner would still emit a distinct
    ``config.id`` and ``kernel_ns`` for such a config and *rank a
    configuration it never actually applied*.

    The check is necessary-but-not-sufficient: it asserts every requested
    per-tensor value *appears* among the constructed layouts (catching the
    "ignored entirely" failure mode).  It does not verify the value landed on
    the correct tensor.  Returns ``None`` when every requested value was
    observed, or when nothing was built (e.g. the run errored earlier).
    """
    if not collected:
        return None

    from d2m_jit._src.tensor_layout import _to_mem_space

    built_blocks = {b for b, _ in collected}
    built_mems = {m for _, m in collected}

    for bs in config.blocks:
        if tuple(bs) not in built_blocks:
            return (
                f"requested block_shape {list(bs)} was never applied "
                f"(built {sorted(list(b) for b in built_blocks)}); the "
                f"materializer likely ignores per-tensor block_shape"
            )

    for ms in config.mems:
        if _to_mem_space(ms.lower()) not in built_mems:
            return (
                f"requested mem_space {ms!r} was never applied; the "
                f"materializer likely ignores per-tensor mem_space "
                f"(pass mem_space to d2m.Layout in the bench's run function)"
            )

    return None


# ---------------------------------------------------------------------------
# Main Autotuner class
# ---------------------------------------------------------------------------


class Autotuner:
    """Sweeps autotuning parameters for KernelBench entries.

    Parameters
    ----------
    knobs:
        ``AutotuneKnobs`` instance.  ``None`` → default ``AutotuneKnobs()``,
        i.e. full-sweep: auto-generated grids/blocks and both ``L1`` and
        ``DRAM`` mem spaces.
    output_dir:
        Root artifacts directory.  Results are stored under
        ``<output_dir>/<bench_name>/``.
    save_profiler_logs:
        If ``True``, the raw ``profile_log_device.csv`` is copied into
        ``<output_dir>/<bench_name>/profiler_logs/<config_id>.csv``.
    check_pcc:
        If ``True``, run a PCC check against the bench golden after each
        timed run.
    n_warmup:
        Number of warm-up iterations before the measured run.  Warm-ups
        compile the kernel and amortize JIT overhead.
    verbose:
        Print per-config progress to stdout.
    """

    def __init__(
        self,
        knobs: Optional[AutotuneKnobs] = None,
        output_dir: str = "autotune-artifacts",
        save_profiler_logs: bool = False,
        check_pcc: bool = False,
        n_warmup: int = 1,
        verbose: bool = True,
    ):
        self.knobs = knobs or AutotuneKnobs()
        self.output_dir = pathlib.Path(output_dir)
        self.save_profiler_logs = save_profiler_logs
        self.check_pcc = check_pcc
        self.n_warmup = n_warmup
        self.verbose = verbose
        # Single profiler directory for the whole Autotuner session.
        # tt-metal initialises its profiler on the first measured run and
        # continues writing to the same path for the process lifetime; using
        # a consistent directory ensures every subsequent run writes its CSV
        # where we expect it, even after the runtime's one-time init.
        self._profiler_dir = str(self.output_dir / ".profiler")

    # ------------------------------------------------------------------
    # Config generation
    # ------------------------------------------------------------------

    def _is_full_sweep(self) -> bool:
        """True when no knob imposes an explicit constraint.

        In this mode grids and blocks are auto-generated from tensor shapes and
        mem_space sweeps both ``L1`` and ``DRAM``.  ``"all"`` counts as
        unconstrained (it *requests* the full auto set), ``None`` is the
        default; any concrete list, or a ``joint_*`` list, is a constraint.
        """
        k = self.knobs
        return (
            k.grid_shapes in (None, "all")
            and k.block_shapes in (None, "all")
            and k.joint_block_shapes is None
            and k.mem_spaces in (None, "all")
            and k.joint_mem_spaces in (None, "all")
        )

    def _resolve_grids(self, bench: KernelBench) -> list[tuple[int, int]]:
        """Grid candidates: explicit list, else auto-generated, else default."""
        k = self.knobs
        if k.grid_shapes not in (None, "all"):
            return [tuple(g) for g in k.grid_shapes]
        if self._is_full_sweep() or k.grid_shapes == "all":
            return valid_grid_shapes(bench, dataclasses.replace(k, grid_shapes=None))
        return [tuple(bench.grid_shape)]

    def _resolve_mem_options(self, n_tensors: int) -> list[list[str]]:
        """Per-tensor mem-space candidates; each entry has one value per tensor."""
        k = self.knobs
        if k.joint_mem_spaces is not None:
            if k.joint_mem_spaces == "all":
                return [
                    list(c) for c in itertools.product(["L1", "DRAM"], repeat=n_tensors)
                ]
            return [list(ms) for ms in k.joint_mem_spaces]
        if k.mem_spaces is not None:
            mems = ["L1", "DRAM"] if k.mem_spaces == "all" else list(k.mem_spaces)
            return [[m] * n_tensors for m in mems]
        if self._is_full_sweep():
            return [["L1"] * n_tensors, ["DRAM"] * n_tensors]
        return [["L1"] * n_tensors]

    def _resolve_block_options(
        self, bench: KernelBench, grid: tuple[int, int], n_tensors: int
    ) -> list[list[list[int]]]:
        """Per-tensor block-shape candidates for *grid*; one [by,bx] per tensor."""
        k = self.knobs
        if k.joint_block_shapes is not None:
            return [[list(bs) for bs in entry] for entry in k.joint_block_shapes]
        if k.block_shapes is not None or self._is_full_sweep():
            knobs_for_blocks = (
                dataclasses.replace(k, block_shapes=None)
                if k.block_shapes == "all"
                else k
            )
            valid = valid_block_shapes(bench, grid, knobs_for_blocks)
            return [[list(bs)] * n_tensors for bs in valid]
        # Focused mode, block unconstrained: each tensor keeps its own default.
        return [[list(ts.block_shape) for ts in bench.tensors]]

    def generate_configs(self, bench: KernelBench) -> list[AutotuneConfig]:
        """Return the deduplicated ``AutotuneConfig``s to evaluate for *bench*.

        The space is the Cartesian product ``grids × block-options(per grid) ×
        mem-options``, where each option is already resolved to a per-tensor
        list (see the ``_resolve_*`` helpers).  ``joint_*`` knobs contribute
        per-tensor entries directly; plain-list / ``"all"`` / full-sweep knobs
        contribute uniform (broadcast) entries.  See ``AutotuneKnobs``.
        """
        n_tensors = len(bench.tensors)
        grids = self._resolve_grids(bench)
        mem_options = self._resolve_mem_options(n_tensors)

        seen: set = set()
        configs: list[AutotuneConfig] = []
        for grid in grids:
            for blocks in self._resolve_block_options(bench, grid, n_tensors):
                for mems in mem_options:
                    cfg = AutotuneConfig(grid_shape=grid, blocks=blocks, mems=mems)
                    # Dedup on a structured key rather than the display id, which
                    # collapses uniform axes and could alias distinct configs.
                    key = (
                        cfg.grid_shape,
                        tuple(tuple(b) for b in cfg.blocks),
                        tuple(cfg.mems),
                    )
                    if key not in seen:
                        seen.add(key)
                        configs.append(cfg)
        return configs

    # ------------------------------------------------------------------
    # Single-config execution
    # ------------------------------------------------------------------

    def run_config(
        self,
        bench: KernelBench,
        config: AutotuneConfig,
        bench_name: str,
        *,
        tmp_dir: Optional[str] = None,
    ) -> AutotuneResult:
        """Run *bench* with *config*, collect profiler data, return result.

        The method:

        1. Stamps each ``TensorSpec`` with its per-tensor ``block_shape`` and
           ``mem_space`` from ``config.blocks`` / ``config.mems``.
        2. Enables in-process device profiling into the session profiler dir.
        3. Executes ``n_warmup`` un-measured warm-up iterations.
        4. Executes one measured iteration (probing every Layout built) and
           parses the profiler CSV.
        5. Verifies the swept knobs actually reached a Layout
           (``_verify_config_applied``); a config that wasn't applied is failed.
        6. Optionally checks PCC and saves the profiler log.

        Note on profiler directory: tt-metal initialises its profiler singleton
        on the first measured run and retains that path for the process lifetime.
        ``self._profiler_dir`` is therefore used consistently across all configs
        so every run's CSV lands where ``_parse_profile_csv`` expects it.
        """
        from d2m_jit._src.builder import _Builder

        # Every config is per-tensor: stamp each tensor's block_shape/mem_space
        # onto its TensorSpec.  The materializer reads these off the specs and
        # builds the corresponding d2m.Layout.
        n_tensors = len(bench.tensors)
        if len(config.blocks) != n_tensors or len(config.mems) != n_tensors:
            raise ValueError(
                f"Config tensor count mismatch: bench has {n_tensors} tensor(s) "
                f"but blocks={len(config.blocks)} mems={len(config.mems)}"
            )
        overridden_tensors = [
            dataclasses.replace(
                bench.tensors[i],
                block_shape=list(config.blocks[i]),
                mem_space=config.mems[i],
            )
            for i in range(n_tensors)
        ]

        profiler_tmp = tmp_dir or self._profiler_dir
        profiler_csv = pathlib.Path(profiler_tmp) / ".logs" / "profile_log_device.csv"
        native_log_dir = self.output_dir / bench_name / "native_logs"
        native_log_path = native_log_dir / f"{config.id}.log"

        t0 = time.perf_counter()
        actual = None
        expected = None
        error_msg: Optional[str] = None
        probed_layouts: list = []

        try:
            native_log_dir.mkdir(parents=True, exist_ok=True)
            with _silence_native_output(str(native_log_path)):
                # The profiling context must wrap *all* runs (including warmup)
                # because TT_METAL_DEVICE_PROFILER=1 needs to be set before the
                # device is first opened.  Warmup data is simply discarded by
                # wiping the .logs dir before the measured pass.
                with _profiling_ctx(profiler_tmp):
                    # Warm-up runs (compile + JIT cache fill; profiler data discarded)
                    for _ in range(self.n_warmup):
                        _Builder.reset()
                        run_bench(
                            bench,
                            tensors=overridden_tensors,
                            grid_shape=config.grid_shape,
                        )

                    # Clean warmup profiler data before the measured pass.
                    stale = pathlib.Path(profiler_tmp) / ".logs"
                    if stale.exists():
                        shutil.rmtree(stale)

                    # Measured run.  Probe every Layout it builds so we can
                    # verify the requested knobs actually reached the device.
                    _Builder.reset()
                    with _layout_probe(probed_layouts):
                        actual, expected = run_bench(
                            bench,
                            tensors=overridden_tensors,
                            grid_shape=config.grid_shape,
                        )

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
        finally:
            _Builder.reset()

        if error_msg is None:
            native_log_path.unlink(missing_ok=True)
        elif native_log_path.exists():
            error_msg = f"{error_msg} (native log: {native_log_path})"

        elapsed_s = time.perf_counter() - t0

        kernel_ns, wall_ns = _parse_profile_csv(profiler_csv, verbose=self.verbose)

        # Guard: reject configs whose swept knobs were silently ignored by the
        # materializer.  Such a config's timing is meaningless (it is a
        # duplicate of some other config), so null it out and record why rather
        # than let it into the ranking.
        if error_msg is None:
            guard_msg = _verify_config_applied(config, probed_layouts)
            if guard_msg is not None:
                error_msg = f"config not applied: {guard_msg}"
                kernel_ns = None
                wall_ns = None

        pcc_val: Optional[float] = None
        if self.check_pcc and actual is not None and expected is not None:
            try:
                pcc_val = compute_pcc(expected.float(), actual.float())
            except Exception as exc:
                if error_msg is None:
                    error_msg = f"PCC error: {exc}"

        # PCC gate: a config that computes wrong numerics is not a valid tuning
        # result, so drop it from the ranking rather than let a fast-but-wrong
        # config win.  ``pcc_val`` is retained on the result so the summary can
        # show *why* it was dropped.  Only active when ``check_pcc`` is set.
        if error_msg is None and pcc_val is not None and pcc_val < bench.pcc:
            error_msg = f"PCC {pcc_val:.5f} < {bench.pcc} threshold"
            kernel_ns = None
            wall_ns = None

        saved_log: Optional[str] = None
        if self.save_profiler_logs and profiler_csv.exists():
            log_dir = self.output_dir / bench_name / "profiler_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            dest = log_dir / f"{config.id}.csv"
            shutil.copy2(profiler_csv, dest)
            saved_log = str(dest)

        # Remove profiler data unless the caller asked to keep it.
        logs_dir = pathlib.Path(profiler_tmp) / ".logs"
        if not self.save_profiler_logs and logs_dir.exists():
            shutil.rmtree(logs_dir)

        # Clean up the per-run temp dir if the caller provided one.
        # Never remove self._profiler_dir — it is reused across configs.
        if tmp_dir is not None and not self.save_profiler_logs:
            shutil.rmtree(profiler_tmp, ignore_errors=True)

        if self.verbose:
            status = (
                f"kernel_ns={kernel_ns:.1f}"
                if kernel_ns is not None
                else "no profiler data"
            )
            if error_msg:
                status = f"ERROR: {error_msg}"
            pcc_str = f"  pcc={pcc_val:.5f}" if pcc_val is not None else ""
            print(f"  [{bench_name}] {config.id:<32} {status}{pcc_str}")

        return AutotuneResult(
            bench_name=bench_name,
            config=config,
            kernel_ns=kernel_ns,
            wall_ns=wall_ns,
            pcc=pcc_val,
            error=error_msg,
            profiler_log=saved_log,
            elapsed_s=elapsed_s,
        )

    # ------------------------------------------------------------------
    # Bench-level sweep
    # ------------------------------------------------------------------

    def run_bench(
        self,
        bench: KernelBench,
        *,
        bench_name: str = "",
        configs: Optional[list[AutotuneConfig]] = None,
        strategy: str = "sweep",
        seed: Optional[AutotuneConfig] = None,
        max_rounds: int = 10,
        tensor_shapes: Optional[list[tuple[int, ...]]] = None,
        tensor_dtypes: Optional[list] = None,
    ) -> list[AutotuneResult]:
        """Run *bench* under the chosen search strategy.

        Parameters
        ----------
        bench:
            ``KernelBench`` to evaluate.
        bench_name:
            Human-readable name (used for output paths).
        configs:
            Explicit list of configs.  ``None`` → auto-generate.  Ignored
            when ``strategy='hill-climb'``.
        strategy:
            ``'sweep'`` (default) – exhaustive Cartesian sweep.
            ``'hill-climb'`` – coordinate-descent hill-climb; much faster
            for large search spaces, but may miss the global optimum.
            ``'default'`` – run only the bench's own per-tensor default config
            (no sweep); a quick correctness/perf spot-check.
        seed:
            Starting config for hill-climbing.  ``None`` → bench default.
        max_rounds:
            Maximum coordinate-descent rounds for hill-climbing.
        tensor_shapes:
            One ``(dim, …)`` tuple per tensor, overriding the shapes defined
            in the bench.  Affects grid/block auto-generation as well as the
            tensors passed to the kernel.  Length must match
            ``len(bench.tensors)``.
        tensor_dtypes:
            One dtype per tensor, overriding the dtypes defined in the bench.
            Length must match ``len(bench.tensors)``.
        """
        name = bench_name or bench.name or "unnamed"
        bench = _override_bench_tensors(bench, tensor_shapes, tensor_dtypes)

        if strategy == "default":
            # Bench's own per-tensor defaults, one config, no sweep.
            cfg = AutotuneConfig(
                grid_shape=tuple(bench.grid_shape),
                blocks=[list(ts.block_shape) for ts in bench.tensors],
                mems=[ts.mem_space for ts in bench.tensors],
            )
            if self.verbose:
                print(f"\n=== {name!r}: default config only ({cfg.id}) ===")
            return [self.run_config(bench, cfg, name)]

        if strategy == "hill-climb":
            if (
                self.knobs.joint_block_shapes is not None
                or self.knobs.joint_mem_spaces is not None
            ):
                print(
                    "  WARNING: hill-climb explores only uniform (grid × block × "
                    "mem_space) configs; joint_block_shapes / joint_mem_spaces are "
                    "ignored.  Use strategy='sweep' to tune per-tensor knobs."
                )
            return self._run_hill_climb(bench, name, seed=seed, max_rounds=max_rounds)

        cfgs = configs if configs is not None else self.generate_configs(bench)

        if self.verbose:
            print(
                f"\n=== Autotuning {name!r}: {len(cfgs)} config(s) "
                f"× {self.n_warmup} warmup(s) ==="
            )

        results: list[AutotuneResult] = []
        for cfg in cfgs:
            results.append(self.run_config(bench, cfg, name))

        return results

    # ------------------------------------------------------------------
    # Hill-climb strategy
    # ------------------------------------------------------------------

    def _run_hill_climb(
        self,
        bench: KernelBench,
        bench_name: str,
        *,
        seed: Optional[AutotuneConfig] = None,
        max_rounds: int = 10,
    ) -> list[AutotuneResult]:
        """Coordinate-descent hill-climb over grid × block × mem_space.

        Each round sweeps one axis at a time while holding the other two
        fixed, moving to the best found value on that axis before continuing.
        Rounds repeat until no axis improves or *max_rounds* is reached.

        This converges in O((G + B + M) × rounds) evaluations rather than
        O(G × B × M) for a full sweep.  In practice it typically converges
        in 1–2 rounds.

        The grid axis adapts the block shape when it changes grids: the
        candidate block with the closest total tile count to the current
        block is used as a proxy.  Once the best grid is found, the block
        axis is swept properly for that grid.
        """
        knobs = self.knobs
        n_tensors = len(bench.tensors)

        # Determine seed config (uniform: hill-climb tunes shared knobs only).
        if seed is None:
            ts = bench.tensors[0]
            seed = AutotuneConfig.uniform(
                grid_shape=tuple(bench.grid_shape),
                block_shape=list(ts.block_shape),
                mem_space="L1",
                n_tensors=n_tensors,
            )

        evaluated: dict[str, AutotuneResult] = {}
        results: list[AutotuneResult] = []

        def _run(cfg: AutotuneConfig) -> AutotuneResult:
            if cfg.id in evaluated:
                return evaluated[cfg.id]
            r = self.run_config(bench, cfg, bench_name)
            evaluated[cfg.id] = r
            results.append(r)
            return r

        def _better(a: AutotuneResult, b: AutotuneResult) -> bool:
            if a.kernel_ns is None:
                return False
            if b.kernel_ns is None:
                return True
            return a.kernel_ns < b.kernel_ns

        def _best_of(candidates: list[AutotuneConfig]) -> AutotuneResult:
            best = None
            for cfg in candidates:
                r = _run(cfg)
                if best is None or _better(r, best):
                    best = r
            return best

        full_sweep = self._is_full_sweep()
        mem_spaces_list = (
            ["L1", "DRAM"]
            if full_sweep or knobs.mem_spaces == "all"
            else (knobs.mem_spaces or ["L1"])
        )

        current = seed
        current_result = _run(current)

        if self.verbose:
            print(
                f"\n=== Hill-climb {bench_name!r}: "
                f"seed={current.id}  max_rounds={max_rounds} ==="
            )

        # Strip "all" sentinels so the valid_* helpers auto-generate rather than
        # iterating the literal string.
        knobs_hc = dataclasses.replace(
            knobs,
            grid_shapes=None if knobs.grid_shapes == "all" else knobs.grid_shapes,
            block_shapes=None if knobs.block_shapes == "all" else knobs.block_shapes,
        )

        for round_idx in range(max_rounds):
            improved = False

            # --- Axis 1: grid ---
            all_grids = valid_grid_shapes(bench, knobs_hc)
            grid_candidates = []
            for g in all_grids:
                blocks = valid_block_shapes(bench, g, knobs_hc)
                if not blocks:
                    continue
                adapted_block = _closest_block(current.blocks[0], blocks)
                grid_candidates.append(
                    AutotuneConfig.uniform(g, adapted_block, current.mems[0], n_tensors)
                )
            best_grid_r = _best_of(grid_candidates)
            if best_grid_r is not None and _better(best_grid_r, current_result):
                current = best_grid_r.config
                current_result = best_grid_r
                improved = True

            # --- Axis 2: block (for current grid) ---
            all_blocks = valid_block_shapes(bench, current.grid_shape, knobs_hc)
            block_candidates = [
                AutotuneConfig.uniform(
                    current.grid_shape, b, current.mems[0], n_tensors
                )
                for b in all_blocks
            ]
            best_block_r = _best_of(block_candidates)
            if best_block_r is not None and _better(best_block_r, current_result):
                current = best_block_r.config
                current_result = best_block_r
                improved = True

            # --- Axis 3: mem_space ---
            mem_candidates = [
                AutotuneConfig.uniform(
                    current.grid_shape, current.blocks[0], m, n_tensors
                )
                for m in mem_spaces_list
            ]
            best_mem_r = _best_of(mem_candidates)
            if best_mem_r is not None and _better(best_mem_r, current_result):
                current = best_mem_r.config
                current_result = best_mem_r
                improved = True

            if self.verbose:
                print(
                    f"  round {round_idx + 1}: best={current.id}  "
                    f"kernel_ns={current_result.kernel_ns}  "
                    f"({'improved' if improved else 'converged'})"
                )

            if not improved:
                break

        return results

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------

    def save_results(
        self,
        bench_name: str,
        results: list[AutotuneResult],
    ) -> pathlib.Path:
        """Persist per-run JSON files for *bench_name*.

        Layout::

            <output_dir>/<bench_name>/
                runs/<config_id>.json   # one per result

        Returns the bench directory path.
        """
        bench_dir = self.output_dir / bench_name
        runs_dir = bench_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        for r in results:
            json_path = runs_dir / f"{r.config_id}.json"
            with open(json_path, "w") as f:
                json.dump(r.as_dict(), f, indent=2)

        if self.verbose:
            print(f"\nResults saved to {bench_dir}/")
            _print_summary(bench_name, results)

        return bench_dir

    def save_summary(
        self,
        all_results: dict[str, list[AutotuneResult]],
    ) -> dict[str, pathlib.Path]:
        """Write a plain-text ``summary.txt`` into each bench's directory.

        Each file contains the best config and a full ranked table for that
        bench.  If *all_results* contains more than one bench, also prints a
        cross-bench best-per-bench table to stdout.

        Returns a dict mapping bench name → summary file path.
        """
        paths: dict[str, pathlib.Path] = {}

        for bench_name, results in all_results.items():
            bench_dir = self.output_dir / bench_name
            bench_dir.mkdir(parents=True, exist_ok=True)
            txt_path = bench_dir / "summary.txt"

            valid, failed = _rank(results)
            best = valid[0] if valid else None

            with open(txt_path, "w") as f:
                f.write(f"d2m-jit Autotune Summary: {bench_name}\n")
                f.write("=" * 40 + "\n\n")

                if best is not None:
                    kns = f"{best.kernel_ns:.1f}" if best.kernel_ns is not None else "-"
                    wns = f"{best.wall_ns:.1f}" if best.wall_ns is not None else "-"
                    pcc = f"{best.pcc:.5f}" if best.pcc is not None else "-"
                    f.write(
                        f"Best: {best.config_id}  kernel_ns={kns}  wall_ns={wns}  PCC={pcc}\n\n"
                    )

                f.write("Ranking (kernel_ns)\n")
                f.write("-" * 40 + "\n")
                if valid:
                    headers = (
                        "Rank",
                        "Config",
                        "kernel_ns",
                        "wall_ns",
                        "PCC",
                        "elapsed_s",
                    )
                    rows = []
                    for rank, r in enumerate(valid, 1):
                        kns = f"{r.kernel_ns:.1f}"
                        wns = f"{r.wall_ns:.1f}" if r.wall_ns is not None else "-"
                        pcc = f"{r.pcc:.5f}" if r.pcc is not None else "-"
                        es = f"{r.elapsed_s:.2f}" if r.elapsed_s is not None else "-"
                        rows.append((str(rank), r.config_id, kns, wns, pcc, es))
                    f.write(_fmt_table(headers, rows) + "\n")
                else:
                    f.write("(no configs produced profiler data)\n")

                if failed:
                    f.write(f"\nFailed / excluded ({len(failed)})\n")
                    f.write("-" * 40 + "\n")
                    frows = [
                        (
                            r.config_id,
                            f"{r.pcc:.5f}" if r.pcc is not None else "-",
                            _drop_reason(r),
                        )
                        for r in failed
                    ]
                    f.write(_fmt_table(("Config", "PCC", "Reason"), frows) + "\n")

            if self.verbose:
                print(f"Summary written to {txt_path}")
            paths[bench_name] = txt_path

        # Cross-bench best table printed to stdout only when there are multiple benches.
        if self.verbose and len(all_results) > 1:
            print("\nBest per bench:")
            headers = ("Bench", "Config", "kernel_ns", "wall_ns", "PCC")
            rows = []
            for bench_name, results in all_results.items():
                best = _best_result(results)
                if best is None:
                    rows.append((bench_name, "-", "-", "-", "-"))
                    continue
                kns = f"{best.kernel_ns:.1f}" if best.kernel_ns is not None else "-"
                wns = f"{best.wall_ns:.1f}" if best.wall_ns is not None else "-"
                pcc = f"{best.pcc:.5f}" if best.pcc is not None else "-"
                rows.append((bench_name, best.config_id, kns, wns, pcc))
            print(_fmt_table(headers, rows))

        return paths


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------


def _fmt_table(headers: tuple, rows: list[tuple]) -> str:
    """Format a plain-text aligned table from headers and rows."""
    all_rows = [headers, *rows]
    widths = [max(len(r[i]) for r in all_rows) for i in range(len(headers))]
    sep = "  ".join("-" * w for w in widths)
    lines = ["  ".join(cell.ljust(w) for cell, w in zip(headers, widths)), sep]
    for row in rows:
        lines.append("  ".join(cell.ljust(w) for cell, w in zip(row, widths)))
    return "\n".join(lines)


def _rank(
    results: list[AutotuneResult],
) -> tuple[list[AutotuneResult], list[AutotuneResult]]:
    """Split *results* into ``(valid_sorted_by_kernel_ns, failed)``.

    A result is *valid* only when it both produced a timing **and** had no
    error — runtime failures, the applied-config guard, and the PCC gate all
    set ``error``.  This is the single source of truth for both the stdout and
    the ``summary.txt`` views, so the ranked table and the ``Best`` line can
    never disagree.
    """
    valid = sorted(
        (r for r in results if r.kernel_ns is not None and r.error is None),
        key=lambda r: r.kernel_ns,
    )
    failed = [r for r in results if not (r.kernel_ns is not None and r.error is None)]
    return valid, failed


def _best_result(results: list[AutotuneResult]) -> Optional[AutotuneResult]:
    valid, _ = _rank(results)
    return valid[0] if valid else None


def _drop_reason(r: AutotuneResult) -> str:
    """One-line reason a result was excluded from the ranking."""
    if r.error:
        return r.error
    if r.kernel_ns is None:
        return "no profiler data"
    return ""


def _print_summary(bench_name: str, results: list[AutotuneResult]) -> None:
    valid, failed = _rank(results)

    print(f"\n=== {bench_name} ranking (kernel_ns) ===")
    if not valid:
        print("  (no valid results)")
    else:
        w = max(len(r.config_id) for r in valid)
        for rank, r in enumerate(valid, 1):
            pcc_str = f"  pcc={r.pcc:.5f}" if r.pcc is not None else ""
            print(f"  #{rank:<2} {r.config_id:<{w}}  {r.kernel_ns:>10.1f} ns{pcc_str}")

    if failed:
        print(f"\n  Failed / excluded ({len(failed)}):")
        w = max(len(r.config_id) for r in failed)
        for r in failed:
            pcc_str = f"pcc={r.pcc:.5f}  " if r.pcc is not None else ""
            print(f"    {r.config_id:<{w}}  {pcc_str}{_drop_reason(r)}")


# ---------------------------------------------------------------------------
# Kernel module loader
# ---------------------------------------------------------------------------


def load_kernel_module(path: str):
    """Import a kernel Python file and return its module object.

    The file is loaded via ``importlib.util`` so it does not need to be on
    ``sys.path``.  Any ``KERNEL_BENCHES`` dict in the module is accessible
    as ``mod.KERNEL_BENCHES``.
    """
    p = pathlib.Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Kernel file not found: {p}")
    spec = importlib.util.spec_from_file_location(p.stem, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# High-level entry point (importable)
# ---------------------------------------------------------------------------


def autotune_kernel(
    kernel_path: str,
    bench_names: Optional[list[str]] = None,
    *,
    knobs: Optional[AutotuneKnobs] = None,
    tensor_shapes: Optional[list[tuple[int, ...]]] = None,
    tensor_dtypes: Optional[list] = None,
    output_dir: str = "autotune-artifacts",
    save_profiler_logs: bool = False,
    check_pcc: bool = False,
    n_warmup: int = 1,
    verbose: bool = True,
    strategy: str = "sweep",
    max_rounds: int = 10,
) -> dict[str, list[AutotuneResult]]:
    """Autotune all (or selected) KernelBench entries in a kernel file.

    Parameters
    ----------
    kernel_path:
        Path to a Python file that defines ``KERNEL_BENCHES``.
    bench_names:
        Names of benches to tune.  ``None`` → tune all benches in the file.
    knobs:
        Knobs controlling the parameter sweep.  ``None`` → auto with defaults.
    tensor_shapes:
        One ``(dim, ...)`` tuple per tensor, overriding bench tensor shapes.
        Length must match ``len(bench.tensors)`` for each tuned bench.
    tensor_dtypes:
        One dtype per tensor, overriding bench tensor dtypes.
        Length must match ``len(bench.tensors)`` for each tuned bench.
    output_dir:
        Root artifacts directory (default ``"autotune-artifacts"``).
    save_profiler_logs:
        Keep raw ``profile_log_device.csv`` per run.
    check_pcc:
        Verify numerics against the golden after each run.
    n_warmup:
        Warmup iterations before the measured run.
    verbose:
        Print progress.
    strategy:
        ``'sweep'`` (default) – exhaustive Cartesian sweep.
        ``'hill-climb'`` – coordinate-descent hill-climb.
        ``'default'`` – run only each bench's own default config (no sweep).
    max_rounds:
        Maximum coordinate-descent rounds for ``strategy='hill-climb'``.

    Returns
    -------
    dict mapping bench name → list of ``AutotuneResult``.
    """
    mod = load_kernel_module(kernel_path)
    benches: dict[str, KernelBench] = getattr(mod, "KERNEL_BENCHES", {})
    if not benches:
        raise ValueError(f"No KERNEL_BENCHES found in {kernel_path!r}")

    if bench_names:
        missing = [n for n in bench_names if n not in benches]
        if missing:
            raise ValueError(
                f"Bench(es) not found in {kernel_path!r}: {missing}. "
                f"Available: {list(benches)}"
            )
        benches = {n: benches[n] for n in bench_names}

    tuner = Autotuner(
        knobs=knobs,
        output_dir=output_dir,
        save_profiler_logs=save_profiler_logs,
        check_pcc=check_pcc,
        n_warmup=n_warmup,
        verbose=verbose,
    )

    all_results: dict[str, list[AutotuneResult]] = {}
    for name, bench in benches.items():
        results = tuner.run_bench(
            bench,
            bench_name=name,
            strategy=strategy,
            max_rounds=max_rounds,
            tensor_shapes=tensor_shapes,
            tensor_dtypes=tensor_dtypes,
        )
        tuner.save_results(name, results)
        all_results[name] = results

    tuner.save_summary(all_results)
    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_shape(s: str) -> list[int]:
    """Parse ``'NxM'`` → ``[N, M]``."""
    parts = s.strip().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Shape must be NxM (e.g. '2x4'), got {s!r}")
    return [int(p) for p in parts]


def _parse_shapes(s: str) -> list[list[int]]:
    """Parse comma-separated shapes ``'1x1,2x2'`` → ``[[1,1],[2,2]]``."""
    return [_parse_shape(tok) for tok in s.split(",")]


def _parse_grid(s: str) -> tuple[int, int]:
    parts = _parse_shape(s)
    return (parts[0], parts[1])


def _parse_grids(s: str) -> list[tuple[int, int]]:
    return [_parse_grid(tok) for tok in s.split(",")]


def _parse_tensor_shapes(s: str) -> list[tuple[int, ...]]:
    """Parse ``'64x96,96x64'`` → ``[(64, 96), (96, 64)]`` (one per tensor)."""
    return [tuple(_parse_shape(tok)) for tok in s.split(",")]


_DTYPE_ALIASES = {
    "float32": "float32",
    "f32": "float32",
    "fp32": "float32",
    "float16": "float16",
    "f16": "float16",
    "fp16": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
}


def _parse_dtypes(s: str) -> list:
    """Parse ``'bf16,float32'`` → ``[torch.bfloat16, torch.float32]``."""
    import torch

    torch_by_name = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    out = []
    for tok in s.split(","):
        key = tok.strip().lower()
        if key not in _DTYPE_ALIASES:
            raise argparse.ArgumentTypeError(
                f"Unknown dtype {tok!r}; choose from {sorted(_DTYPE_ALIASES)}"
            )
        out.append(torch_by_name[_DTYPE_ALIASES[key]])
    return out


def _parse_joint_mems(s: str):
    """Parse joint mem spaces.

    ``'all'`` → the literal ``"all"`` sentinel (every L1/DRAM combo across
    tensors).  Otherwise ``'L1-L1,L1-DRAM'`` → ``[["L1","L1"],["L1","DRAM"]]``
    (dash separates tensors within one combo, comma separates combos).
    """
    if s.strip().lower() == "all":
        return "all"
    return [[m.strip() for m in combo.split("-")] for combo in s.split(",")]


def _parse_joint_blocks(s: str) -> list[list[list[int]]]:
    """Parse joint block shapes.

    ``'1x1_1x1,1x3_3x1'`` → ``[[[1,1],[1,1]],[[1,3],[3,1]]]`` (underscore
    separates tensors within one entry, comma separates entries).
    """
    return [[_parse_shape(bs) for bs in entry.split("_")] for entry in s.split(",")]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=textwrap.dedent(
            """\
            Autotune d2m-jit kernels to maximize performance by minimizing
            kernel_ns.  Sweeps grid_shape, block_shape, and mem_space
            combinations and saves results under autotune-artifacts/.
            """
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--kernel",
        required=True,
        metavar="PATH",
        help="Path to a kernel .py file that defines KERNEL_BENCHES.",
    )
    p.add_argument(
        "--bench",
        metavar="NAME[,NAME…]",
        default=None,
        help=(
            "Comma-separated bench names to tune.  "
            "Omit to tune all benches in the file."
        ),
    )
    p.add_argument(
        "--grid-shapes",
        metavar="NxM[,…]",
        default=None,
        help=(
            "Explicit grid shapes to try, e.g. '1x1,2x2,4x4'.  "
            "Omit to auto-generate from tensor shape."
        ),
    )
    p.add_argument(
        "--block-shapes",
        metavar="NxM[,…]",
        default=None,
        help=(
            "Explicit block shapes to try, e.g. '1x1,2x2'.  "
            "Omit to auto-generate per grid."
        ),
    )
    p.add_argument(
        "--mem-spaces",
        metavar="SPACE[,…]",
        default=None,
        help=(
            "Memory spaces to try: 'L1' and/or 'DRAM'.  "
            "Omit to sweep both in full-sweep mode, or use 'L1' in focused mode."
        ),
    )
    p.add_argument(
        "--joint-mem-spaces",
        metavar="SPEC",
        default=None,
        help=(
            "Per-tensor mem-space combos swept jointly (not Cartesian).  "
            "'all' sweeps every L1/DRAM combo across tensors, or list combos "
            "like 'L1-L1,L1-DRAM,DRAM-DRAM' (dash = per-tensor within a combo)."
        ),
    )
    p.add_argument(
        "--joint-block-shapes",
        metavar="SPEC",
        default=None,
        help=(
            "Per-tensor block-shape entries swept jointly (not Cartesian), "
            "e.g. '1x1_1x1,1x3_3x1' (underscore = per-tensor within an entry).  "
            "Required for kernels like matmul whose operand blocks are coupled."
        ),
    )
    p.add_argument(
        "--tensor-shapes",
        metavar="NxM[,…]",
        default=None,
        help=("Override bench tensor shapes, one per tensor, e.g. '64x96,96x64'."),
    )
    p.add_argument(
        "--tensor-dtypes",
        metavar="DTYPE[,…]",
        default=None,
        help=("Override bench tensor dtypes, one per tensor, e.g. 'bf16,float32'."),
    )
    p.add_argument(
        "--output-dir",
        metavar="DIR",
        default="autotune-artifacts",
        help="Root directory for results (default: autotune-artifacts).",
    )
    p.add_argument(
        "--save-profiler-logs",
        action="store_true",
        default=False,
        help="Save raw profile_log_device.csv per config run.",
    )
    p.add_argument(
        "--check-pcc",
        action="store_true",
        default=False,
        help="Verify PCC against golden after each run.",
    )
    p.add_argument(
        "--strategy",
        metavar="STRATEGY",
        default="sweep",
        choices=["sweep", "hill-climb", "default"],
        help=(
            "Search strategy: 'sweep' (exhaustive), 'hill-climb' (faster, coordinate descent), "
            "or 'default' (run only the bench defaults; no sweep)."
        ),
    )
    p.add_argument(
        "--no-sweep",
        action="store_true",
        default=False,
        help=(
            "Skip the sweep; run only each bench's own default config "
            "(grid_shape, and per-tensor block_shape/mem_space from the "
            "KernelBench definition).  Ignores the sweep knob flags."
        ),
    )
    p.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        metavar="N",
        help="Max coordinate-descent rounds for --strategy hill-climb (default: 10).",
    )
    p.add_argument(
        "--max-cores",
        type=int,
        default=8,
        metavar="N",
        help="Max gy*gx when auto-generating grid shapes (default: 8).",
    )
    p.add_argument(
        "--max-block",
        type=int,
        default=8,
        metavar="N",
        help="Max per-dim block size when auto-generating block shapes (default: 8).",
    )
    p.add_argument(
        "--n-warmup",
        type=int,
        default=1,
        metavar="N",
        help="Warmup iterations before the measured run (default: 1).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress per-config progress output.",
    )
    return p


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    bench_names: Optional[list[str]] = None
    if args.bench:
        bench_names = [n.strip() for n in args.bench.split(",")]

    mem_spaces = (
        [m.strip() for m in args.mem_spaces.split(",")] if args.mem_spaces else None
    )

    # --no-sweep is just the 'default' strategy; everything else flows through
    # the same autotune_kernel entry point.
    strategy = "default" if args.no_sweep else args.strategy

    knobs = AutotuneKnobs(
        grid_shapes=_parse_grids(args.grid_shapes) if args.grid_shapes else None,
        block_shapes=_parse_shapes(args.block_shapes) if args.block_shapes else None,
        mem_spaces=mem_spaces,
        joint_mem_spaces=(
            _parse_joint_mems(args.joint_mem_spaces) if args.joint_mem_spaces else None
        ),
        joint_block_shapes=(
            _parse_joint_blocks(args.joint_block_shapes)
            if args.joint_block_shapes
            else None
        ),
        max_cores=args.max_cores,
        max_block_tiles=args.max_block,
    )

    autotune_kernel(
        kernel_path=args.kernel,
        bench_names=bench_names,
        knobs=knobs,
        tensor_shapes=(
            _parse_tensor_shapes(args.tensor_shapes) if args.tensor_shapes else None
        ),
        tensor_dtypes=(
            _parse_dtypes(args.tensor_dtypes) if args.tensor_dtypes else None
        ),
        output_dir=args.output_dir,
        save_profiler_logs=args.save_profiler_logs,
        check_pcc=args.check_pcc,
        n_warmup=args.n_warmup,
        verbose=not args.quiet,
        strategy=strategy,
        max_rounds=args.max_rounds,
    )


if __name__ == "__main__":
    main()
