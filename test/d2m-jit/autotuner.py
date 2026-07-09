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
* ``grid_shape`` (per-kernel): logical 2-D core grid ``(gy, gx)``.
* ``block_shape`` (per-tensor): tile-block shape ``[by, bx]``.  Passed to
  every ``TensorSpec`` in the bench; materializers that read only
  ``tensors[0].block_shape`` (``eltwise_block_run``, ``rope_materializer``,
  …) will apply it uniformly.  Per-tensor differentiation requires a custom
  ``run`` function.
* ``mem_space`` (per-tensor): tensor memory placement (``"L1"`` or
  ``"DRAM"``).  Applied by temporarily overriding the default argument of
  ``d2m_jit.Layout.__init__`` for the duration of each config run, so that
  every Layout constructed inside the materializer picks up the chosen
  space.  The override is reversed after each run.

Heuristics for sweep narrowing
-------------------------------
When ``grid_shapes`` / ``block_shapes`` are ``None``, valid candidates are
auto-generated from the primary tensor's shape:

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
        [--grid-shapes 1x1,2x2,4x4] \\
        [--block-shapes 1x1,2x2] \\
        [--mem-spaces L1,DRAM] \\
        [--output-dir autotune-artifacts] \\
        [--save-profiler-logs] \\
        [--check-pcc] \\
        [--no-sweep] \\
        [--max-cores 8] \\
        [--max-block 8] \\
        [--n-warmup 1] \\
        [--traits device-zone]

Module usage
------------
::

    import importlib.util, pathlib, sys
    sys.path.insert(0, "test/d2m-jit")
    from autotuner import Autotuner, AutotuneKnobs

    spec = importlib.util.spec_from_file_location(
        "rope", "test/d2m-jit/kernels/prefill/rope.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    bench = mod.KERNEL_BENCHES["rope"]

    knobs = AutotuneKnobs(
        grid_shapes=[(1, 1), (2, 2), (4, 4)],
        block_shapes=[[1, 1], [2, 2]],
        mem_spaces=["L1"],
    )
    tuner = Autotuner(knobs=knobs, output_dir="autotune-artifacts")
    results = tuner.run_bench(bench, bench_name="rope")
    tuner.save_results("rope", results)
    tuner.save_summary({"rope": results})

"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import importlib.util
import json
import math
import os
import pathlib
import shutil
import sys
import textwrap
import time
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Make runner.py importable when running from the repo root
# ---------------------------------------------------------------------------

_RUNNER_DIR = pathlib.Path(__file__).parent
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
        pathlib.Path(__file__).parents[2]
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

    **Full-sweep mode** (default): leave all three knobs as ``None``.  The
    tuner auto-generates every valid ``grid_shape`` and ``block_shape`` from
    the primary tensor's dimensions and tries both ``"L1"`` and ``"DRAM"``
    mem spaces.

    **Focused mode**: set any one (or more) knob explicitly.  Unset knobs
    default to the bench's own value (single grid/block from the
    ``KernelBench`` definition, ``"L1"`` for mem_space) rather than being
    swept.  This avoids an exponential config explosion when you only care
    about one dimension.

    Examples
    --------
    Sweep only grid shapes, keep block and mem at bench defaults::

        AutotuneKnobs(grid_shapes=[(1, 1), (2, 2), (4, 4)])

    Sweep only mem spaces, keep grid and block at bench defaults::

        AutotuneKnobs(mem_spaces=["L1", "DRAM"])

    Attributes
    ----------
    grid_shapes:
        Explicit list of ``(gy, gx)`` tuples.  ``None`` → auto (full-sweep)
        or bench default (focused).
    block_shapes:
        Explicit list of ``[by, bx]`` shapes applied to every tensor.
        ``None`` → auto (full-sweep) or bench default (focused).
    mem_spaces:
        List of memory space strings: ``"L1"`` and/or ``"DRAM"``.
        ``None`` → ``["L1", "DRAM"]`` (full-sweep) or ``["L1"]`` (focused).
    max_cores:
        Maximum ``gy * gx`` when auto-generating grid shapes.
    max_block_tiles:
        Maximum per-dimension block size when auto-generating block shapes.
    """

    grid_shapes: Optional[list[tuple[int, int]]] = None
    block_shapes: Optional[list[list[int]]] = None
    mem_spaces: Optional[list[str]] = None
    max_cores: int = 8
    max_block_tiles: int = 8


@dataclass
class AutotuneConfig:
    """One complete set of parameters to evaluate."""

    grid_shape: tuple[int, int]
    block_shape: list[int]
    mem_space: str = "L1"

    def __post_init__(self):
        self.grid_shape = tuple(self.grid_shape)
        self.block_shape = list(self.block_shape)

    @property
    def id(self) -> str:
        g = f"{self.grid_shape[0]}x{self.grid_shape[1]}"
        b = f"{self.block_shape[0]}x{self.block_shape[1]}"
        return f"g{g}_b{b}_m{self.mem_space}"

    def as_dict(self) -> dict:
        return {
            "grid_shape": list(self.grid_shape),
            "block_shape": self.block_shape,
            "mem_space": self.mem_space,
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
# mem_space override context manager
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _mem_space_ctx(mem_space: str):
    """Temporarily patch ``d2m_jit.Layout.__init__`` default mem_space.

    All ``Layout`` constructions inside this context (e.g. inside a
    materializer that doesn't explicitly pass ``mem_space``) will use the
    given space.  The default is restored on exit.
    """
    if mem_space.upper() == "L1":
        yield
        return

    from d2m_jit._src import tensor_layout as _tl

    original_init = _tl.Layout.__init__
    target_ms = _tl._to_mem_space(mem_space.lower())

    def _patched_init(
        self_inner,
        shape,
        dtype,
        block_shape,
        grid_shape=None,
        tiled=True,
        collapse=True,
        mem_space=None,  # noqa: intentionally shadows outer name
    ):
        # Always use the autotuner-requested mem_space regardless of what
        # the caller specified (or defaulted to).
        original_init(
            self_inner,
            shape,
            dtype,
            block_shape,
            grid_shape,
            tiled,
            collapse,
            target_ms,
        )

    _tl.Layout.__init__ = _patched_init
    try:
        yield
    finally:
        _tl.Layout.__init__ = original_init


# ---------------------------------------------------------------------------
# Profiling context manager and CSV parser
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _profiling_ctx(profiler_dir: str, traits: str):
    """Enable in-process device profiling, directing output to *profiler_dir*.

    Sets the environment variable ``TT_METAL_PROFILER_DIR`` so the tt-metal
    runtime writes ``profile_log_device.csv`` into ``<profiler_dir>/.logs/``.
    Also enables ``config.enable_perf_trace`` and
    ``config.insert_profiler_traces``.  All mutations are reversed on exit.
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
    old_traits = _cfg.profiler_traits
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
        _cfg.profiler_traits = old_traits
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
    csv_path: pathlib.Path,
) -> tuple[Optional[float], Optional[float]]:
    """Return ``(kernel_ns, wall_ns)`` from a ``profile_log_device.csv``.

    Uses ``collect_device_runtimes`` from perf-analyzer for kernel duration
    and ``read_chip_freq_mhz`` to convert cycles → ns.

    Returns ``(None, None)`` if the file does not exist or cannot be parsed
    (e.g. profiling not enabled in the build).
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
    except Exception as e:
        return None, None


# ---------------------------------------------------------------------------
# Main Autotuner class
# ---------------------------------------------------------------------------


class Autotuner:
    """Sweeps autotuning parameters for KernelBench entries.

    Parameters
    ----------
    knobs:
        ``AutotuneKnobs`` instance.  Defaults to all-auto knobs with
        ``mem_spaces=["L1"]``.
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
    traits:
        Profiler trait string forwarded to ``insert-device-zone-scopes``
        (``"device-zone"`` | ``"fpu,sfpu"`` | ``"all"`` | …).
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
        traits: str = "device-zone",
        verbose: bool = True,
    ):
        self.knobs = knobs or AutotuneKnobs()
        self.output_dir = pathlib.Path(output_dir)
        self.save_profiler_logs = save_profiler_logs
        self.check_pcc = check_pcc
        self.n_warmup = n_warmup
        self.traits = traits
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

    def generate_configs(self, bench: KernelBench) -> list[AutotuneConfig]:
        """Return the full list of ``AutotuneConfig``s to evaluate for *bench*.

        The configs are the Cartesian product of::

            grids × blocks(per grid) × mem_spaces

        In **full-sweep mode** (all knobs ``None``): grids and blocks are
        auto-generated from the primary tensor's shape, and mem_spaces is
        ``["L1", "DRAM"]``.

        In **focused mode** (any knob explicitly set): unset knobs fall back
        to the bench's own defaults (single value) rather than being swept.

        Duplicates (identical ``config.id``) are deduplicated.
        """
        knobs = self.knobs
        full_sweep = (
            knobs.grid_shapes is None
            and knobs.block_shapes is None
            and knobs.mem_spaces is None
        )

        if full_sweep:
            grids = valid_grid_shapes(bench, knobs)
            mem_spaces_list = ["L1", "DRAM"]
        else:
            grids = (
                knobs.grid_shapes
                if knobs.grid_shapes is not None
                else [tuple(bench.grid_shape)]
            )
            mem_spaces_list = (
                knobs.mem_spaces if knobs.mem_spaces is not None else ["L1"]
            )

        seen: set[str] = set()
        configs: list[AutotuneConfig] = []

        for grid in grids:
            if full_sweep:
                blocks = valid_block_shapes(bench, grid, knobs)
            elif knobs.block_shapes is not None:
                blocks = valid_block_shapes(bench, grid, knobs)  # filters by grid
            else:
                blocks = [list(bench.tensors[0].block_shape)]

            for block in blocks:
                for mem in mem_spaces_list:
                    cfg = AutotuneConfig(
                        grid_shape=grid,
                        block_shape=block,
                        mem_space=mem,
                    )
                    if cfg.id not in seen:
                        seen.add(cfg.id)
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

        1. Replaces each ``TensorSpec``'s ``block_shape`` with
           ``config.block_shape``.
        2. Overrides the ``d2m_jit.Layout`` default ``mem_space`` for the
           run duration.
        3. Enables in-process device profiling into the session profiler dir.
        4. Executes ``n_warmup`` un-measured warm-up iterations.
        5. Executes one measured iteration and parses the profiler CSV.
        6. Optionally checks PCC and saves the profiler log.

        Note on profiler directory: tt-metal initialises its profiler singleton
        on the first measured run and retains that path for the process lifetime.
        ``self._profiler_dir`` is therefore used consistently across all configs
        so every run's CSV lands where ``_parse_profile_csv`` expects it.
        """
        from d2m_jit._src.builder import _Builder

        overridden_tensors = [
            dataclasses.replace(ts, block_shape=list(config.block_shape))
            for ts in bench.tensors
        ]

        profiler_tmp = tmp_dir or self._profiler_dir
        profiler_csv = pathlib.Path(profiler_tmp) / ".logs" / "profile_log_device.csv"

        t0 = time.perf_counter()
        actual = None
        expected = None
        error_msg: Optional[str] = None

        try:
            with _mem_space_ctx(config.mem_space):
                # The profiling context must wrap *all* runs (including
                # warmup) because TT_METAL_DEVICE_PROFILER=1 needs to be set
                # before the device is first opened.  Warmup data is simply
                # discarded by wiping the .logs dir before the measured pass.
                with _profiling_ctx(profiler_tmp, self.traits):
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

                    # Measured run.
                    _Builder.reset()
                    actual, expected = run_bench(
                        bench,
                        tensors=overridden_tensors,
                        grid_shape=config.grid_shape,
                    )

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
        finally:
            _Builder.reset()

        elapsed_s = time.perf_counter() - t0

        kernel_ns, wall_ns = _parse_profile_csv(profiler_csv)

        pcc_val: Optional[float] = None
        if self.check_pcc and actual is not None and expected is not None:
            try:
                pcc_val = compute_pcc(expected.float(), actual.float())
            except Exception as exc:
                if error_msg is None:
                    error_msg = f"PCC error: {exc}"

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
        seed:
            Starting config for hill-climbing.  ``None`` → bench default.
        max_rounds:
            Maximum coordinate-descent rounds for hill-climbing.
        """
        name = bench_name or bench.name or "unnamed"

        if strategy == "hill-climb":
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

        # Determine seed config.
        if seed is None:
            ts = bench.tensors[0]
            full_sweep = (
                knobs.grid_shapes is None
                and knobs.block_shapes is None
                and knobs.mem_spaces is None
            )
            seed = AutotuneConfig(
                grid_shape=tuple(bench.grid_shape),
                block_shape=list(ts.block_shape),
                mem_space="L1" if not full_sweep else "L1",
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

        full_sweep = (
            knobs.grid_shapes is None
            and knobs.block_shapes is None
            and knobs.mem_spaces is None
        )
        mem_spaces_list = ["L1", "DRAM"] if full_sweep else (knobs.mem_spaces or ["L1"])

        current = seed
        current_result = _run(current)

        if self.verbose:
            print(
                f"\n=== Hill-climb {bench_name!r}: "
                f"seed={current.id}  max_rounds={max_rounds} ==="
            )

        for round_idx in range(max_rounds):
            improved = False

            # --- Axis 1: grid ---
            all_grids = valid_grid_shapes(bench, knobs)
            grid_candidates = []
            for g in all_grids:
                blocks = valid_block_shapes(bench, g, knobs)
                if not blocks:
                    continue
                adapted_block = _closest_block(current.block_shape, blocks)
                grid_candidates.append(
                    AutotuneConfig(g, adapted_block, current.mem_space)
                )
            best_grid_r = _best_of(grid_candidates)
            if _better(best_grid_r, current_result):
                current = best_grid_r.config
                current_result = best_grid_r
                improved = True

            # --- Axis 2: block (for current grid) ---
            all_blocks = valid_block_shapes(bench, current.grid_shape, knobs)
            block_candidates = [
                AutotuneConfig(current.grid_shape, b, current.mem_space)
                for b in all_blocks
            ]
            best_block_r = _best_of(block_candidates)
            if _better(best_block_r, current_result):
                current = best_block_r.config
                current_result = best_block_r
                improved = True

            # --- Axis 3: mem_space ---
            mem_candidates = [
                AutotuneConfig(current.grid_shape, current.block_shape, m)
                for m in mem_spaces_list
            ]
            best_mem_r = _best_of(mem_candidates)
            if _better(best_mem_r, current_result):
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

            valid = sorted(
                [r for r in results if r.kernel_ns is not None],
                key=lambda r: r.kernel_ns,
            )
            failed = [r for r in results if r.kernel_ns is None]
            best = _best_result(results)

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

                f.write("Ranking (kernel_ns, lower is better)\n")
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
                    f.write(f"\nFailed ({len(failed)}): ")
                    f.write(", ".join(r.config_id for r in failed))
                    f.write("\n")

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


def _best_result(results: list[AutotuneResult]) -> Optional[AutotuneResult]:
    valid = [r for r in results if r.kernel_ns is not None and r.error is None]
    return min(valid, key=lambda r: r.kernel_ns) if valid else None


def _print_summary(bench_name: str, results: list[AutotuneResult]) -> None:
    valid = sorted(
        [r for r in results if r.kernel_ns is not None],
        key=lambda r: r.kernel_ns,
    )
    failed = [r for r in results if r.kernel_ns is None]

    print(f"\n=== {bench_name} ranking (kernel_ns, lower is better) ===")
    if not valid:
        print("  (no results with profiler data)")
    else:
        w = max(len(r.config_id) for r in valid)
        for rank, r in enumerate(valid, 1):
            pcc_str = f"  pcc={r.pcc:.5f}" if r.pcc is not None else ""
            print(f"  #{rank:<2} {r.config_id:<{w}}  {r.kernel_ns:>10.1f} ns{pcc_str}")

    if failed:
        print(f"\n  Failed: {', '.join(r.config_id for r in failed)}")


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
    output_dir: str = "autotune-artifacts",
    save_profiler_logs: bool = False,
    check_pcc: bool = False,
    n_warmup: int = 1,
    traits: str = "device-zone",
    verbose: bool = True,
    strategy: str = "sweep",
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
    output_dir:
        Root artifacts directory (default ``"autotune-artifacts"``).
    save_profiler_logs:
        Keep raw ``profile_log_device.csv`` per run.
    check_pcc:
        Verify numerics against the golden after each run.
    n_warmup:
        Warmup iterations before the measured run.
    traits:
        Profiler traits string (``"device-zone"`` | ``"all"`` | …).
    verbose:
        Print progress.
    strategy:
        ``'sweep'`` (default) – exhaustive Cartesian sweep.
        ``'hill-climb'`` – coordinate-descent hill-climb.

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
        traits=traits,
        verbose=verbose,
    )

    all_results: dict[str, list[AutotuneResult]] = {}
    for name, bench in benches.items():
        results = tuner.run_bench(bench, bench_name=name, strategy=strategy)
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
        choices=["sweep", "hill-climb"],
        help="Search strategy: 'sweep' (exhaustive, default) or 'hill-climb' (faster, coordinate descent).",
    )
    p.add_argument(
        "--no-sweep",
        action="store_true",
        default=False,
        help=(
            "Skip the auto-sweep; only run each bench's default config "
            "(grid_shape and block_shape from the KernelBench definition)."
        ),
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
        "--traits",
        default="device-zone",
        metavar="STR",
        help="Profiler traits string (default: 'device-zone').",
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

    if args.no_sweep:
        # Default-config-only: use the KernelBench's own grid/block/mem.
        mod = load_kernel_module(args.kernel)
        benches: dict[str, KernelBench] = getattr(mod, "KERNEL_BENCHES", {})
        if bench_names:
            benches = {n: benches[n] for n in bench_names if n in benches}

        tuner = Autotuner(
            output_dir=args.output_dir,
            save_profiler_logs=args.save_profiler_logs,
            check_pcc=args.check_pcc,
            n_warmup=args.n_warmup,
            traits=args.traits,
            verbose=not args.quiet,
        )
        all_results: dict[str, list[AutotuneResult]] = {}
        for name, bench in benches.items():
            ts = bench.tensors[0]
            cfg = AutotuneConfig(
                grid_shape=tuple(bench.grid_shape),
                block_shape=list(ts.block_shape),
                mem_space=(mem_spaces[0] if mem_spaces else "L1"),
            )
            result = tuner.run_config(bench, cfg, name)
            tuner.save_results(name, [result])
            all_results[name] = [result]
        tuner.save_summary(all_results)
        return

    # Full sweep
    grid_shapes: Optional[list[tuple[int, int]]] = None
    block_shapes: Optional[list[list[int]]] = None

    if args.grid_shapes:
        grid_shapes = _parse_grids(args.grid_shapes)
    if args.block_shapes:
        block_shapes = _parse_shapes(args.block_shapes)

    knobs = AutotuneKnobs(
        grid_shapes=grid_shapes,
        block_shapes=block_shapes,
        mem_spaces=mem_spaces,
        max_cores=args.max_cores,
        max_block_tiles=args.max_block,
    )

    autotune_kernel(
        kernel_path=args.kernel,
        bench_names=bench_names,
        knobs=knobs,
        output_dir=args.output_dir,
        save_profiler_logs=args.save_profiler_logs,
        check_pcc=args.check_pcc,
        n_warmup=args.n_warmup,
        traits=args.traits,
        verbose=not args.quiet,
        strategy=args.strategy,
    )


if __name__ == "__main__":
    main()
