# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Direct-kernel autotuner (Path A) for d2m-jit.

Loads kernels from pattern files (KERNEL_BENCHES), sweeps a config space over
``@d2m.kernel`` driven directly, runs each config on device with profiler traces,
and ranks by ``device_kernel_time.ns`` (parsed from profiler CSV via
``tools/perf-analyzer/perf-analyzer.py::build_report``).

Kernels are defined in ``tools/d2m-jit/patterns/*.py`` files, each exporting a
``KERNEL_BENCHES`` list. The autotuner dynamically loads patterns via
``_load_pattern_file(path)`` which extracts the first KernelBench and passes it
to ``tune()`` or ``search()``.

A config that fails to compile/run stays in the results with ``valid=False``
(useful signal) but is excluded from the leaderboard and from ``best``.

An optional PCC correctness gate (``--check-pcc``, OFF by default) drops any
config whose output deviates from a golden reference. It's cheap (one corrcoef
per config) but off by default for human-driven tuning where kernels are trusted.

Capture mechanics: the in-process device profiler writes
``$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv`` after each
workload. The sweep clears that file before each config and builds the report
from it; with ``--save-profiler-logs`` it also snapshots the CSV into a per-config
``cfg_NNN/`` dir (build output is always discarded).

Requirements (profiler env must be set before device-open):
    TT_METAL_DEVICE_PROFILER=1
    TT_METAL_DEVICE_PROFILER_DISPATCH=0
and a perf-trace runtime build (TT_RUNTIME_ENABLE_PERF_TRACE=ON). ``tune()`` and
``search()`` set these env vars and config flags.

Swept knobs: ``grid_shape`` (core grid), ``block_shape`` (tiles per shard),
``mem_space`` (l1/dram -- where kernel I/O lives). Shape, dtype, and kernel
come from the pattern's KernelBench and are NOT swept. Config is a plain dict.

Two entry points share the same per-config evaluation:
  * ``tune(bench, input_shapes, space)`` -- evaluate every config, rank + Pareto.
  * ``search(bench, input_shapes, proposer)`` -- guided search via proposer,
    using the bundled ``hill_climb_proposer`` heuristic (latency coordinate descent).
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib.util
import json
import os
import pathlib
import shutil
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch

from d2m_jit import config
from d2m_jit.testing import make_inputs, torch_dtype, KernelBench

# --- perf-analyzer (hyphenated filename / sibling tool -> load by path) ------


@contextlib.contextmanager
def _silence_build_output(log_path):
    """Redirect OS-level stdout/stderr (fds 1/2) to ``log_path`` for the duration.

    Kernel JIT compilation runs in subprocesses spawned by tt-metal, so their
    profiler ``#pragma message`` notes reach the terminal through the inherited
    file descriptors, not Python's ``sys.stdout`` -- a ``contextlib.redirect_*``
    won't catch them. Redirecting the fds does. The profiler CSV is written to a
    separate file and is unaffected. Callers pass ``os.devnull`` to discard the
    build output; a genuine compile failure still surfaces via the raised
    exception's message (captured in the config's ``error`` field)."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out, saved_err = os.dup(1), os.dup(2)
    with open(log_path, "w") as log:
        try:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(saved_out)
            os.close(saved_err)


def _load_perf_analyzer():
    """Import ``tools/perf-analyzer/perf-analyzer.py`` (not a normal module)."""
    here = pathlib.Path(os.path.realpath(__file__)).parent  # tools/d2m-jit
    candidates = [here.parent / "perf-analyzer" / "perf-analyzer.py"]
    home = os.environ.get("TT_MLIR_HOME")
    if home:
        candidates.append(pathlib.Path(home) / "tools/perf-analyzer/perf-analyzer.py")
    for cand in candidates:
        if cand.exists():
            spec = importlib.util.spec_from_file_location("d2m_jit_perf_analyzer", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(
        "could not locate tools/perf-analyzer/perf-analyzer.py "
        f"(looked in {[str(c) for c in candidates]}); set TT_MLIR_HOME"
    )


def _device_csv() -> pathlib.Path:
    home = os.environ.get("TT_METAL_HOME")
    if not home:
        raise RuntimeError("TT_METAL_HOME is not set; cannot locate profiler CSV")
    return pathlib.Path(home) / "generated/profiler/.logs/profile_log_device.csv"


# --- Config space ------------------------------------------------------------


def grid_space(shape, max_cores: int = 64) -> list[dict]:
    """Enumerate feasible ``grid_shape`` configs for a 2D ``shape``.

    A grid ``(gy, gx)`` is feasible when it evenly divides the tile counts
    ``(shape[-2]//32, shape[-1]//32)`` and ``gy*gx <= max_cores`` (physical grid
    bound). Returns a list of configs with grid_shape and block_shape=[1,1].
    """
    ty, tx = shape[-2] // 32, shape[-1] // 32
    cfgs = []
    for gy in range(1, ty + 1):
        if ty % gy:
            continue
        for gx in range(1, tx + 1):
            if tx % gx or gy * gx > max_cores:
                continue
            cfgs.append({"grid_shape": [gy, gx], "block_shape": [1, 1]})
    return cfgs


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def block_space(shape, max_cores: int = 64) -> list[dict]:
    """Enumerate feasible ``(grid_shape, block_shape)`` configs for a 2D ``shape``.

    ``block_shape`` ``(bh, bw)`` (in tiles) is the multi-tile shard each
    ``remote_load`` pulls and the kernel computes over. Feasibility:
      * ``bh | tiles_y`` and ``bw | tiles_x`` (block tiles the tensor),
      * the blocked grid ``(tiles_y//bh, tiles_x//bw)`` is divisible by
        ``grid_shape`` (blocks distribute evenly across cores),
      * ``gy*gx <= max_cores``.
    Returns list of configs.
    """
    ty, tx = shape[-2] // 32, shape[-1] // 32
    cfgs = []
    for bh in _divisors(ty):
        for bw in _divisors(tx):
            by, bx = ty // bh, tx // bw  # blocked grid (blocks per dim)
            for gy in _divisors(by):
                for gx in _divisors(bx):
                    if gy * gx > max_cores:
                        continue
                    cfgs.append(
                        {
                            "grid_shape": [gy, gx],
                            "block_shape": [bh, bw],
                        }
                    )
    return cfgs


# --- CLI config-space construction -------------------------------------------
#
# Per-knob candidate pools + a feasibility-filtered product. Both "which knobs
# to parameterize" and "how many values each takes" reduce to pool size: a knob
# omitted on the CLI gets a 1-element pool (pinned); a knob given a list or
# "all" gets a multi-element pool (swept). The same constrained space then feeds
# either tune() (exhaustive) or search() (the proposer's candidates).

_ALL = "all"  # spec keyword: sweep every feasible value of this knob


def _parse_pairs(text: str) -> list[list[int]]:
    """'1x1,2x2,4x4' -> [[1, 1], [2, 2], [4, 4]]."""
    out = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        a, _, b = tok.partition("x")
        out.append([int(a), int(b)])
    return out


def grid_candidates(shape, spec, max_cores: int = 64) -> list[list[int]]:
    """Grid pool from a CLI spec: None (omitted) or ``"all"`` is every feasible
    grid (divides the tile counts, <= max_cores); else an explicit ``"1x1,2x2"``
    list. An unspecified knob means no filter -- sweep everything."""
    if spec is None or spec == _ALL:
        ty, tx = shape[-2] // 32, shape[-1] // 32
        return [
            [gy, gx]
            for gy in _divisors(ty)
            for gx in _divisors(tx)
            if gy * gx <= max_cores
        ]
    return _parse_pairs(spec)


def block_candidates(shape, spec) -> list[list[int]]:
    """Block pool from a CLI spec: None (omitted) or ``"all"`` is every block
    that tiles the tensor; else an explicit ``"1x1,2x2"`` list."""
    if spec is None or spec == _ALL:
        ty, tx = shape[-2] // 32, shape[-1] // 32
        return [[bh, bw] for bh in _divisors(ty) for bw in _divisors(tx)]
    return _parse_pairs(spec)


_MEM_SPACES = ("l1", "dram")


def memspace_candidates(spec) -> list[str]:
    """Mem-space pool: None (omitted) or ``"all"`` is both l1 and dram (l1 is the
    default placement; dram I/O adds DRAM<->L1 movement); else an explicit
    ``"l1,dram"`` list (validated)."""
    if spec is None or spec == _ALL:
        return list(_MEM_SPACES)
    out = [m.strip() for m in spec.split(",") if m.strip()]
    bad = [m for m in out if m not in _MEM_SPACES]
    if bad:
        raise ValueError(f"unknown mem_space(s) {bad}; choose from {list(_MEM_SPACES)}")
    return out


def _feasible(shape, cfg, max_cores) -> bool:
    """A (grid, block) cfg is feasible when the block tiles the tensor,
    the blocked grid divides evenly across the grid, and cores <= max_cores."""
    ty, tx = shape[-2] // 32, shape[-1] // 32
    gy, gx = cfg["grid_shape"]
    bh, bw = cfg["block_shape"]
    if ty % bh or tx % bw:
        return False
    if (ty // bh) % gy or (tx // bw) % gx:
        return False
    return gy * gx <= max_cores


def build_space(
    shape,
    *,
    grid=None,
    block=None,
    mem_space=None,
    max_cores: int = 64,
) -> tuple:
    """Cartesian product of swept knob pools (grid x block x mem_space),
    filtered for feasibility. Returns ``(configs, warnings)`` -- warnings flag
    any *explicitly requested* grid/block value (not ``"all"``) that survived
    in no feasible config, so the drop is surfaced rather than silent. mem_space
    doesn't affect feasibility so it never warns."""
    grids = grid_candidates(shape, grid, max_cores)
    blocks = block_candidates(shape, block)
    mems = memspace_candidates(mem_space)

    configs = []
    for g in grids:
        for b in blocks:
            for mem in mems:
                cfg = {
                    "grid_shape": list(g),
                    "block_shape": list(b),
                    "mem_space": mem,
                }
                if _feasible(shape, cfg, max_cores):
                    configs.append(cfg)

    warnings = []
    explicit = lambda s: s is not None and s != _ALL  # noqa: E731
    if explicit(grid):
        used = {tuple(c["grid_shape"]) for c in configs}
        warnings += [
            f"grid {g[0]}x{g[1]} infeasible for {shape[0]}x{shape[1]} (max_cores"
            f"={max_cores}) -- dropped"
            for g in grids
            if tuple(g) not in used
        ]
    if explicit(block):
        used = {tuple(c["block_shape"]) for c in configs}
        warnings += [
            f"block {b[0]}x{b[1]} infeasible for {shape[0]}x{shape[1]} -- dropped"
            for b in blocks
            if tuple(b) not in used
        ]
    return configs, warnings


# --- Result model ------------------------------------------------------------


@dataclass
class TuneRecord:
    config: dict
    valid: bool
    pcc: Optional[float] = None
    kernel_ns: Optional[float] = None
    diagnostics: dict = field(default_factory=dict)
    report_path: Optional[str] = None
    error: Optional[str] = None


@dataclass
class Objective:
    """One Pareto axis: a name, a direction, and how to read it off a record."""

    name: str
    direction: str  # "min" | "max"
    extract: Callable[[TuneRecord], Optional[float]]

    def at_least(self, a: float, b: float) -> bool:
        """a is no worse than b on this axis."""
        return a <= b if self.direction == "min" else a >= b

    def better(self, a: float, b: float) -> bool:
        """a is strictly better than b on this axis."""
        return a < b if self.direction == "min" else a > b


# Prebuilt axes for kernel tuning. `cores` reads gy*gx off the config; the rest
# read measured fields. Compose any tuple of these (or your own) into pareto().
LATENCY = Objective("kernel_ns", "min", lambda r: r.kernel_ns)
CORES = Objective(
    "cores", "min", lambda r: r.config["grid_shape"][0] * r.config["grid_shape"][1]
)
PCC = Objective("pcc", "max", lambda r: r.pcc)
DEFAULT_OBJECTIVES = (LATENCY, CORES)


@dataclass
class TuneResult:
    objective: str
    records: list[TuneRecord]

    def leaderboard(self) -> list[TuneRecord]:
        """Valid records sorted by the objective (ascending = faster first)."""
        ranked = [r for r in self.records if r.valid and r.kernel_ns is not None]
        return sorted(ranked, key=lambda r: r.kernel_ns)

    @property
    def best(self) -> Optional[TuneRecord]:
        lb = self.leaderboard()
        return lb[0] if lb else None

    def pareto(
        self, objectives: Sequence[Objective] = DEFAULT_OBJECTIVES
    ) -> list[TuneRecord]:
        """Non-dominated (Pareto-optimal) valid records over N objectives.

        Record A dominates B when A is no worse than B on every objective and
        strictly better on at least one. The front is the set dominated by no
        other record, sorted by the first objective. Generalises to any number
        of axes -- pass e.g. ``(LATENCY, CORES, PCC)`` for a 3-way front that
        keeps the accuracy/speed/resource trade-off visible.
        """
        pts = [
            r
            for r in self.records
            if r.valid and all(o.extract(r) is not None for o in objectives)
        ]

        def dominates(a: TuneRecord, b: TuneRecord) -> bool:
            no_worse = all(o.at_least(o.extract(a), o.extract(b)) for o in objectives)
            strictly = any(o.better(o.extract(a), o.extract(b)) for o in objectives)
            return no_worse and strictly

        front = [r for r in pts if not any(dominates(o, r) for o in pts if o is not r)]
        o0 = objectives[0]
        return sorted(front, key=lambda r: o0.extract(r), reverse=o0.direction == "max")

    def to_dict(self, objectives: Sequence[Objective] = DEFAULT_OBJECTIVES) -> dict:
        def point(r: TuneRecord) -> dict:
            return {o.name: o.extract(r) for o in objectives}

        return {
            "objective": self.objective,
            "best": dataclasses.asdict(self.best) if self.best else None,
            "leaderboard": [dataclasses.asdict(r) for r in self.leaderboard()],
            "pareto": {
                "objectives": [
                    {"name": o.name, "direction": o.direction} for o in objectives
                ],
                "front": [
                    {**dataclasses.asdict(r), "point": point(r)}
                    for r in self.pareto(objectives)
                ],
            },
            "records": [dataclasses.asdict(r) for r in self.records],
        }

    def write_json(self, path) -> None:
        pathlib.Path(path).write_text(json.dumps(self.to_dict(), indent=2))


# --- Evaluation core (shared by exhaustive `tune` and guided `search`) -------


def _mean(by_core: dict):
    vals = [v for v in by_core.values() if v is not None]
    return sum(vals) / len(vals) if vals else None


def _diagnostics(rep: dict) -> dict:
    """The actionable scalars from a build_report -- what tells compute-bound
    from wait-bound from memory-bound, for reading why a config is slow (and for
    a guided proposer to steer on)."""
    stats = rep.get("stats", {})
    longest = rep["waits"]["longest"]
    return {
        "device_wall_ns": rep["metadata"]["device_wall_time"]["ns"],
        "wait_share_of_envelope": (
            longest["share_of_kernel_envelope"] if longest else None
        ),
        "mean_sfpu_util": _mean(stats.get("sfpu utilization", {})),
        "mean_fpu_util": _mean(stats.get("fpu utilization", {})),
        "mean_noc_vs_compute": _mean(stats.get("noc vs compute", {})),
    }


@dataclass
class _EvalContext:
    kernel: object
    run: Callable
    canonical: list
    ref: object
    pa: object
    csv: pathlib.Path
    out_root: pathlib.Path
    check_pcc: bool
    pcc: float
    save_profiler_logs: bool


def _setup(
    kernel,
    input_shapes,
    run,
    golden,
    check_pcc,
    pcc,
    out_dir,
    seed,
    save_profiler_logs=False,
) -> _EvalContext:
    """Shared one-time setup: profiler env/flags, perf-analyzer, canonical
    inputs + reference. Used by both `tune` and `search`."""
    # Profiler env must be present before device-open; flip the matching flags.
    os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
    os.environ.setdefault("TT_METAL_DEVICE_PROFILER_DISPATCH", "0")
    config.enable_perf_trace = True
    config.insert_profiler_traces = True

    if check_pcc and golden is None:
        raise ValueError("check_pcc=True requires a golden reference callable")

    out_root = pathlib.Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    # Canonical f32 inputs are fixed across the sweep. When the PCC gate is on,
    # the reference is computed once from them; each config runs on the SAME
    # values cast to its dtype, so the gate measures that config's numerical
    # error against one f32 golden. With the gate off (the default), the golden
    # is skipped entirely.
    canonical = make_inputs(input_shapes, torch.float32, _Seeded(seed))
    ref = (
        golden(*[t.float() for t in canonical])
        if (check_pcc and golden is not None)
        else None
    )
    return _EvalContext(
        kernel,
        run,
        canonical,
        ref,
        _load_perf_analyzer(),
        _device_csv(),
        out_root,
        check_pcc,
        pcc,
        save_profiler_logs,
    )


def _cfg_dirname(idx: int, cfg: dict) -> str:
    """Self-describing per-config dir name, e.g. ``cfg_000_g2x2_b1x1_l1``. The
    numeric prefix preserves evaluation order (meaningful in guided mode); the
    rest names the config so a saved profiler CSV is identifiable without
    cross-referencing results.json."""
    g = "x".join(map(str, cfg["grid_shape"]))
    b = "x".join(map(str, cfg["block_shape"]))
    mem = cfg.get("mem_space", "l1")
    return f"cfg_{idx:03d}_g{g}_b{b}_{mem}"


def _evaluate(cfg: dict, idx: int, ctx: _EvalContext) -> TuneRecord:
    """Run one config: cast inputs -> profiled device run -> report -> PCC gate.

    Never raises: a compile/run failure becomes a ``valid=False`` record (signal),
    so a sweep or search keeps going. Resets the builder afterwards so a failed
    config can't leak MLIR state into the next."""
    # Build output (the profiler #pragma spam from JIT subprocesses) is always
    # discarded to /dev/null. When `save_profiler_logs` is set, each config's
    # device profiler CSV is snapshotted into cfg_NNN/; otherwise the report is
    # built straight from the live CSV and nothing per-config is kept.
    if ctx.save_profiler_logs:
        cfg_dir = ctx.out_root / _cfg_dirname(idx, cfg)
        cfg_dir.mkdir(parents=True, exist_ok=True)

    # tt-metal's device profiler hashes inserted-zone source locations and aborts
    # the *process* (a C++ terminate Python can't catch) when they collide -- which
    # the extra DRAM<->L1 movement ops on the DRAM-I/O path trigger
    # (profiler.cpp "Source location hashes are colliding"). For DRAM configs,
    # drop the inserted fine-grained traces: the objective (kernel_ns) still comes
    # from the always-on kernel zones; only the per-op diagnostics (wait_share,
    # util) go dark. Restored in `finally` so L1 configs keep full diagnostics.
    fine_grained = cfg.get("mem_space", "l1") != "dram"
    prev_insert = config.insert_profiler_traces
    config.insert_profiler_traces = prev_insert and fine_grained
    try:
        cfg_inputs = [t.to(torch_dtype("float32")) for t in ctx.canonical]
        if ctx.csv.exists():
            ctx.csv.unlink()
        with _silence_build_output(os.devnull):
            actual = ctx.run(ctx.kernel, cfg_inputs, cfg)
        if not ctx.csv.exists():
            raise RuntimeError("no profiler CSV produced for this config")
        if ctx.save_profiler_logs:
            snapshot = cfg_dir / "profile_log_device.csv"
            shutil.copy(ctx.csv, snapshot)
        else:
            snapshot = ctx.csv
        rep = ctx.pa.build_report(snapshot)
        cfg_pcc = None
        valid = True
        if ctx.check_pcc:
            cfg_pcc = torch.corrcoef(
                torch.stack([ctx.ref.flatten(), actual.float().flatten()])
            )[0, 1].item()
            valid = cfg_pcc >= ctx.pcc
        diagnostics = _diagnostics(rep)
        diagnostics["fine_grained_traces"] = fine_grained
        return TuneRecord(
            config=cfg,
            valid=valid,
            pcc=cfg_pcc,
            kernel_ns=rep["runtimes"]["device_kernel_time"]["ns"],
            diagnostics=diagnostics,
            report_path=str(snapshot) if ctx.save_profiler_logs else None,
        )
    except Exception as e:  # compile/run failure -> prune, keep as signal
        return TuneRecord(config=cfg, valid=False, error=f"{type(e).__name__}: {e}")
    finally:
        config.insert_profiler_traces = prev_insert
        from d2m_jit._src.builder import _Builder

        _Builder.reset()


# --- Exhaustive sweep --------------------------------------------------------


def tune(
    bench: KernelBench,
    input_shapes: Sequence[tuple],
    space: Sequence[dict],
    *,
    check_pcc: bool = False,
    pcc: float = 0.99,
    objective: str = "device_kernel_time",
    out_dir: str = "autotune-artifacts",
    seed: int = 0,
    save_profiler_logs: bool = False,
) -> TuneResult:
    """Evaluate every config in ``space`` and rank by device kernel time.

    Parameters
    ----------
    bench             : KernelBench with kernel, golden, run (materializer), and input_shapes.
    input_shapes      : one shape per kernel input (fixed across the sweep).
    space             : list of config dicts (see ``grid_space``).
    check_pcc         : gate each config on PCC (default OFF; opt in to verify
                        correctness).
    pcc               : PCC threshold for a config to count as ``valid``.
    out_dir           : output root for ``results.json`` + ``summary.txt``.
    save_profiler_logs: also keep each config's device profiler CSV under
                        ``cfg_NNN/`` (off by default; build output is never kept).
    """
    ctx = _setup(
        bench.kernel,
        input_shapes,
        bench.run,
        bench.golden,
        check_pcc,
        pcc,
        out_dir,
        seed,
        save_profiler_logs,
    )
    records = [_evaluate(cfg, i, ctx) for i, cfg in enumerate(space)]
    result = TuneResult(objective=objective, records=records)
    result.write_json(ctx.out_root / "results.json")
    write_summary(result, ctx.out_root / "summary.txt")
    return result


# --- Guided search -----------------------------------------------------------
#
# Instead of evaluating the whole space, `search` runs propose -> evaluate ->
# observe rounds: a proposer chooses the next configs from the results observed
# so far; returning [] stops the loop. The bundled proposer is the heuristic
# `hill_climb_proposer` (latency-guided coordinate descent) -- it lets you
# explore a large space without running every config. The driver dedups
# re-proposed configs and enforces a `max_evals` budget.
#
# Proposer protocol:
#     proposer(history: list[TuneRecord], meta: dict) -> list[config dict]
#   history : every record so far, in evaluation order (valid + invalid).
#   meta    : {"round", "evaluated", "input_shapes", "objective"}.
#   returns : next configs to evaluate; [] to stop.


def config_key(cfg: dict):
    """Hashable identity of a config (for cross-round dedup)."""
    return (
        tuple(cfg["grid_shape"]),
        tuple(cfg["block_shape"]),
        cfg.get("mem_space", "l1"),
    )


def search(
    bench: KernelBench,
    input_shapes: Sequence[tuple],
    proposer: Callable,
    *,
    check_pcc: bool = False,
    pcc: float = 0.99,
    objective: str = "device_kernel_time",
    max_evals: Optional[int] = None,
    out_dir: str = "autotune-artifacts",
    seed: int = 0,
    on_round: Optional[Callable] = None,
    save_profiler_logs: bool = False,
) -> TuneResult:
    """Guided search: drive ``proposer`` in propose/evaluate/observe rounds.

    Stops when the proposer returns ``[]`` or ``max_evals`` device evaluations
    have run. Already-evaluated configs are skipped (dedup), so a proposer may
    re-propose freely. ``on_round(round_idx, history)`` is an optional progress
    hook. Returns a ``TuneResult`` over the evaluated configs (so leaderboard /
    pareto / JSON all work the same as for an exhaustive ``tune``)."""
    ctx = _setup(
        bench.kernel,
        input_shapes,
        bench.run,
        bench.golden,
        check_pcc,
        pcc,
        out_dir,
        seed,
        save_profiler_logs,
    )
    history: list[TuneRecord] = []
    seen: set = set()
    round_i = 0
    while max_evals is None or len(history) < max_evals:
        meta = {
            "round": round_i,
            "evaluated": len(history),
            "input_shapes": [list(s) for s in input_shapes],
            "objective": objective,
        }
        proposed = proposer(history, meta)
        if not proposed:
            break
        for cfg in proposed:
            if max_evals is not None and len(history) >= max_evals:
                break
            key = config_key(cfg)
            if key in seen:  # robust to a proposer re-proposing a config
                continue
            seen.add(key)
            history.append(_evaluate(cfg, len(history), ctx))
        round_i += 1
        if on_round is not None:
            on_round(round_i, history)

    result = TuneResult(objective=objective, records=history)
    result.write_json(ctx.out_root / "results.json")
    write_summary(result, ctx.out_root / "summary.txt")
    return result


# --- Heuristic proposer: latency hill-climb over (grid, block, mem_space) ----


def knob_neighbors(cfg: dict, candidates: Sequence[dict]) -> list[dict]:
    """Coordinate-step neighbors of ``cfg`` (each differs in ONE knob):

      * grid     : same block & mem, DOUBLE the core count (every aspect),
      * block    : same grid & mem, block tiles doubled OR halved,
      * mem_space: same grid & block, the other mem space.

    Grid is forward-only (more cores trends faster); block is bidirectional
    (the sweet spot for DMA granularity is not monotone). All steps are taken
    from ``candidates``, so only feasible neighbors appear."""
    cores = cfg["grid_shape"][0] * cfg["grid_shape"][1]
    btiles = cfg["block_shape"][0] * cfg["block_shape"][1]
    mem = cfg.get("mem_space", "l1")
    out = []
    for c in candidates:
        cc = c["grid_shape"][0] * c["grid_shape"][1]
        cb = c["block_shape"][0] * c["block_shape"][1]
        same_grid = c["grid_shape"] == cfg["grid_shape"]
        same_block = c["block_shape"] == cfg["block_shape"]
        same_mem = c.get("mem_space", "l1") == mem
        if same_block and same_mem and not same_grid and cc == cores * 2:
            out.append(c)  # grid step: more parallelism
        elif (
            same_grid
            and same_mem
            and not same_block
            and cb in (btiles * 2, max(btiles // 2, 1))
        ):
            out.append(c)  # block step: coarser/finer DMA granularity
        elif same_grid and same_block and not same_mem:
            out.append(c)  # mem_space flip: L1 <-> DRAM I/O
    return out


def hill_climb_proposer(
    candidates: Sequence[dict],
    *,
    objective: Objective = LATENCY,
    seed: Optional[dict] = None,
    neighbors: Callable = knob_neighbors,
) -> Callable:
    """A latency-guided hill climb as a proposer.

    From a seed config, each round proposes the current best's unevaluated
    neighbors; the loop ends when none remain (a local optimum). Stateless --
    it reconstructs its position from ``history`` every round."""
    cands = list(candidates)

    def propose(history, meta):
        evaluated = {config_key(r.config) for r in history}
        if not history:
            return [seed or cands[0]]
        valid = [r for r in history if r.valid and objective.extract(r) is not None]
        if not valid:  # nothing valid yet -> probe the next untried candidate
            return [c for c in cands if config_key(c) not in evaluated][:1]
        pick = min if objective.direction == "min" else max
        best = pick(valid, key=lambda r: objective.extract(r))
        return [
            c for c in neighbors(best.config, cands) if config_key(c) not in evaluated
        ]

    return propose


class _Seeded:
    """Minimal InputSpec-like shim (dist + seed) for make_inputs."""

    def __init__(self, seed: int):
        self.dist = "uniform(-1,1)"
        self.seed = seed


# --- Pareto view -------------------------------------------------------------


def _ascii_scatter(points, front, y_obj, x_obj, width=44, height=12) -> str:
    """Compact log-log scatter: ``#`` = on the front, ``.`` = dominated.

    y_obj on the vertical axis (higher at top), x_obj on the horizontal.
    """
    import math

    xs = [x_obj.extract(r) for r in points]
    ys = [y_obj.extract(r) for r in points]
    xlo, xhi, ylo, yhi = min(xs), max(xs), min(ys), max(ys)

    def pos(v, lo, hi, n):
        if hi <= lo:
            return 0
        frac = (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))
        return max(0, min(n - 1, round(frac * (n - 1))))

    grid = [[" "] * width for _ in range(height)]
    front_ids = {id(r) for r in front}
    for r in points:
        cx = pos(x_obj.extract(r), xlo, xhi, width)
        cy = height - 1 - pos(y_obj.extract(r), ylo, yhi, height)
        mark = "#" if id(r) in front_ids else "."
        if grid[cy][cx] != "#":  # never let a dominated dot hide a front point
            grid[cy][cx] = mark

    out = [f"  {y_obj.name} (log) ^  hi={yhi:g}"]
    out += ["  |" + "".join(row) for row in grid]
    out.append("  +" + "-" * width + f"> {x_obj.name} (log) {xlo:g}..{xhi:g}")
    out.append("  (# = Pareto front, . = dominated)")
    return "\n".join(out)


def format_pareto(result, objectives=DEFAULT_OBJECTIVES, label=str) -> str:
    """Render the Pareto front: a table of front members (one column per
    objective) plus, for exactly two objectives, a log-log scatter."""
    front = result.pareto(objectives)
    n_valid = len([r for r in result.records if r.valid])
    names = ", ".join(f"{o.name}({o.direction})" for o in objectives)
    lines = [
        f"PARETO FRONT over [{names}] -- {len(front)} non-dominated of "
        f"{n_valid} valid configs",
        "",
    ]
    hdr = f"{'config':>16}  " + "  ".join(f"{o.name:>12}" for o in objectives)
    lines += [hdr, "-" * len(hdr)]
    for r in front:
        cells = "  ".join(f"{o.extract(r):>12.5g}" for o in objectives)
        lines.append(f"{label(r.config):>16}  {cells}")
    if len(objectives) == 2:
        pts = [
            r
            for r in result.records
            if r.valid and all(o.extract(r) is not None for o in objectives)
        ]
        lines += ["", _ascii_scatter(pts, front, objectives[0], objectives[1])]
    return "\n".join(lines)


def write_summary(result, path) -> None:
    """Write a concise human-readable digest (``summary.txt``) next to
    ``results.json``: counts, best config, the top of the leaderboard, any
    invalid configs with reasons, and the latency-vs-cores Pareto front."""
    valid = result.leaderboard()
    n_invalid = len(result.records) - len(valid)
    lines = [
        f"objective: {result.objective} (lower = faster)",
        f"configs:   {len(result.records)} evaluated, {len(valid)} valid, "
        f"{n_invalid} invalid",
    ]
    if result.best:
        b = result.best
        pcc = f"  (pcc {b.pcc:.5f})" if b.pcc is not None else ""
        lines.append(f"best:      {_label(b.config)} @ {b.kernel_ns:.1f} ns{pcc}")

    lines += ["", "leaderboard (top 10):"]
    for rank, r in enumerate(valid[:10]):
        pcc = f"  pcc {r.pcc:.5f}" if r.pcc is not None else ""
        wait = r.diagnostics.get("wait_share_of_envelope")
        wtxt = f"  wait {wait:.0%}" if wait is not None else ""
        lines.append(
            f"  {rank:>3}. {_label(r.config):>22}  {r.kernel_ns:>12.1f} ns{pcc}{wtxt}"
        )

    if n_invalid:
        lines += ["", f"invalid ({n_invalid}):"]
        for r in result.records:
            if not r.valid:
                why = r.error or (
                    f"pcc {r.pcc:.5f} < threshold" if r.pcc is not None else "invalid"
                )
                lines.append(f"  {_label(r.config):>22}  {why}")

    if valid:
        lines += ["", format_pareto(result, (LATENCY, CORES), label=_label)]
    pathlib.Path(path).write_text("\n".join(lines) + "\n")


# --- Pattern file loader ---------------------------------------------------


def _load_pattern_file(pattern_path: str) -> KernelBench:
    """Load a pattern file and extract the first KernelBench.

    Parameters
    ----------
    pattern_path : file path to pattern file (e.g., tools/d2m-jit/patterns/eltwise_exp_to_kernel.py)

    Returns
    -------
    KernelBench with kernel, golden, run (materializer), input_shapes, etc.
    """
    pattern_file = pathlib.Path(pattern_path)
    if not pattern_file.exists():
        raise FileNotFoundError(f"pattern file not found: {pattern_path}")

    spec = importlib.util.spec_from_file_location(
        f"pattern_{pattern_file.stem}", pattern_file
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load pattern file: {pattern_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "KERNEL_BENCHES"):
        raise ValueError(
            f"pattern file {pattern_path} does not export KERNEL_BENCHES list"
        )

    benches = mod.KERNEL_BENCHES
    if not benches:
        raise ValueError(f"pattern file {pattern_path} has empty KERNEL_BENCHES list")

    return benches[0]


def _label(cfg):
    g = "x".join(map(str, cfg["grid_shape"]))
    b = "x".join(map(str, cfg["block_shape"]))
    mem = cfg.get("mem_space", "l1")
    return f"{g}|{b}@{mem}"  # grid|block@mem


def _seed_config(space: Sequence[dict]) -> dict:
    """Cheapest pool member (fewest cores, then smallest block) -- the natural
    start for a forward hill climb."""
    return min(
        space,
        key=lambda c: (
            c["grid_shape"][0] * c["grid_shape"][1],
            c["block_shape"][0] * c["block_shape"][1],
        ),
    )


def _print_results(result, mode: str, n_candidates: Optional[int] = None) -> None:
    """Shared reporting for both modes: ranking/eval-path + best + Pareto views."""
    print(f"\nobjective: {result.objective} (lower = faster)\n")
    if mode == "guided":
        print("evaluation path (in order proposed):")
        for i, r in enumerate(result.records):
            if r.valid:
                wait = r.diagnostics.get("wait_share_of_envelope")
                wtxt = f"  wait={wait:.0%}" if wait is not None else ""
                lat = f"{r.kernel_ns:>10.0f} ns"
                print(f"  {i:>2}. {_label(r.config):>22}  {lat}{wtxt}")
            else:
                print(f"  {i:>2}. {_label(r.config):>22}  invalid: {r.error}")
        if n_candidates:
            pct = 100 * len(result.records) / n_candidates
            print(
                f"\nevaluated {len(result.records)} of {n_candidates} configs "
                f"({pct:.0f}% of the space)"
            )
    else:
        hdr = (
            f"{'rank':>4}  {'grid|block@mem':>18}  "
            f"{'kernel_ns':>12}  {'pcc':>9}  {'wait%':>6}"
        )
        print(hdr)
        print("-" * len(hdr))
        for rank, r in enumerate(result.leaderboard()[:12]):
            lat = f"{r.kernel_ns:>12.1f}"
            p = f"{r.pcc:>9.5f}" if r.pcc is not None else f"{'-':>9}"
            wait = r.diagnostics.get("wait_share_of_envelope")
            w = f"{wait:.0%}" if wait is not None else "-"
            print(f"{rank:>4}  {_label(r.config):>22}  {lat}  {p}  {w:>6}")

    if result.best:
        print(f"\nbest: {_label(result.best.config)} @ {result.best.kernel_ns:.1f} ns")
    # 2-way (latency vs cores) with a scatter, then the 3-way front that keeps
    # accuracy in play (bf16's speed vs f32's precision).
    print("\n" + "=" * 54)
    print(format_pareto(result, (LATENCY, CORES), label=_label))
    print("\n" + "=" * 54)
    print(format_pareto(result, (LATENCY, CORES, PCC), label=_label))


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI for autotuning d2m-jit kernels from pattern files.

    grid, block, and mem_space are the swept knobs: each defaults to all feasible
    values (omitted == all); pass ``all``, an explicit list, or a single value to
    pin. A bare run sweeps grid x block x mem_space (at float32, from the pattern).
    The resolved space feeds ``--mode sweep`` (every config) or ``--mode guided``
    (a hill-climb path); use ``--list`` to preview the space
    (and its size) before committing device time. The PCC correctness gate is
    off by default -- pass ``--check-pcc`` to verify each config against a golden."""
    import argparse

    p = argparse.ArgumentParser(
        prog="autotune",
        description="Autotune a d2m-jit kernel from a pattern file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  # sweep exp_fused pattern over all grid/block/mem configs:\n"
            "  python autotune.py --pattern tools/d2m-jit/patterns/eltwise_exp_to_kernel.py\n\n"
            "  # sweep add_exp pattern with correctness gate:\n"
            "  python autotune.py --pattern tools/d2m-jit/patterns/eltwise_add_exp_to_kernel.py --check-pcc\n\n"
            "  # guided search over l1 only, block pinned to 1x1:\n"
            "  python autotune.py --pattern tools/d2m-jit/patterns/eltwise_exp_to_kernel.py --block 1x1 --mem-space l1 --mode guided\n\n"
            "  # preview the config space without running on device:\n"
            "  python autotune.py --pattern tools/d2m-jit/patterns/eltwise_exp_to_kernel.py --grid 1x1,2x2,4x4 --block 2x2 --list\n"
        ),
    )
    p.add_argument(
        "--pattern",
        required=True,
        help="path to pattern file (e.g., tools/d2m-jit/patterns/eltwise_exp_to_kernel.py)",
    )
    p.add_argument(
        "--grid",
        default=None,
        help="grids: 'all' or a list like 1x1,2x2,4x4 (default: all feasible)",
    )
    p.add_argument(
        "--block",
        default=None,
        help="block_shapes in tiles: 'all' or a list like 1x1,2x2 (default: all feasible)",
    )
    p.add_argument(
        "--mem-space",
        default=None,
        help="kernel I/O placement: 'all' or a list like l1,dram (default: all feasible)",
    )
    p.add_argument(
        "--max-cores", type=int, default=64, help="cap on grid cores (default 64)"
    )
    p.add_argument(
        "--mode",
        choices=("sweep", "guided"),
        default="sweep",
        help="'sweep' runs every config; 'guided' hill-climbs a path through "
        "the space without running all of it (default sweep)",
    )
    p.add_argument(
        "--max-evals",
        type=int,
        default=None,
        help="guided mode: cap on device evaluations (budget)",
    )
    p.add_argument(
        "--list",
        action="store_true",
        dest="list_only",
        help="print the resolved config space and exit (no device)",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="output dir (default autotune-artifacts/<pattern_name>)",
    )
    p.add_argument("--seed", type=int, default=0, help="input RNG seed (default 0)")
    p.add_argument(
        "--check-pcc",
        action="store_true",
        help="gate each config on PCC vs a golden, dropping incorrect ones "
        "(off by default)",
    )
    p.add_argument(
        "--save-profiler-logs",
        action="store_true",
        help="keep each config's device profiler CSV under cfg_NNN/ (off by default)",
    )
    args = p.parse_args(argv)

    try:
        bench = _load_pattern_file(args.pattern)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        p.error(str(e))

    # Use pattern's input shapes and dtype (from KernelBench)
    input_shapes = bench.input_shapes
    shape = input_shapes[0]
    try:
        space, warnings = build_space(
            shape,
            grid=args.grid,
            block=args.block,
            mem_space=args.mem_space,
            max_cores=args.max_cores,
        )
    except ValueError as e:
        p.error(str(e))

    for w in warnings:
        print(f"warning: {w}")
    print(f"resolved space: {len(space)} config(s) for shape {shape[0]}x{shape[1]}")

    if args.list_only or not space:
        for cfg in space:
            print(f"  {_label(cfg)}")
        if not space:
            print("  (no feasible configs -- check --grid / --block)")
        return

    pattern_name = pathlib.Path(args.pattern).stem
    out_dir = args.out_dir or f"autotune-artifacts/{pattern_name}"
    if args.mode == "sweep":
        result = tune(
            bench,
            input_shapes,
            space,
            check_pcc=args.check_pcc,
            out_dir=out_dir,
            seed=args.seed,
            save_profiler_logs=args.save_profiler_logs,
        )
        _print_results(result, "sweep")
    else:
        proposer = hill_climb_proposer(space, seed=_seed_config(space))
        result = search(
            bench,
            input_shapes,
            proposer,
            check_pcc=args.check_pcc,
            max_evals=args.max_evals,
            out_dir=out_dir,
            seed=args.seed,
            save_profiler_logs=args.save_profiler_logs,
        )
        _print_results(result, "guided", n_candidates=len(space))
    print(f"\nresults.json + summary.txt written under {out_dir}/")


if __name__ == "__main__":
    main()
