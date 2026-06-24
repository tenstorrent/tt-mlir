# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Direct-kernel autotuner (Path A) for d2m-jit.

Sweeps a config space over a ``@d2m.kernel`` driven directly (the
``eltwise_block_run`` convention), runs each config on device with profiler
traces, and ranks by ``device_kernel_time.ns`` (parsed from the device profiler
CSV via ``tools/perf-analyzer/perf-analyzer.py::build_report``).

Each config is correctness-gated by PCC. The gate is cheap: the reference is
computed once (the inputs are fixed across the sweep), and the device output is
already returned by the same profiled run -- so gating is one ``corrcoef`` per
config. A config that fails to compile/run, or whose PCC is below threshold,
stays in the results with ``valid=False`` (useful signal) but is excluded from
the leaderboard and from ``best``.

Capture mechanics (validated): the in-process device profiler writes
``$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv`` after each
workload. The sweep clears that file before each config and snapshots it into a
per-config directory afterwards, so each report holds exactly one config's zones.

Requirements (the profiler env is read by tt-metal at device-open, so it must be
set before the first device interaction -- run the tuner in a fresh process):
    TT_METAL_DEVICE_PROFILER=1
    TT_METAL_DEVICE_PROFILER_DISPATCH=0
and a perf-trace runtime build (TT_RUNTIME_ENABLE_PERF_TRACE=ON). ``tune()`` sets
the two env vars via ``setdefault`` and flips the matching ``config`` flags.

MVP scope: the swept knob is ``grid_shape`` (block_shape/dtype held fixed). The
config is a plain dict so adding axes (block_shape, dtype, mem_space, ...) later
is additive -- see ``grid_space`` for the enumeration/feasibility pattern.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import pathlib
import shutil
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch

from d2m_jit import config
from d2m_jit.testing import eltwise_block_run, make_inputs, torch_dtype

# --- perf-analyzer (hyphenated filename / sibling tool -> load by path) ------


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


def grid_space(
    shape, dtypes: Sequence[str] = ("float32",), max_cores: int = 64
) -> list[dict]:
    """Enumerate feasible ``(grid_shape, dtype)`` configs for a 2D ``shape``.

    A grid ``(gy, gx)`` is feasible when it evenly divides the tile counts
    ``(shape[-2]//32, shape[-1]//32)`` and ``gy*gx <= max_cores`` (physical grid
    bound). The space is the product of feasible grids and ``dtypes``.

    block_shape is held at ``[1, 1]``: the eltwise kernels load single tiles, so
    a larger block would change the device sharding without the kernel reading
    it. block_shape becomes a real knob only with a block-aware kernel that
    loads multi-tile shards.
    """
    ty, tx = shape[-2] // 32, shape[-1] // 32
    cfgs = []
    for gy in range(1, ty + 1):
        if ty % gy:
            continue
        for gx in range(1, tx + 1):
            if tx % gx or gy * gx > max_cores:
                continue
            for dt in dtypes:
                cfgs.append(
                    {"grid_shape": [gy, gx], "block_shape": [1, 1], "dtype": dt}
                )
    return cfgs


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


# --- Sweep -------------------------------------------------------------------


def tune(
    kernel,
    input_shapes: Sequence[tuple],
    space: Sequence[dict],
    *,
    golden: Optional[Callable] = None,
    run: Callable = eltwise_block_run,
    check_pcc: bool = True,
    pcc: float = 0.99,
    objective: str = "device_kernel_time",
    out_dir: str = "prof_tune",
    seed: int = 0,
) -> TuneResult:
    """Sweep ``space`` over ``kernel`` and rank by device kernel time.

    Parameters
    ----------
    kernel        : the ``@d2m.kernel`` to tune.
    input_shapes  : one shape per kernel input (fixed across the sweep).
    space         : list of config dicts (see ``grid_space``).
    golden        : optional ``(*inputs) -> tensor`` reference for the PCC gate.
                    Computed once. Required when ``check_pcc`` is True.
    run           : ``(kernel, inputs, cfg) -> host tensor`` materializer
                    (default: the eltwise-block convention).
    check_pcc     : gate each config on PCC (default on; near-free, see module doc).
    pcc           : PCC threshold for a config to count as ``valid``.
    out_dir       : root for per-config profiler snapshots + the leaderboard JSON.
    """
    # Profiler env must be present before device-open; flip the matching flags.
    os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
    os.environ.setdefault("TT_METAL_DEVICE_PROFILER_DISPATCH", "0")
    config.enable_perf_trace = True
    config.insert_profiler_traces = True

    if check_pcc and golden is None:
        raise ValueError("check_pcc=True requires a golden reference callable")

    pa = _load_perf_analyzer()
    csv = _device_csv()
    out_root = pathlib.Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Canonical f32 inputs are fixed across the sweep; the reference is computed
    # once from them. Each config runs on the SAME values cast to its dtype, so
    # the PCC gate measures that config's numerical error against one f32 golden
    # (this is what makes dtype a meaningful, gated knob).
    canonical = make_inputs(input_shapes, torch.float32, _Seeded(seed))
    ref = golden(*[t.float() for t in canonical]) if golden is not None else None

    records: list[TuneRecord] = []
    for i, cfg in enumerate(space):
        cfg_dir = out_root / f"cfg_{i:03d}"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        try:
            cfg_inputs = [
                t.to(torch_dtype(cfg.get("dtype", "float32"))) for t in canonical
            ]
            if csv.exists():
                csv.unlink()
            actual = run(kernel, cfg_inputs, cfg)
            if not csv.exists():
                raise RuntimeError("no profiler CSV produced for this config")
            snapshot = cfg_dir / "profile_log_device.csv"
            shutil.copy(csv, snapshot)
            rep = pa.build_report(snapshot)
            kernel_ns = rep["runtimes"]["device_kernel_time"]["ns"]
            longest = rep["waits"]["longest"]
            diagnostics = {
                "device_wall_ns": rep["metadata"]["device_wall_time"]["ns"],
                "longest_wait_share_of_envelope": (
                    longest["share_of_kernel_envelope"] if longest else None
                ),
            }
            cfg_pcc = None
            valid = True
            if check_pcc:
                cfg_pcc = torch.corrcoef(
                    torch.stack([ref.flatten(), actual.float().flatten()])
                )[0, 1].item()
                valid = cfg_pcc >= pcc
            records.append(
                TuneRecord(
                    config=cfg,
                    valid=valid,
                    pcc=cfg_pcc,
                    kernel_ns=kernel_ns,
                    diagnostics=diagnostics,
                    report_path=str(snapshot),
                )
            )
        except Exception as e:  # compile/run failure -> prune, keep as signal
            records.append(
                TuneRecord(config=cfg, valid=False, error=f"{type(e).__name__}: {e}")
            )
        finally:
            # Drop the per-builder state so a failed config can't leak into the next.
            from d2m_jit._src.builder import _Builder

            _Builder.reset()

    result = TuneResult(objective=objective, records=records)
    result.write_json(out_root / "leaderboard.json")
    return result


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


# --- Demo --------------------------------------------------------------------


def _demo() -> None:
    """Tune `exp_fused` over the feasible grids for a 256x256 input."""
    import d2m_jit as d2m

    @d2m.kernel
    def exp_fused(in_t, out_t, m_blocks, n_blocks):
        m_off = core_index(0) * m_blocks  # noqa: F821
        n_off = core_index(1) * n_blocks  # noqa: F821
        for m in range(m_blocks):
            for n in range(n_blocks):
                shard = remote_load(in_t, [m_off + m, n_off + n])  # noqa: F821
                remote_store(out_t, [m_off + m, n_off + n], shard.exp())  # noqa: F821

    shape = (256, 256)
    space = grid_space(shape, dtypes=("float32", "bfloat16"))
    print(f"tuning exp_fused over {len(space)} (grid x dtype) configs for {shape} ...")
    result = tune(
        exp_fused,
        input_shapes=[shape],
        space=space,
        golden=torch.exp,
        out_dir="prof_tune_exp",
    )

    def label(cfg):
        return f'{"x".join(map(str, cfg["grid_shape"]))}/{cfg["dtype"]}'

    print(f"\nobjective: {result.objective} (lower = faster)\n")
    print(f"{'rank':>4}  {'grid/dtype':>16}  {'kernel_ns':>12}  {'pcc':>9}")
    print("-" * 48)
    for rank, r in enumerate(result.leaderboard()):
        print(f"{rank:>4}  {label(r.config):>16}  {r.kernel_ns:>12.1f}  {r.pcc:>9.5f}")
    invalid = [r for r in result.records if not r.valid]
    if invalid:
        print(f"\n{len(invalid)} invalid config(s):")
        for r in invalid:
            why = r.error or f"pcc {r.pcc:.5f} < threshold"
            print(f"  {label(r.config)}: {why}")
    if result.best:
        print(f"\nbest: {label(result.best.config)} @ {result.best.kernel_ns:.1f} ns")

    # Pareto views: 2-way (latency vs cores) with a scatter, then the 3-way
    # front that keeps accuracy in play (where bf16's speed trades against
    # f32's precision instead of one strictly dominating).
    print("\n" + "=" * 48)
    print(format_pareto(result, (LATENCY, CORES), label=label))
    print("\n" + "=" * 48)
    print(format_pareto(result, (LATENCY, CORES, PCC), label=label))
    print("\nleaderboard.json (with pareto front) written under prof_tune_exp/")


if __name__ == "__main__":
    _demo()
