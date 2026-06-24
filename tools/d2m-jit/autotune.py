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

Knobs: ``grid_shape`` (cores), ``block_shape`` (tiles per shard a core processes
per remote_load), and ``dtype``. The config is a plain dict, so further axes
(mem_space, ...) are additive. See ``grid_space`` / ``block_space`` to enumerate.

Two entry points share the same per-config evaluation:
  * ``tune(space)``   -- evaluate every config, rank + Pareto over the results.
  * ``search(proposer)`` -- guided: a proposer chooses the next configs from what
    it has observed, in propose/evaluate/observe rounds. The proposer is the
    optimizer: ``hill_climb_proposer`` (autonomous heuristic) or
    ``make_agent_proposer`` (an LLM reasons over the JSON diagnostics history).
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
from d2m_jit.testing import d2m_dtype, make_inputs, torch_dtype

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

    block_shape is held at ``[1, 1]`` here (one tile per shard). To also tune
    block_shape, use ``block_space`` with the ``block_aware_run`` materializer.
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


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def block_space(
    shape, dtypes: Sequence[str] = ("float32",), max_cores: int = 64
) -> list[dict]:
    """Enumerate feasible ``(grid_shape, block_shape, dtype)`` configs.

    ``block_shape`` ``(bh, bw)`` (in tiles) is the multi-tile shard each
    ``remote_load`` pulls and the kernel computes over. Feasibility:
      * ``bh | tiles_y`` and ``bw | tiles_x`` (block tiles the tensor),
      * the blocked grid ``(tiles_y//bh, tiles_x//bw)`` is divisible by
        ``grid_shape`` (blocks distribute evenly across cores),
      * ``gy*gx <= max_cores``.
    The space is the product over feasible blocks, grids, and dtypes -- so
    block_shape is a genuine third knob alongside grid_shape and dtype.
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
                    for dt in dtypes:
                        cfgs.append(
                            {
                                "grid_shape": [gy, gx],
                                "block_shape": [bh, bw],
                                "dtype": dt,
                            }
                        )
    return cfgs


# --- Materializer ------------------------------------------------------------


def block_aware_run(kernel, inputs, cfg):
    """``(kernel, inputs, cfg) -> host tensor`` honouring ``block_shape``.

    Builds the Layout at the config's block_shape and computes the per-core
    block sweep as ``(tiles // block) // grid`` -- so each ``remote_load`` pulls
    a ``block_shape``-tile shard the kernel computes over. Generalises
    ``testing.eltwise_block_run`` (which assumes 1x1 blocks); identical when
    ``block_shape == [1, 1]``."""
    import d2m_jit as d2m

    ref = inputs[0]
    gy, gx = cfg["grid_shape"]
    bh, bw = cfg["block_shape"]
    layout = d2m.Layout(
        shape=tuple(ref.shape),
        dtype=d2m_dtype(cfg["dtype"]),
        block_shape=[bh, bw],
        grid_shape=[gy, gx],
        tiled=True,
    )
    ins = [d2m.to_layout(t, layout) for t in inputs]
    out = d2m.empty(layout)
    ty, tx = ref.shape[-2] // 32, ref.shape[-1] // 32
    m_blocks = (ty // bh) // gy
    n_blocks = (tx // bw) // gx
    kernel(*ins, out, m_blocks, n_blocks, grid=(gy, gx))
    return out.to_host()


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
    """The actionable scalars from a build_report -- the observation space a
    guided/agentic proposer reasons over to tell compute-bound from wait-bound
    from memory-bound."""
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


def _setup(
    kernel, input_shapes, run, golden, check_pcc, pcc, out_dir, seed
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
    # Canonical f32 inputs are fixed across the sweep; the reference is computed
    # once. Each config runs on the SAME values cast to its dtype, so the PCC
    # gate measures that config's numerical error against one f32 golden.
    canonical = make_inputs(input_shapes, torch.float32, _Seeded(seed))
    ref = golden(*[t.float() for t in canonical]) if golden is not None else None
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
    )


def _evaluate(cfg: dict, idx: int, ctx: _EvalContext) -> TuneRecord:
    """Run one config: cast inputs -> profiled device run -> report -> PCC gate.

    Never raises: a compile/run failure becomes a ``valid=False`` record (signal),
    so a sweep or search keeps going. Resets the builder afterwards so a failed
    config can't leak MLIR state into the next."""
    cfg_dir = ctx.out_root / f"cfg_{idx:03d}"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    try:
        cfg_inputs = [
            t.to(torch_dtype(cfg.get("dtype", "float32"))) for t in ctx.canonical
        ]
        if ctx.csv.exists():
            ctx.csv.unlink()
        actual = ctx.run(ctx.kernel, cfg_inputs, cfg)
        if not ctx.csv.exists():
            raise RuntimeError("no profiler CSV produced for this config")
        snapshot = cfg_dir / "profile_log_device.csv"
        shutil.copy(ctx.csv, snapshot)
        rep = ctx.pa.build_report(snapshot)
        cfg_pcc = None
        valid = True
        if ctx.check_pcc:
            cfg_pcc = torch.corrcoef(
                torch.stack([ctx.ref.flatten(), actual.float().flatten()])
            )[0, 1].item()
            valid = cfg_pcc >= ctx.pcc
        return TuneRecord(
            config=cfg,
            valid=valid,
            pcc=cfg_pcc,
            kernel_ns=rep["runtimes"]["device_kernel_time"]["ns"],
            diagnostics=_diagnostics(rep),
            report_path=str(snapshot),
        )
    except Exception as e:  # compile/run failure -> prune, keep as signal
        return TuneRecord(config=cfg, valid=False, error=f"{type(e).__name__}: {e}")
    finally:
        from d2m_jit._src.builder import _Builder

        _Builder.reset()


# --- Exhaustive sweep --------------------------------------------------------


def tune(
    kernel,
    input_shapes: Sequence[tuple],
    space: Sequence[dict],
    *,
    golden: Optional[Callable] = None,
    run: Callable = block_aware_run,
    check_pcc: bool = True,
    pcc: float = 0.99,
    objective: str = "device_kernel_time",
    out_dir: str = "prof_tune",
    seed: int = 0,
) -> TuneResult:
    """Evaluate every config in ``space`` and rank by device kernel time.

    Parameters
    ----------
    kernel        : the ``@d2m.kernel`` to tune.
    input_shapes  : one shape per kernel input (fixed across the sweep).
    space         : list of config dicts (see ``grid_space``).
    golden        : ``(*inputs) -> tensor`` reference for the PCC gate (required
                    when ``check_pcc``). Computed once.
    run           : ``(kernel, inputs, cfg) -> host tensor`` materializer.
    check_pcc     : gate each config on PCC (default on; near-free, see module doc).
    pcc           : PCC threshold for a config to count as ``valid``.
    out_dir       : root for per-config profiler snapshots + the leaderboard JSON.
    """
    ctx = _setup(kernel, input_shapes, run, golden, check_pcc, pcc, out_dir, seed)
    records = [_evaluate(cfg, i, ctx) for i, cfg in enumerate(space)]
    result = TuneResult(objective=objective, records=records)
    result.write_json(ctx.out_root / "leaderboard.json")
    return result


# --- Guided / agentic search -------------------------------------------------
#
# Instead of evaluating the whole space, `search` runs propose -> evaluate ->
# observe rounds: a proposer chooses the next configs from the results observed
# so far; returning [] stops the loop. The proposer IS the optimizer -- a
# heuristic (`hill_climb_proposer`) for autonomous runs, or an LLM agent
# (`make_agent_proposer`) that reasons over the JSON observation history and the
# per-config diagnostics. The driver dedups re-proposed configs and enforces a
# `max_evals` budget, so any proposer (including a fuzzy agent) is safe to drive.
#
# Proposer protocol:
#     proposer(history: list[TuneRecord], meta: dict) -> list[config dict]
#   history : every record so far, in evaluation order (valid + invalid).
#   meta    : {"round", "evaluated", "input_shapes", "objective"}.
#   returns : next configs to evaluate; [] to stop.


def config_key(cfg: dict):
    """Hashable identity of a config (for cross-round dedup)."""
    return (tuple(cfg["grid_shape"]), tuple(cfg["block_shape"]), cfg["dtype"])


def observations_json(history: Sequence[TuneRecord]) -> list[dict]:
    """The history as plain JSON dicts -- exactly what an agent proposer sees."""
    return [
        {
            "config": r.config,
            "valid": r.valid,
            "pcc": r.pcc,
            "kernel_ns": r.kernel_ns,
            "diagnostics": r.diagnostics,
            "error": r.error,
        }
        for r in history
    ]


def search(
    kernel,
    input_shapes: Sequence[tuple],
    proposer: Callable,
    *,
    golden: Optional[Callable] = None,
    run: Callable = block_aware_run,
    check_pcc: bool = True,
    pcc: float = 0.99,
    objective: str = "device_kernel_time",
    max_evals: Optional[int] = None,
    out_dir: str = "prof_search",
    seed: int = 0,
    on_round: Optional[Callable] = None,
) -> TuneResult:
    """Guided search: drive ``proposer`` in propose/evaluate/observe rounds.

    Stops when the proposer returns ``[]`` or ``max_evals`` device evaluations
    have run. Already-evaluated configs are skipped (dedup), so a proposer may
    re-propose freely. ``on_round(round_idx, history)`` is an optional progress
    hook. Returns a ``TuneResult`` over the evaluated configs (so leaderboard /
    pareto / JSON all work the same as for an exhaustive ``tune``)."""
    ctx = _setup(kernel, input_shapes, run, golden, check_pcc, pcc, out_dir, seed)
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
            if key in seen:  # robust to re-proposals (esp. from an agent)
                continue
            seen.add(key)
            history.append(_evaluate(cfg, len(history), ctx))
        round_i += 1
        if on_round is not None:
            on_round(round_i, history)

    result = TuneResult(objective=objective, records=history)
    result.write_json(ctx.out_root / "search.json")
    return result


# --- Heuristic proposer: latency hill-climb over (grid, block, dtype) --------


def knob_neighbors(cfg: dict, candidates: Sequence[dict]) -> list[dict]:
    """Coordinate-step neighbors of ``cfg`` (each differs in ONE knob):

      * grid  : same block & dtype, DOUBLE the core count (every aspect),
      * block : same grid & dtype, block tiles doubled OR halved (every aspect),
      * dtype : same grid & block, the other dtype.

    Grid is forward-only (more cores trends faster); block is bidirectional
    (the sweet spot for DMA granularity is not monotone). All steps are taken
    from ``candidates``, so only feasible neighbors appear."""
    cores = cfg["grid_shape"][0] * cfg["grid_shape"][1]
    btiles = cfg["block_shape"][0] * cfg["block_shape"][1]
    out = []
    for c in candidates:
        cc = c["grid_shape"][0] * c["grid_shape"][1]
        cb = c["block_shape"][0] * c["block_shape"][1]
        same_grid = c["grid_shape"] == cfg["grid_shape"]
        same_block = c["block_shape"] == cfg["block_shape"]
        same_dtype = c["dtype"] == cfg["dtype"]
        if same_block and same_dtype and not same_grid and cc == cores * 2:
            out.append(c)  # grid step: more parallelism
        elif (
            same_grid
            and same_dtype
            and not same_block
            and cb
            in (
                btiles * 2,
                max(btiles // 2, 1),
            )
        ):
            out.append(c)  # block step: coarser/finer DMA granularity
        elif same_grid and same_block and not same_dtype:
            out.append(c)  # dtype flip
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
    it reconstructs its position from ``history`` every round, so it drops into
    ``search`` exactly where an agent proposer would."""
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


# --- Agentic proposer: an LLM is the optimizer -------------------------------


def make_agent_proposer(
    complete: Callable[[str], str],
    candidates: Optional[Sequence[dict]] = None,
    batch: int = 4,
) -> Callable:
    """Adapt an LLM ``complete(prompt) -> str`` into a search proposer.

    Each round it serializes the observation history (and the optional candidate
    pool) into a prompt, asks the model for up to ``batch`` next configs as a
    JSON list (or ``STOP``), and parses them. The ``search`` driver dedups and
    budgets, so the agent may re-propose freely. ``complete`` is your own model
    call -- this module stays LLM-client-agnostic."""
    pool = list(candidates) if candidates is not None else None

    def propose(history, meta):
        prompt = _agent_prompt(observations_json(history), meta, pool, batch)
        reply = complete(prompt)
        if "{" not in reply and "STOP" in reply.upper():
            return []
        return _parse_configs(reply)

    return propose


def _agent_prompt(observations, meta, pool, batch) -> str:
    parts = [
        "You are autotuning a d2m-jit compute kernel on a Tenstorrent device. "
        f"Objective: MINIMIZE `{meta['objective']}` (device kernel time, ns).",
        'A config is JSON: {"grid_shape": [gy, gx], "block_shape": [bh, bw], '
        '"dtype": "float32" | "bfloat16"}. grid_shape = compute cores (more cores '
        "is usually faster, with diminishing returns); block_shape = tiles per "
        "shard each core processes per remote_load (larger = fewer, coarser DMA "
        "transfers); bfloat16 ~halves data movement vs float32. Read each config's "
        "`diagnostics` to decide the next move: high wait_share_of_envelope => "
        "stalled, add parallelism or coarsen the block; high mean_sfpu_util/"
        "mean_fpu_util => compute-bound (more cores won't help, try dtype); high "
        "mean_noc_vs_compute => memory-bound (coarsen block or try bfloat16). Pick "
        "only from the unevaluated feasible candidates listed below.",
        f"Observations so far ({len(observations)} configs):",
        json.dumps(observations, indent=2),
    ]
    if pool is not None:
        keys = {config_key(o["config"]) for o in observations}
        uneval = [c for c in pool if config_key(c) not in keys]
        parts.append(f"Unevaluated feasible candidates:\n{json.dumps(uneval)}")
    parts.append(
        f"Reply with up to {batch} next configs to evaluate as a JSON list, or "
        "the single word STOP if further search is unlikely to beat the best so "
        "far. Output JSON only."
    )
    return "\n\n".join(parts)


def _parse_configs(text: str) -> list[dict]:
    """Extract a JSON list of config dicts from an LLM reply (tolerant of
    surrounding prose / code fences)."""
    import re

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for c in data if isinstance(data, list) else []:
        if isinstance(c, dict) and "grid_shape" in c:
            c.setdefault("block_shape", [1, 1])
            c.setdefault("dtype", "float32")
            out.append(c)
    return out


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


def _exp_kernel():
    import d2m_jit as d2m

    @d2m.kernel
    def exp_fused(in_t, out_t, m_blocks, n_blocks):
        m_off = core_index(0) * m_blocks  # noqa: F821
        n_off = core_index(1) * n_blocks  # noqa: F821
        for m in range(m_blocks):
            for n in range(n_blocks):
                shard = remote_load(in_t, [m_off + m, n_off + n])  # noqa: F821
                remote_store(out_t, [m_off + m, n_off + n], shard.exp())  # noqa: F821

    return exp_fused


def _label(cfg):
    g = "x".join(map(str, cfg["grid_shape"]))
    b = "x".join(map(str, cfg["block_shape"]))
    return f"{g}|{b}/{cfg['dtype']}"  # grid|block/dtype


def _demo() -> None:
    """Exhaustive tune of `exp_fused` over (grid x block x dtype) for 128x128."""
    shape = (128, 128)
    space = block_space(shape, dtypes=("float32", "bfloat16"))
    print(
        f"tuning exp_fused over {len(space)} (grid x block x dtype) cfgs, {shape} ..."
    )
    result = tune(
        _exp_kernel(), [shape], space, golden=torch.exp, out_dir="prof_tune_exp"
    )

    print(f"\nobjective: {result.objective} (lower = faster)\n")
    print(f"{'rank':>4}  {'grid|block/dtype':>22}  {'kernel_ns':>12}  {'pcc':>9}")
    print("-" * 54)
    for rank, r in enumerate(result.leaderboard()[:12]):
        print(f"{rank:>4}  {_label(r.config):>22}  {r.kernel_ns:>12.1f}  {r.pcc:>9.5f}")
    if result.best:
        print(f"\nbest: {_label(result.best.config)} @ {result.best.kernel_ns:.1f} ns")

    # Pareto views: 2-way (latency vs cores) with a scatter, then the 3-way
    # front that keeps accuracy in play (bf16's speed vs f32's precision).
    print("\n" + "=" * 54)
    print(format_pareto(result, (LATENCY, CORES), label=_label))
    print("\n" + "=" * 54)
    print(format_pareto(result, (LATENCY, CORES, PCC), label=_label))
    print("\nleaderboard.json (with pareto front) written under prof_tune_exp/")


def _demo_search() -> None:
    """Guided search: hill-climb `exp_fused` over (grid x block x dtype) toward
    the best config, evaluating a path through the space instead of all of it."""
    shape = (128, 128)
    candidates = block_space(shape, dtypes=("float32", "bfloat16"))
    seed = {"grid_shape": [1, 1], "block_shape": [1, 1], "dtype": "float32"}
    proposer = hill_climb_proposer(candidates, seed=seed)

    print(
        f"guided search over a {len(candidates)}-config space "
        f"(latency hill-climb from {_label(seed)}) for {shape} ...\n"
    )
    result = search(
        _exp_kernel(), [shape], proposer, golden=torch.exp, out_dir="prof_search_exp"
    )

    print("evaluation path (in order proposed):")
    for i, r in enumerate(result.records):
        if r.valid:
            wait = r.diagnostics.get("wait_share_of_envelope")
            wtxt = f"  wait={wait:.0%}" if wait is not None else ""
            print(f"  {i:>2}. {_label(r.config):>22}  {r.kernel_ns:>10.0f} ns{wtxt}")
        else:
            print(f"  {i:>2}. {_label(r.config):>22}  invalid: {r.error}")

    print(
        f"\nevaluated {len(result.records)} of {len(candidates)} configs "
        f"({100 * len(result.records) / len(candidates):.0f}% of the space)"
    )
    if result.best:
        print(
            f"best found: {_label(result.best.config)} "
            f"@ {result.best.kernel_ns:.0f} ns"
        )
    print("search.json written under prof_search_exp/")


if __name__ == "__main__":
    import sys

    if "--exhaustive" in sys.argv:
        _demo()
    else:
        _demo_search()
