# Runbook: device perf for DS-vs-multicast matmul, per model and per matmul

How the Blackhole measurements in [`../dram-sharding-fleet-case-study.md`](../dram-sharding-fleet-case-study.md)
were produced, written so it can be re-run on **n150 (Wormhole)**. Read
["n150 differences"](#n150-differences) before trusting any absolute number — several of the
constants below are Blackhole-specific, and one of them (the profiler bug) may not exist on your
card at all.

Everything here measures **device kernel time per matmul**, grouped by shape and by projection,
for two compiles of the same model: one with DRAM-sharded matmul enabled, one without.

---

## 0. Prerequisites

**Source `env/activate` from the repo root.** It builds `PYTHONPATH` from `$(pwd)`, so sourcing it
from anywhere else silently resolves `ttrt` to the installed wheel in
`/opt/ttmlir-toolchain/venv/.../site-packages` instead of your `build/python_packages` tree — and
then the profiler outputs land somewhere other than where you look for them. `cd` to the repo root
first, every time.

**A perf-trace build.** Tracy is compiled in, not a runtime flag:

```bash
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=clang++-20 -DCMAKE_C_COMPILER=clang-20 \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DTTMLIR_ENABLE_RUNTIME=ON -DTT_RUNTIME_ENABLE_PERF_TRACE=ON
cmake --build build
```

Check the tracy tools landed where `ttrt` looks for them — building the ttrt wheel is what copies
them into the tt-metal source root (`tools/ttrt/setup.py`):

```bash
ls $TT_METAL_RUNTIME_ROOT/tracy-capture $TT_METAL_RUNTIME_ROOT/tracy-csvexport
```

**The two block-float dtype fixes in `tools/ttrt/common/util.py`.** Only needed if any model's
`main` takes a block-float `<input>` — in practice a bfp8 KV cache, which is most non-Qwen2.5
models. Without them ttrt cannot even start:

- `from_data_type` → `torch.bfloat16` for the six `BFP_*` names (host input generation)
- `ttrt_datatype_to_torch_dtype` → `torch.float32` for the same (output readback; metal unpacks
  block-float to f32 on host, so bfloat16 fails with `shape [...] is invalid for input of size N`
  at double the element count)

Check whether you need them:

```bash
python - <<'EOF'
import re,sys
txt=open(sys.argv[1]).read()
m=re.search(r'func\.func @main\((.*?)\) -> ', txt, re.S)
args=re.split(r'%arg\d+: ', m.group(1))[1:]
bad=[a for a in args if "bfp_" in a.split("loc(")[0] and "argument_type<input>" in a]
print(f"block-float <input> args: {len(bad)}  -> {'ttrt patches REQUIRED' if bad else 'runs as-is'}")
EOF
```

Caveat on the output-side fix: the returned torch tensor has the right size but reinterprets
block-float bytes, so **values are not golden-comparable**. Fine for profiling, never combine with
`--enable-golden` on those ops.

---

## 1. Get the graphs

Two CI runs of the same benchmark, one per configuration. Find the ttnn artifacts:

```bash
RUN=<run id>
gh api "repos/tenstorrent/tt-xla/actions/runs/$RUN/artifacts?per_page=100" --paginate \
  --jq '.artifacts[] | select(.name|startswith("ttnn-mlir-")) | .name'
gh run download $RUN --repo tenstorrent/tt-xla --name "<artifact name>" --dir graphs/ds/<model>
```

Layout the scripts expect:

```
graphs/
  ds/<model>/.../ttnn_runtime_<model>_..._g1_*.mlir
  nods/<model>/.../ttnn_runtime_<model>_..._g1_*.mlir
```

Two things about the artifact contents:

- **`g1` is the decode graph.** Each artifact holds `g0..g3`; graph index 1 is the decode step.
- **Use the `ttnn_runtime_*` file, not `ttnn_*`.** The `ttnn_runtime_*` variant is already past
  `TTNNCommonToRuntimePipeline` and translates straight to a flatbuffer. The plain `ttnn_*` dump is
  mid-pipeline — its `ttcore.cpu_module` still holds TTIR ops and `ttmlir-translate` rejects it with
  a misleading *"operation being parsed with an unregistered dialect"*. If you only have that one,
  run `ttmlir-opt --ttnn-common-to-runtime-pipeline` first.

### Free sanity check before spending device time

```bash
python static_matmul_survey.py graphs/ds/<model>/**/ttnn_runtime_*_g1_*.mlir --only-ds
```

Prints every matmul's shape, program-config kind, `in0_block_w`, in0 shard cores and K-tiles/core.
Confirms the two compiles actually differ, and shows which shapes take the DS path at all. Also
worth checking both compiles are *fully configured* — a model where tt-mlir emitted no program
config leaves ttnn's runtime fallback to choose, and the A/B then measures compile-vs-compile
rather than DS-vs-multicast:

```bash
for v in ds nods; do
  f=$(find graphs/$v/<model> -name "ttnn_runtime_*_g1_*.mlir" | head -1)
  echo "$v: $(grep -o 'matmul_program_config' $f | wc -l) configs / $(grep -o '"ttnn\.\(matmul\|linear\)"' $f | wc -l) matmuls"
done
```

---

## 2. Run it

```bash
export GRAPHS=$PWD/graphs
export MODELS="qwen_2_5_3b llama_3_2_1b ..."
export VARIANTS="ds nods"
bash run_model_fleet.sh          # ~4 min per model per variant
```

The driver translates, profiles, post-processes and joins. What it does per run, and why each
piece is load-bearing:

```bash
export TT_METAL_PROFILER_TRACE_TRACKING=1
ttrt perf graph.ttnn --loops 1 --trace-region-size 268435456 \
    --enable-program-cache --ignore-version
python -O -c "from tracy.process_ops_logs import process_ops; process_ops(None,None,False)"
```

| flag | why it is mandatory |
|---|---|
| `--enable-program-cache` | ttrt registers bool args with `action="store_true"` and never forwards the registered default (`run.py:1249`), so `default=True` reads as `False` on the CLI. Without it a traced graph aborts: *"Program cache must be enabled"* (`capture_or_execute_trace.cpp:214`). |
| `--trace-region-size N` | `run.py:588` assigns `mesh_options.trace_region_size` unconditionally, so the arg default of 0 beats metal's own default — which is also 0. Any traced graph fails to allocate. 256 MB is ample. |
| `TT_METAL_PROFILER_TRACE_TRACKING=1` | `tt_metal_tracy.hpp` emits trace BEGIN/REPLAY only under this flag but END unconditionally. With it off, tracy records an END with no BEGIN and post-processing dies with `KeyError: 0` *after* the whole run has executed. |
| `python -O` for post-processing | `process_ops_logs.py:606` has a bare `assert candidates` that aborts the entire report when any single host op lacks a device row (26 of 3557 for us). `-O` compiles the assert out so those ops are skipped instead. |
| `--ignore-version` | only if the graph's embedded system desc differs from the card. **Diff them first** — see [caveats](#caveats-that-bit-me). |

`ttrt perf` exiting 1 while still producing data is normal: that is the `:606` assert, and `-O`
recovers it.

---

## 3. Decide whether you need the per-core correction

**Do this before analysing anything.** On the Blackhole card used for the study,
`ops_perf_results.csv` reported ~10583 *seconds* for most ops, because
`DEVICE KERNEL DURATION` is `max(ZONE_END across cores) − min(ZONE_START across cores)` and some
cores' cycle counters are zeroed at device init while others are not. Every **multi-core** op
inherited a constant ~1.4288e13-cycle offset; single-core ops were fine.

Check your card:

```bash
# Locate the profiler dir. ttrt prints a warning on stdout, so take the last line;
# and see the env/activate note in section 0 -- the answer depends on your cwd.
PROF=$(python -c "import ttrt.runtime,os;print(os.path.dirname(ttrt.runtime.__file__))" 2>/dev/null | tail -1)/generated/profiler
ls $PROF/reports/ops_perf_results.csv    # if this is missing, find it:
# find / -name ops_perf_results.csv -newermt '-1 hour' 2>/dev/null

python - "$PROF" <<'EOF'
import pandas as pd, sys
df=pd.read_csv(f"{sys.argv[1]}/reports/ops_perf_results.csv", low_memory=False)
k=pd.to_numeric(df["DEVICE KERNEL DURATION [ns]"], errors="coerce")
c=pd.to_numeric(df["CORE COUNT"], errors="coerce")
print(f"rows with device data: {k.notna().sum()}")
print(f"durations > 1 s (impossible): {(k>1e9).sum()}")
print(pd.crosstab(c[k.notna()], (k[k.notna()]>1e9).map({True:'bogus',False:'sane'})).head(8))
EOF
```

- **All sane** → use `ops_perf_results.csv` directly, or the standard `tt-perf-report` tool.
- **Multi-core ops bogus** → use `percore_perf.py`, which recomputes each op as
  `max over cores of (that core's last kernel ZONE_END − its first ZONE_START)` from
  `.logs/profile_log_device.csv`, never mixing two cores' clocks.

Two hypotheses already ruled out on Blackhole, so don't re-chase them: it is **not** profiler
buffer overflow (dropping `--loops` 3→1 halved the log size and changed nothing, and counts were
byte-identical across runs), and `TT_METAL_PROFILER_SYNC=1` has **no effect** —
`profiler_sync_enabled` is only read by `inspector/data.cpp` and the separate realtime profiler.

---

## 4. Analyse

```bash
export FLEET=$PWD/fleet
# per-op durations, epoch-safe (the driver already runs this)
python percore_perf.py --device-log $LOGS/profile_log_device.csv \
                       --ops-data  $LOGS/tracy_ops_data.csv --out percore.csv
# join durations to shapes / program configs on GLOBAL CALL COUNT
python matmul_detail.py --percore percore.csv --report $PROF/reports/ops_perf_results.csv \
                        --out matmuls.csv
# DS vs no-DS per shape, per model, + fleet verdict
python fleet_compare.py --dir $FLEET --models $MODELS
# pivoted by projection role: down / gate+up / qkv / o_proj / lm_head
python by_projection.py $FLEET $MODELS
# where the delta lives: matmul vs layout/reshard vs other, and a per-shape classification
python classify.py $FLEET $MODELS
# every shape where DS beat multicast, separated from the control-noise floor
python ds_wins.py $FLEET $MODELS
# does bandwidth track the read burst on your card?
python burst_check.py $FLEET $MODELS
# score candidate decline criteria against measured outcomes
FLEET=$FLEET python criteria.py
```

`by_projection.py` infers roles from shape and instance count (the IR has no names): `lm_head` is
the single vocab-width matmul, `gate/up` the group appearing twice per layer, `down` the largest-K
per-layer group. **gate and up share a `K x N` and get the same config, so they are physically
indistinguishable here** and reported as one group of two per layer.

---

## 5. What to look at first

1. **`lm_head` as an in-situ bandwidth ceiling.** It stayed on 1D multicast in every compile of
   every model, so its achieved GB/s is the practical DRAM ceiling for *your* card. On Blackhole it
   was 385–390 GB/s in all ten models. Everything else is judged relative to it.
2. **Shapes on the same config in both compiles** are your noise floor. On Blackhole they agreed to
   1–2%, which is what made 1.2x-class differences meaningful.
3. **Op counts should reconstruct the architecture.** 181 matmuls = 36 layers × 5 + lm_head, 36
   SdpaDecode, 72 RotaryEmbedding, 72 PagedUpdateCache. If they don't, the join is wrong.
4. **Bandwidth above the DRAM ceiling means your byte accounting is wrong**, not that the kernel is
   fast. See the bfp4 trap below.

---

## n150 differences

The study card was Blackhole (8 DRAM banks, 10x11 worker grid, 1350 MHz). n150 is Wormhole and
differs in ways that matter here:

| | Blackhole (study) | n150 |
|---|---|---|
| DRAM banks | 8 | **12** |
| worker grid | 10x11 = 110 | 8x8 = 64 |
| DRAM bandwidth | ceiling observed ~390 GB/s | lower — re-derive from `lm_head` |

Consequences to expect:

- **Per-bank weight shards are narrower** (`shard_n = padded_N / 12 / 32` instead of `/8`), so
  `in1CB = in0_block_w × shard_n × tile_bytes × 3` is smaller and the `in0_block_w` search walks
  down less. The collapse-guard patch claims none of Wormhole's 716 DS matmuls walk down at all —
  **so the guard should be a no-op on n150.** If you see DS configs disappearing, that expectation
  is wrong and worth chasing.
- **`kPerCore` is unchanged** (`kNumIn0Cores` is hardcoded to 8 in `MatmulRules.cpp:41`
  independently of bank count), so the prime-`kPerCore` trap still exists in principle —
  Qwen2.5-3B still gives `kPerCore = 43`. Whether it collapses depends on whether `w=43` now fits
  the smaller `in1CB`. Check with `static_matmul_survey.py` before assuming either way.
- **Do not carry over absolute numbers.** The DS plateau (283–316 GB/s), the ~200 KB burst knee,
  and the ~390 GB/s ceiling are all Blackhole measurements. Re-derive each.
- **The 8-reader ceiling hypothesis becomes 12** — if DS's ceiling really is
  `banks × per-core NOC`, n150 should show a *higher* DS ceiling relative to its DRAM bandwidth
  than Blackhole did. That is a genuine test of the hypothesis and the most interesting thing this
  run could produce.
- **The single-op sweep scripts are Blackhole-specific.** `gen_downproj_tests.py` hardcodes core
  ranges for a 10x11 grid and 8-bank weight layouts; it needs regenerating for 8x8 / 12 banks.
- **The profiler epoch bug may not be present.** Run the check in step 3 rather than assuming.

---

## Caveats that bit me

- **System desc mismatch.** `util.py:905` raises on *any* dict difference between the graph's
  embedded desc and the card, bypassable only with `--ignore-version`. Diff them before bypassing:
  a difference in worker grid or L1 size makes the compiled layouts genuinely unsafe, whereas chip
  count or a DRAM-bank-to-core column shift is benign. On the study card the graphs were built for
  a 4-chip host with banks on worker column x=7 while the card had x=6 — benign, but it means small
  matmul deltas against the original machine are NOC distance, not code.
- **The bfp4 trap.** Weight dtypes are not uniform: 2290 bfp8 rows and 128 bfp4 across the fleet,
  and exactly one shape (`llama_3_1_8b` gate/up `4096x14336`) is bfp4. Hardcoding bfp8 bytes gave
  577 GB/s on it — above the DRAM ceiling, which is how the error surfaced. Always read
  bytes-per-element from that shape's own `INPUT_1_DATATYPE` (bfp8 = 1.0625 B/elem, 1088 B/tile;
  bfp4 = 0.5625, 576 B/tile). Verify the dtype is the *same in both compiles*, or the A/B is
  measuring precision rather than the path.
- **`matmuls.csv` is already traced-region only.** `matmul_detail.py` filters on the replay session;
  don't filter again downstream or you get empty tables.
- **`--loops 1` gives no within-run variance.** Repeatability rests on the shapes that share a
  config between compiles agreeing to ~1%. If you want error bars, raise `--loops` and split by
  `loop_number` from `PROGRAM_METADATA` — loop 0 captures the trace, later loops replay it, so they
  are not comparable.
- **Don't background the driver with `&` inside another background job.** It detaches from task
  tracking and looks like it exited instantly while still running.

---

## Script inventory

| script | purpose |
|---|---|
| `run_model_fleet.sh` | driver: translate → `ttrt perf` → post-process → join, per model/variant |
| `percore_perf.py` | epoch-safe per-op durations from `profile_log_device.csv` |
| `matmul_detail.py` | joins durations to shapes/configs/PM figures on `GLOBAL CALL COUNT` |
| `fleet_compare.py` | DS vs no-DS per shape and per model, with a fleet verdict table |
| `by_projection.py` | same data pivoted by projection role across models |
| `classify.py` | delta attribution (matmul vs layout vs other) + per-shape classification |
| `ds_wins.py` | exhaustive check for shapes where DS beat multicast, vs the noise floor |
| `burst_check.py` | tests whether bandwidth tracks `in0_block_w × shard_n` |
| `criteria.py` | scores candidate DS-decline criteria against measured outcomes |
| `static_matmul_survey.py` | config survey straight from the IR, no device needed |

Env vars: `GRAPHS` (downloaded graphs root), `FLEET`/`OUT` (results dir), `MODELS`, `VARIANTS`,
`TTMLIR` (repo root).
