# Llama 3.2 1B, seq 512, forward + loss — optimization-level repro

Standalone repro harness for
[tt-mlir#9189](https://github.com/tenstorrent/tt-mlir/issues/9189): the SST-2
fine-tune step returns loss 27.59 at `optimization-level=2` against 5.84 at
`optimization-level=1`, on the identical batch.

## The graph

`ttir_llama_3_2_1b_512_fwd.mlir` is the forward graph from `321b512.txt`
(the tt-kurbla log attached to the issue on 2026-08-21) — the first
`===== TTIR module =====` block, i.e. embedding → 16 decoder layers → final
norm → tied `lm_head` → softmax → NLL loss. The log's second large module is
the backward pass and is not used here.

Reformatted from the log dump to match a tt-xla style TTIR file: wrapped in
`module { ttcore.device_module { builtin.module { … } } }` and annotated with
`ttcore.argument_type` / `ttcore.local_shape` / `ttcore.shard_status` /
`ttir.name` on all 151 arguments — 147 `parameter` (arg0 embedding,
args 1–144 = 16 layers × 9 weights, arg145 `model.norm`, arg146 rope
`inv_freq`) and 4 `input` (`input_ids`, `attention_mask`, `labels_one_hot`,
`loss_weight`). The log carries no names or locations; argument roles were
derived from how each one is used in the graph, and the `ttir.name` strings are
HF-style labels, not the tt-kurbla dynamo names.

## Pipeline options

From the option table in the issue (agobeljicTT, 2026-08-20). Only
`optimization-level` varies between runs:

| option | value |
| --- | --- |
| `optimization-level` | 0 / 1 / 2 (the variable) |
| `enable-const-eval` | `false` |
| `enable-permute-matmul-fusion` | `true` |
| `compute-cfg-math-fidelity` | `hifi4` |
| `compute-cfg-fp32-dest-acc-en` | `true` |
| `ttnn-perf-metrics-enabled` | `true` |
| `ttnn-perf-metrics-output-file` | `perf_metrics_opt<N>` |
| `mesh-shape` | `1,1` |
| `mesh-topology` | `linear,linear` |

Two deviations from the table, both forced by the CLI:

- `mesh-topology=linear` alone aborts on
  `expected meshTopology size to match meshShape rank`; the mesh is 2-D, so it
  takes one topology per dimension.
- the issue marks the `compute-cfg-*` values as no-op defaults. They are not,
  above level 0: `resolveOptimizationLevelOptions` turns an *unset* fidelity
  into `Undefined` whenever the level is > 0, so leaving them off would compile
  something other than what the experiment compiled.

## Inputs

Two modes.

**Trained weights (`--weights <hf-snapshot>`)** — the real
`meta-llama/Llama-3.2-1B-Instruct` weights from the local HF cache, mapped onto
args 0–145 by name, rope `inv_freq` computed with llama3 scaling from
`config.json`, and 512 real tokens of English prose with the next token as the
label at each position. `lm_head` is tied to the embedding, as the graph
assumes. `real_inputs.py` reads the safetensors and tokenizes byte-level
greedily, so nothing is installed into the toolchain venv and nothing is
downloaded. This is the regime the issue reports from: a confident model whose
logits are peaked, where a small error early in the forward pass gets amplified
by the softmax.

The snapshot is Instruct rather than the base model the experiment used —
identical architecture and shapes, and the cached one needs no gated download.

**Synthetic (default)** — N(0, 0.02) weights, RMSNorm gains at 1, random token
ids, one-hot labels. Cheap and self-contained, but the model is a near-uniform
predictor, so the loss sits at `ln(128256) = 11.76` and the logits are flat.
That turned out to be too forgiving a regime to conclude anything from; it is
kept for smoke-testing the plumbing.

Either way every level is handed byte-identical inputs, so the levels are
comparable to each other.

`--cpu-reference` recomputes the same loss in torch on CPU, op for op and dtype
for dtype against the graph, which gives an independent answer to compare the
device against rather than only comparing levels to each other.

## Files

```
ttir_llama_3_2_1b_512_fwd.mlir   the graph under test
321b512.txt, 321b1024.txt        the tt-kurbla logs it came from (issue attachments)
extract_ttir.py                  rebuilds the .mlir from 321b512.txt
run_loss.py                      compile at each level, run, print the losses
real_inputs.py                   trained weights, tokenizer, CPU reference
wormhole/                        results from this card (see below)
```

Each run writes its artifacts into a directory named after the card —
`wormhole/`, `blackhole/` — resolved from the system descriptor, so results
from different hardware do not overwrite each other. `--outdir` overrides it.
Per level a run leaves `ttnn_opt<N>.mlir`, `opt<N>.ttnn`,
`perf_metrics_opt<N>.json`, `compile_opt<N>.log`, `run_opt<N>.log` and
`loss_opt<N>.json`.

## Running

```sh
source env/activate                     # from the repo root; it reads $(pwd)
ttrt query --save-artifacts             # if ttrt-artifacts/system_desc.ttsys is missing
cd llama_3_2_1b_512_fwd

# synthetic weights, quick plumbing check
python run_loss.py --levels 0 1 2

# the real thing: trained weights, real tokens, CPU golden
python run_loss.py --levels 0 1 2 --weights <hf-snapshot> --cpu-reference

python run_loss.py --levels 2 --keep-compiled   # reuse an existing opt2.ttnn
```

`<hf-snapshot>` is a directory holding `model.safetensors`, `config.json` and
`tokenizer.json` for Llama-3.2-1B (base or Instruct). On this machine:

```
~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6
```

### On a Blackhole card

Nothing to change. Query the system descriptor on that machine so the pipeline
plans against Blackhole, then run the same command — results land in
`blackhole/` next to `wormhole/`, ready to diff:

```sh
ttrt query --save-artifacts
cd llama_3_2_1b_512_fwd
python run_loss.py --levels 1 2 --weights <hf-snapshot> --cpu-reference
```

Level 2 takes ~8.5 minutes to compile here and may take longer there; the issue
reports the same pass failing to finish at all on 8B. What to look for is a
level-2 loss near **27.6** — that is the clamp floor, and it means the vocab
reduction has collapsed to zero (see below). The CPU reference does not touch
the device, so it is the same number on both cards.

The device run needs the harness's own copy of the graph and a built
`ttmlir-opt` / `ttmlir-translate` in `$BUILD_DIR/bin`; `run_loss.py` repairs
`TT_METAL_HOME`, `PATH` and `PYTHONPATH` itself, so it does not matter which
directory `env/activate` was sourced from.

## Results on Wormhole — the bug does not reproduce

Wormhole n150 (1 chip), tt-mlir `827432cf43`, warm kernel cache. Raw artifacts
in `wormhole/`.

Trained weights, 512 real tokens, mean next-token NLL:

| level | loss | compile (s) | submit (s) |
| --- | --- | --- | --- |
| cpu reference | 3.503467 | — | 41 (torch, CPU) |
| 0 | 3.496094 | 1.6 | 11.9 |
| 1 | 3.496094 | 2.8 | 10.7 |
| 2 | 3.496094 | 509.1 | 13.0 |

All three levels agree exactly with each other and land within 0.2 % of the CPU
reference — bf16 noise. Synthetic weights give the same verdict one regime up,
at 12.125 across all levels (seed 24 gives 12.09375, so the harness does track
its inputs).

### The 27.5938 in the report is the clamp floor

The graph clamps the target probability at `9.99999996e-13` before taking its
log, and `-log(9.99999996e-13) = 27.631`. The reported opt2 loss of 27.5938 is
that number to within 0.13 %. So the failure is not a drift in the loss — at
opt2 `sum(softmax × one_hot)` collapsed to zero at nearly every position, and
the loss is just the floor. The suspect op is the vocab-dim reduction, which is
what [PR #9164](https://github.com/tenstorrent/tt-mlir/pull/9164) ("Never give
reductions a sharded output") was written for.

### The suspect pattern *is* exercised here, and it works

Level 2 gives all 35 reductions in the graph a sharded output (18 block, 15
width, 2 height), where level 1 leaves all 35 interleaved. That includes the
one in the clamp-floor path — `%1361`, the sum over the 128256-wide vocab
axis — whose output is height-sharded across 16 cores in L1. It computes
correctly: the loss matches the CPU reference.

So this is not "the optimizer sat the graph out". Level 2 reorganizes the whole
memory plan — 547 added `ttnn.to_memory_config` ops, 39 sharded memory configs,
64 matmul program configs, 180× the compile time (8.5 min vs 2.8 s) for ~20 %
more device time — and produces the right answer, on the same graph and the
same non-default options, including on the exact reduction the report points
at.

### What is left

- **Architecture.** The issue ran on Blackhole (4× p300c); this is Wormhole.
  The memory-layout analysis plans against the system descriptor, and the grid
  differs (8×8 vs 13×10), so BH shards these reductions differently. This is now
  the leading candidate, and the harness is ready to run there unchanged.
- **Batch shape.** The SST-2 batch is mostly padding with 6 valid label tokens;
  this run is 512 dense valid tokens with an all-ones attention mask. A fault
  confined to masked or zero-weighted positions would not show up here.
- **Graph scope.** Only the forward graph. The e2e step also compiles the
  backward + AdamW graph.

The compiler is very nearly the same one: the issue pins `fc339dca`, which HEAD
is only 8 commits ahead of, and the one commit among them touching
`lib/Dialect/TTNN/Analysis` is #9223 (op model interface for TTML ops). PR #9164
is closed and unmerged, so nothing here is explained by a fix having landed.

The next thing to try is this harness on a Blackhole card, then a padded
6-valid-token batch.
