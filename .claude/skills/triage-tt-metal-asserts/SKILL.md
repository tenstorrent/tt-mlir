---
name: triage-tt-metal-asserts
description: >
  Triage a tt-metal uplift diff or digest of TT_FATAL validation changes against what
  tt-mlir guarantees at each optimization level (0: workarounds only, 1: optimizer with
  DRAM-only fallback, 2: L1 sharding with rule books), sorting each assert into IGNORE /
  COVERED / LOOSE (we can violate it, tighten) / TIGHT (we over-constrain, relax). Use
  whenever the user pastes a list of added/modified/deleted TT_FATALs, an uplift preview or
  digest, a tt-metal PR touching *_device_operation.cpp validate() functions, or asks
  whether new metal asserts are "already covered", whether "we send the right data",
  whether a workaround or rule-book filter can now be dropped, or what will break on the
  next tt-metal uplift.
---

# Triage tt-metal assert changes against tt-mlir coverage, per optimization level

A tt-metal uplift changes the `TT_FATAL` checks ops run in `validate()`. Each one
constrains a tensor or attribute that tt-mlir chose at compile time, and tt-mlir chooses
differently at each `optimization_level`. Sort every assert into exactly one bucket, and
for LOOSE and TIGHT say which level it applies to:

| Bucket | Meaning | Output |
|---|---|---|
| **IGNORE** | We cannot violate it. Structural property of tt-mlir, not of the assert. | one line, no work |
| **COVERED** | Something forces or catches it at every level, *and* it actually runs. | file:line per level |
| **LOOSE** | At some level we can emit a violating config and nothing catches it. Uplift turns a working compile into a device fault, or a compile failure. Fix = tighten. | named fix + mechanism + level |
| **TIGHT** | We constrain harder than metal requires: forcing DRAM, rejecting sharding, casting dtype where the new `validate()` would accept L1 or the native dtype. Fix = relax. | constraint to drop + what it unlocks + level |

A deleted or relaxed assert whose only tt-mlir consequence was a constraint is TIGHT with
the tag `stale`. A TIGHT finding never gets applied by this skill: relaxing a constraint
changes codegen for every model using the op and wants its own change and test run.

## Definition of Done

Every in-scope assert lands in one bucket with a citation per optimization level, for the
mechanism that covers it or for the mechanism that should have. No assert left as
"probably fine". Truncated conditions resolved, not guessed. Indirect and
level-conditional coverage labelled as such, never as plain COVERED. Every in-scope op
also gets a TIGHT pass (Stage 6) even if none of its asserts changed in a way that hurts.

## What each optimization level can do

Resolve every claim below against `origin/main`; the level toggles live in
`resolveOptimizationLevelOptions()` in `include/ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h`
and the pass order in `createTTNNPipelineAnalysisPasses` in
`lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp`.

| | Level 0 | Level 1 | Level 2 |
|---|---|---|---|
| Optimizer | off | `GreedyMemoryLayoutPropagation`, sharded **and** L1-interleaved candidates cleared | same, with L1 interleaved + sharded candidates and `GreedyL1SpillManagement` |
| Layouts we emit | `TTNNLayout` default: DRAM interleaved, tiled per `shouldTilizeResult(op)` | optimizer emits DRAM interleaved only, tile or row-major (RM siblings); workarounds still insert L1/sharded `to_tensor_spec` (e.g. NLPConcatHeadsDecodeInput, DistributedRMSNormWidthShardInput) | anything, subject to rule books and constraint queries |
| dtype | as in IR, after the pre-optimizer dtype passes; nothing changes it later | optimizer keeps the IR dtype; `OperationValidationAndFallback` may cast a failing op's inputs to bf16 / f32 / u32 / i32, DRAM interleaved only | same as level 1; no dtype change is possible on any L1 candidate (tt-mlir #9299) |
| Operand (layout/dtype) workarounds | **all** ops | only ops in `enabledOpsForWorkaroundWithOptimizer` | same |
| Decomposition workarounds | all | all (`PagedUpdateCacheOpRewritePattern` only below level 2) | all except level-gated ones |
| Rule books | not consulted | consulted, DRAM-interleaved candidates only: input page-layout filters (tile vs RM) and op-specific attrs (e.g. Matmul program config via `applyOpSpecificAttrs`) decide the query; sharding/buffer-type filters, output hints and reshard exploration are inert with no sharded candidate to pick | fully live: input filters, output hints, reshard exploration |
| Compile-time check of metal asserts | **none** | `GreedyMemoryLayoutPropagation` queries each candidate (`evaluateHint` -> `validateOperation`), then `OperationValidationAndFallback` queries + fallbacks | same, plus L1/sharded candidates queried, then `GreedyL1SpillManagement` |
| Fallback set on query failure | n/a | original IR config (including a bfp8 dtype) tried first; then inputs: {bf16, f32, u32, i32} x {tile, row-major} x DRAM interleaved; output forced DRAM interleaved only when the op has a tensor result (null-config in-place ops keep theirs) | same |
| Compute kernel config | pipeline defaults `hifi4` + `fp32_dest_acc_en=true` applied to every op that does not already set the knob | `resolveOptimizationLevelOptions()` resets both to unset unless the CLI or frontend set them explicitly; unset knobs are never applied, so TTNN's per-op defaults decide | same as level 1 |

This table is the **default effective mapping** for a stock `optimization_level`.
`resolveOptimizationLevelOptions()` derives the toggles only when they are unset, so explicit
options (`enable-optimizer`, `memory-layout-analysis-enabled`, a forced compute-config or
dtype) move behavior between columns: level 0 with `enable-optimizer=true` runs the
optimizer, level 1 with sharding enabled gets L1 candidates. Always read the actual
invocation before trusting a row.

Consequences that decide buckets:

- **Level 0 has no layout/constraint net, but is not pass-free.** A violated layout assert
  is a device fault: only operand workarounds, decomposition workarounds, and TTIR verifiers
  stand between the IR and the kernel, and buffer-type/memory-layout asserts are mostly moot
  because everything is DRAM interleaved unless a workaround sets otherwise. But the
  always-on pre-optimizer passes still run here (weight/KV/activation dtype conversion,
  `TTNNSetComputeKernelConfig`), so dtype and compute-kernel-config asserts are shaped by
  them at level 0 too.
- **Levels 1 and 2 catch what the constraint query can see.** The query runs metal's
  `validate()` and program creation for the exact layouts in the IR. A new TT_FATAL is
  caught at compile time if, and only if, (a) the op has a working OpModel, (b) the query
  passes the same arguments the runtime passes, and (c) metal checks the condition in
  `validate()` or program creation rather than deep in a kernel. Being caught is not the
  same as being fixed: the fallback set must contain a config the kernel accepts, and
  fallback success still costs L1 residency and possibly a dtype cast.
- **The optimizer never changes dtype; only the fallback pass does, and only into DRAM
  interleaved.** A dtype assert at level >= 1 is satisfied by a whitelisted operand
  workaround, by a pre-optimizer dtype pass (`TTNNWeightDtypeConversion`,
  `TTNNKVCacheDtypeConversion`, activation dtype lowering), or by
  `OperationValidationAndFallback` casting a failing op's inputs to one of its four dtypes
  with DRAM-interleaved layouts. So a dtype requirement can be met at levels 1 and 2, but
  never on an L1 tensor: an op that needs both L1 and a different dtype has no compile-time
  path except a whitelisted workaround. The op's original config is validated first, so a
  bfp8 operand set by a pre-optimizer dtype pass is tried as bfp8; only when that fails
  does the fallback cast, and its set contains no bfp8, so a failing bfp8 tensor gets
  widened to one of the four.
- **Silent wrong results are never caught by queries.** If metal returns garbage instead
  of asserting (its comment or the linked issue will say "incorrect results"), a rule-book
  filter or a workaround owns it, not the query (e.g. the `where` operand workaround forces
  the predicate dtype for exactly this reason). The uplift diff cannot tell you when it is
  safe to drop such a constraint, so those TIGHT candidates need a silicon check.
- **Compute-kernel-config asserts (math fidelity, fp32 dest acc) are reachable only where a
  knob is actually applied.** By default that is level 0 (`hifi4`, `fp32_dest_acc_en=true`
  on every op without its own value, `TTNNSetComputeKernelConfig`), or any level when the
  frontend forces a knob (tt-crank and tt-xla both forward `math_fidelity` /
  `fp32_dest_acc_en` into `computeCfgMathFidelity` / `computeCfgFp32DestAccEn`). At
  levels >= 1 with nothing forced, TTNN's own defaults are what the kernel sees, so an
  assert there is on metal's defaults, not on anything tt-mlir sent.
- **Whitelisted operand workarounds insert `ttnn.to_tensor_spec` before the optimizer.**
  An L1 result of that op is unplaceable by both L1 spill trackers (tt-mlir #9299). For a
  layout-only requirement at level >= 1, prefer a rule-book filter over a whitelist entry;
  reserve whitelisting for dtype requirements the optimizer cannot express.

## Stage 0: baseline

**Hard rule: resolve everything against `origin/main`, never the checked-out branch.**

```bash
git fetch origin main -q
git rev-list --count HEAD..origin/main   # nonzero => working tree is stale
```

A feature branch tens of commits behind will show ops as "not called" that main calls,
producing a confidently wrong "nothing to do here". Use `git grep <pat> origin/main -- <paths>`
and `git show origin/main:<path>` for every lookup below. Audit a branch instead only if
the user asks, and say so.

## Stage 1: parse the diff into claims

One row per assert: **(metal file, op, condition text, added | modified | deleted)**.

- For `modified`, recover *both* old and new condition. The delta is the whole point.
- If a digest truncates a condition (`…`, `&amp;&amp;`, a cut line), **fetch the full text
  before classifying**. Guessing the tail is how a gap gets certified as covered.
- Keep the surrounding comment. Metal's comments state the *reason* (CB page sizing,
  scaler computed from padded shape). That decides whether a violation is a loud crash or
  silent numerical corruption, which drives priority.

## Stage 2: scope filter by symbol, never by path

**Do not filter by directory.** Ops under `tt-train/sources/ttml/` look like a separate
consumer of tt-metal, but tt-mlir calls several directly (`ttml::metal::adamw`,
`ttml::metal::sdpa_fw`, `ttml::metal::sdpa_bw`). Path scoping silently drops real exposure.

```bash
git grep -l "<op_name>" origin/main -- runtime lib include
```

Three call paths, all counting:

| Path | Where |
|---|---|
| Flatbuffer runtime dispatch | `runtime/lib/ttnn/operations/**` |
| EmitC generated C++ | `lib/Conversion/TTNNToEmitC/TTNNToEmitC.cpp` |
| EmitPy generated Python | `lib/Conversion/TTNNToEmitPy/TTNNToEmitPy.cpp` |

**Grep trap:** `ttml` is a substring of `ttmlir`, so `grep ttml` matches nearly the repo.

```bash
git grep -nE "\bttml::|ttml/" origin/main -- runtime lib | grep -v ttmlir
```

List zero-hit ops in the report as out-of-scope rather than omitting them, so the reader
can audit the filter. Scope is usually the largest single reduction; apply it first.

## Stage 3: drop the IGNORE tier

Everything in a `validate()` is nominally about caller data, so "internal to tt-metal" is
almost never the right dismissal. The real noise tier is asserts we cannot violate:

- **Device identity**: `x.device() == query.device()`. One program targets one
  device/mesh and every tensor comes from that program's pool.
- **Allocation / liveness**: `buffer() != nullptr`, `storage_type() == DEVICE`. The
  runtime tensor pool guarantees this (`DEBUG_ASSERT(ttnnTensor.is_allocated())` on insert).
  Exception: `TTNNLayout` deliberately keeps conv2d weight arguments and MeshShard inputs in
  system memory, so a `storage_type() == DEVICE` assert on those paths is live, not IGNORE.
- **Derived-quantity restatements**: e.g. `target_pages == input_nc_pages` sitting beside
  a shape assert that already pins the relation; it guards metal's own page indexing.

Report these as one grouped line. **State the reason they are noise**, because it is a
property of tt-mlir, not of the assert: multi-device programs would make device-identity
checks live again. Never silently delete them from the digest.

## Stage 4: classify the rest by constraint kind

The kind names the mechanism that can own it. Find the mechanism first, then in Stage 5
check whether it is in force at each level.

| Assert constrains | Owned by | Find it with |
|---|---|---|
| `dtype()` | operand workaround in `TTNNOperandsWorkaroundsFactory::create<Op>OperandsWorkarounds`, `lib/Dialect/TTNN/IR/TTNNWorkaroundsPass.cpp`; or a pre-optimizer dtype pass | grep the factory fn; check the whitelist (5a) |
| `memory_config().memory_layout()`, `buffer_type()`, sharded vs interleaved, grid | rule book `getInputLayoutFilter` / `getOutputHints` / `shouldExploreReshards`, `lib/Dialect/TTNN/Analysis/OpRules/*.cpp`; or an operand workaround forcing buffer type / memory layout | registry `getRuleBook()` in `OpRules/OpRuleBook.cpp`; factory fn |
| `layout()` (TILE / ROW_MAJOR) | `requireTiled` / `requireRowMajor` plus `generatesRowMajorInputSiblings` in the rule book, or an operand workaround | same |
| shape, rank, dim relations, padding | TTIR op verifier, `lib/Dialect/TTIR/IR/TTIROps.cpp`; decomposition workaround under `include/ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/`; **also the OpModel constraint query at level >= 1**, which catches these even with no verifier | `git grep -n "<Op>::verify" origin/main`; ls the Decomposition dir |
| scalar attribute values (`dropout_probability == 0.0F`) | the same verifier (a `.td` default is **not** enforcement); the constraint query also catches these at level >= 1 | `TTIROps.td` + verifier |
| program config / compute kernel config fields | `applyOpSpecificAttrs` in the rule book, `TTNNSetComputeKernelConfig`, decomposition workaround (e.g. `RMSNormConfig`, `ReduceScatterConfig`) | grep the config type |

Ops with no entry in `getRuleBook()` use `defaultRules`: accept every input layout, null
output hint plus sharded fallbacks, reshards explored. That is the loosest possible setting
and the first thing to check for a LOOSE finding at level 2.

Filter helpers (`layout_filter_utils::`, `include/ttmlir/Dialect/TTNN/Analysis/OpRules/LayoutFilterUtils.h`):
`requireTiled`, `requireRowMajor`, `rejectAllSharded`, `rejectL1Interleaved`,
`rejectWidthSharded`, `requireDRAMInterleaved`, `allowOnlyShardingType`, `isFullBboxSharded`;
output-hint helpers `nullHintOnly`, `nonShardedOutputHints`, `dramInterleavedOnlyOutputHints`.

## Stage 5: per-level matrix

For every in-scope assert fill three cells, one per level, each with exactly one of:

- `forced(<mechanism file:line>)`: a workaround, rule book, verifier or pass makes the
  violating config unreachable at that level.
- `caught(<fallback outcome>)`: the constraint query would reject the violating config at
  compile time. State whether the fallback set contains an accepted config
  (`caught, fallback ok`) or not (`caught, compile fails`).
- `unprotected`: the violating config is reachable and nothing checks it.

Then derive the bucket: all cells `forced` or `caught, fallback ok` gives COVERED, with the
per-level caveats in the table. Any `unprotected` or `caught, compile fails` gives LOOSE at
that level. Fallback ok at level 1 and 2 with a dtype cast or a lost L1 residency is COVERED
for correctness but is also a TIGHT/perf note in Stage 6.

Four ways a `forced` or `caught` cell is wrong:

### 5a. Operand workarounds are gated by an opt-level whitelist

At `optimization_level >= 1`, operand workarounds run **only** for ops in
`enabledOpsForWorkaroundWithOptimizer`
(`lib/Dialect/TTNN/Transforms/Workarounds/TTNNWorkaroundsPatterns.cpp`, list at the file
end). At level 0, all TTNN dialect ops get theirs.

```bash
git grep -n "create<Op>OperandsWorkarounds" origin/main -- lib          # 1. exists?
git show origin/main:lib/Dialect/TTNN/Transforms/Workarounds/TTNNWorkaroundsPatterns.cpp \
  | grep -n "ttnn::<Op>::getOperationName"                             # 2. whitelisted?
```

Exists but not whitelisted: `forced` at level 0, `unprotected` or `caught` at levels 1 and 2
depending on the OpModel. The whitelist comments are the model for justifying an entry.
Absence is sometimes deliberate (ArgMax is out because its rule book supplies RowMajor input
siblings instead); check for a comment before proposing an addition, and prefer a rule-book
filter for layout-only requirements (see the `to_tensor_spec` note above).

### 5b. The constraint query may not run for this op

```bash
git show origin/main:lib/Dialect/TTNN/Interfaces/TTNNOpModelInterface.cpp | awk '
  /[A-Za-z0-9]+Op::getOp(Constraints|Runtime)\(/ {
    match($0, /[A-Za-z0-9]+Op::getOp(Constraints|Runtime)/); sig = substr($0, RSTART, RLENGTH);
    fn = sig; sub(/::getOp.*/, "", fn); inC = (sig ~ /Constraints/) }
  inC && /ReasonForLackOfSupport::/ { match($0, /ReasonForLackOfSupport::[A-Za-z]+/); r[fn] = r[fn] " " substr($0, RSTART + 24, RLENGTH - 24) }
  inC && /constraintsDispatch|OpModel</ { real[fn] = 1 }
  END { for (f in r) print f, r[f], (real[f] ? "(conditional)" : "(always)") }' | sort
```

Only refusals inside `getOpConstraints` matter for validation; ignore `getOpRuntime`
refusals (creation and weight-preparation ops), since the optimizer does not call
`getOpRuntime` at all today. Do not read the file with a plain `grep -B`, the window lands
on the previous function's name. On main the always-refusing set is `DeallocateOp`,
`MoeGptOp`, `PrepareConv3dWeightsOp`, `ToTensorSpecOp`; `TypecastOp` refuses only for a
system-memory input. Ops without the OpModel interface at all are skipped the same way.

A refusing op is skipped by `OperationValidationAndFallback`, so its metal asserts are not
compile-time checked. But "not checked" is not the same as `unprotected`, and the two levels
differ:

- **Level 1 has no spill pass.** A refusing op's cell is `unprotected` unless a workaround
  or verifier forces the condition.
- **Level 2 runs `GreedyL1SpillManagement` first.** On a NotImplemented op it evicts every
  live L1 tensor to DRAM interleaved (`L1SpillManagement.cpp`, the `isNotImplemented` path).
  So a "must be DRAM/interleaved" assert on such an op is `forced` by the spill at level 2
  even though no query ran. Classify the cell after checking whether the spill produces the
  layout the assert wants.

For ops that do have a query, compare the argument list `OpModel<Op>::getOpConstraints`
passes in `lib/OpModel/TTNN/TTNNOpModel.cpp` with what `runtime/lib/ttnn/operations/**`
passes at dispatch. A program config, compute kernel config or optional tensor present at
runtime but defaulted in the query means the query validates a different call. Mark
`caught (arg mismatch: <field>)` and treat as LOOSE.

### 5c. Coverage is often transitive

When the consumer op has no rule of its own for an assert, the guarantee lives on the
tensor's *producer side*. Which producer side depends on whether producer and consumer are
in the same compiled program. Decide that first, then run the matching checklist.

**Step 1: where does the operand come from in the compile that matters?**

- Inference graphs are one program: the operand is an op result, use path A.
- Training: the frontends compile forward and backward as separate programs. A backward op
  consuming a forward result (`intermediates`, saved activations) sees a **function
  argument**, use path B. Any rule on the forward op is not in that compile.
- A lit test that passes the tensor as a `func.func` argument exercises path B even if the
  real model would be path A. Say which path you evaluated; when unsure, report both.

**Path A, same program: the layout is fixed where the tensor is produced.** Verify each
link; all are needed at level 2 (levels 0 and 1 have no sharded candidates at all):

1. Producer rule book: `getInputLayoutFilter` (does it ignore `operandIdx`?) and
   `getOutputHints`. Read `nullHintOnly` correctly: the optimizer sends only a NULL output
   memory config and takes whatever the backend returns. The result **is sharded whenever
   metal derives a sharded output from the inputs**, so it constrains nothing by itself;
   the input filter plus metal's derivation rule (link 2) decide the output.
2. Metal: which input's `memory_config()` the producer's `compute_output_specs` copies
   into that result, and whether tt-mlir passes a preallocated output that would override it.
3. Consumer rule book: `shouldExploreReshards()`; if `true`, propagation may insert a
   reshard on the edge, and only the consumer's own filter stops a sharded one.
4. Passes that touch the edge afterwards: spill (L1 to DRAM interleaved only) and fallback
   (DRAM interleaved only). Fine for "must be interleaved", fatal for "must be L1".

**Path B, function argument: the compiled argument layout is the guarantee.**

1. `TTNNLayout` leaves Input-typed arguments at the DRAM-interleaved default
   (`lib/Dialect/TTNN/Transforms/TTNNLayout.cpp`, `shouldForceInputSystemMemory` and
   `shouldForceInputRowMajor` list the exceptions).
2. `MemoryLayoutPropagation` has no producer in beam for an argument and keeps its current
   layout as the only candidate; a reshard can still be added unless the consumer's
   `shouldExploreReshards()` is `false` or its filter rejects the reshard target.
3. The frontend normalizes each argument to the consumer program's compiled layout, so the
   producing program's layout choice never survives the boundary. This is the crux of path
   B: when the tensor was produced by another program (the forward graph's `intermediates`
   feeding the backward graph), the backward op does **not** inherit whatever layout the
   forward program gave it. At runtime the frontend reads the layout the backward program
   was compiled to expect for that argument and converts the incoming tensor to it before
   submit, so the forward program's rule book has no say in what the backward kernel sees.
   tt-crank (the in-tree frontend, `tt-crank/`) reads `tt::runtime::getLayout(binary, 0, i)`
   per input at compile time (`src/engine/compile.cpp`) and, in `bind_tensor`
   (`src/engine/execution_payload.cpp`), calls `tt::runtime::toLayout` whenever `hasLayout`
   is false before the tensor enters the input slot that `run()` submits. tt-xla does the
   same via `ensure_layout`
   (`pjrt_implementation/src/api/flatbuffer_loaded_executable_instance.cc`). Only layout is
   normalized: `bind_tensor` asserts shape and dtype match (a bfp8/bfp4 target is the one
   dtype it converts, since torch has no block format), so those still cross the boundary
   unchanged. A frontend that skipped the layout conversion, e.g. a zero-copy device
   handoff, would make the compiled argument layout no guarantee and the assert LOOSE at the
   program boundary, so confirm it if a new frontend appears.

Worked example: upstream `sdpa_bw_q` / `sdpa_bw_kv` assert `intermediates` INTERLEAVED;
`TTMLSDPABackwardRuleBook`'s filter is `requireTiled` only. Path B (separate programs):
argument is DRAM interleaved by default, `shouldExploreReshards()` is `false`, and the
frontend converts on the way in; the fw rule book is irrelevant here
and only protects `sdpa_fw`'s own Q/K/V asserts. Path A (single graph):
`TTMLSDPAForwardRuleBook` rejects sharded on every operand, `sdpa_fw` copies
`query.memory_config()` into `intermediates` so a non-sharded query yields a non-sharded
result under the NULL hint, and the bw `shouldExploreReshards()` is `false`. Before the
assert lands, a sharded `intermediates` would stream garbage gradients with no error
(metal's comment says so), which is the silent-corruption priority case.

Mark these **COVERED (indirect, path A | path B)** and name every load-bearing link, not
just the first. They break silently when someone relaxes one link for performance, with no
signal at the edit site. Every TIGHT recommendation in Stage 6 must be checked against
these links, and a producer-side TIGHT relaxation cannot break a path-B consumer, while a
consumer-side reshard rule can.

### 5d. Downstream passes can re-break the invariant

- `shouldExploreReshards()` returning `false` stops layout propagation inserting reshards
  on that op's operands. If it returns `true`, confirm reshard targets stay legal.
- `GreedyL1SpillManagement` can still move a tensor L1 to DRAM interleaved. That preserves
  "interleaved" but breaks "must be L1" or "must be DRAM-sharded".
- `OperationValidationAndFallback` rewrites a failing op's inputs to DRAM interleaved and
  its output to DRAM interleaved. A consumer that required the original layout gets a
  revert `to_tensor_spec`, so consumer-side asserts stay satisfied, but producer-side
  assumptions ("input is sharded") do not.

Also: **`getInputLayoutFilter(unsigned operandIdx)` frequently ignores `operandIdx`** and
returns one lambda for every operand. Read the body; do not infer per-operand targeting
from the signature.

## Stage 6: TIGHT hunt, for every in-scope op

Run this for every op that appears in the diff, not only for deleted asserts. The uplift
is the moment someone actually reads the op's current `validate()`, so it is the cheapest
moment to find constraints we no longer need.

1. **Inventory tt-mlir's constraints on the op**, each with its stated reason:

   ```bash
   git grep -n -B6 "<Op>RuleBook::" origin/main -- lib/Dialect/TTNN/Analysis/OpRules/   # filters, hints, reshards
   git grep -n -A25 "create<Op>OperandsWorkarounds" origin/main -- lib/Dialect/TTNN/IR/TTNNWorkaroundsPass.cpp
   git grep -n "<Op>" origin/main -- include/ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/
   git grep -n "<Op>::getOperationName" origin/main -- lib/Dialect/TTNN/Transforms/Workarounds/TTNNWorkaroundsPatterns.cpp
   ```

   Constraints worth listing: `requireDRAMInterleaved`, `rejectAllSharded`,
   `rejectL1Interleaved`, `nonShardedOutputHints`, `dramInterleavedOnlyOutputHints`,
   `shouldExploreReshards() == false`, an operand workaround setting `BufferType::DRAM` or
   `TensorMemoryLayout::Interleaved` or a dtype, a whitelist entry, a decomposition
   workaround that moves tensors to DRAM. On main these include the Slice, ConcatenateHeads,
   Matmul operand 1, SDPA-decode K/V/cache, AdamW, LayerNormForward and CrossEntropyForward
   rule books, and the CrossEntropyForward, LayerNormForward, MoeCompute, MoeGpt and
   RotaryEmbedding operand workarounds.

2. **Find the reason.** Rule books and factories cite it in a comment: a tt-metal issue
   link, a `TT_FATAL` quote, "produces incorrect results", "not supported by our program
   config generation". No comment is itself a finding: report it, since a constraint
   without a reason cannot be relaxed safely.

3. **Compare the reason with the new `validate()`:**

   | Reason | New validate() | Verdict |
   |---|---|---|
   | quotes an assert | assert deleted or relaxed | **TIGHT (stale)** |
   | quotes an assert | assert still present, narrower than our constraint (assert rejects width-sharded, we `rejectAllSharded`) | **TIGHT (over-broad)** |
   | quotes an assert | assert unchanged and matches | keep, no finding |
   | metal issue link | issue closed and no assert covers the symptom | **TIGHT (candidate)**, needs silicon verification |
   | "incorrect results", no assert ever existed | anything | **TIGHT (candidate)**, silicon verification mandatory; the diff cannot prove safety |
   | our own limitation ("program config generation") | irrelevant | not a metal-uplift finding; note as tt-mlir debt |
   | dtype requirement enforced by whitelist | assert unchanged | keep; blocked by tt-mlir #9299, do not propose removal |

4. **State what relaxing unlocks and at which level.** Level 2 only for anything about L1
   or sharding; level 1 and 2 for a dropped dtype cast; level 0 for a dropped operand
   workaround. Say what the optimizer would now try (L1 interleaved input, sharded output,
   native dtype) and what has to be re-verified: the constraint query accepting the new
   candidates in a lit test at `optimization_level=2`, plus a silicon run whenever the
   original reason was numerics.

5. **Check Stage 5c links before recommending.** A filter that looks redundant on its own
   op may be the load-bearing guarantee for a consumer's assert.

## Stage 7: report

```
## LOOSE: will break on uplift
- <op>: <condition>
  L0: <forced(file:line) | unprotected>
  L1: <forced | caught, fallback ok | caught, compile fails | unprotected>
  L2: <same>
  Fix:    <operand workaround | whitelist entry | rule-book filter | verifier check |
           decomposition workaround | OpModel arg fix> at <file:line>
  Risk:   <which shapes/models trip it; device fault, compile failure, or silent corruption>

## TIGHT: constraint can be relaxed
- <constraint at file:line> (reason: <comment / issue / assert>) -> <stale | over-broad | candidate>
  Unlocks: <L1 input | sharded output | dropped dtype cast> at level <n>
  Verify:  <lit test at opt level 2 | silicon run because reason was numerics>
  Blocked: <tt-mlir #9299 for dtype whitelist entries, or none>

## COVERED
| Assert | L0 | L1 | L2 | Caveat |
|---|---|---|---|---|
| ... | forced(file:line) | caught, fallback ok | forced(rule book) | direct / indirect (path A via <op> | path B argument) / fallback costs L1 |

## IGNORE: cannot violate
<grouped one-liner + why it is a tt-mlir property, not an assert property>

## Out of scope
<ops from the diff with no tt-mlir call site>
```

Judgement calls to make explicitly:

- **Priority.** An assert that previously permitted silent wrong numerics (metal's comment
  will say so: mis-scaled softmax, overrun CB pages) outranks one that was already a loud
  failure. The uplift *fixes* the former; the latent bug predates it and affects everything
  currently compiled. Among loud failures, a device fault at level 0 outranks a compile
  failure at level 1, because level 0 has no message pointing at the cause.
- **Level.** Never report a bare "covered" when coverage holds only at level 0, and never
  report "caught" as covered when the fallback set has no accepted config. Production
  compiles run level >= 1.
- **Fallback is not free.** `caught, fallback ok` means the op runs DRAM interleaved with
  possibly a dtype cast. If the kernel would accept an L1 layout at the right dtype, that is
  a TIGHT/perf note as well as a COVERED row.
- **Rule book before whitelist.** For a layout-only requirement at level >= 1, propose a
  rule-book filter. A whitelist entry inserts a `to_tensor_spec` the spill trackers cannot
  place, and is the right tool only for dtype requirements until tt-mlir #9299 lands.
