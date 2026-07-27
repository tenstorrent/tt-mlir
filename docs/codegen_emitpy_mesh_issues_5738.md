# Codegen (EmitPy) problems hit while debugging tt-xla #5738

While using the codegen emit/load path (`TTXLA_CODEGEN_EXPORT_DIR` /
`TTXLA_CODEGEN_LOAD_DIR`, handled in `pjrt_implementation/src/api/compile_options.cc:54-55`)
to iterate on the llama-3.1-70B qb2 2×2 accuracy bug, I hit two **codegen tooling bugs** that
are independent of the accuracy issue. Both block the emit→edit→reload workflow on the 2×2
(multi-device) mesh at realistic layer counts. Documented here so they can be filed/fixed.

Context: model `test_llama_3_1_70b_tp_qb2`, `QB2_MESH=2d` (2×2 mesh), `QB2_OPT=2`,
`QB2_NORM=shard`, Blackhole qb2 (4 chips). Emit is one graph per traced region:
`graph_0` = prefill, `graph_1` = decode.

---

## Bug A — decode-graph Python export segfaults at scale

**Symptom.** Exporting the emit with `TTXLA_CODEGEN_EXPORT_DIR` writes `graph_0/main.py`
(prefill) completely, then the process **segfaults (exit 139, "Fatal Python error:
Segmentation fault", "dumped core")** while emitting `graph_1` (decode) — `graph_1/main.py`
is never written (only the empty `graph_1/` dir is created).

**Scale-dependent.** 
- 3-layer decode graph exports fine (`emit_opt2/graph_1/main.py` exists, ~2185+ lines).
- 40-layer decode graph segfaults mid-emission of `graph_1`.
- Prefill `graph_0` at 40 layers exports fine (252,702 lines, 201 matmuls, ends cleanly).

So it is a **size/scale limit in the decode-graph Python emission**, not a per-op problem.
The compiled graph itself is fine (the normal benchmark runs the 40- and 80-layer decode
graph to completion); only the *EmitPy export* of the large decode graph crashes.

**Also reproduced** on 2×2 with **default** norm even at 3 layers (`emit_2x2_def/graph_1`
missing) — i.e. certain 2×2 decode configs crash the export regardless of layer count.

**Repro.**
```
TTXLA_CODEGEN_EXPORT_DIR=<dir> QB2_MESH=2d QB2_OPT=2 QB2_NORM=shard \
  pytest -q -s tests/benchmark/test_llms.py::test_llama_3_1_70b_tp_qb2 \
  --num-layers 40 --pcc-decode
# -> graph_0/main.py written; segfault; graph_1/main.py missing.
```

**Workaround.** Iterate on the **prefill** graph only (`--pcc-prefill`), which also exhibits
the accuracy collapse; prefill exports fine at 40 layers.

---

## Bug B — CPU-hoisted const-eval calls `to_torch` on a multi-device tensor (load fails)

**Symptom.** Loading an emitted 2×2 graph (`TTXLA_CODEGEN_LOAD_DIR`) fails at runtime with:
```
RuntimeError: ... in cpu_hoisted_const_eval_... -> execute_cpu_hoisted_function
  -> ttnn ... to_torch
info:
Can't convert a tensor distributed on MeshShape([2, 2]) mesh to row-major logical tensor.
Supply a mesh composer to concatenate multi-device shards.
```
Call path: `forward` → `consteval_forward` → `main_const_eval_N` →
`cpu_hoisted_const_eval_<hash>` → `utils.execute_cpu_hoisted_function` → `ttnn ... to_torch`
(`graph_0/main.py`, `graph_0/utils.py:116`).

**Cause (precise).** `graph_0/utils.py::execute_cpu_hoisted_function` *does* have a
multi-device branch (splits inputs into per-device shards) — but it is gated on
`mesh_device = DeviceGetter._instance`. On the codegen **load** path that singleton is
**None** when the const-eval runs, so it falls into the `if mesh_device is None:` branch and
calls `ttnn.to_torch(tensor)` on a tensor that is actually mesh-distributed → the
`MeshShape([2,2]) … supply a mesh composer` error. So the root is a **setup-ordering bug: the
mesh device (`DeviceGetter._instance`) is not registered before CPU-hoisted const-evals run on
the load path** (the emitted mesh-aware code is fine; it just never sees the device). The
prefill graph alone has **636** CPU-hoisted const-evals, so this fires immediately.
Why the singleton is null: `forward` opens the mesh via `DeviceGetter.get_device((2,2), …)`
at its top (main.py ~248271, idempotent — opens once, caches `_instance`), but on the
**load** path the const-eval graph (`consteval_forward`, 636 `main_const_eval_*`) runs
**before** that, so `_instance` is still None. Note the `main_const_eval_*(…, device)`
functions are even *handed* a `device`, but `execute_cpu_hoisted_function(function, inputs)`
doesn't receive it and reads the singleton instead.

Fix direction: ensure the load harness opens/sets the mesh device before const-evals (mirror
the runtime ProgramContext), or thread the passed `device` into
`execute_cpu_hoisted_function`, or derive the mesh from the tensor.

**Local unblock used for the bisection:** patched the emitted `graph_0/utils.py` so
`execute_cpu_hoisted_function` opens the mesh idempotently when the singleton is null:
```python
mesh_device = DeviceGetter._instance
if mesh_device is None:
    mesh_device = DeviceGetter.get_device((2,2), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING)
```
(get_device is idempotent, so forward's later call reuses it.) This is a per-emit patch
(wiped on re-emit); the real fix belongs in the codegen/load harness.

**Repro.**
```
TTXLA_CODEGEN_LOAD_DIR=<dir-with-const-eval-emit> QB2_MESH=2d QB2_OPT=2 QB2_NORM=shard \
  pytest -q -s ...test_llama_3_1_70b_tp_qb2 --num-layers 40 --pcc-prefill
# -> RuntimeError: Can't convert a tensor distributed on MeshShape([2,2]) ...
```

**Workaround.** Re-emit with **`QB2_CONST_EVAL=0`** (`tests/benchmark/benchmarks/
llm_benchmark.py:535` → sets `enable_const_eval=false`, `enable_const_eval_on_cpu=false`).
Const-eval is compile-time constant folding, accuracy-neutral, so disabling it is safe for
the investigation. (Testing whether this also avoids Bug A's segfault — the segfault may be
inside the same const-eval export.)

**Proper fix direction.** The EmitPy const-eval codegen (CPU-hoisted branch) must pass a mesh
composer to `to_torch` (concatenate shards) when the tensor is on a >1-device mesh, or skip
CPU-hoisting for mesh-distributed tensors.

**`QB2_CONST_EVAL=0` is NOT a clean workaround** — it trades Bug B for an OOM: with
const-folding disabled, the folded constants are materialized at runtime and the 40-layer 2×2
run aborts in the tt-metal allocator:
```
terminate called after throwing 'std::runtime_error'
  what(): TT_THROW @ .../tt_metal/impl/allocator/allocator.cpp:295: tt::exception
Fatal Python error: Aborted   (exit 134)
```
Neither `graph_0` nor `graph_1` is written. So on 2×2 the emit/load path is boxed in:
const-eval **on** → Bug B on load (to_torch mesh); const-eval **off** → allocator OOM.
The real unblock is to fix Bug B (patch the emitted `to_torch` to take a mesh composer) or
fix the const-eval codegen upstream.

---

## Non-issues (ruled out as codegen bugs)

- **`LOAD REPORT ... UNEXPECTED (layers 77-79) / MISSING (layers 1-39)`** — a HuggingFace
  weight-mapping warning for the truncated (`--num-layers`) model; appears in normal runs too.
  Not a codegen bug.
- **`venv/activate: No such file or directory`** — my own harness error (ran `source
  venv/activate` without `cd /home/mvasiljevic/tt-xla` first), not a codegen bug.
