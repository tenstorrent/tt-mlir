<!-- SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# d2m-jit CCL spec: `all_gather`

A design/scoping doc for adding collective-communication (CCL) support to
the `d2m-jit` testbed, using a hand-written `all_gather` kernel as the
driving example. Tracked here alongside [TODO.md](TODO.md) because the work
spans the DSL layer (`api.py`, `_src/ast.py`, `_src/builder.py`,
`_src/tensor_layout.py`), the runtime marshaling in `_execute`, and a couple
of compiler-side lowerings that are currently unexercised.

Status legend: 🔴 blocker · 🟡 missing surface · 🟢 nice to have · ✅ exists.

---

## 0. The target kernel

The pseudocode we want to compile and run (the `ccl.d2m` sketch, rewritten to
the keyword-argument surface chosen below):

```python
@d2m.kernel
def all_gather(in0, out0, ccl_start_semaphore, ccl_end_semaphore):
    dy = mesh_position(0)
    dx = mesh_position(1)
    cy = core_index(0)
    cx = core_index(1)
    device_synchronize(
        ccl_start_semaphore,
        start_device=[dy, 0],
        mcast_shape=[1, 8],
        num_receivers=7,
        core_indices=[cy, cx],
    )

    buf = empty([1, 1])                 # in-kernel L1 scratch shard
    remote_load(buf, in0, [cy, 0])      # explicit destination buffer

    remote_store(
        out0,
        [dx * 2 + cy, 0],
        buf,
        start_device=[dy, 0],
        device_mcast_shape=[1, 8],
        semaphore=ccl_end_semaphore,
        semaphore_indices=[cy, 0],
    )
    semaphore_wait(ccl_end_semaphore, 7)
    semaphore_set(ccl_start_semaphore, 0)
```

This runs on a **1×8 mesh**: `in0` is a per-device shard, `out0` is the
gathered result. Every device runs the same program; `device_synchronize`
makes receivers signal senders before any cross-device `remote_store`.

---

## 1. Primitive inventory

Each pseudocode line mapped to its D2M op and current status. All dialect ops
exist and have Python bindings (verified signatures in §2).

| Pseudocode | D2M op | Status |
| --- | --- | --- |
| `ccl_*_semaphore` kernel params | `!d2m.global_semaphore` GenericOp operands | 🔴 no semaphore arg kind |
| `mesh_position(0/1)` | `d2m.mesh_position {dim}` → index | 🟡 |
| `core_index(0/1)` | `d2m.core_index` | ✅ |
| `device_synchronize(...)` | `d2m.device_synchronize` | 🟡 |
| `buf = empty([1,1])` | `d2m.empty : tensor<…x!tile>` (L1 scratch) | 🟡 (host-only today) |
| `remote_load(buf, in0, [cy,0])` | `d2m.remote_load` w/ explicit `localBuffer` | 🟡 (auto-allocates; no explicit-buffer form) |
| `remote_store(..., start_device=, device_mcast_shape=, semaphore=, semaphore_indices=)` | `d2m.remote_store` cross-device form | 🟡 (op supports it; `api.py` hardcodes empty + no semaphore) |
| `semaphore_wait(sem, 7)` | `d2m.semaphore_wait` | 🟡 (only `.wait()` method) |
| `semaphore_set(sem, 0)` | `d2m.semaphore_set` | 🟡 (only `.set()` method) |
| 1×8 mesh + sharded host I/O | module `ttcore.meshes`, register-device `mesh-shape`, runtime mesh I/O, host `d2m.mesh_shard` | 🔴 single-device only |

Reference IR to mimic:
- `test/ttmlir/Dialect/D2M/generic/generic_global_semaphores.mlir` — the
  canonical hand-written `d2m.generic` with a `create_global_semaphore`
  backing buffer, the semaphore as an additionalArg, and
  `semaphore_wait`/`reset_global_semaphore` + buffer dealloc.
- `test/ttmlir/Conversion/D2MToTTMetal/all_gather_virtual_grid_untilize.mlir`
  — the `1x8` module mesh attribute and the `mesh_shard`
  (`full_to_shard`/`shard_to_full`) framing around `all_gather`.
- `lib/Conversion/TTIRToD2M/TTIRToD2M.cpp` — `D2MAllGatherRewriter`, the
  reference for the generic body the DSL is replicating by hand.

---

## 2. Verified Python op signatures

From `install/python_packages/ttmlir/dialects/_d2m_ops_gen.py`:

```
MeshPositionOp(dim, *, results)                     # dim: I64Attr -> index
DeviceSynchronizeOp(sync_semaphore, senderStartDevice,
                    senderDeviceMcastShape, numReceivers, coreIndices)
                                                    # numReceivers: I32Attr; rest: Variadic<Index>
CreateGlobalSemaphoreOp(input, *, value)            # input: backing tensor -> !d2m.global_semaphore
ResetGlobalSemaphoreOp(semaphore, value)
SemaphoreSetOp(semaphore, value, dstCoreIndex, mcastShape, startDevice, deviceMcastShape)
SemaphoreIncOp(...)                                 # same shape as SemaphoreSetOp
SemaphoreWaitOp(semaphore, value, *, resetValue)
```

`d2m.remote_store` already accepts `start_device`, `device_mcast_shape`,
`semaphore`, `semaphore_indices` (see the
`(resultType, memref, indices, localBuffer, startDevice, deviceMcastShape,
semaphore, semaphoreIndices)` builder in `D2MGenericRegionOps.td`).
`d2m.remote_load` already accepts an explicit `localBuffer`
(`(resultType, localBuffer, memref, indices)` builder).

---

## 3. Decisions (locked)

- **Op surface = keyword args.** New kernel-body ops use explicit kwargs, not
  the loose positional form from the original sketch. This disambiguates
  `num_receivers` (a compile-time attribute, not a runtime index) and is
  self-documenting. The §0 pseudocode reflects this.
- **Mesh sharding = host-scope `d2m.mesh_shard`.** Inputs get a
  `full_to_shard` `mesh_shard` and outputs a `shard_to_full`, mirroring the
  TTIR `all_gather` pipeline. The host sees full tensors; the compiler
  distributes shards across the mesh. Runtime marshaling stays single-tensor
  per arg.

---

## 4. Workstreams

### A. In-kernel syntax ops (small, mechanical)

Registered via `@syntax(...)` in `api.py`, dispatched by `D2MCompiler`.
`_src/ast.py` already handles the straight-line body and index arithmetic
(`dx * 2 + cy` lowers through `visit_BinOp` → `arith.muli`/`arith.addi` on
index types). No control-flow changes needed.

- **A1 `mesh_position`** — free function, `dim` as an I64 attribute (model on
  `core_index`):
  ```python
  @syntax("mesh_position",
          args_as_attr=[lambda n: IntegerAttr.get(IntegerType.get_signless(64), n.value)])
  def mesh_position(dim):
      return d2m.mesh_position(dim)
  ```

- **A2 `device_synchronize`** — free function, keyword surface. `num_receivers`
  must be a Python literal (it is an `I32Attr`), so it takes an `args_as_attr`
  callback; the three index lists are resolved at runtime via `_asindex`:
  ```python
  device_synchronize(sem, start_device=[dy,0], mcast_shape=[1,8],
                     num_receivers=7, core_indices=[cy,cx])
  ```
  Emits `d2m.device_synchronize(sem, start_device, mcast_shape,
  I32Attr(num_receivers), core_indices)`.

- **A3 in-kernel `empty(shape)`** — free function mirroring the existing
  `_zeros_op`, but emits a bare `d2m.empty(tensor<shape x !tile>)`
  (uninitialized L1 scratch) instead of the `tile_fill` generic. Reuse the
  `_shape_literal` callback (shape must be a static literal — it sizes a
  tensor type).

- **A4 `remote_load` explicit-buffer form** — extend the wrapper to accept an
  optional leading buffer; keep the auto-alloc form for back-compat:
  ```python
  remote_load(buf, src, indices, ...)   # buf is the localBuffer; result aliases buf
  remote_load(src, indices, ...)        # current auto-alloc form
  ```
  `indices` length must equal N/2 (grid dims only).

- **A5 `remote_store` cross-device kwargs** — surface the four fields the op
  already has:
  ```python
  @syntax("remote_store")
  def remote_store(dst, indices, src, *, start_device=None,
                   device_mcast_shape=None, semaphore=None, semaphore_indices=None):
      return d2m.remote_store(
          dst.type, dst, indices, src,
          start_device=[_asindex(v) for v in (start_device or [])],
          device_mcast_shape=[_asindex(v) for v in (device_mcast_shape or [])],
          semaphore=semaphore,
          semaphore_indices=[_asindex(v) for v in (semaphore_indices or [])])
  ```

- **A6 free-function `semaphore_set` / `semaphore_wait` / `semaphore_inc`** —
  the kernel calls these as free functions; today only the `Semaphore`
  *method* forms exist, and method dispatch keys off the receiver's MLIR
  type-string (`!d2m.semaphore`) which will not match `!d2m.global_semaphore`.
  Add free-function `@syntax` wrappers delegating to `d2m.semaphore_*`.
  Confirm/also-register the method forms under the global-semaphore
  type-string if method syntax on semaphore params is desired.

### B. Global semaphores as a new kernel-argument kind (medium) 🔴

The central new concept. Today kernel args are only `Layout`-backed
`LazyTensor` or `int`.

- **B1 host-side handle + creation.** Add a `GlobalSemaphore` host object and
  a constructor `d2m.global_semaphore(grid_shape, init=0)` that, on the
  builder, emits:
  - a backing `d2m.empty()` of `tensor<GY x GX x 1 x 1 x ui32, #sem_layout>`
    (a `#ttcore.metal_layout` over the worker grid — see `#sem_layout` in
    `generic_global_semaphores.mlir:13`), then
  - `d2m.create_global_semaphore(backing) {value=init}` →
    `!d2m.global_semaphore`.

  Requires a **ui32 path** in `tensor_layout.py`: `get_scalar_type`,
  `_TTCORE_TO_TORCH`, and `_to_runtime_data_type` (in `builder.py`) only map
  f32/f16/bf16 today; add `UInt32`.

- **B2 pass into the kernel.** Extend the arg-splitting in
  `_emit_kernel_generic` (`builder.py:~1078-1104`) to recognize the semaphore
  handle as a third arg category that becomes a GenericOp **additionalArg** of
  type `!d2m.global_semaphore`. Mirror `add_scalar_input` with an
  `add_semaphore_input` that wires the `create_global_semaphore` result in as
  the operand.

- **B3 compile-time arg typing.** `_emit_entry` (`ast.py:~249-266`) builds func
  operand types from `Layout`/`int` only. Add a branch mapping a semaphore arg
  to `d2m.ir.GlobalSemaphoreType.get(ctx)` so the kernel func signature and
  block args type-check and the body names resolve.

- **B4 operand ordering.** GenericOp `operandSegmentSizes` is
  `[numInputs, numOutputs, numAdditional]`; semaphores join `additionalArgs`
  (cf. `generic_global_semaphores.mlir:34-39`,
  `operandSegmentSizes = array<i32: 1, 1, 1>`). Decide scalar-vs-semaphore
  order within additionalArgs and keep the block-arg replacement loop
  (`builder.py:~1180-1184`) consistent.

- **B5 teardown.** After the kernel call, emit
  `d2m.reset_global_semaphore(sem) {value=0}`; liveness/dealloc reclaims the
  backing buffer (the reference test depends on this).

### C. Multi-device mesh support (largest; the real blocker) 🔴

The kernel is meaningless on 1×1.

- **C1 module mesh attribute.** `_Builder.__init__` (`builder.py:~230-246`)
  sets `module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x8>]>}`
  when a mesh is requested.

- **C2 register-device mesh-shape.** `_run_pipeline` (`builder.py:~862-865`)
  must add `mesh-shape=1,8` to `ttcore-register-device` (matches the lit RUN
  line `mesh-shape=1,8 mesh-topology=linear,ring`).

- **C3 Layout gains a mesh/shard concept + host `mesh_shard`.** Per the locked
  decision, emit `d2m.mesh_shard` at host scope: `full_to_shard` on inputs and
  `shard_to_full` on outputs (cf.
  `all_gather_virtual_grid_untilize.mlir:10-25`). The kernel sees per-device
  shards; the host sees full tensors. `tensor_layout.py`/`builder.py` need a
  way to describe the shard dims + mesh shape for the `mesh_shard` attrs
  (`shard_dims`, `shard_shape`, `shard_direction`, `shard_type`).

- **C4 runtime mesh open + I/O.** `_execute` (`builder.py:~889-944`) already
  opens the mesh from `fbb.get_program_mesh_shape` and marshals one borrowed
  host tensor per arg. With C1/C2 the program mesh shape becomes 1×8
  automatically, and with C3's `mesh_shard` the host I/O stays single-tensor.
  **Verify** the runtime `submit` path handles a multi-device program built
  directly by d2m-jit the same way it handles the full TTIR pipeline's output.

### D. Pipeline / compiler prerequisites (verify, don't assume)

- **D1 multi-synchronizable unified region** 🔴 — this kernel puts
  `remote_load` + `remote_store` + `device_synchronize` +
  `semaphore_wait`/`set` in one unified region (multiple synchronizable ops).
  This is exactly the `SplitUnifiedThread::wrapComputeInSynchronizedRegion`
  assertion in [TODO.md](TODO.md) ("synchronized scope must be unambiguous").
  The `split-unified-thread-v2` work on this branch
  (`config.use_split_unified_thread_v2`) is the likely prerequisite and must
  be validated against this kernel shape.

- **D2 unexercised lowerings** 🔴 — there are **no** existing lit/silicon
  tests referencing `device_synchronize` or `mesh_position`, and only
  `test/ttmlir/Conversion/D2MToTTKernel/dma_ops.mlir` exercises
  `startDevice`/`deviceMcastShape` (via `dma_write`, not `remote_store`). So
  `D2MToTTKernel` lowering of `device_synchronize`, cross-device
  `remote_store` + semaphore, and `mesh_position` is effectively untested
  through this front. Budget time for lowering fixes, not just DSL plumbing.

---

## 5. Milestones

1. **Single-device scaffolding (no mesh).** A2/A3/A4/A5/A6 + B (global
   semaphores) + free-function semaphore ops. Anchor test: a 1×1 "fake CCL"
   kernel that creates a global semaphore, does
   `remote_load`→`remote_store`→`semaphore_wait`/`set` with no cross-device
   fields, modeled on `generic_global_semaphores.mlir`. Validates B
   end-to-end without the mesh blocker.
2. **`mesh_position` + `device_synchronize` lowering (D2).** Lit-only
   IR-shape tests via `config.print_ir_before_pipeline` to lock the emitted
   IR, then push through `D2MToTTKernel` and fix lowering bugs.
3. **Multi-device (C).** Mesh attribute + register-device mesh-shape +
   host `mesh_shard` + runtime mesh I/O. Anchor: the real 1×8 `all_gather`,
   PCC-checked against `torch` gather.
4. **SplitUnifiedThread v2 (D1).** Validated as part of milestone 3 (it
   blocks the full kernel).

---

## 6. Files touched

| File | Changes |
| --- | --- |
| `api.py` | A1–A6 syntax ops; free-function semaphore ops; `GlobalSemaphore` re-export |
| `_src/ast.py` | B3 semaphore func-arg typing in `_emit_entry` |
| `_src/builder.py` | B1 create_global_semaphore + backing buffer; B2/B4 additionalArg plumbing; B5 reset/dealloc; C1 mesh attr; C2 register-device mesh-shape; C3 host `mesh_shard`; C4/ui32 runtime marshaling |
| `_src/tensor_layout.py` | B1 ui32 dtype path; C3 mesh/shard descriptor fields |
| `test/d2m-jit/` | milestone anchor tests (lit IR-shape + on-device PCC) |

---

## 7. Workstream C: detailed scope — multi-device mesh

Status: A and B landed (commits on `nsmith/d2m-ccl`); the full `ccl.d2m`
all_gather kernel verifies and lowers on a degenerate 1x1 mock device. C is
what makes it actually gather across devices. Researched against the codebase
and this dev box.

### Findings (grounding)

- **Mesh-shape plumbing is cheap.** The module attr
  `ttcore.meshes = #ttcore.meshes<[<"mesh"=RxC>]>` (or the
  `ttcore-register-device` `mesh-shape` / `mesh-topology` options) flows
  `determineMeshShape` → `DeviceAttr.meshShape` → flatbuffer `Dim2d` →
  runtime `get_program_mesh_shape` → `open_mesh_device`. Module attr wins;
  setting both to *conflicting* values errors.
  (`TTCore/Utils/Mesh.h`, `Target/Utils/MLIRToFlatbuffer.h:52`,
  `TTCoreRegisterDevice.cpp`, `TTCore/Transforms/Passes.td:54-86`.)
- **`d2m.mesh_shard` already lowers in our pipeline.** It survives
  bufferization (`D2MOps.cpp` `MeshShardOp::bufferize`) and is rewritten to
  `ttmetal.mesh_shard` by `d2m-to-ttmetal-pipeline`
  (`D2MToTTMetal.cpp:429`, `D2MMeshShardRewriter`) — already in
  `_build_pipeline`. No new pass needed for mesh_shard I/O.
- **Runtime distributes a full host tensor.** With `full_to_shard` on
  inputs and `shard_to_full` on outputs, the runtime's `MeshShardCommand`
  (`tensorFullToShard` / `tensorShardToFull`) distributes/gathers, so
  `_execute` can keep marshaling one **full** `create_borrowed_host_tensor`
  per input (Option A). A pre-sharded API
  (`create_multi_device_host_tensor`) exists but is not needed for v1.
- **Our hand-built generic mirrors `D2MAllGatherRewriter`**
  (`TTIRToD2M.cpp:3775` — two global semaphores, to-layout'd I/O, the same
  body), so we do not need `ttir.all_gather` or its
  `TTIRMultiDeviceTensorAnnotation` pass. Risk: the TTIR-frontend annotations
  we skip might still matter — must verify a directly built multi-device
  program runs (medium risk).
- **This box has ~2 chips, not 8.** UMD reports `local chip {0}` +
  `remote chip {1}`. So end-to-end validation here is a **1x2** mesh, not
  1x8. The kernel's mesh constants (`mcast_shape=[1,8]`, `num_receivers=7`
  in `ccl.d2m`) must be **derived from the mesh shape**, not hardcoded
  (1x2 → mcast `[1,2]`, `num_receivers=1`).

### Tasks

- **C1 — Mesh configuration (builder).** A way to declare the mesh (e.g.
  `d2m.mesh(shape=(1,2), topology=("linear","ring"))` stored on the builder)
  that sets the module `ttcore.meshes` attr and passes
  `mesh-shape` / `mesh-topology` to `_register_device`. Pick module-attr OR
  pass-option (both-conflict errors). ~0.5 day. Risk: low.
- **C2 — `mesh_shard` host ops + Layout mesh support (tensor_layout +
  builder).** `Layout` gains mesh fields (`mesh_shape`, `shard_dims`,
  `shard_shape`) or a sibling descriptor; the per-device shard shape is
  derived from the full logical shape (256x2048, shard [1,8] → 256x256). New
  host op `mesh_shard(lt, shard_dims, shard_shape, direction)` (or a
  `to_layout` variant) emitting `d2m.mesh_shard` with `shard_type=devices`;
  inputs `full_to_shard`, outputs `shard_to_full`. ~2 days. Risk: medium
  (Layout is single-device today).
- **C3 — Runtime mesh I/O (`_execute`).** Once C1 lands,
  `get_program_mesh_shape` returns the mesh and `open_mesh_device` opens N
  devices. Keep full-tensor input marshaling (Option A); allocate full-shape
  outputs for `shard_to_full`. Validate `submit` accepts a full borrowed
  tensor for a mesh_shard'd input. ~1 day + validation. Risk: medium
  (untested d2m-jit multi-device submit).
- **C4 — Mesh-shape-driven kernel constants.** The all_gather body's mcast
  shape / num_receivers / output-index math must come from the mesh shape
  (kernel scalars or captures), not literals, so it works on 1x2 and 1x8
  alike. ~0.5 day.
- **C5 — End-to-end all_gather test (1x2).** Full host input → `mesh_shard`
  → all_gather kernel → `mesh_shard` → full output, PCC vs `torch`. This is
  where the cross-device `semaphore_wait` is finally signaled (no deadlock).
  ~1 day incl. debugging.

### Risks / open questions (ranked)

1. **Untested multi-device path (D2 at mesh scale).** `mesh_position`,
   `device_synchronize`, cross-device `remote_store`, and the d2m-jit
   multi-device submit are exercised by no test. Highest chance of
   compiler/runtime bugs. Mitigate with a tiny `mesh_shard` round-trip
   (phase C-a) before the full gather.
2. **Skipped TTIR-frontend annotations.** Directly built d2m multi-device
   programs may miss `TTIRMultiDeviceTensorAnnotation` / LocalShape metadata.
   Verify; add the pass to `_build_pipeline` if needed.
3. **Hardware ceiling.** 1x2 here; 1x8 needs an 8-chip system. Keep the
   kernel mesh-shape-driven so it scales when run elsewhere.
4. **Topology / system-desc agreement.** register-device with a real system
   desc + `mesh-shape` must match the physical cluster (ring on the cluster
   axis). Confirm the auto-discovered mesh on the target supports the
   requested shape.

### Suggested phasing

- **C-a (de-risk):** mesh config (C1) + runtime (C3) + a *no-compute*
  `mesh_shard` round-trip (full → shard → full identity) on 1x2. Validates
  mesh plumbing + runtime I/O in isolation, before any CCL semantics.
- **C-b:** Layout mesh support + `mesh_shard` host ops (C2); PCC round-trip.
- **C-c:** mesh-driven kernel constants (C4) + end-to-end all_gather (C5).

### Progress & findings

- **C-a (done):** `d2m.mesh(shape, topology)` config; 1x2 mesh_shard identity
  validated through pipeline + runtime.
- **C-b (done, with a caveat):** `mesh_shard` / `mesh_gather` host ops +
  `LazyTensor.mesh` (`MeshShard`) metadata + mesh-aware `to_host` gather. The
  **identity** round-trip (`mesh_shard(full)` → `to_host`) works end-to-end on
  1x2. The host func arg is the full tensor; `mesh_shard` emits `full_to_shard`,
  `to_host` emits `shard_to_full`.

- **C-c task 0 (done): compute on a mesh shard.** Previously a round-trip with
  a compute generic between f2s and s2f failed at runtime
  (`LOG_ASSERT meshBuffer.size() == tensorDesc.sizeBytes()`,
  `runtime/lib/ttmetal/executor_utils.h:384`): the `mesh_shard` `full_to_shard`
  result was a plain single-device `TensorDesc` (e.g. 64x64 = 16 KB) while the
  device mesh buffer spans the mesh (2 x 16 KB). **Fix:** tag the per-device
  shard at the mesh_shard boundary with `#ttcore.tensor_mesh<name>` (option (a)
  — replicating what `ttir-multi-device-tensor-annotation` does, but for
  `d2m.mesh_shard`). It bufferizes to `#ttcore.host_layout<..., <name>>`, so the
  runtime sizes the distributed host buffer to match the mesh device buffer.
  Only the host-side shard boundary tensors are tagged (f2s result, s2f input);
  the full tensor and the per-device device buffers stay plain. Verified on 1x2:
  `mesh_shard → eltwise kernel → mesh_gather → to_host` matches torch
  (`test_mesh_compute_roundtrip_1x2`).

### C5: the all_gather kernel — precise scope

Transcribed from `D2MAllGatherRewriter` (`TTIRToD2M.cpp:3775-4052`), the
authoritative implementation. The `ccl.d2m` sketch is a rough 1x8 approximation
and is **wrong/incomplete** in three ways: it has a trailing `semaphore_set`
the rewriter does not (the start semaphore is reset at host scope via the
auto-emitted `reset_global_semaphore`); it omits the **fabric connection
config**; and it omits the **input/output view-grid reblocking**.

**Algorithm (general).** Given `cluster_axis` (the mesh axis being gathered
over), `all_gather_dim` (the tensor dim that grows), ring topology:
- `num_devices = meshShape[cluster_axis]`; `num_cores = 2*num_links` (ring,
  so 2 for `num_links=1`).
- `workerCoreSplitDim` = first input dim whose physical (tiled) size is
  divisible by `num_cores` — work is split across `num_cores` cores along it.
- `inputViewGrid = [1]*rank; inputViewGrid[splitDim] *= num_cores`.
  `outputViewGrid = [1]*rank; outputViewGrid[splitDim] *= num_cores;
  outputViewGrid[all_gather_dim] *= num_devices`.
- Reblock input/output device tensors to those grids via `d2m.view_layout`
  with `d2m.ir.calculate_reblock_map(oldShape, newShape, ctx)` (the same helper
  `Layout.build_blocked_view` already uses).
- Emit a `d2m.generic` over `grid = inputViewGrid`, operands
  `ins(inputStream) outs(outputStream) additionalArgs(startSem, endSem)`,
  **with `fabricConnectionConfig =
  #ttcore.fabric_connection_config<noc_index = noc0, topology = ring,
  cluster_axis = <ca>, routing_mode = unidir_ring_torus, num_links = 1>`**, and
  empty block_factors/indexing_maps/iterator_types (explicit-datamovement form).

**Kernel body (the d2m.generic region):**
```
startDevice[d]      = (d == cluster_axis) ? 0          : mesh_position(d)
deviceMcastShape[d] = (d == cluster_axis) ? num_devices : 1
coreIndices         = [core_index(i) for i in range(rank)]
device_synchronize(startSem, startDevice, deviceMcastShape,
                   num_receivers = num_devices - 1, coreIndices)
inputIndices[i]     = core_index(splitDim) if i == splitDim else 0
load                = remote_load(inputStream, inputIndices)
# output index along the gather dim: this device's slot + local shard index
shardOffset         = outputViewGrid[all_gather_dim] // num_devices
outputIndices[i]    = (i == all_gather_dim)
                      ? mesh_position(cluster_axis) * shardOffset + inputIndices[i]
                      : inputIndices[i]
remote_store(outputStream, outputIndices, load,
             start_device=startDevice, device_mcast_shape=deviceMcastShape,
             semaphore=endSem, semaphore_indices=inputIndices)
semaphore_wait(endSem, num_devices - 1)        # no reset (legal in any thread)
yield(store)
```
Note: **no `semaphore_set` inside** — so this is legal in unified OR
datamovement form. The rewriter uses unified; the d2m-jit path should try
`@d2m.kernel(thread="datamovement")` first (it sidesteps SplitUnifiedThread and
matches our validated datamovement lowering), falling back to unified.

**Concrete 1x2 target** (`cluster_axis=1`, `all_gather_dim=0`, ring,
`num_links=1` → `num_devices=2`, `num_cores=2`):
- host full input `(256, 512)` → `mesh_shard [1,2] dims [0,1]` → `(256, 256)`
  per device.
- `splitDim`: tiled `(256,256)` → `(8,8)` tiles; `8 % 2 == 0` → `splitDim = 0`.
- `inputViewGrid = [2, 1]` (grid = 2x1); `outputViewGrid = [4, 1]`
  (split×devices on dim 0 = 2*2).
- per-device output `(512, 256)`; `mesh_gather`/`shard_to_full [1,2]` →
  `(512, 512)`.
- constants: `num_receivers = 1`, `deviceMcastShape = [1, 2]` on the cluster
  axis, `semaphore_wait(endSem, 1)`, `shardOffset = 4 // 2 = 2`.

**d2m-jit gaps to close (C5 tasks):**
- **C5.1 fabric config:** add a `fabric=` option to `@d2m.kernel` /
  `_emit_kernel_generic` that builds the `FabricConnectionConfigAttr` and passes
  it to `GenericOp(..., fabricConnectionConfig=...)` (the Python builder already
  accepts the kwarg). Small.
- **C5.2 view-grid reblock:** a host helper that emits `d2m.view_layout` with a
  `calculate_reblock_map` to the input/output view grids, returning a
  LazyTensor the kernel consumes. The existing `view`/`view_layout` don't take
  an explicit target grid; add a thin `reblock(lt, grid)` (or reuse
  `Layout.build_blocked_view` machinery). Medium.
- **C5.3 the kernel + wiring:** author the all_gather kernel (above) with 1x2
  constants, wire `mesh_shard(input) → reblock → all_gather kernel → reblock →
  mesh_gather → to_host`, PCC vs torch all_gather. Iterate on silicon — this is
  where cross-device `device_synchronize`/`semaphore_wait` correctness is first
  exercised end to end.
- **C5.4 (C4) generalize:** replace 1x2 literals with mesh-shape-derived values
  (kernel scalars / captures) so it scales to 1x8.

**Risk ranking:** C5.3 on-silicon cross-device sync (highest — untested
anywhere); C5.2 getting the reblock maps exactly right (the index math must
match the affine map the rewriter builds); C5.1 (low). The reblock + index math
are the subtle parts; the op vocabulary and mesh I/O are all validated.

**Open question:** whether to author the kernel against the reblocked *stream*
tensors (as the rewriter does, requiring C5.2) or whether a simpler grid layout
avoids the reblock for the 1x2 case. Worth a short spike: try the kernel without
the view-grid reblock first (grid = the natural sharded grid) and see if the
gather is still expressible — it may simplify C5.2 away for the first cut.

### C5 progress & findings (in-flight)

- **C5.1 (done): fabric config.** `@d2m.kernel(...)` accepts `fabric=` and
  `_emit_kernel_generic` passes it to `GenericOp(fabricConnectionConfig=...)`;
  `d2m.fabric_config(cluster_axis, topology="ring", num_links=1, ...)` builds the
  `#ttcore.fabric_connection_config<...>` attr. Verified: builds, no regression
  on non-CCL paths.
- **Kernel authored correctly.** Generating the 1x2 all_gather via the DSL
  (mesh_position / device_synchronize(num_receivers=1) / explicit-buffer
  remote_load / cross-device remote_store with `semaphore increment` /
  semaphore_wait, output index `dx*2 + cy`) produces a `d2m.generic` that is
  **structurally identical** to `D2MAllGatherRewriter`'s output (verified by
  diffing the pre-pipeline IR): same fabric config, grid 2x1, operands,
  additionalArgs, ops, and indices.
- **Reference lowers through our pipeline.** The rewriter's D2M IR
  (`ttir-to-d2m` of a 1x2 ttir.all_gather) lowers cleanly through `_build_pipeline`
  **with `config.use_split_unified_thread_v2 = True`** — produces the final
  `ttmetal.enqueue_program` with the FabricConnectionManager. So our backend
  pipeline is CCL-capable; the v1 split-unified-thread asserts (D1) and **v2 is
  required** for the all_gather.
- **🔴 The remaining blocker: the generic's operands must be `view_layout`
  reblocked *streams*, not direct `to_layout` buffers.** The "spike" (choosing
  `block_shape=[1,2]` so `to_layout` yields the 2x1 / 4x1 grids directly) makes
  the *generic* identical, but skips the `d2m.view_layout` reblock the rewriter
  emits (input `2x2x1x1 → 2x1x1x2`, output `4x2x1x1 → 4x1x1x2`, via
  `calculate_reblock_map`). Without those stream views the downstream fabric/DMA
  lowering fails (`D2MToTTKernel getFabricConnectionManager`: no fabric
  `dma_write` found in the func). So **C5.2 (the reblock) is required after all**
  — the streams carry the buffer↔stream relationship the DMA lowering needs.

**Remaining C5 work (precise):**
1. A `reblock(lt, grid)` host helper: `ViewLayoutOp` with
   `d2m.ir.calculate_reblock_map(old_device_shape, new_device_shape, ctx)` to the
   stream grid. Input: `mesh_shard → L(grid=[2,2]) → reblock([2,1])`. Output:
   `empty(L(grid=[4,2])) → reblock([4,1])`.
2. Thread the reblocked-view output back through `mesh_gather` / `to_host`
   (the generic result is a stream view; `_emit_returns_and_finalise` must
   un-view it before `from_device` + `shard_to_full` — the view↔buffer + mesh
   interaction is the fiddly part).
3. Enable `config.use_split_unified_thread_v2` for the all_gather (or detect a
   fabric kernel and set it).
4. Run on a healthy 2-chip mesh and PCC-check vs the reference all_gather
   (`hstack(vstack(in[:,0:64], in[:,64:128]), ...)` for the 1x2 256x512 case).

**Env note:** repeated multi-device open/close put the dev box's chip into an
`ARC startup` firmware error; on-device runs need a `tt-smi` reset.
