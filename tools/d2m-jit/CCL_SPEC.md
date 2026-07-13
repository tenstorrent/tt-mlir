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
Note: **no `semaphore_set` inside** — so it works in the unified form that is
now the only `@d2m.kernel` form. The backend splits the unified generic and
pins the `device_synchronize` barrier to a single datamovement thread
(ScheduleDMA), so the all_gather lowers and runs correctly. (The explicit
`thread="datamovement"` form has been removed.)

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

### C5.2 done: `reblock` implemented + threaded through the gather path

- `d2m.reblock(lt, grid)` emits `d2m.view_layout` with
  `d2m.ir.calculate_reblock_map` to a stream grid; it picks `block_shape` so the
  new layout's blocked grid equals `grid` (self-consistent, so the later
  device-view/to_host is a no-op). Verified: produces the same `2x2x1x1 →
  2x1x1x2` view + `#map` as the rewriter.
- Threaded: `mesh_shard(full, L(grid=[2,2])) → reblock([2,1]) → all_gather
  kernel → reblock'd output → mesh_gather → to_host`. Kernel-output rebinding
  now clears `is_view` (a reblocked stream becomes a real result once written).
- The full all_gather **builds** with the right structure (reblocked streams,
  fabric config, kernel body).

**🔴 Still not lowering — narrowed to IR-shape deltas vs the rewriter.** Diffing
my pre-pipeline IR against `D2MAllGatherRewriter`'s output (the reference lowers
cleanly through our pipeline with split-v2), the remaining differences are:
  1. **Output buffer shape.** Rewriter: plain `empty(128x64)` → `to_layout`
     (4x2) → `view_layout` (4x1). Mine: `d2m.empty` directly at the device grid
     (4x2) → `reblock`, skipping the plain-tensor + to_layout step.
  2. **Load buffer op.** Rewriter uses `tensor.empty` for the per-core load
     buffer; my in-kernel `empty([1,2])` emits `d2m.empty`.
  3. **No `d2m.yield`.** The rewriter yields the `remote_store` result; d2m-jit
     kernels end without a yield (works for non-CCL, maybe not for the fabric
     path).
  4. **Output index op.** Rewriter emits `affine.apply` (`dx*2+cy`); mine emits
     `arith.muli/addi` (equivalent value).
  The crash is `D2MToTTKernel getFabricConnectionManager` (no fabric `dma_write`
  found in the mesh_position's func) — one of the above changes the
  thread-split grouping or the remote_store→dma_write lowering so the fabric
  write isn't where the mesh_position rewriter expects it.

**Next:** reproduce the rewriter's exact IR shape (likely #1 output-buffer
structure and/or #2 `tensor.empty` matter most), iterating against a healthy
device. The reblock + fabric + mesh plumbing are all in place; this is now a
"match the rewriter's IR byte-for-byte" exercise on the fabric path.

### C5.3 done: the all_gather LOWERS end-to-end

Root cause of the `getFabricConnectionManager` crash, found by diffing the
thread-split: my all_gather generic split into **two** datamovement threads;
the second got a `mesh_position` (from `device_synchronize`) but **no** fabric
`dma_write`, so the per-func fabric-manager walk didn't create an `fcm` for it.
The reference splits into **one** datamovement thread (mesh_position + fabric
write together).

**Fix (one line):** the in-kernel `empty([...])` op now emits **`tensor.empty`**
instead of `d2m.empty` (matching the rewriter's load buffer). The `d2m.empty`
scratch buffer was making the backend schedule the datamovement work onto a
second NOC thread; `tensor.empty` keeps the fabric chain on one thread. (The
`d2m.yield` and output-buffer-structure deltas turned out **not** to matter —
`tensor.empty` alone fixes lowering.)

With that, the full 1x2 all_gather — `mesh_shard → reblock → all_gather kernel
(unified + `config.use_split_unified_thread_v2 = True`) → reblock → mesh_gather
→` build+lower — **compiles to a ttmetal flatbuffer** (program mesh shape
`(1,2)`, fabric `enqueue_program` with a single datamovement kernel). Test:
`test_all_gather_1x2_lowers` (device/desc-gated, build+lower only).

**Remaining (C5.4 / validation):**
- **🔴 On-device execution blocked by fabric topology on this box.** The
  all_gather compiles + lowers, but `to_host` (full execute) fails at
  `runtime.submit` with `FATAL: Backward direction is missing on mesh
  coordinate MeshCoordinate([0, 0])`. The all_gather requires **ring** topology
  (D2MAllGatherRewriter asserts ring-only; fabric config uses
  `unidir_ring_torus`), but the auto-discovered 1x2 (2-chip n300) fabric mesh
  doesn't provide the ring's backward link. This is a **fabric/hardware
  constraint, not a compiler/DSL bug** — the non-fabric mesh_shard round-trips
  (host-side distribute/gather) execute fine on this box; only the inter-device
  fabric path needs a real ring. Validating execution needs an 8-chip ring (the
  original 1x8 target) or a 2-chip box wired/configured as a bidirectional
  ring. Expected result for the 1x2 256x512 case (when runnable):
  `hstack(vstack(in[:,0:64], in[:,64:128]), <same>)`.
- **Auto-enable split-v2 for fabric kernels** (currently the caller sets
  `config.use_split_unified_thread_v2`); detect a `fabric=`/CCL kernel and set
  it automatically.
- **Generalise the 1x2 literals** to mesh-shape-derived values (C5.4) for 1x8.

### Line config on a 1x2 mesh (n300) — progress

d2m **does** support a line config: `Topology` has `linear` and `RoutingMode`
has `bidir_line_mesh` (vs ring / `unidir_ring_torus`), both exposed on the
`#ttcore.fabric_connection_config` attr. Because the DSL hand-builds the kernel
(not via `D2MAllGatherRewriter`, which hard-asserts ring-only), we can target
the 2-chip n300's bidirectional line. Two changes vs the ring version:
- `d2m.mesh((1,2), topology=("linear","linear"))` and
  `fabric_config(cluster_axis=1, topology="linear", routing="bidir_line_mesh")`.
- **One core per device**, not two: a line uses `num_cores = num_links*1 = 1`
  (ring uses `*2`). So `grid=(1,1)`, the whole per-device shard on one core
  (`buf = empty([2,2])`, `remote_load(buf, in0, [0,0])`), output index
  `out_i = mesh_position(1)` (shardOffset 1), no work-split reblock.

Progress on the 1x2 box (each step cleared the previous error):
1. ring topology → `FATAL: Backward direction is missing` (2-chip has no ring
   backward link). Fixed by linear topology.
2. 2-core kernel → `FATAL: Number of cores (2) exceeds routing planes (1)`.
   Fixed by the single-core (grid 1x1) line kernel.
3. single-core line kernel → **compiles, lowers, and passes fabric init**, but
   **execution hangs** (deadlock, killed at timeout). The hand-written
   cross-device sync (device_synchronize / mcast remote_store + semaphore_wait,
   copied from the ring rewriter) does not complete on a 2-device line.

**🔴 Remaining: the line all_gather sync protocol.** There is no reference for
a *line* all_gather (the rewriter is ring-only), so the device_synchronize /
semaphore mcast protocol for a 2-device bidirectional line needs to be worked
out (num_receivers, mcast start/shape, wait values, and whether a 2-device line
even forms the expected mcast group). This is genuine cross-device-protocol
work, not a DSL gap — the DSL can express it; the protocol values are unknown.
The dev box's chip also wedges (ARC `0xdeadc0de`) on repeated fabric runs,
needing `tt-smi -r` (sometimes twice), which makes iteration slow.

### ✅ RESOLVED: line all_gather works e2e via `D2MAllGatherRewriter`

The 1x2 line all_gather now passes through the *rewriter* (not just the DSL):
`test/python/golden/d2m/test_allgather.py` with `mesh_shape=(1,2)` and
`mesh-topology=linear,linear` passes all 8 cases (gather dims 0/1, 128x128 &
256x256, bf16 & f32) with `pcc=1.0` on the n300. Four changes:

1. **`D2MAllGatherRewriter` accepts linear topology** (`TTIRToD2M.cpp`): the
   ring-only assert now allows `Linear`, and routing mode is selected by
   topology (`Ring → UnidirRingTorus`, `Linear → BidirLineMesh`). `num_cores`
   already yields 1 for non-ring (the single-core line kernel).
2. **Fabric line-mcast routing fix** (`experimental_fabric_topology_info.h`,
   `get_line_regions`): the "remove my_idx from endpoints" remap moved `my_idx`
   itself, which on a 2-device range *inverts* the send direction for the
   end-of-line sender (it sent forward into its non-existent forward link, so
   the increment never arrived → deadlock). The forward/backward counts already
   exclude self, so the remap is unnecessary for the non-wrapped (line) case and
   is now confined to the wrapped/torus branch. (`get_ring_regions` already does
   this correctly by shrinking the range endpoints, not `my_idx`.)
3. **Conditional direction asserts** (`experimental_fabric_1d_routing.h`,
   `get_mcast_params_line`): edge devices legitimately have one direction =
   `COUNT`, so the `fwd_dir/bwd_dir != COUNT` asserts are gated by the
   corresponding `range != 0` (would otherwise trip under watcher).
4. **End-semaphore off-by-one** (`TTIRToD2M.cpp`): `remote_store` increments
   `endSemaphore` on every device in the mcast range *including the sender* (a
   local self-inc for the shard written locally), so each device receives
   `num_devices` increments (`num_devices - 1` remote + 1 self), but the wait
   targeted `num_devices - 1`. With an exact-equality `semaphore_wait` the count
   overshoots by one and deadlocks. The end-wait target is now `num_devices`.
   (The `device_synchronize` start-barrier correctly stays `num_devices - 1` —
   it has no local self-inc.)

**🟡 Caveat — ring path not re-validated on hardware.** Fix #4 changes the
end-wait value on the *shared* all_gather path, so it also affects the ring
config (e.g. 1x8). The same local self-inc happens for ring, so `num_devices` is
believed correct there too (`num_devices - 1` was a latent off-by-one the ring
path never exercised on-device — `(1,8)` is filtered out on this 2-chip n300).
**This must be validated on an ≥8-chip ring system (CI) before merge.**

### d2m-jit DSL port of the all_gather (`test_all_gather_1x2_roundtrip`)

`test/d2m-jit/test_mesh.py::test_all_gather_1x2_roundtrip` hand-builds the same
1x2 line all_gather via the DSL (the rewriter mirror). Two things surfaced:

1. **✅ Fixed: datamovement-thread segfault.** A `@d2m.kernel(thread=
   "datamovement")` CCL kernel emits `remote_load`/`remote_store` in *implicit
   (local-buffer)* form (no CB — the unified path's `SplitUnifiedThread` is what
   inserts CBs). `D2MLowerLoadStoreOpsToDMA` only handled explicit-CB form and
   segfaulted on `getCbType().getUnderlyingAs<MemRefType>()` (null `CBType`).
   Both `D2MLowerRemoteStoreRewritePattern` and `D2MLowerRemoteLoadRewritePattern`
   now lower the implicit form directly to `dma_write`/`dma_read` on the local
   buffer (no CB reserve/wait/push/pop). `datamovement` now lowers + reaches
   execution. (There was no prior datamovement-through-full-pipeline test;
   `ccl_all_gather.py` only register-device + verify + prints IR.)

2. **🔴 Open: DSL kernel device-hangs at execution.** With the segfault fixed,
   the hand-built kernel deadlocks on the 2-device line on **both**
   `datamovement` and `unified`+split-v2 (103% host CPU busy-poll, no compiler
   running), while the **rewriter** path passes e2e with the same protocol
   values (`num_receivers = 1`, end-wait `num_devices = 2`, single-core line,
   linear/bidir_line_mesh). So it is *not* the end-wait off-by-one. Suspected: a
   structural / semaphore-init discrepancy vs the rewriter.

   **D2M IR diff (done).** Dumped the rewriter's post-`TTIRToD2M` D2M IR and the
   DSL kernel's constructed D2M IR and compared. The sync ops are identical
   (`device_synchronize` numReceivers=1, mcast `[1,2]`; `semaphore_wait %end, 2`;
   matching `start_device`/`mcast_shape`/`semaphore_indices`). Two divergences
   found — **both addressed, neither cleared the hang**:
   1. *Store source.* The rewriter threads `remote_load`'s result into
      `remote_store` (`%16 = remote_load …; remote_store … %16`); the DSL kernel
      passed the pre-load buffer (`remote_load(buf,…); remote_store(…, buf)`),
      leaving the load result dead and dropping the load→store dependency. Fixed
      in the test (`buf = remote_load(buf, in0, [0,0])`). Real correctness fix,
      but the hang persists.
   2. *reset_global_semaphore.* The DSL emits two `reset_global_semaphore` after
      the generic (builder.py, for backing-buffer deadness); the rewriter emits
      none. Ruled out: suppressing them (env-gated patch, reverted) did not clear
      the hang.
   The cause is therefore **below the D2M level**.

   **TTKernel IR diff (done, print-ir-after-all both paths).** Compared the
   post-`ConvertD2MToTTKernel` kernels. The **cross-device sync ops are
   byte-for-byte identical**: barrier `fabric_mcast_sem_inc` + `semaphore_wait(.,
   1)`; end `fabric_mcast_sem_inc` + local `noc_semaphore_inc` +
   `semaphore_wait(., 2)`. (An apparent "extra wait" in the rewriter was a
   print-after-all duplication artifact — one end wait each.) So the hang is
   **not** a sync-op/value difference.

   The real divergence is **structural data-path**: the rewriter's all_gather
   lowers to a *CB-pipelined multi-kernel* — a `compute` kernel that `tilize`s
   (`cb_reserve`/`tilize_block`/`cb_push`) feeding a `datamovement` kernel that
   consumes via `cb_wait_front`/`cb_pop_front` around the fabric write, plus an
   aux datamovement kernel. The DSL `datamovement` kernel is a single thread
   doing direct `dma_read`→`dma_write` on the local buffer (the implicit-form
   lowering), no tilize / no CB producer-consumer. The `unified`+v2 DSL variant
   *does* get a CB/compute split yet still hangs, so the CB pipeline alone isn't
   the fix either — the splitter's exact fabric/CB/buffer scaffolding differs
   from the hand kernel in a way static IR can't pin down.

   **Static analysis is exhausted** (sync ops match; data-path structure differs
   but both internally consistent).

   **On-device `dprint` attempted (inconclusive).** Instrumented the kernel with
   stage markers (S0 before sync … S4 after the end wait) via a dynamically
   registered `dprint` syntax op (`d2m.PrintOp` → `ttkernel::dprint` →
   `DEVICE_PRINT`). The markers compile into the kernel and the DPRINT server
   attaches on both devices, but **no marker is emitted at all — not even S0**
   (the first op, before any sync). `DEVICE_PRINT` auto-drains via the host
   server (no manual flush needed), so the absence of even S0 means either (a)
   the datamovement thread never reaches its first instruction — the hang is at
   program launch / dispatch, before the kernel body — or (b) the first
   `DEVICE_PRINT` itself stalls waiting on the server's start-magic
   (dprint.h: "device code stalls waiting on the host to flush it"), i.e. dprint
   perturbs the hang. Either way it didn't localise the stall.

   **Watcher localised it (done, `TT_METAL_WATCHER=1` with
   `TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1 TT_METAL_WATCHER_DISABLE_STACK_USAGE=1`
   — full/`DUMP_ALL` overflows the idle-erisc ELF, so disable the heavy
   features).** On the live hang both devices' worker core (0,0) sit at the same
   waypoints: `NTW, NWFD, W, W, W` — BRISC at `NTW` (firmware idle, normal),
   NCRISC (the datamovement thread running the kernel) at `NWFD`, the three
   TRISCs idle. `NWFD` = `noc_async_writes_flushed` *completed* (dataflow_api.h),
   and the kernel's `experimental::semaphore_wait` (experimental_semaphore.h) is
   an un-instrumented `while (*sem != val)` spin, so the last waypoint stays
   `NWFD` from the `fabric_mcast_sem_inc` just before it.

   **Conclusion: the hang is the END `semaphore_wait` (`while (sem != 2)`), on
   both devices** — *not* launch/dispatch (the dprint ambiguity is resolved: the
   kernel runs), *not* the start barrier (it's past `device_synchronize` and the
   data write — `NWFD` means the NOC writes flushed). The end semaphore never
   reaches exactly `num_devices = 2`. Since the data fabric write *does* land
   (PCC=1.0 in the sync-disabled run), fabric delivery works for writes; what
   fails is the end-semaphore increment count converging to 2 on the 2-device
   line.

   **Discriminating tests (done) — the end semaphore stays at 0.** Ran the DSL
   kernel varying the end-wait value and `device_synchronize`:

   | device_synchronize | end `semaphore_wait` | result |
   | --- | --- | --- |
   | removed | removed | **PASS, PCC=1.0** |
   | removed | `wait(1)` | hang |
   | removed | `wait(2)` | hang |
   | present | `wait(1)` | hang |
   | present | `wait(2)` | hang |

   The presence of the end `semaphore_wait` alone flips PASS→hang, *independent
   of `device_synchronize`* — so the hang is the **end wait**, not the start
   barrier (correcting the earlier `NWFD`-ambiguity read: `noc_async_writes_flushed`
   runs after *every* fabric op, so that waypoint didn't prove "past the data
   write"). And it hangs at **both `wait(1)` and `wait(2)`**, so the end semaphore
   never reaches even 1 — effectively **stuck at 0**.

   So `remote_store`'s `SemaphoreInc` — which the TTKernel IR shows lowering to a
   local `noc_semaphore_inc` (self) **plus** `fabric_mcast_sem_inc` (peer) — does
   **not** increment the semaphore that `semaphore_wait` polls: *neither* the
   local self-inc *nor* the remote arrives at the wait's slot. This is a
   **semaphore-plumbing bug in the DSL path**, not a wait-value error. (The
   `num_devices` end-wait value is correct — it is verified on the rewriter, which
   reaches exactly `num_devices`; the DSL fails because *no* increment lands, so
   no wait value would pass.)

   The clean rows (device_synchronize *removed*, so no start-barrier confound)
   hang at **both** `wait(1)` and `wait(2)`, so the end semaphore is genuinely
   **stuck at 0** — not undershoot-at-2 / overshoot-at-1.

   **Slot trace (done, no handle mismatch).** Dumped the lowered EmitC for the
   datamovement kernel and traced the end-semaphore ops:
   - end `semaphore_wait` polls `reinterpret_cast<uint32_t*>(%11)` where `%11 =
     get_compile_time_arg_val(1)` = end_sem's L1 address (LOCAL core pointer).
   - end `fabric_mcast_sem_inc` and the local `noc_semaphore_inc` both write
     `noc_addr = get_noc_unicast_addr(convert_logical_x/y_to_translated(0),
     %11)` — i.e. the **same** semaphore `%11`, at translated logical core (0,0).
   - the local self-inc's in-range gate folds to `mesh_pos(axis1) <= 1`, always
     true, so it *does* execute; and the watcher confirms the kernel runs on
     worker (0,0).

   So inc and wait reference the **same semaphore and (nominally) the same
   core**, yet the sem stays 0. The local `noc_semaphore_inc`'s NOC write isn't
   landing at the L1 slot the `semaphore_wait` polls. Prime suspect: a
   **coordinate-space mismatch** — the inc addresses the core via
   `convert_logical_to_translated(0)` (a fabric/translated coord) while the wait
   reads the core's local L1 pointer directly; if translated(0) ≠ the kernel's
   physical/virtual core, the NOC self-inc lands on a different core's L1. The
   rewriter avoids this (its single-thread→split CB structure / coord handling
   differs).

   **Next step:** read the end-sem L1 value on the hang to confirm 0, and compare
   the *physical* core the `noc_semaphore_inc` resolves to against the core the
   `semaphore_wait` reads (the logical→translated vs logical→virtual coordinate
   path for a self-targeted global-semaphore inc). This is deep NOC-coordinate /
   tt-metal plumbing work. The rewriter path works e2e and is the supported route
   today; the DSL port (`test_all_gather_1x2_roundtrip`) stays skipped pending
   this follow-up.

   The test is `@pytest.mark.skip`ped (a device-hang can't be `xfail`'d — it
   would time out the suite) with the correct algorithm + the load→store fix
   captured for when the hang is resolved.

   **Kernel-C++ diff (done) — the datamovement kernel is identical; the
   divergence is program/runtime-level.** Cleared the program-kernel cache, ran
   the golden rewriter cold to get its CCL datamovement kernel C++
   (`kernel_includes.hpp`), and diffed it against the DSL's. The end-semaphore
   handling is the same: local `noc_semaphore_inc(addr, 1)` to the running core
   + `fabric_mcast_sem_inc(..., 1)` to the peer, then
   `experimental::semaphore_wait(reinterpret_cast(ct_arg), 2)`; the ct_arg→end_sem
   mapping is correct in both. The *only* difference was the self-inc's Y coord:
   rewriter uses `convert_logical_y_to_translated(get_absolute_logical_y())`
   (running core), DSL used `...(0)` (literal, from `semaphore_indices=[0,0]`).
   **Tested the fix** (`semaphore_indices=[core_index(0), 0]`, making the DSL's
   coord dynamic == the rewriter's): **still hangs**. (For grid 1x1 on row 0 the
   two coincide anyway — confirming the kernel runs on (0,0) and the self-inc
   already targets the correct core.)

   So with the coord fix the datamovement kernel C++ is byte-equivalent to the
   rewriter's, yet the DSL hangs and the rewriter passes → **the bug is not in
   the datamovement kernel codegen.** It is in the surrounding *program/runtime*:
   the rewriter lowers to a compute(`tilize`)+datamovement CB-pipelined
   multi-kernel program, while the DSL is datamovement-only; and the two use
   different runtime program-assembly / global-semaphore allocation+init / fabric
   setup. The end sem staying at 0 (local self-inc not visible to the local wait
   despite identical code + same core) points at the **runtime semaphore
   allocation / device-side zero-init / kernel placement**, not the kernel.

   **Next step:** compare the two *programs* (not the datamovement kernel): the
   kernel list (rewriter has a compute kernel the DSL lacks), the global-semaphore
   allocation + whether it is zero-initialised on device before launch, the
   fabric program config, and the per-kernel runtime args (the actual semaphore
   L1 addresses passed). i.e. move from kernel-C++ diffing to runtime/program
   diffing (flatbuffer / `_execute` vs the ttmetal runtime).

   **Flatbuffer program diff (done) — the all_gather program is missing a
   compute kernel.** Dumped both flatbuffers (`Binary.as_json()`) and compared
   the all_gather `EnqueueProgramCommand` (the one carrying the
   `fabric_connection_config`, identical in both: noc0 / Linear /
   bidir_line_mesh / num_links 1). Everything matches except the **kernel list**:
   - **rewriter all_gather program: 2 kernels** — a `NocConfig` (datamovement) on
     core (0,0) **+ a `ComputeConfig` (tilize) kernel** on core (0,0).
   - **DSL all_gather program: 1 kernel** — only the `NocConfig` datamovement on
     core (0,0).
   Both datamovement kernels: same core (0,0, size 1x1), same `Noc0`, same
   `arg_refs = [BufferRef, BufferRef, GlobalSemaphoreRef, GlobalSemaphoreRef]`,
   1 cb. So the *only* structural difference in the all_gather program is the
   presence of the compute kernel.

   The rewriter gets the compute kernel because its unified generic carries a
   `tilize` (row-major→tiled data path), which `SplitUnifiedThread` peels into a
   compute thread; the rewriter datamovement kernel then consumes the tilized
   data via CBs (`cb_wait_front`). The DSL authors a `datamovement`-only kernel
   on pre-tiled data (direct `dma_read`), so no compute kernel and no CB
   producer. **This correlates exactly with the hang**: compute-kernel-present
   (rewriter) works; compute-kernel-absent (DSL) hangs — and the `unified`+v2 DSL
   variant also hung because its pure-datamovement body produces no compute op to
   split out either.

   **Semaphore-creation check (done) — identical.** `CreateGlobalSemaphoreCommand`
   in both flatbuffers: `initial_value = 0`, `core_range_set = 8x8` (full worker
   grid, includes core (0,0)). So the semaphores are zero-initialised on the core
   the kernel runs on in both paths — this rules out a semaphore init / core-range
   cause and confirms the **compute kernel presence is the SOLE structural
   difference** in the all_gather program.

   **🎯 Strong hypothesis:** a tt-metal fabric/CCL program (or just the
   program-launch / NOC-init / go-signal path on a Tensix core) needs the compute
   kernel present; without it the datamovement kernel's local `noc_semaphore_inc`
   never becomes visible to its own `semaphore_wait` (end sem stuck at 0).
   **Next test:** make the DSL all_gather program include a compute kernel on the
   same core — e.g. route the data through a trivial tilize/compute (so the
   generic splits into compute+datamovement like the rewriter), or add a no-op
   compute thread — and see if the end semaphore then advances. If that fixes it,
   the DSL CCL path needs to emit the compute-kernel half, not just datamovement.

   **Tried it (compile failure — a *second* gap).** Authored the all_gather as a
   `unified` kernel with a compute op in the data path
   (`buf = remote_load(...); buf = sigmoid(buf); remote_store(..., buf)`,
   `use_split_unified_thread_v2=True`) so `SplitUnifiedThread` would emit a
   compute kernel alongside the datamovement one. It **fails to lower**:
   `getFabricConnectionManager: Assertion 'fcm' failed, Expected fabric
   connection manager op` (`D2MToTTKernel.cpp:174`). So when a unified CCL kernel
   is split into datamovement+compute, the fabric-connection-manager setup is not
   propagated to the split datamovement thread — the d2m-jit fabric path doesn't
   support a compute thread coexisting with the fabric/CCL datamovement thread.

   So the DSL CCL path has two layered gaps: (a) it can't currently emit the
   compute-kernel half that the rewriter's all_gather program has (fcm assertion
   on the unified+compute+CCL split), and (b) the datamovement-only program it
   *can* emit hangs (end sem stuck at 0). Both are genuine d2m-jit
   fabric/SplitUnifiedThread infrastructure work. The rewriter path remains the
   supported, working route; the DSL port stays skipped. A focused fix would
   start by teaching the fabric-connection-manager setup to survive the
   compute/datamovement split (gap a), then re-test whether the resulting
   2-kernel program clears the hang (gap b / the hypothesis).

### ✅✅ ROOT CAUSE FOUND & FIXED — `_execute` never enabled the device fabric

The hang was **not** in the kernel, the program, or the compute-kernel split — it
was the **runtime**. The golden harness opens the mesh device *after* calling
`runtime.set_fabric_config(FabricConfig.FABRIC_1D_RING)` (conftest.py); the
d2m-jit `_execute` opened the mesh device with **no `set_fabric_config` call at
all**, so the device fabric defaulted to `DISABLED`. With the fabric disabled,
the program's cross-device fabric ops (the `device_synchronize` mcast sem-inc,
the `remote_store` fabric write, and the fabric semaphore increments) **silently
no-op**, so the end semaphore is never incremented across devices and the kernel
spins forever on `semaphore_wait` (end sem stuck at 0 — exactly what the watcher
and the wait-value sweep showed). The "compute kernel present ⇒ works" was a red
herring (the rewriter happened to also be on the fabric-enabled golden runtime).

**Fix** (`tools/d2m-jit/_src/builder.py`):
- track `_Builder._fabric_used` (set when a kernel is invoked with `fabric=`),
- in `_execute`, if a fabric kernel was used, `set_fabric_config(FABRIC_1D_RING)`
  before `open_mesh_device` and reset to `DISABLED` after `close_mesh_device`.

With this, `test_all_gather_1x2_roundtrip` runs end-to-end and matches torch
(maxdiff ~2e-3) — the test is now **unskipped and passing**. The kernel uses
`semaphore_indices=[core_index(0), 0]` (matching the rewriter's running-core
self-inc address, vs the earlier literal `[0,0]`).

Secondary fix kept (`D2MToTTKernelPass.cpp`): the fabric-connection-manager
insertion now detects fabric *semaphore* ops (`DeviceSynchronizeOp`,
`SemaphoreInc/SetOp` with a startDevice), not only fabric `DMAWriteOp`. This
closes gap (a) — a unified CCL+compute kernel now lowers past the
`getFabricConnectionManager` assertion (compile-validated). It does not affect
the datamovement-only path (that func already has a fabric write).
