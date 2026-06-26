# Porting all_gather_minimal_matmul_async to d2m-jit — design + status

Goal: port the concept of
`third_party/tt-metal/.../models/tt_dit/tests/models/wan2_2/test_all_gather_minimal_matmul_async.py`
into d2m-jit, starting small and scaling up.

## The reference op (what it computes)

`ttnn.experimental.all_gather_minimal_matmul_async(input, weight, ...)` is a
**fused all_gather + matmul** for tensor/sequence-parallel linears (Wan2.2 DiT
qkv / dense-attn / ff1 layers). On a 2D mesh (`sp_axis`, `tp_axis`):

- `input` (activations `[M, K]`) is **sequence-parallel** sharded along M: device
  has `[M/sp, K]`.
- **all_gather over the SP axis** -> every device holds the full `[M, K]`.
- `weight` (`[K, N]`) is **tensor-parallel** sharded along N: device has `[K, N/tp]`.
- **matmul**: gathered `[M, K] @ [K, N/tp] = [M, N/tp]` per device.
- Fused/async: the all_gather is overlapped with the matmul (start matmul on
  chunks as they arrive); plus optional bias / gelu / addcmul / chunked-N.

Reference torch (test line 146): `torch_output = torch_input @ weight_input`.
Non-fused path also exists: `all_gather_async` then `minimal_matmul` as two ops.
Test shapes are large (M up to 115200, K=5120, N up to 4096) for a (2,4)/(4,8)
cluster.

## d2m-jit port plan (1D, this 1x4 Blackhole)

The box is a 1x4 ring (1D mesh), so the port is a 1D specialization: all_gather
activations over the 4-device ring, matmul with a weight. With no separate TP
axis the weight is **replicated** (each device computes the full `[M, N]` =
`gathered @ weight`); a real TP split needs a 2D mesh.

Start small ([M,K,N] a few tiles), scale up.

## Status

WORKING END TO END (small): `scratchpad/agmm4.py` -- all_gather (sp, 1x4) ->
TP-sharded matmul -> per-device N-slice -> mesh_gather to full `[M, N]`. PASS
(M=128, K=64, N=256, rel 0.0017). KEY INSIGHT: the weight must be **TP-sharded**
along N (`mesh_shard(W, shard_dims=[0,1], shard_shape=[1,N])`, each device its
N-slice -- the real AGMM form), NOT replicated. Replication is what failed:
`to_layout` (single-device) on a mesh -> buffer-size assert; `mesh_shard
shard_shape=[1,1]` -> compiler crash ("incorrect fold result type"). With a
TP-sharded weight (sharded operands, like the ring kernels) there is no
replication and the AG-kernel -> MM-kernel chaining works. The MM is row-parallel
(grid (M_tiles,1)); each core does `gathered_row @ weight_slice`; the activation
(gathered, per-device-full from the AG) and weight (TP shard) are both sharded.

(superseded) early notes:
- **Step 1 — 1x4 all_gather** (`scratchpad/agmm1.py`): each device mcasts its
  shard to all devices' gathered buffer (`remote_store(out,[dx,0],buf,
  device_mcast_shape=[1,N], semaphore=es)`, then `semaphore_wait(es,N-1)`); every
  device ends with the full gathered `[M,K]`. PASS (shape 128x128, rel 0.002).
  This is the all_gather building block (the existing test_ccl_all_gather is 1x8;
  this is the 1x4 single-core form). Gather layout: out buffer is `[1,K]` row
  blocks on an `[N,1]` grid (one block per device's row).

BLOCKERS (the matmul half):
- **matmul on a multi-device mesh fails** at runtime:
  `LOG_ASSERT meshBuffer.size() == tensorDesc.sizeBytes()`
  (runtime/lib/ttmetal/executor_utils.h:384). Repro `scratchpad/mmonly.py`: a
  row-parallel matmul (`grid=(N,1)`, each core `gathered_row[cy] @ weight`) on
  `d2m.mesh((1,4))` with `to_layout`-replicated operands. The existing matmul
  tests all use single-device `d2m.mesh((1,1))`; distributing/replicating matmul
  operands across a multi-device mesh has a buffer-descriptor mismatch that needs
  sorting (likely the replicated `to_layout` operand or the `mesh_gather` of the
  output on a `[1,N]` shard_shape). THIS is the first thing to fix to get a
  non-fused AG+MM.
- **on-device kernel chaining** (AG kernel output -> MM kernel input) hits the
  same buffer-size assert (`scratchpad/agmm2.py`/`agmm3.py`) -- but since MM-only
  fails too, the root cause is the multi-device-mesh matmul, not composition per
  se. Re-test chaining once the mesh matmul works.

## Fusing AG+MM into ONE kernel — attempted, blocked at framework level

`scratchpad/agf.py` — single-core fused kernel: AG mcasts the shard to all
devices' gathered `g` (`[N,1]` row-block grid), `semaphore_wait(N-1)`, then a
`static_range(N)` loop matmuls each gathered row `g[r] @ weight_slice` and stores
`out[r]`. Findings:

- **matmul reads `g` via `remote_load`**: COMPILES + RUNS (no hang/crash) but
  WRONG (rel ~0.54). Two root issues: (a) no compute<->DM barrier within one
  kernel -- the compute thread's `remote_load(g)` races the DM thread's fabric
  writes (`semaphore_wait` is DM-side; compute doesn't wait for it); (b) `g` is
  both the AG output (remote_store/DM) and the matmul input (remote_load/DM-read)
  -- split-v2 likely allocates these as DIFFERENT buffers, so the matmul reads an
  unwritten `g`. The non-fused version is correct only because the AG kernel fully
  completes (program boundary) before the MM kernel.
- **matmul reads `g` via `fabric_recv`** (`fabric_recv(g,[r]) @ weight`, the
  proper fabric-write consume that the ring uses): FAILS to legalize
  `memref.reinterpret_cast`. Same with `copy_(tmp, fabric_recv(g,[r]))` then
  matmul. `fabric_recv` composes only with the eltwise tile ops (the ring's
  `acc += r`), not with matmul or `copy_`.

IR INSIGHT (`scratchpad/agfdump2.py`): a single `@d2m.kernel` with AG +
`static_range(N)` matmuls does NOT become one generic -- it DECOMPOSES into the AG
generic + N matmul generics (one per unrolled iter), with `g` flowing between them
*within one program*. That is exactly why the non-fused two-kernel version is
correct (the AG kernel hits a program boundary / device_synchronize before the MM
kernel) while this is not (no ordering between the AG generic's fabric writes and
the matmul generics' reads of `g` inside one program). So "fuse into one kernel"
splits into two sub-problems: (i) one TRUE fused generic (DM gathers, compute
matmuls via fabric_recv) -- blocked by the fabric_recv->matmul legalization wall
above; or (ii) keep AG-generic -> matmul-generic but add a cross-generic
fabric-write->compute-read barrier within the program (analogous to the
InsertSpillAndScratch fusion barrier in [[d2m-ring-interleaved-fabric-hang]]).

CONCLUSION: fusing CCL all_gather + matmul in ONE d2m-jit kernel needs framework
support, one of:
1. make `fabric_recv`'s result a legal matmul operand (fix the
   `memref.reinterpret_cast` left after lowering a fabric_recv -> tile_matmul), OR
2. a compute<->DM barrier so a compute `remote_load` of a fabric-written operand
   is ordered after the DM writes (and the AG-output `g` and matmul-input `g`
   resolve to the SAME buffer).
This mirrors how the TTNN op solves it: dedicated fabric workers gather into
compute-readable buffers with explicit cross-core sync, rather than one core
doing both. The WORKING port stays the non-fused two-kernel
`test_all_gather_matmul.py`.

## Designs for the fused single-kernel AGMM (after the mesh matmul works)

Two reconciliation points for AG (writes `[1,K]` rows on an `[N,1]` grid) vs MM:
1. **Row-parallel MM** on the `[N,1]` grid (no reblock): `grid=(N,1)`, core cy
   does `gathered[cy] @ weight -> out[cy]`. Natural fit for the gather layout.
2. **Reblock** the `[N,1]` gathered to one `[M,K]` block then a normal matmul
   (hit a buffer mismatch via chaining; revisit).
Fusing AG+MM in ONE kernel additionally needs compute to read the
fabric-gathered buffer (fabric_recv-style) and, for a grid, the gathered data
available to each matmul core (cores don't share L1) -- this is what the TTNN op
solves with dedicated fabric workers + careful routing.

## Hardware note

The every-core-does-fabric ring all_reduce already converged at <=4 fabric cores
(2 eth channels between adjacent devices, num_links<=2; see
d2m-ccl-send-only-forwarding memory + test_all_reduce_grid.py). Scaling AGMM to
the reference's large shapes ultimately needs the dedicated-fabric-worker CCL
design, not every-core-fabric.

## Next steps

1. Fix the multi-device-mesh matmul buffer-descriptor mismatch (mmonly.py) ->
   non-fused AG+MM end to end on small shapes.
2. Scale M/K/N; add bias, gelu.
3. Fuse AG+MM into one kernel (row-parallel design #1).
4. Add async overlap; then a 2D-mesh TP weight split (needs a 2D mesh box).
