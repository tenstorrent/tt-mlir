# Fused 8×8 matmul + all_gather on a 1×2 mesh — design sketch

Scaling the fused matmul+all_gather from the current grid-`(1,1)` single-tile
proof (`test_matmul_all_gather_fused_1x2_roundtrip`) to a real matmul on the
full 8×8 worker grid, then all_gather the per-device result across a 1×2 mesh.

## The constraint that shapes everything

Fabric APIs — the cross-device send/mcast (`remote_store` with `startDevice`),
the `device_synchronize` barrier, and the fabric semaphore increments — can only
be issued from a **single Tensix core per mesh-link direction**. A 1×2 mesh
(`cluster_axis=1`) has one link direction, so *all* cross-device datamovement
must originate from one worker core.

The CCL machinery already encodes this: `num_cores = 2*num_links` (ring) or
`num_links` (line) *per direction*, so a 1×2 line with `num_links=1` already
runs the collective on a single core. The constraint is satisfied automatically
**as long as the fabric work lives in a grid-1 generic.**

### Consequence: this cannot be one generic

A `d2m.generic` runs its body on every core in its grid. A grid-8×8 generic runs
its body — *including any fabric op* — on all 64 cores, which violates the
single-fabric-core rule. The matmul wants 64 cores; the fabric wants exactly 1.
They therefore cannot share a per-core body.

So the "fused" op is **two generics chained in one program**:

1. **matmul** — grid 8×8, no fabric. Produces device *d*'s output shard `C_d`.
2. **all_gather** — grid 1×1 (the link core). Gathers `C_d` and fabric-sends it.

`C_d` is the matmul generic's output and the all_gather generic's input; that
data dependency is the fusion. Per-core fusion (one generic doing both, as in
the 1×1 proof) is impossible here once the matmul spans more than the link core.

This is *not* a regression from the 1×1 case — at grid `(1,1)` the matmul core
and the link core are the same core, so one generic worked. At 8×8 they are
disjoint sets, so the program must express two roles.

## Prior art: tt-metal's fused CCL ops (what to steal)

`third_party/.../ttnn/operations/experimental/ccl` has a family of fused
compute+CCL ops — `all_gather_matmul_async`, `all_gather_minimal_matmul_async`,
`matmul_reduce_scatter_async`, `llama_all_gather_matmul_async`,
`llama_reduce_scatter_matmul`. `matmul_reduce_scatter_async` is the direct dual
of what we want (matmul **then** a collective). How they actually do it:

- **One program on the full grid, not two sequential ops.** The program factory
  *composes* the matmul program factory (`matmul_multicore_reuse_mcast_2d/1d`)
  and the CCL program into a single `Program`. Both span the compute grid; cores
  are specialized by **compile-time role flags** baked into one kernel binary —
  `is_injector_core`, `is_output_writer`, `is_sink_core`
  (`all_gather_minimal_matmul_async/device/kernels/dm_in1_sender_out.cpp`). Same
  kernel everywhere, branch on role.

- **A "fused-op signaler" overlaps compute and communication.**
  `MatmulFusedOpSignaler` / `AllGatherFusedOpSignaler`
  (`ttnn/operations/ccl/ccl_op_fusion.hpp`) carry the *receiver cores* and
  *per-core signal semaphores* of the partner op. For matmul→CCL
  (`matmul_reduce_scatter`), the matmul kernel **increments a semaphore on the
  CCL receiver cores as each output block is produced**, so the collective
  streams that block over the fabric while the matmul computes the next one —
  block-granular pipelining, not a barrier between phases.

- **Hand-off is through L1 circular buffers, not DRAM.** The CCL datamovement
  kernel pushes gathered/!produced tiles into the matmul's input CB and the
  compute kernel `cb_wait_front`s on it (`compute.cpp`); chunking is per K-block
  (`fused_receiver_utils.hpp`'s `compute_device_chunk_stats`). No DRAM round trip.

- **The single-fabric-endpoint rule is realized with a `fabric_mux` + injector
  core.** A dedicated fabric-mux router kernel owns the link; worker/injector
  cores connect to it over channels (`fabric_mux_connection_*` in the program
  factory) rather than each issuing raw fabric ops. So "one core per link
  direction" is satisfied by funneling traffic through the mux/injector while all
  64 cores still run matmul.

**What this means for us.** tt-metal validates that the *fully fused* form is one
program with per-core roles + a signaler + L1-CB overlap — richer than the
two-generic sketch above. But that pattern leans on hand-written per-core role
divergence and a host-built signaler, which the d2m `generic` model (one body
per grid, lowered) does not express natively. So:

- **Keep the two-generic structure as the d2m first cut** — it's the natural way
  to get matmul-on-64 + fabric-on-1 without per-core `scf.if`, and it reuses the
  working all_gather generic unchanged.
- **But take three cues now:** (1) prefer the **L1 gather** hand-off (Option 1
  below), not DRAM, matching their CB approach; (2) treat the eventual
  **compute/comm overlap** (matmul blocks signalling the fabric core as they land)
  as the optimization target — it's the d2m analog of `MatmulFusedOpSignaler`,
  and would need either the single-generic conditional-fabric form or a new
  cross-generic signalling primitive; (3) our grid-1 all_gather generic *is* the
  injector/mux core — the constraint maps cleanly.

## Worked example (per-shard / block-diagonal matmul)

Mirrors the 1×1 test's semantics, scaled up. Full tensors `A`, `B` are
column-sharded across the 2 devices; device *d* computes `C_d = A_d @ B_d` on its
shard, then all_gather stacks the per-device results.

| tensor | full | per-device shard | tile-grid on 8×8 |
|---|---|---|---|
| `A` | 256×512 | `A_d` 256×256 | input to matmul |
| `B` | 256×512 | `B_d` 256×256 | input to matmul |
| `C_d` | — | 256×256 | **8×8 tiles, 1 tile/core** |
| gathered | — | vstack(`C_0`,`C_1`) 512×256 | on every device |
| `mesh_gather` | 512×512 | — | col-concat of the two halves |

Each core `(cy,cx)` computes one 32×32 output tile `C_d[cy,cx] = Σ_k A_d[cy,k] @
B_d[k,cx]` — a K=8 reduction (a `1×8 @ 8×1` tile-block matmul per core). `C_d`
ends up distributed one tile per core across the 8×8 grid.

## The crux: handing `C_d` (on 64 cores) to the one fabric core

After the matmul, `C_d` lives in L1 distributed across 64 cores. The link core
must send *all* of `C_d` over the fabric. Two ways to bridge that:

### Option 1 — `reblock` + gathering `remote_load` (reuse today's machinery)

This is exactly what the existing 1×2 all_gather already does, just at larger
fan-in. `reblock` is a **metadata `view_layout` (no data movement)**: it
re-expresses the 8×8-sharded `C_d` as a single logical block addressable by the
link core. The link core's `remote_load` of that view then performs the actual
NOC reads, gathering all 64 tiles into its L1; it then fabric-sends.

```
C_d (grid 8×8, L1)  --reblock([link_grid])-->  view  --remote_load on link core-->  link L1  --remote_store(fabric)-->  peer
```

- **Pro:** no new machinery — the all_gather generic is structurally the current
  one, with its input being `reblock(matmul_out, link_grid)` instead of
  `reblock(mesh_shard(...), link_grid)`.
- **Con / hard limit:** the link core's L1 must hold the whole gathered shard.
  256×256 f32 = 256 KB; L1 is ~1.5 MB, so this size fits but leaves little room,
  and it does **not** scale to large `C_d`. Past L1 capacity this must stream in
  chunks (gather a slab, send it, repeat) — a loop in the all_gather generic.

### Option 2 — stage `C_d` to DRAM, fabric-stream from DRAM

Have the matmul write `C_d` to a DRAM tensor (interleaved/sharded). DRAM is
globally addressable, so the link core reads `C_d` from DRAM and fabric-writes to
the peer's DRAM — no 64→1 L1 gather, no L1-capacity wall. This is how tt-metal
CCL conventionally works (the EDM/fabric routers move DRAM↔DRAM).

- **Pro:** scales to any size; matches the native CCL model.
- **Con:** a DRAM round-trip for `C_d` (write after matmul, read in all_gather);
  extra bandwidth and latency vs. staying in L1.

**Recommendation:** start with **Option 1** for parity with the working 1×2
all_gather and to validate the two-generic structure end-to-end at 8×8 (256×256
shard fits L1). Move to **Option 2** (or chunked-streaming Option 1) once shard
sizes exceed L1 or when matching the production CCL path.

## Synchronization

Two independent ordering requirements:

1. **matmul-done → fabric-read.** The link core must not gather `C_d` until all
   64 matmul cores have written their tile. Because the matmul and all_gather are
   separate generics with a true tensor dependency (`C_d`), the pipeline already
   serializes them — the all_gather generic is scheduled after the matmul
   generic completes. No explicit cross-generic barrier is needed; this is
   ordinary d2m generic sequencing. (Within Option 2, the DRAM write/read
   dependency enforces it; within Option 1, the all_gather's `remote_load`
   depends on the matmul output tensor.)

2. **cross-device barrier.** Inside the all_gather generic, `device_synchronize`
   gates this device's fabric send on the peer having started — identical to the
   current all_gather. The end-semaphore (`remote_store(semaphore=...)` +
   `semaphore_wait(end_sem, num_devices)`) signals completion, also unchanged.

The single-fabric-core constraint plus the semaphore-pinning work already landed
(`ScheduleDMA` keeps the barrier and explicit mutations on one DM thread) means
the all_gather generic needs no new sync machinery.

## Kernel sketches

**matmul generic** (grid 8×8, no fabric):

```python
@d2m.kernel
def matmul_shard(lhs, rhs, out, k_blocks):
    cy = core_index(0)
    cx = core_index(1)
    c = zeros([1, 1])
    for k in range(k_blocks):           # K-reduction, 8 k-tiles
        a = remote_load(lhs, [cy, k])   # A_d[cy, k]
        b = remote_load(rhs, [k, cx])   # B_d[k, cx]
        c += a @ b
    remote_store(out, [cy, cx], c)      # local store, no fabric
```

**all_gather generic** (grid 1×1 = link core, fabric — structurally today's):

```python
@d2m.kernel
def all_gather(in0, out0, start_sem, end_sem):
    dy = mesh_position(0); dx = mesh_position(1)
    cy = core_index(0); cx = core_index(1)
    device_synchronize(start_sem, start_device=[dy, 0], mcast_shape=[1, 2],
                       num_receivers=1, core_indices=[cy, cx])
    buf = empty([8, 8])                 # whole C_d shard (8×8 tiles) on link core
    buf = remote_load(buf, in0, [0, 0]) # gathers the 64 distributed tiles
    remote_store(out0, [dx, 0], buf, start_device=[dy, 0],
                 device_mcast_shape=[1, 2], semaphore=end_sem,
                 semaphore_indices=[cy, 0])
    semaphore_wait(end_sem, 2)
```

Driver (one program):

```python
d2m.mesh((1, 2), topology=("linear", "linear"))
a_s = d2m.mesh_shard(A, L_in, shard_dims=[0, 1], shard_shape=[1, 2])  # grid 8×8
b_s = d2m.mesh_shard(B, L_in, shard_dims=[0, 1], shard_shape=[1, 2])
c_d = d2m.empty(L_in)                              # 8×8 matmul output
matmul_shard(a_s, b_s, c_d, 8, grid=(8, 8))
in_s  = d2m.reblock(c_d, [1, 1])                   # view: gather to link core
out_s = d2m.reblock(d2m.empty(L_out), [2, 1])
all_gather(in_s, out_s, ss, es, grid=(1, 1),
           fabric=d2m.fabric_config(cluster_axis=1, topology="linear",
                                    routing="bidir_line_mesh"))
out = d2m.mesh_gather(out_s, shard_dims=[0, 1], shard_shape=[1, 2])
```

(`L_in` = 256×256 on grid 8×8; `L_out` = 512×256 on grid 2×1, the stacked shards.)

## Alternative: single fused generic with a conditional fabric core

It is *possible* to keep one grid-8×8 generic and gate the fabric ops to the link
core with `scf.if (cy==0 and cx==0)`. Core (0,0) would then: matmul its own tile,
local-gather the other 63 tiles (NOC `remote_load`s), fabric-send, while the
other 63 cores only matmul and signal a 64→1 local barrier when their tile is
ready.

- **Pro:** no DRAM round-trip; truly one generic.
- **Con:** needs (a) per-core control-flow divergence (`scf.if` on `core_index`
  in the kernel body — not currently exercised by the CCL path), (b) an explicit
  64→1 local barrier so the link core waits for every matmul tile, and (c) the
  same L1-capacity wall as Option 1.

### Probe results (2026-06): blocked in the current d2m model

Three probes (single device, minimal scale) drove this from "maybe" to "blocked":

1. **scf.if-gated fabric — WORKS (fixed).** `if core_index == 0 { device_synchronize }`
   in a multi-core generic lowered onto the *compute* thread until the split
   passes were taught to recurse into `scf.if` (commit "Recurse into scf.if when
   classifying thread-split ops"). After the fix the gated fabric op lands on a
   datamovement thread inside the guard. (a) is solved.
2. **Intra-generic cross-core gather of *input* — WORKS.** In a grid-(1,2)
   generic each core `remote_load`ed both shards `[0,0]`/`[0,1]` and summed them;
   runs on device, correct. A grid-indexed `remote_load` of another core's shard
   within one generic is fine.
3. **Intra-generic gather of *produced* data — BLOCKED.** The killer. A tensor
   used as **both** a per-core produce-target (`remote_store(out,[cy,cx],…)`) and
   a cross-core gather-source (`remote_load(out,[0,j])`) in one generic resolves
   to **two inconsistent memrefs** — the produce store loses its `#ttcore.shard`
   device layout and is rejected: `'d2m.remote_store' op memref/tensor must be
   remote (have a device layout)`. The 2→1 `semaphore_inc`/`wait` barrier itself
   was fine; the wall is the read+write dual-use of one operand. There is no
   `operand_alias` surface in the DSL, and no existing kernel re-reads an output.

So the single fused generic is **not expressible in the current d2m operand
model**: the matmul output can't be both written per-core and gathered cross-core
inside the same generic. Unblocking it needs real compiler work — operand
aliasing / a consistent device layout for a read+write operand, and verifying the
downstream allocate/DMA lowering accepts a cross-core RAW on a generic operand.

**Therefore: use the two-generic structure** (matmul generic → all_gather
generic). There the matmul output is a clean *output* of generic 1 and a clean
*input* to generic 2 — no intra-generic read+write of one operand, so the wall
disappears, and the validated gather (probe 2) + scf.if fabric gating (probe 1)
still apply. The single-generic form is a future optimization gated on the d2m
operand-model work above.

## Open questions / risks

- **L1 capacity** is the binding constraint for the in-L1 gather (Option 1). Pin
  down the largest `C_d` shard that fits on the link core alongside the CCL
  scratch/semaphore buffers; beyond that, chunked streaming or DRAM (Option 2).
- **Does `reblock([1,1])` over an 8×8-sharded L1 tensor lower to a correct
  gathering `remote_load`?** The 2×2→1×1 case works today; confirm the
  `reblock_map` / view addressing is correct at 8×8 fan-in (64 source cores).
- **Matmul K-reduction on f32** routes through SFPU fp19 (≈1% error per the
  existing matmul tests) — use PCC, not tight abs-diff, for the larger reduction.
- **Generic sequencing guarantee:** verify the scheduler really orders the
  all_gather generic strictly after the matmul generic via the `C_d` dependency
  (no overlap that lets the link core gather a half-written shard).

## Incremental plan

1. **Two-generic structure, Option 1, small.** matmul on a 2×2 grid (4 tiles) +
   the existing 1-core all_gather reading `reblock(matmul_out)`. Smallest step
   that proves the matmul→fabric hand-off and the 4→1 gather. Validate PCC on the
   1×2 mesh.
2. **Scale the matmul grid to 8×8** (64→1 gather), same structure. Confirm L1
   fits the 256×256 shard; add a PCC test.
3. **Large shards:** add chunked streaming in the all_gather generic, or switch
   to Option 2 (DRAM staging), once `C_d` exceeds L1.
4. **(Optional) single-generic conditional-fabric variant** if the DRAM round
   trip / two-generic overhead ever matters.
