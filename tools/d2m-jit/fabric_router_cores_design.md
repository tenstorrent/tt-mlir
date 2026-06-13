# Fabric router cores: fabric on a subset of a generic's grid

## Status

Design proposal, not yet implemented. Written to unblock the *fused*
single-generic matmul + all_gather (the tt-metal overlap pattern), which is
currently impossible because fabric is wired to a generic's whole grid.

## The problem

A `d2m.generic` runs its body on every core in its grid. Fabric (cross-device
`remote_store` with `startDevice`, `device_synchronize`, fabric semaphore incs)
can only be issued from a limited set of cores per mesh link — `num_links *
cores_per_link` cores per direction (1 on a 1x2 line). Today fabric is coupled
to the *whole* grid:

- **Runtime** (`runtime/lib/common/fabric_config.cpp:206`): `appendFabricConfigArgs`
  builds fabric connection args for every core of `corerange_to_cores(coreRangeSet)`,
  where `coreRangeSet` is the kernel's grid (`runtime/lib/ttmetal/executor.cpp:405`),
  and asserts:
  ```
  LOG_ASSERT(cores.size() <= num_links * cores_per_link, ...)
  ```
- **Kernel**: `create_fabric_connection_manager` / `setup_fabric_connections`
  are emitted at the top of the datamovement kernel, so *every* grid core opens
  a connection manager.

So a grid-`(2,1)` generic with a `fabricConnectionConfig` fails at dispatch:

```
FATAL | Number of cores (2) to connect to fabric routers exceeds routing planes available (1)
```

A grid-1 fabric generic (the current all_gather) works because its grid *is*
the single fabric core. The consequence is the design constraint from
`fused_matmul_allgather_8x8_design.md`: a >1-core matmul and the single-core
fabric send cannot share a generic, so they must be **two generics**
(matmul grid N -> all_gather grid 1). That works (`test_streaming_matmul_all_gather`,
`test_multicore_matmul_all_gather`) but **serializes** the matmul and the
collective — there is a hard ordering barrier between the two generics. The
*fused* form (the matmul cores stream tiles to a router core that forwards them
over fabric while the rest of the grid is still computing — tt-metal's
`MatmulFusedOpSignaler` pattern) needs fabric on a **subset** of one generic's
grid. That is the capability this doc designs.

`scf.if (core_index(1) == 0) { ...fabric... }` already gates fabric *execution*
to one core, and the matmul + `core_read` gather + producer-done fence all work
single-device (`test_matmul_core_read_gather`). The gating does **not** help the
two failures above: the runtime still sets up fabric for the whole grid, and the
kernel still creates a connection manager on every core. The `scf.if` predicate
is also opaque to the compiler/runtime — nothing ties "the cores where this
branch is taken" to "the cores the runtime should wire to fabric".

## The proposal under review

> A routing config exposed at the API layer, wired to the kernel's kwargs. The
> config configures, at dispatch time, which cores do routing, so the runtime
> knows what to do. Special intrinsics like `is_router_core()` let the kernel
> branch on whether this core was assigned a routing task, instead of the kernel
> hard-coding cores.

This is the right shape. Two things make it work where `scf.if(cx==0)` does not:

1. It makes the router-core set **explicit and declarative** (a config value),
   so there is a single source of truth the *runtime* can read to restrict
   fabric setup — the missing piece for the runtime assert.
2. `is_router_core()` is a **compiler-recognized** predicate, not an opaque
   integer comparison. The compiler can (a) gate the connection-manager
   lifecycle to exactly those cores and (b) emit the router-core set into the
   flatbuffer for the runtime. An arbitrary `scf.if` condition gives neither.

So `is_router_core()` is doing double duty: a kernel-author branch *and* the
contract that links the user's gating to the compiler's fabric setup and the
runtime's per-core wiring.

### What the proposal as stated under-specifies

- **It is not just kernel branching.** Branching on `is_router_core()` gates
  *execution*. The fix must also (a) restrict the runtime `appendFabricConfigArgs`
  to the router subset and (b) gate `create_fabric_connection_manager` /
  `setup_fabric_connections` to the router cores in the *lowering*. All three
  must agree on the same set or the program will hang / assert. The config is
  the single source of truth that keeps them in sync.
- **One router core per link direction, not one core total.** A bidirectional
  collective has a forward and a backward router (per link); the design must
  carry a per-`(link, direction)` router-core assignment and let a router core
  learn its direction (`router_direction()`), not just whether it is one. See
  "Router cores are per (link, direction)" below.
- **How `is_router_core()` evaluates.** Two options (below); the config-constant
  comparison is preferred over a runtime-pushed per-core flag.
- **Interaction with ScheduleDMA.** The fused kernel has a multicore matmul
  *and* fabric. ScheduleDMA's current "pin to a single DM thread when >1 fabric
  send" collapses the whole DM region to one NOC thread, which previously broke
  the matmul reader/writer split (`test_matmul_all_gather_fused` PCC nan under
  the broad pin). The router-core mechanism does not by itself resolve this; see
  Risks.

## Router cores are per (link, direction) — not a single core

The fabric is organized by **link direction**, and a router/sender core is
assigned per `(link, direction)`. This must be first-class in the design, not an
afterthought. From the metal repo:

- Directions are `eth_chan_directions{EAST, WEST, NORTH, SOUTH}`
  (`hostdevcommon/api/hostdevcommon/fabric_common.h:38`). A cluster axis maps to
  two opposed directions: `directions[axis] = {{N,S},{W,E}}` with positive =
  `E`/`S`, negative = `W`/`N` (`ttnn/.../ccl/common/host/moe_utils.cpp:113`).
- Routing planes (links) are counted **per direction**:
  `get_num_available_routing_planes_in_direction(fabric_node_id, direction)`
  (`control_plane.hpp:219`, `moe_utils.cpp`).
- The fused/async CCL program factories are parameterized by `num_links` ×
  **`num_directions_per_link`** × `num_workers_per_direction` (+
  `num_mux_cores_per_direction_per_link`), with explicit `forward_coord` /
  `backward_coord` neighbors
  (`all_gather_async/device/all_gather_async_default_program_factory.cpp:40-108`).
  A bidirectional all_gather therefore has a **forward sender and a backward
  sender** (per link).
- Our own runtime already reflects this: `cores_per_link = (routing ==
  UnidirRingTorus) ? 2 : 1`, `connection_directions` collects forward (`.first`)
  and backward (`.second`) per dim, the plane index is `i / cores_per_link`, and
  the direction is chosen by `i % 2` for the 2-core case
  (`runtime/lib/common/fabric_config.cpp:215-262`).

So the count of router cores is `num_links × num_directions` (num_directions is 1
when a single core drives both directions — `bidir_line_mesh`, `cores_per_link=1`
— or 2 when each direction has its own core — `unidir_ring_torus`,
`cores_per_link=2`). Two consequences for the design:

1. `router_cores` is a **per-(link, direction) assignment**, addressable as such,
   not a flat anonymous list. The ordering must line up with the runtime's plane
   (`i / cores_per_link`) and direction (`i % cores_per_link`) indexing.
2. A router core needs to know **which direction it serves**, so it addresses
   the correct neighbor (forward vs backward device) in its `remote_store`. The
   kernel needs a `router_direction()` query, not just a boolean
   `is_router_core()`.

## Refined design

A **router-core set, indexed by (link, direction)**, carried on the fabric
config and threaded through every layer, with `is_router_core()` /
`router_direction()` as the kernel-side queries. Single source of truth: the
config.

### 1. API

```python
# bidir line: one core drives both directions of one link (cores_per_link=1)
fabric = d2m.fabric_config(
    cluster_axis=1, topology="linear", routing="bidir_line_mesh", num_links=1,
    router_cores=[(0, 0)],            # one entry == one (link, direction) slot
)

# unidir ring / per-direction: a forward and a backward router (cores_per_link=2)
fabric = d2m.fabric_config(
    cluster_axis=1, topology="ring", routing="unidir_ring_torus", num_links=1,
    router_cores=[(0, 0), (1, 0)],    # [plane0/dir0, plane0/dir1] == [fwd, bwd]
)
```

`router_cores` is a list of `(y, x)` grid coordinates, one per
`(link, direction)` slot, in the runtime's order: entry `i` maps to routing
plane `i / cores_per_link` and direction `i % cores_per_link`, matching the loop
at `fabric_config.cpp:215-262`. Length must equal `num_links * cores_per_link`
(`cores_per_link` is 1 for `bidir_line_mesh`, 2 for `unidir_ring_torus`). The
same core may appear for multiple slots only if it can host that many fabric
connections. Default (when omitted) = the whole grid, i.e. exactly today's
behavior, so existing grid-1 fabric kernels are unchanged.

(A future ergonomic form can take a direction-keyed map,
`router_cores={"forward": [(0,0)], "backward": [(1,0)]}`, lowering to the same
ordered list; the positional list is the canonical IR form.)

### 2. IR / attribute

Extend `#ttcore.fabric_connection_config` with an optional `router_cores`
attribute (array of 2xi64). It already rides on the generic; this is the natural
home (fabric-specific, one source of truth for both the kernel predicate and the
runtime subset).

### 3. `is_router_core()` / `router_direction()` intrinsics

New `d2m` query ops, lowered in D2MToTTKernel like `core_index` /
`mesh_position`. Because `router_cores` are compile-time constants, both lower to
a comparison of `my_logical_{y,x}_()` against the configured coords — no new
runtime-arg plumbing, and trivially consistent with the runtime (both read the
same `router_cores`):

```
is_router_core()  := OR over slots i of (my_logical == router_cores[i])
router_direction() := the direction of the slot this core matches
                      (forward/backward; or an index into the (link,direction)
                       table). Undefined / NONE on non-router cores.
```

A router core uses `router_direction()` to address the correct neighbor in its
fabric `remote_store` (forward vs backward device). Kernels typically branch
once on `is_router_core()` and, inside, switch on `router_direction()` (or write
a per-direction block guarded by `router_direction() == FORWARD`). For the
common `bidir_line_mesh` case (one core, both directions via the mcast range),
`router_direction()` is a single value and the existing `device_mcast_shape`
addressing is unchanged.

(Alternative considered: a runtime-set per-core boolean/enum arg instead of the
config-constant comparison. More plumbing and more ways for kernel and runtime
to disagree; rejected unless the predicate must depend on runtime-only state.)

### 4. Lowering: gate the connection-manager lifecycle

Today `create_fabric_connection_manager` / `setup_fabric_connections` /
`close_fabric_connections` are hoisted to the kernel top (every core). They must
move **inside** the router region so only router cores open a connection:

```
if is_router_core():
    fcm = create_fabric_connection_manager()
    setup_fabric_connections(fcm)
    device_synchronize(...)
    ...core_read + fabric remote_store...     # all fcm users live here
    close_fabric_connections(fcm)
```

`getFabricConnectionManager` (D2MToTTKernel.cpp:166) currently finds/creates the
fcm at a common point; it must instead anchor it inside the `is_router_core()`
region, and every fcm user (`mesh_position`, `device_synchronize`, fabric
`remote_store`, fabric sem incs) must be dominated by it. Since authors already
put all fabric ops inside the gate, the constraint is: **all fcm users are in
the router region** — the verifier should enforce this.

### 5. Flatbuffer

Add `router_cores: [Dim2dRange]` to `FabricConnectionConfig`
(`Common/types.fbs:195`) — or to the `EnqueueProgramCommand` next to
`fabric_connection_config`. On the config is cleaner (co-located with the rest
of the fabric parameters).

### 6. Runtime

`appendFabricConfigArgs` uses `router_cores` (mapped to physical `CoreCoord`s)
instead of `corerange_to_cores(coreRangeSet)` for the per-core loop. Everything
else there is already per-(link, direction): plane `= i / cores_per_link`,
direction from `connection_directions` / `i % 2`. Feeding it the explicit
`router_cores` (in slot order) just replaces the implicit "all grid cores"
enumeration with the intended subset, and the `cores.size() <= num_links *
cores_per_link` assert becomes an equality the config already guarantees.
`SetRuntimeArgs` applies fabric args only to router cores; non-router cores get
the normal (non-fabric) rt args. The executor caller (`executor.cpp:405`) passes
the router subset through. Because the runtime derives each slot's direction the
same way the kernel's `router_direction()` does (slot order), the two agree by
construction.

## Alternatives considered

1. **Infer the router set from the `scf.if` predicate.** Reject: general
   predicate analysis ("which (x,y) satisfy this condition") is intractable and
   fragile. The explicit config + recognized intrinsic is the point.
2. **Always place fabric on a fixed canonical core (e.g. (0,0)), no config.**
   Simpler, but cannot express multiple router cores (needed for `num_links > 1`
   / ring `cores_per_link = 2`) and bakes in policy. The config generalizes; the
   default can still be "(0,0) for a 1-core fabric".
3. **Keep two generics but run them concurrently (overlap without fusion).**
   Reject: cross-generic ordering in d2m is a barrier (generic 2 reads generic
   1's output only after it completes); overlap fundamentally needs one program
   with per-core roles, i.e. this design. Two generics remain the correct
   *non-overlapped* implementation and stay supported.
4. **Separate fabric kernel on a router sub-grid (per-thread core ranges).**
   The most interesting alternative. Instead of one kernel deployed to the whole
   grid that branches on `is_router_core()`, lower the fabric datamovement to its
   *own* kernel whose core range is just the router cores, while the matmul
   kernels span the full grid — all in **one program** (one
   `EnqueueProgramCommand`, multiple kernels with different `core_range_set`s,
   which tt-metal programs already support). The matmul cores hand tiles to the
   router kernel via `core_read` + the producer-done semaphore, exactly as in the
   role-flag design.
   - **Pro:** no `is_router_core()` branch (the fabric kernel only exists where
     fabric runs); the runtime naturally wires fabric to just that kernel's core
     range; and — crucially — it **sidesteps the ScheduleDMA tension**, because
     the fabric kernel is physically separate from the matmul's DM threads, so
     the matmul reader/writer split is untouched.
   - **Con:** a larger d2m model change. A `d2m.generic`'s threads currently all
     share the generic's grid; giving a thread its own (sub)core-range is new
     structure (per-thread core ranges, or a fabric thread that lowers to a
     distinct kernel). It also diverges from tt-metal, which deliberately uses
     **one role-flagged kernel** (`is_injector_core` / `is_output_writer` /
     `is_sink_core` baked into a single binary) rather than separate kernels.
   - **Verdict:** keep the role-flag (`is_router_core`) design as the primary —
     it matches tt-metal and reuses the one-kernel-per-grid model — but if the
     surgical ScheduleDMA fabric-thread separation (Risks) proves too invasive,
     this separate-kernel decomposition is the fallback that makes the threading
     conflict disappear by construction. Worth prototyping the schema/runtime
     side, since steps 5-6 below already need the runtime to accept a fabric core
     range distinct from the compute grid either way.

## Risks and open questions

- **ScheduleDMA vs the matmul (the hard one).** The fused kernel is a multicore
  matmul + multiple fabric sends. The current `keepSingleDMThread` (pin when >1
  fabric send) collapses the whole DM region to one NOC thread, which broke the
  matmul reader/writer split (`test_matmul_all_gather_fused` nan under the broad
  pin). Router cores do not change this: the pin is per-generic. The real fix is
  **surgical scheduling** — keep the *fabric* ops on one thread (one connection
  manager) while letting the matmul's non-fabric DM (reader feeding compute,
  writer draining the matmul CB) use the other thread. This is a separate,
  non-trivial ScheduleDMA change and is the main implementation risk. Until it
  exists, the fused kernel may need to fall back to a single fabric send per
  router core (gather into one buffer, send once), which avoids the pin.
  Alternative 4 (separate fabric kernel on a router sub-grid) makes this risk
  disappear by construction — the fabric kernel is not on the matmul's DM
  threads at all — and is the recommended escape hatch if the surgical
  scheduling change is too invasive.
- **Three-way consistency.** Kernel predicate, lowering fcm gating, and runtime
  subset must use the identical set. Enforced by deriving all three from the one
  `router_cores` config; a verifier should reject fabric ops not dominated by an
  `is_router_core()`-gated fcm.
- **Physical vs logical coords.** `router_cores` are logical grid coords;
  `is_router_core()` compares logical (`my_logical_{x,y}_`), and the runtime maps
  to physical for `append_routing_plane_connection_manager_rt_args`. Keep the
  logical->physical mapping in one place (runtime), consistent with how
  `core_index`/`mesh_position` already work.
- **Routing-plane order.** `router_cores[i]` must map to plane `i / cores_per_link`
  to match the runtime loop; document and validate this ordering.
- **Is the overlap worth it?** The two-generic path already gives a correct,
  bounded-L1 streaming all_gather. The fused form's win is overlapping the
  collective with the matmul tail (and avoiding the inter-generic barrier) — real
  for large matmuls, modest for small. Land the router-core mechanism because it
  is the general capability (and the only way to express fused CCL), but treat
  the perf payoff as the motivation, not a correctness requirement.

## Incremental implementation plan

1. **Attribute + API + intrinsics, default-compatible.** Add `router_cores`
   (a per-`(link, direction)` slot list) to the fabric config attr and
   `fabric_config(...)`; add the `d2m.is_router_core` and `d2m.router_direction`
   ops with their config-constant lowerings. With `router_cores` defaulting to
   the whole grid, all existing tests are unchanged. (No runtime change yet.)
   Validate the slot count against `num_links * cores_per_link`.
2. **Lowering: gate the fcm lifecycle** into the `is_router_core()` region; add
   the verifier that all fcm users are dominated by a gated fcm. Confirm the
   grid-1 all_gather still lowers identically (router set == whole grid == the
   one core).
3. **Flatbuffer + runtime subset.** Add `router_cores` to the schema; restrict
   `appendFabricConfigArgs` / `SetRuntimeArgs` to the subset. Now a grid-`(2,1)`
   generic with `router_cores=[(0,0)]` dispatches without the routing-plane
   assert.
4. **Resolve the ScheduleDMA tension** (surgical fabric-thread separation) or
   adopt the single-send fallback. Bring up `test_fused_matmul_core_read_all_gather`
   on the 1x2 mesh: one generic, grid `(2,1)`, all cores matmul + producer-done
   fence, router core `(0,0)` `core_read`s each tile and fabric-forwards it.
5. **Scale + generalize.** `num_links > 1` (multiple router cores, plane order),
   8x8 grid, and fold the router-core config into the higher-level CCL lowering
   so user-facing collectives pick router cores automatically.

## Relationship to existing work

- `core_read` / `core_write` + the producer-done fence
  (`unified_semaphore_design.md`) are the in-generic gather/signal primitives the
  fused kernel uses; they already work single-device
  (`test_matmul_core_read_gather`).
- The multi-fabric-send single-DM-thread fix (commit keeping fabric kernels on
  one DM thread) is what made the *two-generic* per-tile streaming all_gather
  work; the fused kernel inherits its tension with the matmul (see Risks).
- This design is the missing piece that turns the two-generic
  (`fused_matmul_allgather_8x8_design.md`) structure into a genuinely fused,
  overlapped single-generic op.
