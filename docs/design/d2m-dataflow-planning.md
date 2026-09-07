# D2M Dataflow Planning

The D2M dataflow planner owns compiler decisions that cross individual
`d2m.generic` operations. It runs after TTIR-to-D2M conversion and constant
scalarization, while generic operations still have tensor semantics, and
before grid selection fixes layout and placement decisions.

## Pipeline boundary

```text
TTIR decomposition
  -> TTIRToD2M
  -> candidate graph construction
  -> kernel variant enumeration
  -> feasibility filtering
  -> cost evaluation and plan selection
  -> selected-plan materialization
  -> grid/layout/buffer selection
  -> internal D2M scheduling and DMA lowering
  -> TTKernel / TTMetal
```

The generic operations produced by TTIR-to-D2M are kernel candidates, not
necessarily final physical kernels. A selected variant can eventually fuse
multiple candidates, while staged variants can split one candidate after their
interface and coverage rules are modeled explicitly.

Internal planning happens in two phases. Before external planning it enumerates
variants and records estimates without lowering the payload IR. After external
planning selects kernel boundaries, placement, and communication, downstream
D2M passes materialize tile loops, circular buffers, compute and data-movement
threads, synchronization, and DMA.

## Framework objects

- `DataflowCandidateGraph` is an immutable function-block scope containing
  top-level `d2m.generic` candidates and producer-consumer dependencies.
- `DataflowKernelVariant` describes a candidate implementation, including its
  member computations, grid, block factors, resource estimate, and optional
  cycle estimate.
- `DataflowMappingPlan` records the selected temporal, fused, or spatial
  kernels and their connections. Search constructs plans out of band.
- `DataflowFeasibilityModel` rejects invalid plans before ranking. The initial
  structural model checks graph coverage, variants, and connections; hardware
  models will add L1, CB, DST, core-range, and communication constraints.
- `DataflowCostModel` evaluates feasible plans. The analytical baseline does
  not invent missing cycle data: temporal latency and spatial initiation
  interval remain unknown until all participating kernels have estimates.
- `DataflowPlanMaterializer` applies only the selected plan. The temporal
  fallback is already represented by the input IR and is therefore a no-op.

Feasibility and cost are separate by design. An impossible plan is rejected
with a structured reason rather than assigned a large penalty. Feasible plans
retain multiple metrics, including latency, initiation interval, DRAM and NoC
traffic, L1 pressure, occupied cores, spills, and program count, so later
search policies can preserve a Pareto frontier instead of depending on one
fragile weighted score.

## Initial policy

The first policy creates one kernel variant per candidate, preserves program
order and materialized producer-consumer edges, and leaves existing D2M IR
unchanged. This provides a behavior-preserving baseline for adding variant
enumeration, hardware resource checks, fused plans, and `d2m.spatial`
materialization incrementally.
