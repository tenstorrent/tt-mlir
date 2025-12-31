# TTNN Optimizer

## TL;DR

TTNNOptimizer performs two key optimizations for TTNN operations:
1. **Maximizes L1 memory usage** — keeps intermediate tensors in fast on-chip L1 memory instead of slow DRAM
2. **Optimizes op-specific configurations** — selects optimal parameters for operations (e.g., Conv2d block sizes, activation handling)

It achieves this via:
1. **Layout Generation**: Enumerate valid tensor layouts per op
2. **DFShardingPolicy**: Build L1 chains in DFS order
3. **ShardSolver**: Constraint satisfaction to find compatible configs
4. **Graph Transformation**: Apply configs and insert reshards

Key limitation: Only tracks first operand; single edge failure breaks entire chain.

---

## Table of Contents

1. [Introduction & Goals](#1-introduction--goals)
2. [Architecture Overview](#2-architecture-overview)
3. [Current Design - Core Components](#3-current-design---core-components)
   - 3.1 [Layout Generation Pipeline](#31-layout-generation-pipeline)
   - 3.2 [DFShardingPolicy](#32-dfshardingpolicy)
   - 3.3 [ShardSolver](#33-shardsolver)
      - 3.3.9 [Shortcomings and Limitations](#339-shortcomings-and-limitations)
   - 3.4 [OpModel Integration](#34-opmodel-integration)
   - 3.5 [Graph Transformation](#35-graph-transformation)
4. [Future Work / Proposed Refactors](#4-future-work--proposed-refactors)
   - 4.1 [DFSharding 2.0: Chain Merging and L1 Saturation](#41-dfsharding-20-chain-merging-and-l1-saturation)

---

## 1. Introduction & Goals

### 1.1 Purpose

The **TTNNOptimizer** pass determines optimal memory layouts and op configurations for TTNN operations to maximize performance on Tenstorrent hardware. The fundamental goal is to **maximize data residency in L1 memory** while maintaining correctness.

### 1.2 Memory Hierarchy Context

Tenstorrent devices feature a two-level memory hierarchy:

```
┌───────────────────────────────────────────────┐
│  Tensix Cores: [L1] [L1] [L1] ... [L1]        │
│                  │     │     │       │        │
│                  └─────┴─────┴───────┘        │
│                          │                    │
│                 NoC (Network on Chip)         │
│                          │                    │
│                   ┌──────┴──────┐             │
│                   │    DRAM     │             │
│                   └─────────────┘             │
└───────────────────────────────────────────────┘
```

| Memory | Size | Latency | Use Case |
|--------|------|---------|----------|
| **L1 (SRAM)** | ~1.5 MB per core | Low | Hot data, intermediate tensors within compute chains |
| **DRAM** | ~12 GB total | High | Large tensors, model weights, spill buffer |

### 1.3 Optimization Goals

1. **Maximize L1 Residency**: keep intermediate tensors in L1 as long as possible to avoid costly DRAM round-trips.

2. **Enable Sharding**: distribute tensors across multiple cores' L1 to enable parallel computation and fit larger tensors.

3. **Minimize Resharding**: when sharding layouts differ between producer and consumer ops, avoid unnecessary `ToLayoutOp` insertions.

4. **Maximize Core Utilization**: prefer configurations that use more cores (larger grids) for better parallelism.

5. **Maintain Correctness**: only choose configurations validated by the backend (via OpModel).

### 1.4 Key Trade-offs

| Decision | Benefit | Cost |
|----------|---------|------|
| L1 Sharded | Fastest compute, parallel execution | Limited capacity, layout constraints |
| L1 Interleaved | Simpler, no sharding constraints | Less parallelism than sharded |
| DRAM Interleaved | Unlimited capacity | Slowest, memory bandwidth bound |

---

## 2. Architecture Overview

### 2.1 High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TTNNOptimizer Pass                                │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                      Analysis Pipeline                                 │ │
│  │                                                                        │ │
│  │  ┌─────────────────────┐     ┌─────────────────────┐                   │ │
│  │  │ ScalarDataType      │ ───▶│ LegalTensorLayout   │                   │ │
│  │  │ Analysis            │     │ Analysis            │                   │ │
│  │  │                     │     │                     │                   │ │
│  │  │ Collects all scalar │     │ Generates ALL       │                   │ │
│  │  │ types in graph      │     │ possible layouts    │                   │ │
│  │  └─────────────────────┘     │ for each tensor     │                   │ │
│  │                              │ type                │                   │ │
│  │                              └──────────┬──────────┘                   │ │
│  │                                         │                              │ │
│  │                                         ▼                              │ │
│  │                         ┌───────────────────────────┐                  │ │
│  │                         │  LegalOpLayoutAnalysis    │                  │ │
│  │                         │  (per-op)                 │                  │ │
│  │                         │                           │                  │ │
│  │                         │  Filters layouts via      │                  │ │
│  │                         │  OpModel validation       │                  │ │
│  │                         └─────────────┬─────────────┘                  │ │
│  │                                       │                                │ │
│  │                                       ▼                                │ │
│  │                         ┌───────────────────────────┐                  │ │
│  │                         │  LegalOpConfigAnalysis    │                  │ │
│  │                         │  (per-op)                 │                  │ │
│  │                         │                           │                  │ │
│  │                         │  Expands op-specific      │                  │ │
│  │                         │  configs (Conv2dConfig)   │                  │ │
│  │                         └─────────────┬─────────────┘                  │ │
│  │                                       │                                │ │
│  └───────────────────────────────────────┼────────────────────────────────┘ │
│                                          │                                  │
│                                          ▼                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                     MemoryLayoutAnalysis                               │ │
│  │                                                                        │ │
│  │   ┌──────────────────────────────────────────────────────────────────┐ │ │
│  │   │              Memory Layout Policy (Pluggable)                    │ │ │
│  │   │                                                                  │ │ │
│  │   │  ┌──────────────────┐                                            │ │ │
│  │   │  │ DFShardingPolicy │◀── Default, production-ready               │ │ │
│  │   │  │                  │                                            │ │ │
│  │   │  │  • DFS scheduling│                                            │ │ │
│  │   │  │  • L1 chain      │                                            │ │ │
│  │   │  │    building      │     ┌──────────────────────┐               │ │ │
│  │   │  │  • ShardSolver   │────▶│     ShardSolver      │               │ │ │
│  │   │  │    resolution    │     │                      │               │ │ │
│  │   │  └──────────────────┘     │  • Constraint SAT    │               │ │ │
│  │   │                           │  • Bitset tracking   │               │ │ │
│  │   │                           │  • Reshard insertion │               │ │ │
│  │   │                           │  • Core usage max    │               │ │ │
│  │   │                           └──────────────────────┘               │ │ │
│  │   └──────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                        │ │
│  └────────────────────────────────────────┬───────────────────────────────┘ │
│                                           │                                │
│                                           ▼                                │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                     OpConfigAnalysis                                  │ │
│  │                                                                       │ │
│  │   Picks single final config per op from remaining valid set           │ │
│  └────────────────────────────────────────┬──────────────────────────────┘ │
│                                           │                                │
│                                           ▼                                │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                     Graph Transformation                              │ │
│  │                                                                       │ │
│  │   • Apply layout attributes to ops                                    │ │
│  │   • Set op-specific configs (Conv2dConfigAttr)                        │ │
│  │   • Insert ToLayoutOp for resharding                                  │ │
│  │   • Process spill-to-DRAM ops                                         │ │
│  │   • L1 Interleaved fallback (optional upgrade from DRAM)              │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```


---

## 3. Current Design - Core Components

### 3.1 Layout Generation Pipeline

The layout generation pipeline creates the search space of valid configurations for each operation.

#### 3.1.1 ScalarDataTypeAnalysis

**Purpose**: Collect all scalar element types used across the graph.

```
Input Graph:
  %0 = ttnn.conv2d(...) : tensor<1x32x32x64xbf16>
  %1 = ttnn.relu(%0) : tensor<1x32x32x64xbf16>
  %2 = ttnn.matmul(...) : tensor<1x32x32x128xf32>

Output:
  scalarTypes = {bf16, f32}
```

This analysis respects layout overrides specified by the user.

#### 3.1.2 LegalTensorLayoutAnalysis

**Purpose**: Generate all possible `TTNNLayoutAttr` combinations for each tensor type.

For each `(TensorType, ScalarType)` pair, generates layouts across these dimensions:

```
TensorPageLayout:
  ├── Tiled      (32x32 tiles)
  └── RowMajor   (if enabled via --row-major-enabled)

TensorMemoryLayout:
  ├── Interleaved (data spread across cores round-robin)
  └── Sharded     (data explicitly partitioned per core)

BufferType:
  ├── L1    (SRAM, local to Tensix core)
  └── DRAM  (shared, accessed via NoC)

Grid (for sharded only):
  Various grid dimensions based on device worker grid
  e.g., 1x1, 1x8, 8x1, 8x8, etc.
```

#### 3.1.3 LegalOpLayoutAnalysis

**Purpose**: Select tensor layouts for each operation from the pre-generated layout pool.

This per-op analysis picks layouts from the tensor type layouts generated in the previous step, associating them with specific operations. Results are bounded by `maxLegalLayouts` to limit the search space.

#### 3.1.4 LegalOpConfigAnalysis

**Purpose**: Expand layouts with op-specific configuration parameters.

For operations like Conv2d, there are additional configuration knobs beyond just the output layout (e.g., activation block sizes, memory deallocation options). This analysis expands each legal layout by generating the cartesian product with op-specific parameter values, producing the full configuration search space.

#### 3.1.5 Configuration Flow Diagram

```
┌────────────────────┐
│ tensor<1x64x64xbf16>│
└─────────┬──────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│           LegalTensorLayoutAnalysis                     │
│                                                         │
│  Generates ~100s of layouts:                            │
│  • L1-Sharded-1x8, L1-Sharded-8x1, L1-Sharded-8x8, ...  │
│  • L1-Interleaved                                       │
│  • DRAM-Interleaved                                     │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│           LegalOpLayoutAnalysis (per-op)                │
│                                                         │
│  For ttnn.matmul:                                       │
│    Valid: L1-Sharded-8x8, L1-Interleaved, DRAM-Inter... │
│    Invalid: L1-Sharded-1x1 (not enough parallelism)     │
│                                                         │
│  Filtered to maxLegalLayouts = 8                        │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│           LegalOpConfigAnalysis (per-op)                │
│                                                         │
│  For ops with extra config knobs (e.g., Conv2d):        │
│    Expand each layout with op-specific parameters       │
│                                                         │
│  N layouts × M parameter combinations = configs         │
└─────────────────────────────────────────────────────────┘
          │
          ▼
      legalConfigs[op] = vector<OpConfig>
```

---

### 3.2 DFShardingPolicy

**Purpose**: Build and resolve L1 chains by processing ops in DFS-schedulable order.

#### 3.2.1 L1 Chain Concept

An **L1 chain** is a sequence of operations whose intermediate tensors can reside in L1 memory. The goal is to identify maximal chains where data flows through L1 without spilling to DRAM.

```
Example L1 Chain:

  DRAM Input
       │
       ▼
  ┌─────────┐
  │ Conv2d  │  ─┐
  └─────────┘   │
       │        │
       ▼        │  L1 Chain
  ┌─────────┐   │  (all intermediates in L1)
  │  ReLU   │   │
  └─────────┘   │
       │        │
       ▼        │
  ┌─────────┐  ─┘
  │  Add    │
  └─────────┘
       │
       ▼
  DRAM Output (spill)
```

#### 3.2.2 Chain Building Algorithm

The policy walks the graph in DFS (Depth-First Search) schedulable order:

```
Algorithm: DFShardingPolicy.run()

1. Schedule ops in DFS order
2. For each op:
   a. Add to chain if shardable with legal configs
   b. Continue chain if: (1) next op uses this as operand[0], AND (2) single use
   c. Otherwise: finalize chain, start new one
3. Resolve each chain with ShardSolver
4. Select config maximizing core usage
```

#### 3.2.3 Chain Building Rules

| Rule | Rationale |
|------|-----------|
| **Single-use only** | If `currentOp.hasOneUse()` is false (fork), chain breaks to avoid complex dataflow |
| **First operand** | `nextOp.operand[0]` must be `currentOp` - ensures linear chain structure |
| **Shardable ops only** | Only specific ops support L1 sharding (Conv2d, Matmul, elementwise ops, etc.) |
| **Has legal configs** | Op must have at least one valid sharded configuration |

#### 3.2.4 L1ChainConfig State Machine

```
┌──────────────┐
│   InBuild    │  Initial state, ops can be added
└──────┬───────┘
       │ build()
       ▼
┌──────────────┐
│    Built     │  Ops finalized, ready for ShardSolver
└──────┬───────┘
       │ resolveWithSolver()
       ▼
┌──────────────┐         ┌──────────────┐
│   Resolved   │────────▶│    Failed    │  If solver finds no solution
└──────┬───────┘         └──────────────┘
       │ complete()
       ▼
┌──────────────┐
│  Completed   │  Single config selected per op
└──────────────┘
```

---

### 3.3 ShardSolver

**Purpose**: Solve the constraint satisfaction problem of finding compatible sharding configurations for adjacent operations in an L1 chain.

#### 3.3.1 Problem Formulation

Given an L1 chain of N operations, each with K_i valid configurations, find a selection of one configuration per operation such that:

1. **Edge Compatibility**: For each producer→consumer edge, the producer's output layout is compatible with the consumer's input requirements
2. **Memory Fit**: All selected configurations fit within L1 capacity
3. **Maximize Core Usage**: Among valid solutions, prefer those using more cores

#### 3.3.2 Bitset-Based Constraint Tracking

ShardSolver uses fixed-size bitsets to efficiently track which configurations remain valid for each operation. Each bit position represents a configuration index—a set bit means the config is still valid, a cleared bit means it has been eliminated.

The bitset size is chosen to accommodate the expanded configuration space. While `maxLegalLayouts` bounds the number of tensor layouts, op-specific attributes (e.g., Conv2d block sizes) multiply the total configs per operation into the hundreds.

#### 3.3.3 PathSet Graph

The solver maintains a graph of valid "paths" between adjacent operations. A path connects a producer config index to a consumer config index, indicating that the producer's output layout is compatible with the consumer's input requirements for those specific configurations.

For each edge in the chain, the solver stores a PathSet containing all valid producer-consumer config pairs. As constraints propagate, paths are removed when either endpoint's config becomes invalid.

#### 3.3.4 Resolution Algorithm

The solver processes the chain in order, building PathSets for each edge:

1. **Preprocess first op**: Determine which configs can accept external (typically DRAM interleaved) input
2. **Build paths**: For each edge, validate all producer-consumer config combinations via OpModel
3. **Handle incompatibility**: If no valid paths exist for an edge, attempt to insert a reshard
4. **Update bitsets**: Narrow each operation's valid configs based on which have valid paths
5. **Propagate constraints**: Iteratively propagate until the constraint graph stabilizes

If any edge ends up with zero valid paths, the entire chain fails.

#### 3.3.5 First Op Preprocessing

The first operation receives input from outside the chain. The solver checks which of its configs can accept interleaved input and produce sharded output. If none can, a reshard is inserted at chain entry.

#### 3.3.6 Reshard Insertion

When adjacent operations have incompatible layouts, the solver inserts a reshard (ToLayoutOp). It searches all possible sharded layouts that could bridge the gap, recording which consumer configs each reshard layout enables. Multiple valid reshard layouts may exist per consumer config, providing flexibility during final resolution.

#### 3.3.7 Constraint Propagation

After initial path construction, constraints propagate bidirectionally through the graph until convergence. When a config is eliminated from one operation, all paths involving that config are removed, which may eliminate configs from adjacent operations, triggering further propagation.

This continues until no more changes occur or an edge loses all paths (failure).

#### 3.3.8 Core Usage Maximization

After constraints are resolved, the solver selects the final configuration for each operation. It computes accumulated core usage by walking the chain backward, summing each operation's grid volume with its downstream usage. The configuration path with highest total core usage is selected, preferring solutions that maximize parallelism across the entire chain.

#### 3.3.9 Shortcomings and Limitations

The current ShardSolver design has several significant limitations:

**1. Reshard Insertion is Reactive (Primary Limitation)**

Reshards are only inserted when direct compatibility fails:
- No proactive optimization of reshard placement
- Cannot reason about whether a reshard earlier in the chain would enable better configurations downstream
- A strategically placed reshard might unlock better overall configurations, but the solver only reacts to failures

**2. First Operand Only**

ShardSolver only tracks and resolves constraints along the **first operand (operand[0])** of each operation. This means:
- Operations with multiple activation tensor inputs (e.g., binary ops where both inputs come from the chain) cannot have both inputs properly constrained
- The second operand's layout is not considered during constraint propagation
- This fundamentally limits the solver to linear chains where data flows through the first operand

**3. Single Edge Failure Breaks Entire Chain**

If constraint resolution fails on **any single edge** in the chain, the entire chain fails:
- No partial solutions are possible
- A chain of 10 ops will completely fall back to DRAM if one edge cannot be resolved
- This leads to suboptimal results for graphs that could benefit from partial L1 placement

**4. Local Optimization Only**

Each L1 chain is solved independently:
- No global view of memory pressure across chains
- Cannot make trade-offs between chains (e.g., shrink one chain to benefit another)
- Chains are built greedily without considering downstream implications

**5. Complex Constraint Propagation**

The PathSet-based constraint propagation adds significant complexity:
- Bidirectional updates between producer and consumer bitsets
- Iterative propagation until convergence
- Difficult to reason about and debug when constraints conflict
- The `updateSolver` loop can visit the same operations multiple times

---

### 3.4 OpModel Integration

**Purpose**: Query the tt-metal backend for operation validity, memory requirements, and actual output layouts.

The OpModel provides a validation interface that checks if an operation can execute with given input layouts and configuration. It returns whether the configuration is valid, memory usage information, and the actual output layout the backend will produce. This validation is called extensively during layout generation and constraint resolution to ensure only valid configurations are considered.

---

### 3.5 Graph Transformation

**Purpose**: Apply the chosen configurations to the IR and insert necessary memory reconfiguration operations.

After all analysis phases complete, the optimizer transforms the IR by applying the resolved layout and op-specific configurations to each operation. Where adjacent operations have incompatible layouts (as determined by ShardSolver), `ToLayoutOp` reshards are inserted to bridge the gap. Chain outputs that cannot remain in L1 are spilled to DRAM. Finally, operations are reordered according to the computed schedule to ensure memory-efficient execution.

#### Transformation Example

```
Before Optimization:

  %0 = ttnn.conv2d(%input, %weight) : tensor<1x64x64x128xbf16>  [DRAM Interleaved]
  %1 = ttnn.relu(%0) : tensor<1x64x64x128xbf16>  [DRAM Interleaved]
  return %1

After Optimization:

  %0 = ttnn.conv2d(%input, %weight)
         {conv2d_config = #ttnn.conv2d_config<actBlockHOverride = 32>}
       : tensor<1x64x64x128xbf16, #layout<L1, Sharded, 8x8>>

  %1 = ttnn.relu(%0) : tensor<1x64x64x128xbf16, #layout<L1, Sharded, 8x8>>

  // Inserted spill to DRAM for function return
  %2 = ttnn.to_layout(%1) {memory_config = #ttnn.memory_config<DRAM, Interleaved>}
       : tensor<1x64x64x128xbf16, #layout<DRAM, Interleaved>>

  return %2
```

---

## 4. Future Work / Proposed Refactors

### 4.1 DFSharding 2.0: Chain Merging and L1 Saturation

**Goal**: Maximize L1 utilization by keeping chain outputs in L1 when possible, avoiding unnecessary DRAM spills.

#### 4.1.1 Motivation

The current DFSharding policy builds isolated L1 chains, where each chain's output spills to DRAM before the next chain begins. This wastes L1 capacity and introduces DRAM latency when:

1. **Chain A → Chain B**: Chain A's output could stay in L1 for Chain B to consume directly
2. **Fork operations**: An op with multiple users spills to DRAM, causing all consumers to read from slow DRAM
3. **Concat operations**: All input chains spill to DRAM, then concat reads them back

```
Current Behavior (wasteful):

  Chain A: [Conv2d → Relu]
       │
       ▼ spill to DRAM
  ─────────────────────
       │
       ▼ reload from DRAM
  Chain B: [Add → Matmul]


Desired Behavior (Chain Merging):

  Chain A: [Conv2d → Relu]
       │
       │ stays in L1 ────────┐
       ▼                     │
  Chain B: [Add → Matmul] ←──┘
```

#### 4.1.2 Chain Merging Types

**1. Simple A→B Merge**

Chain A's output stays in L1 and is consumed by Chain B on any operand.

```
Chain A ────┐
            │ (operand 1)
            ▼
       ┌─────────┐
       │  Add    │  Chain B first op
       └─────────┘
            │
            ▼
         [...]    Chain B continues
```

**Validation**: All ops scheduled between Chain A's last op and the join point must be re-validated with Chain A's output size as additional L1 usage. This includes any scheduled op in that window, regardless of whether it belongs to an L1 chain.

**2. 3-Way Merge**

When Chain B's first op has two operands from different chains, both can stay in L1.

```
Chain A (operand 0) ───┐
                       │
                  ┌────┴────┐
                  │  Add    │  Chain B first op
                  └────┬────┘
                       │
Chain C (operand 1) ───┘
```

**Execution order**: Chain A executes first, then Chain C executes while Chain A's output stays in L1, then Chain B's first op consumes both.

**Validation**: Chain C must be validated to execute with Chain A's output as additional L1 pressure.

**3. N-Way Merge (Concat)**

Concat operations consume multiple inputs. All input chains can stay in L1 if:
- All input chains complete successfully (state = Completed)
- All inputs have compatible sharding (based on concat dimension)
- Concat can consume all L1-sharded inputs directly

```
Chain 1 ────┐
            │
Chain 2 ────┼───▶ [Concat] ───▶ Chain output
            │
Chain 3 ────┘
```

**Concat sharding constraints**:
- Width concat (dim = last): requires HEIGHT_SHARDED inputs
- Height concat (dim = second-to-last): requires WIDTH_SHARDED inputs
- BLOCK_SHARDED is NOT supported for concat

**Validation**: Chains feeding into concat execute in schedule order. All scheduled ops between the first chain's completion and concat execution must be validated with cumulative L1 pressure:
- Chain 1 executes normally
- All ops scheduled after Chain 1 (including Chain 2) are validated with Chain 1's output as additional L1
- All ops scheduled after Chain 2 (including Chain 3) are validated with (Chain 1 + Chain 2) outputs as additional L1
- And so on until concat, which must fit all N inputs in L1 simultaneously

#### 4.1.3 L1 Reservation Timeline

To validate merges, we track L1 memory usage across the schedule timeline. Each reservation records which operation's output is being held in L1, the schedule positions where the reservation is active (from production to last use), and the size in bytes. When validating any operation, we query active reservations at that schedule position to determine total L1 pressure from merged chain outputs.

#### 4.1.4 Fork Op L1 Optimization

Operations with multiple users (forks) traditionally spill to DRAM, causing all consumers to read from slow memory. This refactor tries to keep forked outputs in L1.

**Algorithm**:
```
For each chain that spills to DRAM with forked output:
  1. Try keeping SHARDED layout:
     - Check all consumers can accept sharded input
     - Validate memory pressure across fork span
     - If valid: spillLocation = None, create L1 reservation

  2. Fallback to L1 INTERLEAVED:
     - Validate op can produce L1 interleaved output
     - Check all consumers can accept L1 interleaved input
     - Validate memory pressure
     - If valid: spillLocation = L1Interleaved, create L1 reservation

  3. If both fail: keep DRAM spill
```


#### 4.1.5 Merge Validation Process

**Critical Insight**: Chain merging validation must cover ALL scheduled ops between the source chain's last op and the join point, not just ops in L1 chains. This is implemented via `validateScheduleRangeWithReservation`.

**L1 Residents Layout Map**:

Tracks layouts of chain outputs that stay in L1 after merging. Updated incrementally as merges are applied, so subsequent validations see actual sharded layouts from merged chains instead of IR's DRAM layouts.

**validateScheduleRangeWithReservation()** (core validation function):
```
For each op in schedule range [startPos, endPos]:
  1. Calculate total additional L1 at this position:
     totalAdditionalL1 = getActiveL1Reservations(pos) + additionalL1

  2. If op is in a chain:
     - Build input layouts from resolved configs + l1ResidentsLayoutMap
     - Validate with chain's resolved config

  3. If op is NOT in a chain:
     - Extract layouts from IR
     - Override inputs with l1ResidentsLayoutMap where applicable
     - Validate with IR config

  4. If validation fails: reject merge
```

**validateChainBWithMergedInput()**:
```
1. Validate intermediate ops between source chain and join op:
   validateScheduleRangeWithReservation(startPos+1, joinOpPos-1, sourceOutputSize)

2. For each op in Chain B (up to join op):
   - Build input layouts from resolved configs + l1ResidentsLayoutMap
   - At join op: replace operand layout with source chain's output
   - Calculate total additional L1
   - Validate operation
```

**validateChainWithPredecessorInL1()** (for 3-way merge):
```
For each op in Chain C:
  1. Build input layouts from resolved configs + l1ResidentsLayoutMap
  2. Calculate total additional L1:
     additionalL1 = predecessorOutputSize + getActiveL1Reservations(opPos)
  3. Call validateOperation(op, inputLayouts, config, additionalL1)
  4. If validation fails: reject merge
```

**validateThreeWayMergeJoinOp()** (3-way merge join validation):
```
The join op must be revalidated with BOTH sharded inputs:
  1. Build input layouts starting from IR
  2. Replace operand 0 with Chain A's sharded output layout
  3. Replace operand 1 with Chain C's sharded output layout
  4. Validate join op with both sharded inputs
```
This is critical because the join op was originally validated with interleaved inputs during chain resolution.

#### 4.1.6 Concat Chain Resolution

Concat ops are isolated into single-op chains and resolved after regular chains:

**Chain Building**:
```
When encountering ConcatOp:
  1. Finalize current chain
  2. Create single-op chain with isConcatChain = true
  3. Start new chain for subsequent ops
```

**setConcatChainPreferences()** (pre-resolution):
```
For each concat chain:
  1. Determine required input memory layout from concat dim
  2. Set preferredOutputMemLayout on all input chains
  3. Set preferredOutputMemLayout on consumer chain
```

**resolveConcatChains()** (post regular chain resolution):
```
For each concat chain:
  1. Check all input chains are Completed
  2. Check all inputs have compatible sharding
  3. Validate concat with L1-sharded inputs
  4. If valid:
     - Set spillLocation = None on input chains
     - Complete concat chain with L1-sharded output
```

#### 4.1.7 Data Flow with Merging

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DFShardingPolicy with Chain Merging                  │
│                                                                         │
│  1. Build chains (unchanged, but ConcatOp gets isolated chains)         │
│                          │                                              │
│                          ▼                                              │
│  2. setConcatChainPreferences()                                         │
│     Set preferred sharding for chains feeding into concat               │
│                          │                                              │
│                          ▼                                              │
│  3. Resolve regular chains with ShardSolver                             │
│     (preferredOutputMemLayout influences config selection)              │
│                          │                                              │
│                          ▼                                              │
│  4. resolveConcatChains()                                               │
│     Validate concat can consume L1-sharded inputs                       │
│                          │                                              │
│                          ▼                                              │
│  5. applyL1ReservationsForReshapes()                                    │
│     Keep reshape outputs in L1 when feeding interleaved consumers       │
│                          │                                              │
│                          ▼                                              │
│  6. applyL1ReservationsForForkOps()                                     │
│     Keep forked outputs in L1 (sharded or interleaved)                  │
│                          │                                              │
│                          ▼                                              │
│  7. applyChainMerges()                                                  │
│     Merge chains where outputs can stay in L1                           │
│     - Simple A→B merges                                                 │
│     - 3-way merges                                                      │
│                          │                                              │
│                          ▼                                              │
│  Output: Chains with spillLocation set (None, L1Interleaved, or DRAM)   │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 4.1.8 Future: Towards Global Optimization

Chain merging is a step towards global optimization but still operates on pre-built chains. Future work may:

1. **Remove L1 chain concept entirely** - treat graph as a whole
2. **Global memory pressure analysis** - consider all ops simultaneously
3. **Cost-based optimization** - evaluate trade-offs between L1 placement and reshards
4. **Deprecate ShardSolver** - replace with simpler per-edge validation

The L1 reservation timeline mechanism introduced here provides the foundation for global memory tracking.
