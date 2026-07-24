# TTNN Optimizer

## TL;DR

The TTNN optimizer performs two key optimizations for TTNN operations:
1. **Maximizes L1 memory usage** — keeps intermediate tensors in fast on-chip L1 memory instead of slow DRAM
2. **Optimizes op-specific configurations** — selects optimal parameters for operations (e.g., Conv2d block sizes, activation handling)

It is implemented as a greedy, pass-based architecture consisting of two passes:
1. **GreedyMemoryLayoutPropagation** (Pass 1): Enumerate valid tensor layouts per op and greedily propagate op configurations / layouts through the graph, inserting reshards where needed.
2. **GreedyL1SpillManagement** (Pass 2): Enforce the L1 budget by spilling selected tensors to DRAM.

This greedy architecture is described in [section 5](#5-pass-based-architecture-current-design).

> **Status:** The original chain-based optimizer (`TTNNOptimizer` pass,
> `DFShardingPolicy`, `ShardSolver`, `L1ChainConfig` and the
> `MemoryLayoutAnalysis` policy dispatcher) has been **removed**. It is
> superseded by the greedy pass-based architecture above, which now serves every
> optimization level. Sections 2–4 below describe the removed chain-based design
> and are retained only for historical context.

---

## Table of Contents

1. [Introduction & Goals](#1-introduction--goals)
2. [Architecture Overview (Removed — Historical)](#2-architecture-overview-removed--historical)
3. [Chain-Based Design - Core Components (Removed — Historical)](#3-chain-based-design---core-components-removed--historical)
   - 3.1 [Layout Generation Pipeline](#31-layout-generation-pipeline)
   - 3.2 [DFShardingPolicy](#32-dfshardingpolicy)
   - 3.3 [ShardSolver](#33-shardsolver)
      - 3.3.9 [Shortcomings and Limitations](#339-shortcomings-and-limitations)
   - 3.4 [OpModel Integration](#34-opmodel-integration)
   - 3.5 [Graph Transformation](#35-graph-transformation)
4. [Future Work / Proposed Refactors (Removed — Historical)](#4-future-work--proposed-refactors-removed--historical)
   - 4.1 [DFSharding 2.0: Chain Merging and L1 Saturation](#41-dfsharding-20-chain-merging-and-l1-saturation)
5. [Pass-Based Architecture (Current Design)](#5-pass-based-architecture-current-design)
   - 5.1 [Summary](#51-summary)
   - 5.2 [Motivation](#52-motivation)
   - 5.3 [Design Philosophy](#53-design-philosophy)
   - 5.4 [Problems with the Removed Chain-Based Approach](#54-problems-with-the-removed-chain-based-approach)
   - 5.5 [Empirical Findings](#55-empirical-findings)
   - 5.6 [Architecture](#56-architecture)
      - 5.6.0 [Legal-Layout Generation (shared analyses)](#560-legal-layout-generation-shared-analyses)
      - 5.6.1 [Pass 1: GreedyMemoryLayoutPropagation](#561-pass-1-greedymemorylayoutpropagation)
      - 5.6.2 [Pass 2: GreedyL1SpillManagement](#562-pass-2-greedyl1spillmanagement)
   - 5.7 [Simplicity Benefits](#57-simplicity-benefits)
   - 5.8 [Optimization Strategies](#58-optimization-strategies)

---

## 1. Introduction & Goals

### 1.1 Purpose

The **TTNN optimizer** (the greedy `GreedyMemoryLayoutPropagation` +
`GreedyL1SpillManagement` passes) determines optimal memory layouts and op
configurations for TTNN operations to maximize performance on Tenstorrent
hardware. The fundamental goal is to **maximize data residency in L1 memory**
while maintaining correctness.

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

> **Note:** The capacity/latency figures and per-core budgets in this document
> are Wormhole values. Other architectures (e.g. Blackhole) differ; generalizing
> these numbers across architectures is tracked as follow-up work.

### 1.3 Optimization Goals

1. **Maximize L1 Residency**: keep intermediate tensors in L1 as long as possible to avoid costly DRAM round-trips.

2. **Enable Sharding**: distribute tensors across multiple cores' L1 to enable parallel computation and fit larger tensors.

3. **Minimize Resharding**: when layouts differ between producer and consumer
   ops, avoid unnecessary reshard insertions. A reshard that only changes the
   memory config (buffer type / memory layout / grid) is emitted as a
   `ttnn.to_memory_config` (`ToMemoryConfigOp`); a `ttnn.to_layout`
   (`ToLayoutOp`) is emitted only when a tile ↔ row-major re-tile is required,
   since `to_memory_config` cannot retile.

4. **Maximize Core Utilization**: prefer configurations that use more cores (larger grids) for better parallelism.

5. **Maintain Correctness**: only choose configurations validated by the backend (via OpModel).

### 1.4 Key Trade-offs

| Decision | Benefit | Cost |
|----------|---------|------|
| L1 Sharded | Fastest compute, parallel execution | Limited capacity, layout constraints |
| L1 Interleaved | Simpler, no sharding constraints | Less parallelism than sharded |
| DRAM Interleaved | Unlimited capacity | Slowest, memory bandwidth bound |

---

## 2. Architecture Overview (Removed — Historical)

> The single-pass `TTNNOptimizer` architecture diagrammed in this section (with
> the `MemoryLayoutAnalysis` policy dispatcher, `DFShardingPolicy`, and
> `ShardSolver`) has been **removed** from the codebase. See
> [section 5](#5-pass-based-architecture-current-design) for the current greedy
> pass-based optimizer.

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

## 3. Chain-Based Design - Core Components (Removed — Historical)

> The chain-based design described in this section (and section 4) has been
> removed from the codebase. See [section 5](#5-pass-based-architecture-current-design)
> for the current greedy pass-based optimizer.

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

## 4. Future Work / Proposed Refactors (Removed — Historical)

> This section captured proposed refactors of the now-removed chain-based
> `DFShardingPolicy`. It is retained only for historical context; the ideas that
> carried forward (all-operand handling, fork liveness, treating the graph as a
> whole) are realized by the greedy pass-based optimizer in
> [section 5](#5-pass-based-architecture-current-design).

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

---

## 5. Pass-Based Architecture (Current Design)

### 5.1 Summary

The greedy pass-based architecture is the **current and only** optimizer. It
replaced the `DFShardingPolicy`, `ShardSolver`, `L1ChainConfig`, and the
`MemoryLayoutAnalysis` policy dispatcher (all now removed). Empirical analysis
across 50+ models showed that this approach matches or exceeds the old
performance while being significantly easier to understand and maintain.

It is implemented as two passes — `GreedyMemoryLayoutPropagation` (Pass 1) and
`GreedyL1SpillManagement` (Pass 2) — wired into the pipeline by
`createTTNNPipelineAnalysisPasses` (`lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp`).
Despite the "greedy" name, Pass 1 defaults to a bounded **beam search**
(`beam-width=8`) scored by a fixed heuristic; see [Section 5.8.1](#581-beam-search-for-layout-propagation-pass-1).
Pass 2 remains a greedy, liveness-driven spill manager.

### 5.2 Motivation

The removed chain-based architecture had grown complex over time. The
combination of L1 chain building, chain merging, and `ShardSolver` constraint
propagation created a system that was difficult to reason about, debug, and
extend. More importantly, the analysis below revealed that this complexity did
not translate to better results—the sophisticated backtracking mechanism rarely
provided practical benefit.

### 5.3 Design Philosophy

- **Simple mental model:** "Keep data in L1 unless an operation requires otherwise, then return to L1 as soon as possible."
- **Defer to the backend:** Let the backend decide optimal configs and layouts via its query APIs. The optimizer's job is to respect those choices, not second-guess them.

### 5.4 Problems with the Removed Chain-Based Approach

These are the limitations of the removed `ShardSolver`/`DFShardingPolicy` design
that motivated the current architecture. Each is addressed by the greedy passes
as noted below.

#### 5.4.1 Operand 0 Limitation

The `ShardSolver` only propagated sharding decisions through operand 0 edges.
For binary operations like `subtract(a, b)`, if operand 0 could not be sharded,
the solver ignored operand 1 entirely—even when it was perfectly valid to keep
operand 1 in L1. This caused significant unnecessary spills. **Fixed:** Pass 1
enumerates candidates for *every* tensor operand (`getInputCandidateSets`), so
layout decisions consider all operands.

#### 5.4.2 All-or-Nothing Chain Failure

When any edge in an L1 chain failed validation, the entire chain spilled to
DRAM. There was no mechanism for partial success—a single incompatible operation
forced all connected operations out of L1. **Fixed:** the greedy passes make
per-op decisions with no chain concept, so one op falling back to DRAM does not
evict its neighbors.

#### 5.4.3 Designed for Linear Chains

The `ShardSolver`'s bitset-based constraint propagation assumed linear chain
structure. Complex graph topologies like forks, joins, and diamonds required
special-case handling outside the solver. **Fixed:** Pass 1 processes the graph
op-by-op and resolves forks during beam consolidation
([Section 5.8.1](#581-beam-search-for-layout-propagation-pass-1)); Pass 2 tracks
fork liveness directly.

#### 5.4.4 Solving a Problem That Rarely Exists

The `ShardSolver` was designed to solve the case where an operation has multiple
valid output layouts for a given sharded input and the choice matters because it
affects downstream compatibility. In practice most operations produce a single
valid output layout for a given input—there is rarely a meaningful choice to
optimize over, so the constraint-propagation/backtracking machinery addressed a
problem that empirically almost never arose.

#### 5.4.5 Precomputed Layout Pool Discards Valid Results

The chain-based pipeline filtered every backend result against a fixed,
precomputed pool of candidate layouts (`LegalTensorLayoutAnalysis`): if the
backend's op-constraints API returned a valid sharded output layout that was not
in the pool, the optimizer silently discarded it. The greedy Pass 1 mitigates
this by treating the primary output "hint" as *null* — it lets the backend
report whatever output layout it produces from the given inputs and accepts that
result (`OpRuleBook::getOutputHints`), rather than requiring the output to be a
member of the precomputed set. The legal-layout pool is still generated (see
[Section 5.6.1](#561-pass-1-greedymemorylayoutpropagation)) but is used to seed
*candidate* layouts and reshard targets, not to gate the backend's own output.

### 5.5 Empirical Findings

We analyzed the compiled IR for 50+ models including Segformer, ResNet50, and 45 LLM variants (Llama, Falcon, Gemma, Phi, Qwen). Key findings:

**Memory pressure is rare at current batch sizes.** All tested models show 40-94% L1 headroom. The L1 budget of ~1364 KB per core is rarely stressed. This could change for CNN models at larger batch sizes, where activation tensors grow significantly. LLM decode paths should remain comfortable since activations stay relatively small regardless of batch size.

**Spills are constraint-driven, not memory-driven.** The vast majority of spills occur because specific operations require DRAM inputs (reduce ops, permute, reshape), not because L1 is full.

**Unnecessary spills have clear causes:**
- Operand 0 limitation: 36-65 spills per model in patterns where one operand is DRAM while another could stay in L1
- Fork handling: 15 spills in ResNet50 residual connections that could remain in L1 with proper liveness tracking

**Greedy decisions would have been correct.** In every case we analyzed, the optimal choice was apparent from local information—no backtracking was needed to find it.

### 5.6 Architecture

The optimizer is two independent passes with clear responsibilities.

#### 5.6.0 Legal-Layout Generation (shared analyses)

Both the removed design and the current one build a per-op search space with the
same multi-analysis stack, and the greedy Pass 1 still runs all of it (invoked
from `GreedyMemoryLayoutPropagation::runOnOperation`):

1. **ScalarDataTypeAnalysis** — collects the unique scalar element types in the
   graph.
2. **LegalTensorLayoutAnalysis** — for each `(TensorType, ScalarType)` pair,
   generates all candidate `TTNNLayoutAttr`s (tiled/row-major × interleaved/
   sharded × L1/DRAM × grids). Sharded/L1-interleaved candidates are filtered out
   when `enable-l1-sharding-layouts` is off.
3. **LegalOpLayoutAnalysis** — per op, selects the legal output layouts from that
   pool (bounded by `max-legal-layouts`, default 64).
4. **LegalOpConfigAnalysis** — per op, expands each legal layout with op-specific
   configs. Config expansion is restricted to the ops that actually carry extra
   knobs — `Conv2dOp`, `ConvTranspose2dOp`, `Conv3dOp`, `MatmulOp`, `LinearOp`
   (e.g. Conv2d block sizes via `Conv2dConfigSearchSpace`).

The expanded set (`legalConfigs[op]`) is the per-op candidate search space
consumed by Pass 1. (The `OpConfigAnalysis` used by the old single-config
selection step is no longer part of this path.)

#### 5.6.1 Pass 1: GreedyMemoryLayoutPropagation

An edge-based layout picker (`MemoryLayoutPropagation`) that walks the function in
schedule order and, for each op that implements the `OpModel` interface, selects
the best valid layout through backend validation. For each operation:

1. Build a candidate set for **every** tensor operand from its producer's chosen
   layout, plus L1-interleaved fallbacks and reshard targets.
2. Take output "hints" from the rulebook — the primary hint is *null* ("let the
   backend decide the output from the inputs"); sharded legal configs are added
   as fallback hints, tried only if the null hint does not yield a sharded
   output.
3. Validate each (operand-candidates × hint) combination against the backend via
   `getOpConstraints` (OpModel). Validation checks feasibility and memory; it does
   **not** run the op or query runtime.
4. Score valid candidates with a fixed heuristic (`LayoutScore`), roughly:
   L1 over DRAM → sharded over interleaved → fewer input DRAM bytes → no-reshard
   over reshard → more cores → lower L1 usage.
5. Keep the top-K candidates (beam search, default `beam-width=8`).

If no candidate validates, the op falls back to L1 Interleaved and then to DRAM
Interleaved (`getDRAMInterleavedFallback`). This pass considers all operands
(fixing the operand 0 limitation) but has no notion of memory pressure across
multiple live tensors — it only validates that each individual op-to-op
transition is valid.

**Reshard exploration:** Even when a reshard-free path exists between two ops, a
reshard may enable a better downstream layout (e.g. more cores). Pass 1 generates
sharded reshard targets proactively (`addReshardCandidates`, gated by
`shouldExploreReshards` — disabled for ops like reshape/permute) and lets the
beam ([Section 5.8.1](#581-beam-search-for-layout-propagation-pass-1)) pick
between direct and reshard paths. Interleaved→sharded reshards are only added
when no surviving candidate already offers a sharded layout.

**What op is inserted for a reshard:** a reshard that only changes the memory
config (buffer type / memory layout / grid) is emitted as a
`ttnn.to_memory_config` (`ToMemoryConfigOp`). A `ttnn.to_layout` (`ToLayoutOp`)
is emitted only when a tile ↔ row-major re-tile is required, because
`to_memory_config` cannot retile. Reshard feasibility itself is validated as a
`ToMemoryConfigOp`.

**Note:** Because this pass does not track global L1 pressure, it may leave the
graph in a state where OOM is expected at runtime—multiple simultaneously live
tensors may each be assigned L1 layouts that are individually valid but
collectively exceed the L1 budget. This is by design; Pass 2 (L1 Spill
Management) resolves these conflicts.

**Op-specific configs:** This pass also selects op-specific configs (conv2d,
matmul, compute configs) from the `LegalOpConfigAnalysis` candidates. Today we
generate these configs ourselves because the backend's query APIs use a dummy
allocator that cannot auto-select optimal configs. Once the allocator is
integrated into the query path, both layouts and op configs could become fully
backend-driven, and this pass would simply ask the backend "given these inputs,
what is the best config?" instead of enumerating candidates.

#### 5.6.2 Pass 2: GreedyL1SpillManagement

Pass 1 validates each op-to-op edge in isolation—it confirms that a single
producer-consumer pair can both fit in L1, but does not account for *other*
tensors that are simultaneously live. Pass 2 (`L1SpillManagement`) takes the L1
layout assignments from Pass 1 and adjusts them based on global memory pressure.
It runs only when memory-layout analysis is enabled.

**Core strategy:**
- Walk the schedule tracking all live tensors and their per-core L1 sizes
  (`SumL1MemoryTracker`), freeing tensors past their last use at each step.
- Validate each op against the memory budget; when the output does not fit
  (`ensureFitsL1`) or the op OOMs (`handleOOM`), evict tensors until it fits.
- Choose the eviction victim greedily: the live tensor whose **last use is
  farthest away** (`evictFarthestUse`, a max-heap on last-use position). This is
  the "furthest next use" heuristic; there is no DP-based selection.
- Evicted tensors are spilled to DRAM Interleaved via `ttnn.to_memory_config`,
  reconnecting only the uses at/after the spill point so earlier uses keep
  reading L1.

**Fork handling:** liveness is fork-aware — a result's last-use position is the
max over *all* its users, so a forked tensor stays in L1 until its last consumer.
When a spilled producer feeds multiple consumers, Pass 2 re-materializes L1 for
the consumers that still require it (`insertReshardForConsumer`) and adjusts op
configs as needed (e.g. conv2d `deallocate_activation=false` for fork inputs).

**Key distinction from Pass 1:** Pass 1 only falls back to DRAM when no valid L1
layout exists for an operation. Pass 2 may *undo* an L1 decision that Pass 1
made, because the cumulative L1 pressure from multiple simultaneously live
tensors exceeds the budget—even though each individual op-to-op edge was valid in
isolation.

### 5.7 Simplicity Benefits

The greedy architecture is substantially simpler than the removed chain-based
one:

- No chain state machine or chain merging logic
- No bitset-based constraint solver
- Each pass has a single, well-defined responsibility
- Decisions are local and easy to trace
- Fewer special cases for graph patterns

This simplicity translates to faster development, easier debugging, and more predictable behavior.

### 5.8 Optimization Strategies

Pure greedy allocation provides a functional baseline; the pass-based
architecture enables more sophisticated strategies. Beam search
([5.8.1](#581-beam-search-for-layout-propagation-pass-1)) is **implemented and
enabled by default** (`beam-width=8`). Cost-mode scoring
([5.8.1](#581-beam-search-for-layout-propagation-pass-1), cost mode) and
DP-based spill selection ([5.8.2](#582-dynamic-programming-for-optimal-spill-selection-pass-2-enhancement))
are not yet implemented and remain future work.

#### 5.8.1 Beam Search for Layout Propagation (Pass 1)

Pure greedy (K=1) can lock in suboptimal choices early. For example, an early op might choose 32-core sharding because it avoids a reshard, but this propagates forward and forces downstream matmuls to also use 32 cores—losing significant compute throughput.

**The problem:** Pure-greedy's strategy is "use working config without reshard, fall back to reshard only if none exists." This avoids reshards but may miss globally better paths.

**What is implemented:** beam search with K candidates per op, default `beam-width=8` (`GreedyMemoryLayoutPropagation`; `beamWidth` option). The backward consolidation phase runs only when K > 1. Beam search has two phases:

**Forward phase (candidate selection):** Process ops in schedule order. For each op:
1. Enumerate candidates from configs compatible with input layouts (no reshard) and configs requiring reshards but enabling more cores
2. For binary ops, consider combinations from both inputs (K × K pairs)
3. Score all candidates, keep top K
4. Store back-pointers to parent candidates

**Backward phase (trace-back):** Starting from leaf nodes, trace back through best candidates to reconstruct the optimal path. At fork points, resolve conflicts (see below).

**Scoring (heuristic mode — current):** the fixed `LayoutScore` heuristic
described in [Section 5.6.1](#561-pass-1-greedymemorylayoutpropagation) (L1 over
DRAM, sharded over interleaved, fewer input DRAM bytes, no-reshard over reshard,
more cores, lower L1 usage). No device access is required; feasibility comes from
`getOpConstraints`.

**Scoring (cost mode — future, not implemented):** with device access, use
`getOpRuntime()` for actual runtime estimates so score = accumulated runtime.
This would enable precise tradeoffs but is slower and requires device. It is
*not* wired into the greedy passes today — the beam is scored purely by the
`LayoutScore` heuristic above.

**Complexity:** O(K² × n) where K = beam width, n = number of ops. The K² factor comes from binary ops where we evaluate K × K input combinations. For ops with more inputs (e.g., concat with 4-5 operands), the combinations remain tractable since K is small (default 8).

**Why this reaches parity with the removed optimizer:** the `ShardSolver` explored configurations via constraint propagation and backtracking. Beam search achieves similar exploration with bounded complexity, but considers all operands and doesn't suffer from chain-level failures.

**Handling fork points:** During backward trace-back, different consumer paths may prefer different layouts from a forked tensor:

```
        fork_op (keeps K candidates: [HS, BS, WS, ...])
           /              \
      consumer_A      consumer_B
      (path wants HS) (path wants BS)
```

At each fork during trace-back:
1. Collect what layout each consumer path wants
2. For each of fork's K candidates, compute total reshard cost to satisfy all consumers
3. Pick the candidate with minimum total reshard cost

This is a local decision—no tree traversal needed. Beam search reduces the global problem to local decisions by preserving K candidates at each op.

#### 5.8.2 Dynamic Programming for Optimal Spill Selection (Pass 2 Enhancement)

> **Status: future work, not implemented.** Pass 2 currently uses the greedy
> farthest-last-use eviction described in
> [Section 5.6.2](#562-pass-2-greedyl1spillmanagement).

When L1 pressure exists and multiple tensors compete for limited space, the spill decision becomes a classic register allocation problem. A DP-based approach can find the globally optimal set of tensors to keep in L1:

**Problem formulation:** Given a schedule of operations and their tensor lifetimes, select which tensors to keep in L1 at each point such that total memory never exceeds budget and total spill cost is minimized.

**DP state:** At each operation in the schedule, track which subset of live tensors are in L1. Transitions occur when tensors become live (allocate or spill) or die (deallocate).

**Cost model:** Assign costs to spills based on tensor size and access patterns. Tensors accessed multiple times have higher spill cost than single-use tensors.

This approach guarantees optimal allocation but has exponential complexity in the number of simultaneously live tensors. For most models this is tractable (typically 5-15 live tensors), but may need pruning heuristics for complex graphs.

#### 5.8.3 Progression Path

Each phase delivers a complete optimizer (both Pass 1 and Pass 2). The phases
represent increasing sophistication in the strategies used within each pass.
**Phases 1–2 are implemented and shipping; Phases 3–4 are future work.**

**Phase 1 - Greedy (MVP) — ✅ done:** Pass 1 pure greedy layout propagation (K=1). Pass 2 greedy spill management with liveness tracking. Together this fixes the operand 0 limitation (Pass 1 considers all operands) and fork handling (Pass 2 tracks tensor lifetimes). Validated the pass-based architecture with minimal complexity.

**Phase 2 - Beam Search with Heuristics (Parity) — ✅ current default:** Pass 1 runs beam search (default `beam-width=8`) with the `LayoutScore` heuristic: prefer L1/sharded, break ties by reshard count and core usage. Explores reshard paths even when reshard-free paths exist. No device access needed, fast. Pass 2 remains greedy. This is the configuration used by the pipeline today.

**Phase 3 - Beam Search with Cost Mode (Opt Level 3) — ⏳ future:** Upgrade Pass 1's scoring to use `getOpRuntime()` for actual runtime estimates. Precise cost-based tradeoffs between reshards and compute. Requires device access, slower, but more accurate for complex models. Not yet implemented.

**Phase 4 - DP Extensions (Edge Cases) — ⏳ future:** Upgrade Pass 2 to use DP-based spill selection for models with genuine memory pressure. The empirical data (40-94% headroom) suggests this is rarely needed, but the architecture supports it. Not yet implemented.
