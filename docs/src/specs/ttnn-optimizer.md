# TTNN Optimizer

**Component**: TTNNOptimizer
**Location**: `lib/Dialect/TTNN/Transforms/OptimizerPasses/Optimizer.cpp`
**Authors**: Compiler Team
**Status**: Current Design + Future Refactors (WIP)

---

## TL;DR

TTNNOptimizer maximizes L1 memory usage for TTNN operations via:
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
│  │   │  ┌──────────────────┐     │  • Reshard insertion │               │ │ │
│  │   │  │GreedyL1Interleaved│    │  • Core usage max    │               │ │ │
│  │   │  │ (deprecated)     │     └──────────────────────┘               │ │ │
│  │   │  └──────────────────┘                                            │ │ │
│  │   │                                                                  │ │ │
│  │   │  ┌──────────────────┐                                            │ │ │
│  │   │  │ BFInterleavedPolicy│                                          │ │ │
│  │   │  │ (deprecated)     │                                            │ │ │
│  │   │  └──────────────────┘                                            │ │ │
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

### 2.2 Key Files and Responsibilities

| File | Responsibility |
|------|----------------|
| `Optimizer.cpp` | Main pass orchestration, applies final configs to IR |
| `DFShardingPolicy.cpp` | L1 chain identification, DFS scheduling, invokes ShardSolver |
| `ShardSolver.cpp` | Constraint satisfaction, bitset-based config tracking, reshard insertion |
| `L1ChainConfig.h` | Data structure for L1 chain state and op specs |
| `OpConfig.h` | Config structure: `TTNNLayoutAttr` + op-specific attrs |
| `LegalTensorLayoutAnalysis.cpp` | Generates all possible tensor layouts |
| `LegalOpLayoutAnalysis.cpp` | Filters layouts via OpModel validation |
| `LegalOpConfigAnalysis.cpp` | Expands op-specific config search space |
| `OpConfigAnalysis.cpp` | Picks final single config per op |
| `MemoryLayoutAnalysis.cpp` | Dispatches to appropriate policy |
| `OpConstraintValidation.cpp` | Interface to OpModel for validation |

### 2.3 Policy Selection

The optimizer supports pluggable memory layout policies via `--memory-layout-analysis-policy`:

| Policy | Status | Description |
|--------|--------|-------------|
| `DFSharding` | **Production** | DFS-based L1 chain building with ShardSolver |
| `GreedyL1Interleaved` | Deprecated | Greedy L1 interleaved placement |
| `BFInterleaved` | Deprecated | Breadth-first interleaved placement |

> **Note**: `GreedyL1Interleaved` and `BFInterleaved` remain for experimental purposes only.

---

## 3. Current Design - Core Components

### 3.1 Layout Generation Pipeline

The layout generation pipeline creates the search space of valid configurations for each operation.

#### 3.1.1 ScalarDataTypeAnalysis

**Purpose**: Collect all scalar element types used across the graph.

**Location**: `lib/Dialect/TTNN/Analysis/ScalarDataTypeAnalysis.cpp`

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

**Location**: `lib/Dialect/TTNN/Analysis/LegalTensorLayoutAnalysis.cpp`

For each `(TensorType, ScalarType)` pair, generates layouts across these dimensions:

```
TensorPageLayout:
  ├── Tiled      (32x32 tiles)
  └── RowMajor   (if enabled via --row-major-enabled)

TensorMemoryLayout:
  ├── Interleaved (data spread across cores round-robin)
  └── Sharded     (data explicitly partitioned per core)

BufferType:
  ├── L1    (on-chip SRAM)
  └── DRAM  (off-chip)

Grid (for sharded only):
  Various grid dimensions based on device worker grid
  e.g., 1x1, 1x8, 8x1, 8x8, etc.
```

**Output Structure**:

```cpp
using TensorTypeLayoutsMap =
  DenseMap<RankedTensorType,                    // Key: tensor shape + element type
           DenseMap<Type,                        // Key: scalar type (bf16, f32, ...)
                    std::array<                  // Index: TensorPageLayout
                      std::array<                // Index: TensorMemoryLayoutIndex
                        SmallVector<TTNNLayoutAttr>,  // The layouts
                      kNumMemLayoutValues>,
                    kNumPageLayoutValues>>>;
```

#### 3.1.3 LegalOpLayoutAnalysis

**Purpose**: Filter tensor layouts to those valid for a specific operation.

**Location**: `lib/Dialect/TTNN/Analysis/LegalOpLayoutAnalysis.cpp`

This is a **per-op analysis** that queries the OpModel backend to validate each candidate layout:

```
for each candidate layout in tensorTypePossibleLayouts[op.outputType]:
    result = OpModel.validateOperation(op, inputLayouts, OpConfig(layout))
    if result.isSuccess():
        legalLayouts.append(layout)
    # Stop at maxLegalLayouts (default: 8) to bound search space
```

The validation checks:
- Can the op produce output with this layout?
- Does the configuration fit in L1?
- Are all data type and tiling constraints satisfied?

#### 3.1.4 LegalOpConfigAnalysis

**Purpose**: Expand layouts with op-specific configuration parameters.

**Location**: `lib/Dialect/TTNN/Analysis/LegalOpConfigAnalysis.cpp`

For operations like `Conv2d`, there are additional configuration knobs beyond just the output layout:

```cpp
struct Conv2dAttrs {
  std::optional<Conv2dConfigAttr> conv2dConfig;
  std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig;
};

// Conv2dConfigAttr contains:
// - actBlockHOverride: activation block height
// - deallocateActivation: free activation after use
// - reshardIfNotOptimal: allow reshard within conv
// - etc.
```

**Search Space Generation**:

```
For Conv2d ops:
  searchSpace.actBlockHOverride = {0, 64, 32}   // 0 = max flexibility
  searchSpace.deallocateActivation = {true}

  For each layout in legalLayouts:
    For each (actBlockH, dealloc, ...) in cartesian_product(searchSpace):
      configs.append(OpConfig(layout, Conv2dAttrs{actBlockH, dealloc, ...}))
```

The cartesian product uses an "odometer" algorithm to enumerate all combinations.

#### 3.1.5 Configuration Flow Diagram

```
┌────────────────────┐
│ tensor<1x64x64xbf16>│
└─────────┬──────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│           LegalTensorLayoutAnalysis                      │
│                                                          │
│  Generates ~100s of layouts:                            │
│  • L1-Sharded-1x8, L1-Sharded-8x1, L1-Sharded-8x8, ... │
│  • L1-Interleaved                                       │
│  • DRAM-Interleaved                                     │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│           LegalOpLayoutAnalysis (per-op)                 │
│                                                          │
│  For ttnn.matmul:                                       │
│    Valid: L1-Sharded-8x8, L1-Interleaved, DRAM-Inter... │
│    Invalid: L1-Sharded-1x1 (not enough parallelism)     │
│                                                          │
│  Filtered to maxLegalLayouts = 8                        │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│           LegalOpConfigAnalysis (per-op)                 │
│                                                          │
│  For ttnn.conv2d:                                       │
│    Expand each layout with:                             │
│    • actBlockHOverride ∈ {0, 64, 32}                   │
│    • deallocateActivation ∈ {true}                     │
│                                                          │
│  8 layouts × 3 actBlockH = 24 configs                   │
└─────────────────────────────────────────────────────────┘
          │
          ▼
      legalConfigs[op] = vector<OpConfig>
```

---

### 3.2 DFShardingPolicy

**Purpose**: Build and resolve L1 chains by processing ops in DFS-schedulable order.

**Location**: `lib/Dialect/TTNN/Analysis/DFShardingPolicy.cpp`

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
| **Shardable ops only** | Only specific ops support L1 sharding (Conv2d, Matmul, Add, Relu, etc.) |
| **Has legal configs** | Op must have at least one valid sharded configuration |

**Shardable Operations** (hardcoded list):
```cpp
isa<Conv2dOp, ConvTranspose2dOp, AddOp, MultiplyOp, ReluOp, Relu6Op,
    TypecastOp, SiluOp, MatmulOp, LinearOp, MinimumOp, RMSNormOp,
    RotaryEmbeddingOp, GeluOp>(currentOp)
```

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

**Location**: `lib/Dialect/TTNN/Analysis/ShardSolver.cpp`

#### 3.3.1 Problem Formulation

Given an L1 chain of N operations, each with K_i valid configurations, find a selection of one configuration per operation such that:

1. **Edge Compatibility**: For each producer→consumer edge, the producer's output layout is compatible with the consumer's input requirements
2. **Memory Fit**: All selected configurations fit within L1 capacity
3. **Maximize Core Usage**: Among valid solutions, prefer those using more cores

#### 3.3.2 Bitset-Based Constraint Tracking

ShardSolver uses 512-bit bitsets to efficiently track which configurations remain valid:

```cpp
using Bitset = std::bitset<512>;  // kNumBitsetBits = 512

// Each bit position represents a configuration index
// Bitset[i] = 1 means config[i] is still valid
// Bitset[i] = 0 means config[i] has been eliminated

// Per-operation bitset tracking
DenseMap<Operation*, BitsetId> bitsetIds;
std::vector<Bitset> bitsets;
```

**Why 512 bits?** While `maxLegalLayouts` defaults to 8, op-specific configuration attributes (e.g., Conv2dConfig) multiply the number of configs. For example, 8 layouts × multiple `actBlockHOverride` values × other Conv2d parameters can produce hundreds of configs per operation.

#### 3.3.3 PathSet Graph

The solver maintains a graph of valid paths between adjacent operations:

```cpp
struct Path {
  std::uint64_t producerId;   // Index into producer's config vector
  std::uint64_t consumerId;   // Index into consumer's config vector
};

struct PathSet {
  BitsetId producerBitsetId;
  BitsetId consumerBitsetId;
  Operation *producerOp;
  Operation *consumerOp;
  Paths paths;  // SmallVector<Path, 16>

  // Update paths based on current bitset state
  bool update(const std::vector<Bitset> &bitsets);
};
```

A `Path(i, j)` means "producer config[i] is compatible with consumer config[j]".

#### 3.3.4 Resolution Algorithm

```
Algorithm: ShardSolver.resolveStep()

1. preprocessFirstOp()
   # Handle chain entry: can first op accept DRAM interleaved input?

2. for each op in chain (in order):
     consumerBitset = getOrInsertBitset(op, ALL_ONES)
     consumerConfigs = legalConfigs[op]

     for each edge in operandOpEdges[op]:
       producerOp = edge.producerOp
       producerBitset = getOrInsertBitset(producerOp, ALL_ONES)
       producerConfigs = legalConfigs[producerOp]

       if reshardOnEdge:
         # Reshard handles layout conversion
         paths = generatePathsFromReshardMap(edge)
       else:
         paths = []
         for producerId in valid(producerBitset):
           for consumerId in valid(consumerBitset):
             result = OpModel.validate(op, producerConfigs[producerId],
                                           consumerConfigs[consumerId])
             if result.isSuccess():
               paths.append(Path(producerId, consumerId))

       if paths.isEmpty():
         # Try inserting reshard
         if not insertReshard(edge):
           return FAILED
         paths = generatePathsFromReshardMap(edge)

       # Update bitsets based on valid paths
       producerBitset &= pathsToProducerBitset(paths)
       consumerBitset &= pathsToConsumerBitset(paths)

       pathSets[edge] = PathSet(paths)

3. Propagate constraints through path graph
   updateSolver(root, expand=false)

4. return SUCCESS
```

#### 3.3.5 First Op Preprocessing

The first operation in an L1 chain receives input from outside the chain (typically DRAM interleaved). Special handling determines which configs work with external input:

```
preprocessFirstOp():
  firstOpBitset = empty

  # Check which configs accept interleaved input
  for each config in firstOpConfigs:
    if OpModel.supportsInterleavedInputShardedOutput(firstOp, config):
      firstOpBitset.set(config.index)

  if firstOpBitset.any():
    return SUCCESS

  # No config works with interleaved input - must reshard
  return insertReshard(chainInputEdge)
```

#### 3.3.6 Reshard Insertion

When adjacent operations have incompatible layouts, the solver inserts a reshard (ToLayoutOp). This finds all valid (reshardLayout → consumerConfig) combinations:

```
insertReshard(edge):
  consumerBitset = empty
  reshardMap = {}

  # Try all possible sharded layouts as reshard output
  for each reshardLayout in possibleShardedLayouts(edge.inputTensor):
    # Check which consumer configs work with this reshard layout
    for each config in consumerConfigs:
      if OpModel.validate(consumerOp, reshardLayout, config):
        consumerBitset.set(config.index)
        reshardMap[config.index].append(reshardLayout)

  if reshardMap.empty():
    return FAILED

  memReconfigMap[edge] = reshardMap
  return SUCCESS
```

The reshard map stores multiple valid reshard layouts per consumer config, allowing flexibility during final resolution.

#### 3.3.7 Constraint Propagation

After initial path construction, constraints propagate bidirectionally through the graph until convergence:

```
updateSolver(root):
  worklist = [root]

  while worklist not empty:
    op = worklist.pop()
    changed = false

    # Propagate to producers (backward)
    for each incoming edge (producer → op):
      if pathSet.update() removes any paths:
        worklist.add(producer)
        changed = true
      if pathSet becomes empty:
        return FAILED  # No valid solution

    # Propagate to consumers (forward)
    for each outgoing edge (op → consumer):
      if pathSet.update() removes any paths:
        worklist.add(consumer)
        changed = true
      if pathSet becomes empty:
        return FAILED

  return SUCCESS
```

The `pathSet.update()` removes paths where either endpoint's config bit has been cleared, then updates the corresponding bitsets to reflect remaining valid configs.

#### 3.3.8 Core Usage Maximization

After constraints are resolved, the solver computes accumulated core usage to guide config selection:

```
Algorithm: produceMaxCoreUsage()

# Walk chain from tail to head (reverse order)
for op in reversed(shardSpecs):
  configs = legalConfigs[op]

  # Initialize with own grid volume
  for i, config in enumerate(configs):
    accCoreUsage[op][i] = config.outputLayout.getGrid().getGridVolume()

  # Add downstream core usage via valid paths
  for pathSet in getUserPathSetsPts(op):
    consumerOp = pathSet.getConsumerOp()

    for path in pathSet.paths:
      if bitset[op].test(path.producerId):
        downstream = accCoreUsage[consumerOp][path.consumerId]
        accCoreUsage[op][path.producerId] = max(
            accCoreUsage[op][path.producerId],
            accCoreUsage[op][path.producerId] + downstream / forkFactor
        )

return accCoreUsage
```

**Example**:

```
    Op0 ──── grid: 4x2=8 cores
     │
    Op1 ──── grid: 8x8=64 cores
    / \
   /   \
  Op2  Op3 ─ grid: 4x4=16 cores each
   \   /
    \ /
    Op4 ──── grid: 2x2=4 cores

Accumulated (walking backward):
  Op4: 4
  Op2: 16 + 4/2 = 18  (fork factor = 2 for join)
  Op3: 16 + 4/2 = 18
  Op1: 64 + max(18,18) = 82
  Op0: 8 + 82 = 90

Config with highest accumulated core usage is selected.
```

#### 3.3.9 Shortcomings and Limitations

The current ShardSolver design has several significant limitations:

**1. First Operand Only**

ShardSolver only tracks and resolves constraints along the **first operand (operand[0])** of each operation. This means:
- Operations with multiple activation tensor inputs (e.g., binary ops where both inputs come from the chain) cannot have both inputs properly constrained
- The second operand's layout is not considered during constraint propagation
- This fundamentally limits the solver to linear chains where data flows through the first operand

**2. Single Edge Failure Breaks Entire Chain**

If constraint resolution fails on **any single edge** in the chain, the entire chain fails:
- No partial solutions are possible
- A chain of 10 ops will completely fall back to DRAM if one edge cannot be resolved
- This leads to suboptimal results for graphs that could benefit from partial L1 placement

**3. Complex Constraint Propagation**

The PathSet-based constraint propagation adds significant complexity:
- Bidirectional updates between producer and consumer bitsets
- Iterative propagation until convergence
- Difficult to reason about and debug when constraints conflict
- The `updateSolver` loop can visit the same operations multiple times

**4. Local Optimization Only**

Each L1 chain is solved independently:
- No global view of memory pressure across chains
- Cannot make trade-offs between chains (e.g., shrink one chain to benefit another)
- Chains are built greedily without considering downstream implications

**5. Reshard Insertion is Reactive**

Reshards are only inserted when direct compatibility fails:
- No proactive optimization of reshard placement
- Cannot reason about whether a reshard earlier in the chain would enable better configurations downstream

---

### 3.4 OpModel Integration

**Purpose**: Query the tt-metal backend for operation validity, memory requirements, and actual output layouts.

**Location**: `lib/Dialect/TTNN/Validation/OpConstraintValidation.cpp`

#### 3.4.1 Validation Interface

The OpModel provides a validation interface that checks if an operation can execute with given input layouts and configuration:

```cpp
struct ValidationResult {
  enum Status {
    Success,
    OutOfMemory,
    MetalBackendError,
    NotImplemented
  };

  Status status;
  std::string errorMessage;
  TTNNLayoutAttr actualOutputLayout;  // Backend-determined output layout
  size_t configIndex;                  // Index of validated config
};

ValidationResult validateOperation(
    Operation *op,
    const std::vector<TTNNLayoutAttr> &inputLayouts,
    const OpConfig &config);
```

#### 3.4.2 OpConstraints Structure

The backend returns detailed memory information:

```cpp
struct OpConstraints {
  size_t cbL1PeakSize;         // Circular buffer L1 allocation (bytes)
  size_t tensorL1PeakSize;     // Tensor L1 allocation (bytes)
  size_t peakL1MemorySize;     // Peak total = CB + tensors
  size_t outputL1BufferSize;   // Output buffer allocation
  TTNNLayoutAttr outputLayout; // Actual output layout from backend
};
```

---

### 3.5 Graph Transformation

**Purpose**: Apply the chosen configurations to the IR and insert necessary memory reconfiguration operations.

**Location**: `lib/Dialect/TTNN/Transforms/OptimizerPasses/Optimizer.cpp`

#### 3.5.1 Transformations Applied

After analysis completes, the optimizer applies the following transformations to the IR:

**1. Configuration Application (per-op)**
- Update the tensor type encoding with the chosen `TTNNLayoutAttr`
- Set the layout attribute on ops implementing `TTNNLayoutOpInterface`
- Set op-specific configuration attributes (e.g., `Conv2dConfigAttr` for Conv2d ops)
- Update data type attributes on ops implementing `TTNNDtypeOpInterface`

**2. Memory Reconfiguration Insertion**
- For each edge in `memReconfigEntryMap`, insert a `ToLayoutOp` between producer and consumer
- Select the best reshard layout that preserves tiling properties when possible
- If producer is already a `ToLayoutOp`, modify it in-place instead of inserting a new one

**3. Spill Processing**
- For ops marked in `spillToDramOps`, insert a `ToLayoutOp` that converts output to DRAM interleaved
- Rewire all uses to read from the spilled DRAM tensor
- Optimization: if a memory reconfig op exists on one branch, that branch can continue reading from L1

**4. L1 Interleaved Fallback** (optional, via `--l1-interleaved-fallback-analysis-enabled`)
- After main optimization, scan DRAM-placed ops
- Upgrade to L1 interleaved layout where the op supports it and L1 capacity allows

**5. Schedule Application**
- Reorder operations according to the schedule produced by the policy
- Ensures memory-efficient execution order

**6. Function Type Update**
- Update function return types to match the transformed operation result types

#### 3.5.2 Transformation Example

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

**Branch**: `origin/rpavlovic/pr1-chain-merging`
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

To validate merges, we need to track L1 memory usage across the schedule timeline.

**L1Reservation Structure**:
```cpp
struct L1Reservation {
  Operation *sourceOp;  // Op whose output is reserved in L1
  int64_t startPos;     // Schedule position where reservation starts
  int64_t endPos;       // Schedule position where reservation ends (last user)
  uint64_t sizeBytes;   // L1 size reserved in bytes
};
```

**Querying active reservations**:
```
getActiveL1Reservations(schedulePos, reservations):
  total = 0
  for each reservation:
    if schedulePos in [reservation.startPos, reservation.endPos]:
      total += reservation.sizeBytes
  return total
```

When validating an op, total additional L1 = chain merge overhead + active reservations at that schedule position.

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

**SpillLocation Enum** (replaces `spillEndToDRAM` bool):
```cpp
enum class SpillLocation {
  None,          // No spill - output stays in current layout
  L1Interleaved, // Spill to L1 interleaved (for reshape consumers, etc.)
  DRAM           // Spill to DRAM interleaved (default)
};
```

#### 4.1.5 Merge Validation Process

**Critical Insight**: Chain merging validation must cover ALL scheduled ops between the source chain's last op and the join point, not just ops in L1 chains. This is implemented via `validateScheduleRangeWithReservation`.

**L1 Residents Layout Map**:
```cpp
llvm::DenseMap<Operation *, TTNNLayoutAttr> l1ResidentsLayoutMap;
```
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

## Appendix A: Key Data Structures

### OpConfig

```cpp
struct OpConfig {
  TTNNLayoutAttr outputLayout;

  using OpSpecificAttrs = std::variant<
      UninitializedAttrs,  // Default, no op-specific config
      Conv2dAttrs          // Conv2d-specific parameters
  >;
  OpSpecificAttrs opSpecificAttrs;
};
```

### TTNNLayoutAttr

```cpp
// Encoded on tensor types, describes memory placement and layout
TTNNLayoutAttr {
  BufferType bufferType;        // L1 or DRAM
  TensorMemoryLayout memLayout; // Interleaved or Sharded
  Layout layout;                // Tile or RowMajor
  GridAttr grid;                // Shard grid (if sharded)
  DataType dataType;            // bf16, f32, etc.
}
```

### L1ChainConfig

```cpp
class L1ChainConfig {
  std::vector<OpL1MemSpec> opL1MemSpecs;
  llvm::DenseSet<Operation *> l1ChainedOps;
  llvm::DenseMap<Edge, MemReconfigEntry> memReconfigEntryMap;
  L1ChainState state;  // InBuild, Built, Resolved, Completed, Failed
  bool spillEndToDRAM;
};

struct OpL1MemSpec {
  Operation *op;
  uint tensorSplitFactor;
  OpConfig config;
};
```

**Proposed additions (Section 4.1)**:
- `SpillLocation spillLocation` — replaces `spillEndToDRAM` bool
- `bool isConcatChain` — true for single-op concat chains
- `std::optional<TensorMemoryLayout> preferredOutputMemLayout`

### Edge

```cpp
struct Edge {
  Operation *producerOp;
  Operation *consumerOp;
  int64_t operandIndex;  // Which operand of consumer
};
```

---

## Appendix B: CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--memory-layout-analysis-enabled` | false | Enable memory layout optimization |
| `--memory-layout-analysis-policy` | DFSharding | Policy: DFSharding, GreedyL1Interleaved, BFInterleaved |
| `--l1-interleaved-fallback-analysis-enabled` | false | Try upgrading DRAM to L1 interleaved |
| `--max-legal-layouts` | 8 | Max sharded layouts to consider per op |
| `--tensor-l1-usage-cap` | 1.0 | Scale L1 capacity (0.0-1.0) |
| `--row-major-enabled` | false | Enable row-major layout generation |
| `--override-output-layout` | "" | Manual layout overrides per op |
| `--override-conv2d-config` | "" | Manual Conv2d config overrides |

---

## Appendix C: Chain Merging Implementation Details

### Helper Data Structures

**Schedule Position Map**:
```cpp
llvm::DenseMap<Operation *, int64_t> buildSchedulePositionMap(schedule)
```
Maps each operation to its position in the execution schedule for O(1) lookup during validation.

**Op to Chain Map**:
```cpp
llvm::DenseMap<Operation *, size_t> buildOpToChainMap(l1ChainConfigs)
```
Maps each operation to the index of its containing chain.

**Resolved Layout Map**:
```cpp
llvm::DenseMap<Operation *, TTNNLayoutAttr> buildResolvedLayoutMap(chain)
```
Maps each op in a chain to its resolved output layout, used for building input layouts during validation.

**All Chain Layout Maps**:
```cpp
llvm::DenseMap<size_t, llvm::DenseMap<Operation *, TTNNLayoutAttr>>
    chainLayoutMaps = buildAllChainLayoutMaps(l1ChainConfigs);
```
Pre-built resolved layout maps for all completed chains, used for efficient lookup during `validateScheduleRangeWithReservation`.

### Merge Candidate Selection

- For chains with multiple merge candidates, select the one with largest output size (maximizes L1 utilization benefit)
- One-level merge limit: a chain can only receive one merge to avoid complex cascading effects

### SpillLocation Transformations

With the `SpillLocation` enum, the optimizer applies:

| SpillLocation | Action |
|---------------|--------|
| `None` | No spill - output stays in L1 (sharded or interleaved) |
| `L1Interleaved` | Insert `ToLayoutOp` converting sharded → L1 interleaved |
| `DRAM` | Insert `ToLayoutOp` converting to DRAM interleaved |
