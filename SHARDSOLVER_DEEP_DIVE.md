# ShardSolver Deep Dive: Complete Internal Mechanics

## Overview

ShardSolver is a constraint satisfaction solver that finds compatible sharding configurations for operations in an L1 chain. It uses:
- **Bitsets**: Each operation has a bitset where bit `i` = 1 means config `i` is still valid
- **PathSets**: Represent valid (producer_config, consumer_config) pairs for each edge
- **Constraint Propagation**: Iteratively narrows down valid configs by propagating constraints

---

## 1. Constructor and Initialization

**File**: `lib/Dialect/TTNN/Analysis/ShardSolver.cpp`
**Function**: `ShardSolver::ShardSolver()` (Line 59-99)

### Parameters

```cpp
ShardSolver(
    tensorTypePossibleLayouts,  // All possible layouts for each tensor type
    legalConfigs,                // Map: Operation* -> vector<OpConfig>
    shardSpecs,                  // Vector of OpL1MemSpec (ops in chain)
    shardedOps,                  // Set of operations that can be sharded
    overrideReshardEdges,        // User-specified reshard edges
    overrideOutputLayout,         // User-specified output layouts
    customCheckShardCompatible,  // Custom compatibility checker (nullptr)
    solveForOptimalFirstOpInput  // TRUE for dispatch consumer chains
)
```

### Initialization Steps

**Line 70-75**: Store parameters
```cpp
tensorTypePossibleLayouts = tensorTypePossibleLayouts;
legalConfigs = &legalConfigs;
shardSpecs = &shardSpecs;
shardedOps = &shardedOps;
memReconfigEdges = overrideReshardEdges;
overrideOutputLayout = overrideOutputLayout;
customCheckShardCompatible = customCheckShardCompatible;
solveForOptimalFirstOpInput = solveForOptimalFirstOpInput;
```

**Line 76-79**: Reserve space
```cpp
pathSets.reserve(shardSpecs.size());      // PathSets for edges
pathSetIds.reserve(shardSpecs.size());    // Edge -> PathSetId map
bitsets.reserve(shardedOps.size());       // Bitsets for each op
bitsetIds.reserve(shardedOps.size());     // Operation* -> BitsetId map
```

**Line 83**: Cache device attribute
```cpp
deviceAttr = ttcore::lookupDevice(shardSpecs.front().op);
```

**Line 85-98**: **Build Edge Maps** - Critical for constraint propagation

```cpp
for (const auto &shardSpec : shardSpecs) {
  Operation *op = shardSpec.op;  // e.g., %4, %5
  for (size_t operandIndex = 0; operandIndex < op->getNumOperands(); operandIndex++) {
    Value operand = op->getOperand(operandIndex);
    Operation *operandOp = operand.getDefiningOp();  // Producer op

    // Only track edges between sharded ops
    if (operandOp && shardedOps.count(operandOp) > 0) {
      Edge edge(operandOp, op, operandIndex);
      operandOpEdges[op].emplace_back(edge);      // Consumer -> edges
      userOpEdges[operandOp].emplace_back(edge);  // Producer -> edges
    }
  }
}
```

**Example for Chain1 (`%4, %5`)**:
```cpp
// %4 = matmul(%dispatch, %arg2)
//   - %dispatch is NOT in shardedOps (it's dispatch_d2m)
//   - %arg2 is a block arg (no defining op)
//   - operandOpEdges[%4] = []  (no edges to sharded ops)

// %5 = matmul(%4, %arg3)
//   - %4 is in shardedOps
//   - operandOpEdges[%5] = [Edge(%4, %5, 0)]
//   - userOpEdges[%4] = [Edge(%4, %5, 0)]
```

**Initial State**:
```cpp
bitsets = []                    // Empty, created on-demand
bitsetIds = {}                  // Empty map
pathSets = []                   // Empty, created during resolveStep
pathSetIds = {}                 // Empty map
operandOpEdges = {%4: [], %5: [Edge(%4, %5, 0)]}
userOpEdges = {%4: [Edge(%4, %5, 0)]}
selectedOpConfig = {}           // Empty, filled by set()
memReconfigMap = {}             // Empty, filled if reshard needed
```

---

## 2. Bitset Structure

**Type**: `std::bitset<512>` (Line 45-46 in ShardSolver.h)

### What is a Bitset?

A bitset is a bitmask where:
- **Size**: 512 bits (supports up to 512 configs per operation)
- **Bit `i` = 1**: Config at index `i` is **still valid** (not eliminated)
- **Bit `i` = 0**: Config at index `i` is **eliminated** (incompatible)

### Example

For operation `%4` with 10 legal configs:
```cpp
legalConfigs[%4] = [
  config0: {outputLayout: #layout_2x2, ...},
  config1: {outputLayout: #layout_2x4, ...},
  config2: {outputLayout: #layout_4x2, ...},
  ...
  config9: {outputLayout: #layout_1x1, ...}
]

// Initial bitset (all configs valid)
bitset = 0b1111111111  // Bits 0-9 set, rest 0

// After constraint propagation (only config0 and config5 valid)
bitset = 0b1000010000  // Only bits 0 and 5 set
```

### Key Operations

**Line 930-946**: `getOrInsertBitset(Operation *op, const Bitset &init)`
```cpp
// First time: Create new bitset with all bits set
bitsetIds[%4] = 0
bitsets[0] = kBitsetAll  // All 512 bits set to 1

// Subsequent: Return existing bitset
return &bitsets[bitsetIds[%4]]
```

**Line 306-307**: Constrain bitsets (AND operation)
```cpp
*producerBitset &= edgeProducerBitset;  // Keep only valid producer configs
*consumerBitset &= edgeConsumerBitset;  // Keep only valid consumer configs
```

---

## 3. PathSet and Path Structure

### Path Structure

```cpp
struct Path {
  std::uint64_t producerId;  // Producer config index
  std::uint64_t consumerId;  // Consumer config index
};
```

A `Path` represents: **"Producer config `producerId` is compatible with consumer config `consumerId`"**

### PathSet Structure

```cpp
class PathSet {
  BitsetId producerSetId;      // Index into bitsets[] for producer
  BitsetId consumerSetId;      // Index into bitsets[] for consumer
  Operation *producerOperation;
  Operation *consumerOperation;
  Paths paths;                 // Vector of valid (producerId, consumerId) pairs
};
```

**Example for Edge(%4, %5, 0)**:
```cpp
PathSet {
  producerSetId = 0,           // bitsets[0] is %4's bitset
  consumerSetId = 1,           // bitsets[1] is %5's bitset
  producerOperation = %4,
  consumerOperation = %5,
  paths = [
    Path(0, 2),  // %4 config0 compatible with %5 config2
    Path(0, 5),  // %4 config0 compatible with %5 config5
    Path(3, 2),  // %4 config3 compatible with %5 config2
    ...
  ]
}
```

### PathSet::update() - The Core Constraint Filter

**File**: `include/ttmlir/Dialect/TTNN/Analysis/ShardSolver.h` (Line 188-216)

```cpp
bool PathSet::update(std::vector<Bitset> &bitsets) {
  Bitset validProducerSet = 0;  // Will contain valid producer configs
  Bitset validConsumerSet = 0;  // Will contain valid consumer configs
  Bitset producer = bitsets[producerSetId];  // Current producer bitset
  Bitset consumer = bitsets[consumerSetId];  // Current consumer bitset

  // Step 1: Filter paths - keep only paths where BOTH configs are still valid
  for (size_t i = 0; i < paths.size(); i++) {
    const Path &path = paths[i];
    if (consumer[path.consumerId] and producer[path.producerId]) {
      // Both configs still valid - keep this path
      validProducerSet.set(path.producerId);
      validConsumerSet.set(path.consumerId);
    } else {
      // One or both configs eliminated - remove path
      paths[i] = paths.back();
      paths.pop_back();
      i--;
    }
  }

  // Step 2: Check if bitsets need updating
  bool isProducerSub = isSubset(producer, validProducerSet);
  bool isConsumerSub = isSubset(consumer, validConsumerSet);
  bool unchanged = isProducerSub and isConsumerSub;

  // Step 3: Update bitsets if needed
  if (!unchanged) {
    bitsets[producerSetId] &= validProducerSet;  // Eliminate invalid producer configs
    bitsets[consumerSetId] &= validConsumerSet;  // Eliminate invalid consumer configs
  }

  return not unchanged;  // Return true if bitsets changed
}
```

**Example**:
```cpp
// Initial state
producer bitset = 0b1111  // %4 configs 0,1,2,3 all valid
consumer bitset = 0b1111  // %5 configs 0,1,2,3 all valid
paths = [Path(0,0), Path(0,1), Path(1,2), Path(2,3)]

// After some constraint eliminated %4 config1
producer bitset = 0b1101  // %4 config1 eliminated
consumer bitset = 0b1111

// PathSet::update() called:
//   - Path(0,0): producer[0]=1, consumer[0]=1 → KEEP
//   - Path(0,1): producer[0]=1, consumer[1]=1 → KEEP
//   - Path(1,2): producer[1]=0 → REMOVE
//   - Path(2,3): producer[2]=1, consumer[3]=1 → KEEP
//   validProducerSet = 0b1101 (configs 0,2,3)
//   validConsumerSet = 0b1111 (configs 0,1,2,3)
//   producer bitset unchanged (already 0b1101)
//   Return: false (no change)

// Later, %5 config3 eliminated
consumer bitset = 0b0111  // %5 config3 eliminated

// PathSet::update() called again:
//   - Path(0,0): KEEP
//   - Path(0,1): KEEP
//   - Path(2,3): consumer[3]=0 → REMOVE
//   validProducerSet = 0b0011 (configs 0,2)
//   validConsumerSet = 0b0111 (configs 0,1,2)
//   producer bitset &= 0b0011 → 0b0001 (only config0 left!)
//   Return: true (producer bitset changed)
```

---

## 4. resolve() Entry Point

**File**: `lib/Dialect/TTNN/Analysis/ShardSolver.cpp` (Line 753-764)

```cpp
bool ShardSolver::resolve() {
  reset();  // Clear previous state

  bool resolved = resolveStep();  // Main solving logic
  if (earlyExit) {
    assert(!resolved);
    return false;
  }

  assert(resolved);
  return resolved;
}
```

**Line 101-106**: `reset()`
```cpp
pathSets.clear();
pathSetIds.clear();
bitsets.clear();
bitsetIds.clear();
```

---

## 5. resolveStep() - Main Solving Logic

**File**: `lib/Dialect/TTNN/Analysis/ShardSolver.cpp` (Line 108-331)

### Phase 1: Preprocess First Op (Line 116)

```cpp
if (!preprocessFirstOp()) {
  return false;  // Failed to preprocess
}
```

**For Chain1 (`%4, %5`) with `solveForOptimalFirstOpInput = TRUE`**:

**Line 412-560**: `preprocessFirstOp()`

**Line 413**: `Operation *firstOp = %4`

**Line 461**: `if (solveForOptimalFirstOpInput)` → **TRUE**

**Line 467-477**: Find dispatch operand
```cpp
// Iterate through %4's operands
for (unsigned i = 0; i < %4->getNumOperands(); ++i) {
  Value operand = %4->getOperand(i);
  if (Operation *defOp = operand.getDefiningOp()) {
    if (isa<DispatchD2MOp>(defOp)) {
      dispatchOperandIdx = i;  // Found at index 0
      break;
    }
  }
}
```

**Line 491-493**: Get candidate input layouts
```cpp
candidateInputLayouts = getShardedLayoutsForTensorTypeAndScalarType(...)
// Returns ~17 candidate layouts: 2x2, 2x4, 4x2, 4x4, etc.
```

**Line 507-545**: Validate each (input, output) combination
```cpp
for (const TTNNLayoutAttr &candidateInput : candidateInputLayouts) {
  inputLayouts[0] = candidateInput;  // Set dispatch operand to candidate

  for (size_t i = 0; i < firstOpConfigs.size(); ++i) {
    // Try each output config
    OpConfig config = OpConfig(firstOpConfigs[i].outputLayout, ...);

    // Validate with OpModel
    ValidationResult result = validateOperation(%4, inputLayouts, config);

    if (result.isSuccess() && result.actualOutputLayout == desiredOutput) {
      validCombinations.emplace_back(candidateInput, i);
    }
  }
}
```

**Line 547-559**: Pick first valid combination
```cpp
if (!validCombinations.empty()) {
  auto [bestInput, bestOutputIdx] = validCombinations[0];
  resolvedFirstOpInputLayout = bestInput;  // Store for dispatch output
  firstOpBitset->set(bestOutputIdx);      // Constrain %4 to only this config
  return true;
}
```

**Bitset State After preprocessFirstOp()**:
```cpp
bitsetIds[%4] = 0
bitsets[0] = 0b0000000001  // Only bit 0 set (config0 valid)
bitsetIds[%5] = (not created yet)
```

### Phase 2: Process Each Operation (Line 122)

```cpp
for (const auto &shardSpec : *shardSpecs) {
  // shardSpecs = [{op: %4}, {op: %5}]

  Operation *consumerOp = shardSpec.op;
  Bitset *consumerBitset = getOrInsertBitset(consumerOp, kBitsetAll);
  const std::vector<OpConfig> &consumerConfigs = getLegalConfigs(consumerOp);

  auto edges = operandOpEdges.find(consumerOp);
  // For %4: edges = [] (no edges to sharded ops)
  // For %5: edges = [Edge(%4, %5, 0)]
```

### Phase 3: Process Each Edge (Line 135)

**For `%5` processing Edge(%4, %5, 0)**:

**Line 136-137**: Check for reshard
```cpp
bool reshardOnEdge = memReconfigEdges.count(edge) > 0 ||
                     memReconfigMap.count(edge) > 0;
// Initially: FALSE (no reshard yet)
```

**Line 144-147**: Get producer bitset and configs
```cpp
Operation *producerOp = %4;
Bitset *producerBitset = getOrInsertBitset(%4, kBitsetAll);
// Returns existing bitset[0] = 0b0000000001 (from preprocessFirstOp)
const std::vector<OpConfig> &producerConfigs = getLegalConfigs(%4);
```

**Line 151-156**: Initialize edge bitsets
```cpp
PathSet::Paths paths;
Bitset edgeProducerBitset = 0;  // Will contain valid producer configs for this edge
Bitset edgeConsumerBitset = 0;  // Will contain valid consumer configs for this edge
std::uint64_t producerCount = producerConfigs.size();  // e.g., 10
```

**Line 179-254**: **Find Valid Paths** (no reshard case)

**Line 180-182**: Get unique test configs
```cpp
llvm::SmallVector<OpConfig> testConfigs =
    optimizer_utils::getUniqueTestConfigs(consumerConfigs, ...);
// Gets configs with unique output layouts (may be fewer than consumerConfigs)
```

**Line 185-186**: Extract input layouts template
```cpp
std::vector<TTNNLayoutAttr> inputLayouts = utils::extractInputLayouts(%5);
// inputLayouts = [%4's output layout, %arg3's layout]
```

**Line 188-254**: Try each producer config
```cpp
for (std::uint64_t producerId = 0; producerId < producerCount; ++producerId) {
  if (!producerBitset->test(producerId)) {
    continue;  // Skip if producer config already eliminated
  }
  // For %4: Only producerId=0 is valid (from preprocessFirstOp)

  TTNNLayoutAttr inputLayout = producerConfigs[producerId].outputLayout;
  // inputLayout = %4's config0 output layout

  inputLayouts[0] = inputLayout;  // Set %5's operand 0 layout

  // Validate %5 with this input layout and all output configs
  std::vector<ValidationResult> results =
      validateWithMultipleAttributes(%5, inputLayouts, testConfigs, consumerConfigs);

  for (std::size_t i = 0; i < results.size(); ++i) {
    if (results[i].isSuccess()) {
      // Found valid (producerId, consumerId) pair
      edgeProducerBitset.set(producerId);
      edgeConsumerBitset.set(results[i].configIndex);
      paths.push_back(Path(producerId, results[i].configIndex));
    }
  }
}
```

**Example Result**:
```cpp
// %4 config0 (2x2 sharded) → %5 config2 (2x1 sharded) → VALID
// %4 config0 (2x2 sharded) → %5 config5 (2x8 sharded) → VALID
paths = [Path(0, 2), Path(0, 5)]
edgeProducerBitset = 0b0000000001  // Only config0
edgeConsumerBitset = 0b0010010000  // Configs 2 and 5
```

**Line 257-296**: Handle no valid paths (insert reshard if needed)

**Line 298-304**: Check if producer bitset needs constraining
```cpp
if (!isSubset(*producerBitset, edgeProducerBitset)) {
  // Producer bitset has configs not in edgeProducerBitset
  // Need to constrain producer → add to processor queue
  opProcessor.addOp(producerOp);
}
```

**Line 306-307**: **Constrain Bitsets** (Critical!)
```cpp
*producerBitset &= edgeProducerBitset;
*consumerBitset &= edgeConsumerBitset;
```

**Example**:
```cpp
// Before:
producerBitset[%4] = 0b1111111111  // All configs valid
edgeProducerBitset = 0b0000000001  // Only config0 valid

// After AND:
producerBitset[%4] = 0b0000000001  // Only config0 remains

// Before:
consumerBitset[%5] = 0b1111111111  // All configs valid
edgeConsumerBitset = 0b0010010000  // Configs 2,5 valid

// After AND:
consumerBitset[%5] = 0b0010010000  // Only configs 2,5 remain
```

**Line 309-313**: Create PathSet
```cpp
PathSetId pathSetId = pathSets.size();
pathSets.emplace_back(
    bitsetIds[%4],      // Producer bitset ID
    bitsetIds[%5],      // Consumer bitset ID
    %4,                 // Producer op
    %5,                 // Consumer op
    paths               // Valid paths
);
pathSetIds[Edge(%4, %5, 0)] = pathSetId;
```

**State After Edge Processing**:
```cpp
bitsets[0] = 0b0000000001  // %4: only config0
bitsets[1] = 0b0010010000  // %5: configs 2,5
pathSets[0] = PathSet(producerSetId=0, consumerSetId=1, paths=[Path(0,2), Path(0,5)])
pathSetIds[Edge(%4, %5, 0)] = 0
```

**Line 316**: Process operations in queue
```cpp
opProcessor.process(this);
// If %4 was added to queue, process it (may further constrain %4's bitset)
```

### Phase 4: Final Constraint Propagation (Line 319)

```cpp
for (const auto &shardSpec : *shardSpecs) {
  Operation *op = shardSpec.op;
  bool updateSuccess = updateSolver(op, false /* expand_root */);
  assert(updateSuccess);
}
```

**Calls `updateSolver()` for each op to propagate constraints**

---

## 6. updateSolver() - Iterative Constraint Propagation

**File**: `lib/Dialect/TTNN/Analysis/ShardSolver.cpp` (Line 834-920)

### Purpose

Propagate constraint changes through the graph. When one op's bitset changes, update connected ops.

### Algorithm

```cpp
bool updateSolver(Operation *root, bool expand_root, bool invokedBySet) {
  std::vector<Operation *> needsUpdate = {root};

  if (expand_root) {
    // Update path sets for root's operands and users
    auto operandPathSets = getOperandPathSetsPts(root);
    auto userPathSets = getUserPathSetsPts(root);

    for (auto *path_set : operandPathSets) {
      path_set->update(bitsets);  // Filter paths, update bitsets
    }

    for (auto *path_set : userPathSets) {
      path_set->update(bitsets);
    }

    addOperandsAndUsers(root, needsUpdate);  // Add connected ops to queue
  }

  // Iteratively process ops until no more changes
  while (not needsUpdate.empty()) {
    auto *op = needsUpdate.back();

    auto operandPathSets = getOperandPathSetsPts(op);
    auto userPathSets = getUserPathSetsPts(op);

    // Update operand path sets (producer → op)
    std::vector<bool> producersChanged(operandPathSets.size());
    for (size_t i = 0; i < operandPathSets.size(); i++) {
      producersChanged[i] = operandPathSets[i]->update(bitsets);
      // Returns true if bitsets changed
    }

    // Update user path sets (op → consumer)
    std::vector<bool> consumers_changed(userPathSets.size());
    for (size_t i = 0; i < userPathSets.size(); i++) {
      consumers_changed[i] = userPathSets[i]->update(bitsets);
    }

    // If producer bitsets changed, add producers to queue
    for (size_t i = 0; i < producersChanged.size(); i++) {
      if (producersChanged[i]) {
        Operation *producerOp = operandPathSets[i]->getProducerOp();
        needsUpdate.push_back(producerOp);
        addOperandsAndUsers(producerOp, needsUpdate, op);
      }
    }

    // If consumer bitsets changed, add consumers to queue
    for (size_t i = 0; i < consumers_changed.size(); i++) {
      if (consumers_changed[i]) {
        Operation *consumerOp = userPathSets[i]->getConsumerOp();
        needsUpdate.push_back(consumerOp);
        addOperandsAndUsers(consumerOp, needsUpdate, op);
      }
    }

    if (not edge_changed) {
      needsUpdate.pop_back();  // No changes, remove from queue
    }
  }

  return true;
}
```

### Example Propagation

**Initial State**:
```cpp
bitsets[0] = 0b0000000001  // %4: config0
bitsets[1] = 0b0010010000  // %5: configs 2,5
pathSets[0] = PathSet(paths=[Path(0,2), Path(0,5)])
```

**Call**: `updateSolver(%5, expand_root=false)`

**Step 1**: Get path sets
```cpp
operandPathSets[%5] = [pathSets[0]]  // Edge(%4, %5, 0)
userPathSets[%5] = []  // %5 has no sharded consumers
```

**Step 2**: Update operand path set
```cpp
pathSets[0]->update(bitsets)
// Checks: producer[0]=1, consumer[2]=1 → Path(0,2) valid
//         producer[0]=1, consumer[5]=1 → Path(0,5) valid
// validProducerSet = 0b0000000001
// validConsumerSet = 0b0010010000
// No change needed (already match)
// Return: false (no change)
```

**No propagation needed** - state is consistent.

**If %5's bitset was later constrained to only config2**:
```cpp
bitsets[1] = 0b0000010000  // Only config2

pathSets[0]->update(bitsets)
// Path(0,2): valid → KEEP
// Path(0,5): consumer[5]=0 → REMOVE
// validConsumerSet = 0b0000010000
// bitsets[1] &= 0b0000010000 → already 0b0000010000
// Return: false (no change)
```

**If %4's bitset was constrained**:
```cpp
bitsets[0] = 0b0000000000  // No valid configs!

pathSets[0]->update(bitsets)
// Path(0,2): producer[0]=0 → REMOVE
// Path(0,5): producer[0]=0 → REMOVE
// paths.empty() → return handleNoPathsLeftOnUpdate()
// → FAIL (no solution)
```

---

## 7. set() - Selecting a Config

**File**: `lib/Dialect/TTNN/Analysis/ShardSolver.cpp` (Line 968-1012)

Called by `pickOpShardConfigs()` to select a specific config for an operation.

```cpp
void ShardSolver::set(Operation *op, const OpConfig &config) {
  assert(selectedOpConfig.count(op) == 0);  // Not already set

  selectedOpConfig[op] = config;  // Store selection

  // Find config index
  const std::vector<OpConfig> &configs = getLegalConfigs(op);
  size_t selection = configs.size();
  for (size_t i = 0; i < configs.size(); ++i) {
    if (configs[i] == config) {
      selection = i;
      break;
    }
  }

  Bitset *op_bitset = getBitset(op);

  // Constrain bitset to ONLY this config
  op_bitset->reset();  // Clear all bits
  op_bitset->set(selection);  // Set only selected bit

  // Update reshard map if edge has reshard
  for (int64_t operandIdx = 0; operandIdx < op->getNumOperands(); ++operandIdx) {
    Value operand = op->getOperand(operandIdx);
    Operation *producerOp = operand.getDefiningOp();

    auto it = memReconfigMap.find(Edge(producerOp, op, operandIdx));
    if (it != memReconfigMap.end()) {
      it->second.setSelectedReshardOutputConfigBitIndex(selection);
    }
  }

  // Propagate constraint changes
  bool updateSuccessful = updateSolver(op, true /*expand_root*/, true /*invokedBySet*/);
  assert(updateSuccessful);
}
```

**Example**: `set(%4, config0)`
```cpp
// Before:
bitsets[0] = 0b0000000001  // Config0 valid

// After:
bitsets[0] = 0b0000000001  // Only config0 (no change, already constrained)

// Propagate to %5:
updateSolver(%4, expand_root=true, invokedBySet=true)
// Updates pathSets[0], may further constrain %5's bitset
```

---

## 8. finish() - Extract Solution

**File**: `lib/Dialect/TTNN/Analysis/ShardSolver.cpp` (Line 1116-1119)

```cpp
ShardSolverSolution ShardSolver::finish() const {
  assert(selectedOpConfig.size() == shardedOps->size());
  return ShardSolverSolution(selectedOpConfig, memReconfigMap);
}
```

**Returns**:
```cpp
{
  selectedOpConfig: {
    %4: OpConfig{outputLayout: #layout_2x2, ...},
    %5: OpConfig{outputLayout: #layout_2x1, ...}
  },
  memReconfigEntryMap: {
    // Empty if no reshard needed
    // Or: Edge(%4, %5, 0): MemReconfigEntry{...} if reshard needed
  }
}
```

---

## 9. Complete State Tracking for Chain1

### After Constructor

```cpp
operandOpEdges = {%4: [], %5: [Edge(%4, %5, 0)]}
userOpEdges = {%4: [Edge(%4, %5, 0)]}
bitsets = []
bitsetIds = {}
pathSets = []
pathSetIds = {}
```

### After preprocessFirstOp()

```cpp
bitsets[0] = 0b0000000001  // %4: only config0 (2x2 sharded)
bitsetIds = {%4: 0}
resolvedFirstOpInputLayout = #ttnn_layout4  // 2x2 block_sharded
```

### After Processing %4 in resolveStep()

```cpp
// %4 has no edges to sharded ops
// No path sets created
// Bitset unchanged: bitsets[0] = 0b0000000001
```

### After Processing %5 in resolveStep()

```cpp
bitsets[0] = 0b0000000001  // %4: config0
bitsets[1] = 0b0010010000  // %5: configs 2,5 (after constraint)
bitsetIds = {%4: 0, %5: 1}
pathSets[0] = PathSet(
  producerSetId=0,
  consumerSetId=1,
  paths=[Path(0,2), Path(0,5)]
)
pathSetIds = {Edge(%4, %5, 0): 0}
```

### After updateSolver() Calls

```cpp
// No further changes (state is consistent)
bitsets[0] = 0b0000000001  // %4: config0
bitsets[1] = 0b0010010000  // %5: configs 2,5
```

### After pickOpShardConfigs() → set()

```cpp
// set(%4, config0)
bitsets[0] = 0b0000000001  // %4: config0 (no change)

// set(%5, config2)  // Picked config2 (higher core usage)
bitsets[1] = 0b0000010000  // %5: only config2
selectedOpConfig = {%4: config0, %5: config2}
```

### After finish()

```cpp
ShardSolverSolution {
  selectedOpConfig: {
    %4: OpConfig{outputLayout: #layout_2x2, ...},
    %5: OpConfig{outputLayout: #layout_2x1, ...}
  },
  memReconfigEntryMap: {
    // If %4 config0 output != %5 config2 input:
    Edge(%4, %5, 0): MemReconfigEntry{
      reshardOutputConfigMap: {
        0: [configs that convert %4 output to %5 input]
      },
      selectedReshardOutputConfigBitIndex: 0
    }
  }
}
```

---

## 10. Key Insights

1. **Bitsets are constraints**: Each bit represents a valid config. Constraint propagation eliminates invalid configs.

2. **PathSets are compatibility matrices**: They store which (producer_config, consumer_config) pairs are valid.

3. **Constraint propagation is iterative**: When one op's bitset changes, connected ops are updated until no more changes.

4. **preprocessFirstOp() is special**: For dispatch consumer chains, it solves for optimal input by validating all (input, output) combinations.

5. **set() locks in a choice**: Once a config is selected, the bitset is constrained to only that config, and constraints propagate.

6. **Reshard edges are escape hatches**: If no valid paths exist, a reshard edge is inserted, allowing any producer config with specific consumer configs.

This design allows ShardSolver to efficiently explore the configuration space while ensuring compatibility constraints are satisfied.
