# Complete Line-by-Line Walkthrough: d2m.mlir Optimization Flow

## Input IR State (After d2m-fusing Pass)

```
func.func @dispatch_with_matmul_producer(
    %arg0: tensor<64x128xbf16, #layout>,      // L1 interleaved
    %arg1: tensor<128x256xbf16, #layout>,     // L1 interleaved
    %arg2: tensor<256x64xbf16, #layout_256x64>, // L1 interleaved
    %arg3: tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout> {

  %0 = "ttnn.matmul"(%arg0, %arg1) : ... -> tensor<64x256xbf16, #layout>
  %1 = "ttnn.exp"(%0) : ... -> tensor<64x256xbf16, #layout>
  %2 = "ttnn.neg"(%1) : ... -> tensor<64x256xbf16, #layout>
  %3 = "ttnn.abs"(%2) : ... -> tensor<64x256xbf16, #layout>
  // After d2m-fusing: exp, neg, abs are fused into dispatch_d2m
  %dispatch = "ttnn.dispatch_d2m"(%0, %empty) : ... -> tensor<64x256xbf16, #layout>
  %4 = "ttnn.matmul"(%dispatch, %arg2) : ... -> tensor<64x64xbf16, #layout_64x64>
  %5 = "ttnn.matmul"(%4, %arg3) : ... -> tensor<64x256xbf16, #layout>

  return %5 : tensor<64x256xbf16, #layout>
}
```

---

## Phase 1: DFShardingPolicy::run() - buildL1Chains()

**File: `lib/Dialect/TTNN/Analysis/DFShardingPolicy.cpp`**
**Function: `DFShardingPolicy::run()` starting at line 1371**

### Initial State

```cpp
// Line 1372-1384: Initialize
funcToProcess = nullptr
deviceAttr = (unset)
scheduler = Scheduler(&func)  // Creates DFS scheduler
l1ChainConfigs = [L1ChainConfig()]  // Vector with one empty chain
scheduleableOps = []
currentOp = nullptr
```

### Step 1: Walk Functions (Line 1374)

**Line 1374**: `rootOp->walk([&](func::FuncOp func) {`
- Finds `@dispatch_with_matmul_producer`
- **Line 1375**: Checks `isForwardDeviceFunc(func)` → **TRUE**
- **Line 1379**: `funcToProcess = func`
- **Line 1380**: `deviceAttr = ttcore::lookupDevice(func)` → Gets device grid (8x8)

### Step 2: Build L1 Chains Loop (Line 1392)

**Line 1392**: `while (scheduler.hasUnscheduledOps()) {`

#### Iteration 1: Process `%0 = ttnn.matmul(%arg0, %arg1)`

**Line 1393**: `scheduleableOps = scheduler.getSchedulableOps()`
- Returns: `[%0 = ttnn.matmul]` (only schedulable op)

**Line 1402**: `if (l1ChainConfigs->back().isEmpty())` → **TRUE** (chain is empty)
- **Line 1403-1408**: Check for ToLayoutOp → **NONE**

**Line 1411**: `if (currentOp == nullptr)` → **TRUE**
- **Line 1412**: `currentOp = scheduleableOps[0]` → `currentOp = %0 (matmul)`

**Line 1417**: `scheduler.scheduleOp(currentOp)` → Marks `%0` as scheduled

**Line 1421**: `if (l1ChainConfigs->back().isEmpty() && isa<ToLayoutOp>(currentOp))` → **FALSE**

**Line 1430**: `if (isa<DispatchD2MOp>(currentOp))` → **FALSE**

**Line 1446**: `bool validForSharding = ... && legalConfigs.lookup(currentOp).size() > 0`
- Checks if `%0` is a MatmulOp → **TRUE**
- Checks `legalConfigs[%0].size() > 0` → **TRUE** (has legal configs)
- **Result**: `validForSharding = TRUE`

**Line 1456**: `if (scheduler.hasUnscheduledOps())` → **TRUE**
- **Line 1458**: `scheduleableOps = scheduler.getSchedulableOps()`
  - Returns: `[%dispatch = dispatch_d2m]` (next schedulable)
- **Line 1462-1469**: Find nextOp that uses currentOp's output
  - Checks `%dispatch` operands → `%dispatch` uses `%0` as operand 0
  - **Line 1465**: `nextOp = %dispatch`

**Line 1472**: `if (validForSharding)` → **TRUE**
- **Line 1473-1479**: Create OpL1MemSpec for `%0`
  ```cpp
  OpL1MemSpec shardSpec;
  shardSpec.op = %0;
  shardSpec.tensorSplitFactor = 1;
  ```
- **Line 1480**: `l1ChainConfigs->back().addOpL1MemSpec(std::move(shardSpec))`
  - **State**: `l1ChainConfigs[0] = { opL1MemSpecs: [%0] }`

**Line 1482**: `if (nextOp && currentOp->hasOneUse())`
- `nextOp = %dispatch` → **TRUE**
- `%0->hasOneUse()` → Check if `%0` has exactly one use
  - `%0` is used by `%dispatch` → **TRUE**
- **Line 1485**: `if (nextOp->getOperand(0).getDefiningOp() != currentOp)`
  - `%dispatch->getOperand(0).getDefiningOp() == %0` → **TRUE**
  - Condition is **FALSE** (they match)
- **Line 1494**: `else if (mlir::isa<DispatchD2MOp>(nextOp))` → **TRUE**
  - **Line 1496-1500**: Debug log: "Breaking L1 chain at op %0 as next op %dispatch is DispatchD2MOp"
  - **Line 1501**: `currentOp = nullptr` → Break chain

**Line 1509**: `currentOp = nullptr`

**Line 1511**: `if (!l1ChainConfigs->back().isEmpty())` → **TRUE**
- **Line 1512**: `l1ChainConfigs->back().build()` → Sets state to `Built`
- **Line 1513**: `l1ChainConfigs->push_back(L1ChainConfig())` → Start new chain
  - **State**: `l1ChainConfigs = [Chain0: {%0}, Chain1: {}]`

#### Iteration 2: Process `%dispatch = dispatch_d2m`

**Line 1393**: `scheduleableOps = [%dispatch]`

**Line 1412**: `currentOp = %dispatch`

**Line 1417**: `scheduler.scheduleOp(%dispatch)`

**Line 1430**: `if (isa<DispatchD2MOp>(currentOp))` → **TRUE**
- **Line 1432**: `if (!l1ChainConfigs->back().isEmpty())` → **TRUE** (Chain1 is empty, but check anyway)
  - Chain1 is empty, so condition is **FALSE**
- **Line 1436**: `currentOp = nullptr`
- **Line 1437**: `continue` → Skip to next iteration

**Result**: `%dispatch` is NOT added to any chain (chain boundary)

#### Iteration 3: Process `%4 = ttnn.matmul(%dispatch, %arg2)`

**Line 1393**: `scheduleableOps = [%4]`

**Line 1412**: `currentOp = %4`

**Line 1417**: `scheduler.scheduleOp(%4)`

**Line 1446**: `validForSharding = TRUE` (%4 is MatmulOp with legal configs)

**Line 1456**: `if (scheduler.hasUnscheduledOps())` → **TRUE**
- **Line 1458**: `scheduleableOps = [%5]`
- **Line 1462-1469**: Find nextOp
  - `%5` uses `%4` as operand 0 → **TRUE**
  - `nextOp = %5`

**Line 1472**: `if (validForSharding)` → **TRUE**
- **Line 1480**: Add `%4` to Chain1
  - **State**: `l1ChainConfigs[1] = { opL1MemSpecs: [%4] }`

**Line 1482**: `if (nextOp && currentOp->hasOneUse())`
- `nextOp = %5` → **TRUE**
- `%4->hasOneUse()` → Check: `%4` is used by `%5` → **TRUE**
- **Line 1485**: `if (nextOp->getOperand(0).getDefiningOp() != currentOp)` → **FALSE** (%5 uses %4 as operand 0)
- **Line 1494**: `else if (mlir::isa<DispatchD2MOp>(nextOp))` → **FALSE**
- **Line 1503**: `currentOp = nextOp` → `currentOp = %5`
- **Line 1504**: `continue` → Continue growing chain

#### Iteration 4: Process `%5 = ttnn.matmul(%4, %arg3)` (continuing chain)

**Line 1446**: `validForSharding = TRUE`

**Line 1456**: `if (scheduler.hasUnscheduledOps())` → **FALSE** (no more ops)

**Line 1472**: `if (validForSharding)` → **TRUE**
- **Line 1480**: Add `%5` to Chain1
  - **State**: `l1ChainConfigs[1] = { opL1MemSpecs: [%4, %5] }`

**Line 1482**: `if (nextOp && currentOp->hasOneUse())`
- `nextOp = nullptr` → **FALSE**
- Skip to line 1509

**Line 1509**: `currentOp = nullptr`

**Line 1511**: `if (!l1ChainConfigs->back().isEmpty())` → **TRUE**
- **Line 1512**: `l1ChainConfigs->back().build()` → Chain1 state = `Built`
- **Line 1513**: `l1ChainConfigs->push_back(L1ChainConfig())` → Create Chain2 (empty)
  - **State**: `l1ChainConfigs = [Chain0: {%0}, Chain1: {%4, %5}, Chain2: {}]`

**Line 1392**: `while (scheduler.hasUnscheduledOps())` → **FALSE** (all ops scheduled)

### Step 3: Cleanup (Line 1520)

**Line 1520**: `if (!l1ChainConfigs->empty() && l1ChainConfigs->back().isEmpty())` → **TRUE**
- **Line 1522**: `l1ChainConfigs->pop_back()` → Remove empty Chain2
  - **Final State**: `l1ChainConfigs = [Chain0: {%0}, Chain1: {%4, %5}]`

**Line 1517**: `(*schedule)[func] = scheduler.getSchedule()`
- **State**: `schedule[func] = [%0, %dispatch, %4, %5]`

---

## Phase 2: Build Dispatch Relationships

**Line 1530-1536**: `buildDispatchChainRelationships()`

```cpp
std::vector<DispatchChainRelationship> dispatchRelationships;
dispatchRelationships = buildDispatchChainRelationships(
    schedule[func], l1ChainConfigs);
```

### Inside `buildDispatchChainRelationships()`:

For each `DispatchD2MOp` in schedule:
1. Find producer chain: Chain0 (contains `%0`)
2. Find consumer chain: Chain1 (contains `%4, %5`)
3. Create relationship:
   ```cpp
   DispatchChainRelationship {
     dispatchOp = %dispatch,
     producerChainIdx = 0,
     consumerChainIdx = 1
   }
   ```

**Result**: `dispatchRelationships = [{dispatchOp: %dispatch, producerChainIdx: 0, consumerChainIdx: 1}]`

---

## Phase 3: Compute Chain Solving Order

**Line 1545**: `computeChainSolvingOrder(l1ChainConfigs->size(), dispatchRelationships)`

### Inside `computeChainSolvingOrder()`:

1. Build dependency graph:
   - Chain0 → Chain1 (via dispatch)
2. Topological sort:
   - Chain0 has no dependencies → solve first
   - Chain1 depends on Chain0 → solve second
3. **Result**: `solvingOrder = [0, 1]`

**Line 1567-1573**: Build `chainToDispatchProducer` map
```cpp
llvm::DenseMap<size_t, Operation *> chainToDispatchProducer;
// For Chain1 (consumer), map to dispatch op
chainToDispatchProducer[1] = %dispatch;
```

---

## Phase 4: Solve Chains in Topological Order

### Solve Chain 0 (Producer Chain: `%0`)

**Line 1575**: `for (size_t chainIndex : solvingOrder)` → `chainIndex = 0`

**Line 1576**: `L1ChainConfig &l1ChainConfig = l1ChainConfigs[0]`

**Line 1579**: `numOpsInChain = 1` (`%0`)

**Line 1580**: `firstOp = %0`

**Line 1585**: `bool consumesFromDispatch = chainToDispatchProducer.count(0) > 0` → **FALSE**
- Chain0 does NOT consume from dispatch

**Line 1596**: `ShardSolver shardSolver = l1ChainConfig.resolveWithSolver(...)`
- **Parameter**: `solveForOptimalFirstOpInput = FALSE` (Chain0 doesn't consume from dispatch)

#### Inside `L1ChainConfig::resolveWithSolver()` (Line 26-45):

**Line 38**: Create ShardSolver:
```cpp
ShardSolver shardSolver(
    tensorTypePossibleLayouts,
    legalConfigs,
    opL1MemSpecs,  // [{op: %0}]
    l1ChainedOps,
    overrideReshardEdges,
    overrideOutputLayout,
    nullptr,  // customCheckShardCompatible
    false     // solveForOptimalFirstOpInput = FALSE
);
```

**Line 43**: `state = shardSolver.resolve() ? L1ChainState::Resolved : L1ChainState::Failed`

#### Inside `ShardSolver::resolve()` (Line 105-120):

**Line 116**: `if (!preprocessFirstOp())` → Call `preprocessFirstOp()`

#### Inside `ShardSolver::preprocessFirstOp()` (Line 412-611):

**Line 413**: `Operation *firstOp = %0`

**Line 415**: `Edge firstOpEdge = Edge(%arg0->getDefiningOp(), %0, 0)`
- `%arg0` is a block argument → `getDefiningOp() = nullptr`
- `firstOpEdge = Edge(nullptr, %0, 0)`

**Line 417**: `reshardOnEdgeExists = FALSE` (no reshard edge)

**Line 461**: `if (solveForOptimalFirstOpInput)` → **FALSE** (skip dispatch consumer logic)

**Line 570**: **Default path**: Assume DRAM interleaved input

**Line 571**: `for (size_t i = 0; i < firstOpConfigs.size(); ++i)`
- Iterate through legal configs for `%0`
- For each config, check if it supports interleaved input → sharded output
- **Line 587-590**: `supportsInterleavedInputShardedOutput(%0, config)`
  - Calls OpModel to validate
  - Finds valid configs

**Line 603**: `if (actualLayout == layoutForComparison)` → **TRUE** (found valid config)
- **Line 604-610**: Set bitset and return TRUE

**Return**: `preprocessFirstOp() = TRUE`

**Line 122**: `for (const auto &shardSpec : *shardSpecs)` → Only one spec: `{op: %0}`

**Line 126**: `Operation *consumerOp = %0`

**Line 128**: `consumerBitset = getOrInsertBitset(%0, kBitsetAll)`
- Creates bitset with all bits set (all configs initially valid)

**Line 130**: `consumerConfigs = getLegalConfigs(%0)` → Gets all legal configs for matmul

**Line 135**: `auto edges = operandOpEdges.find(%0)`
- Finds edges: `[{producer: nullptr (block arg), consumer: %0, operandIdx: 0}, ...]`

**Line 135-194**: Process each edge, constrain bitset based on compatibility

**Result**: ShardSolver finds valid configs for `%0`

**Line 1600**: `if (l1ChainConfig.getState() == L1ChainState::Failed)` → **FALSE**

**Line 1607**: `pickOpShardConfigs(shardSolver, l1ChainConfig)`

#### Inside `pickOpShardConfigs()` (Line 1700-1722):

**Line 1704**: `accMaxCoreUsage = shardSolver.produceMaxCoreUsage()`
- Computes core usage for each config

**Line 1708**: `lastOp = %0` (only op in chain)

**Line 1709**: `preferredMemLayout = std::nullopt` (no preference)

**Line 1716**: `if (preferredMemLayout)` → **FALSE**

**Line 1722-1750**: Process ops in order
- For `%0`: Pick config with highest core usage
- **Line 1750**: `shardSolver.set(%0, selectedConfig)`
  - Sets `selectedOpConfig[%0] = config`
  - Updates bitset

**Line 1609**: `ShardSolverSolution resolvedShardSolution = shardSolver.finish()`
- Returns: `{selectedOpConfig: {%0: config}, memReconfigEntryMap: {}}`

**Line 1610**: `l1ChainConfig.complete(resolvedShardSolution.selectedOpConfig, ...)`
- **State**: Chain0 completed with config for `%0`

**Line 1615**: `if (consumesFromDispatch && producerDispatchOp)` → **FALSE** (skip)

**Line 1630**: `Operation *lastOp = %0`

**Line 1631-1643**: Check if `%0` feeds `DispatchD2MOp`
- `%0->getResults()[0].getUsers()` → `{%dispatch}`
- `isa<DispatchD2MOp>(%dispatch)` → **TRUE**
- `feedsDispatchD2M = TRUE`

**Line 1645**: `if (feedsDispatchD2M)` → **TRUE**
- **Line 1650**: `l1ChainConfig.spillLocation = SpillLocation::None`
- **Line 1652**: Debug: "Chain ending at %0 feeds DispatchD2MOp - keeping in L1"

**Result**: Chain0 solved, output kept in L1 (not spilled to DRAM)

---

### Solve Chain 1 (Consumer Chain: `%4, %5`)

**Line 1575**: `chainIndex = 1`

**Line 1576**: `L1ChainConfig &l1ChainConfig = l1ChainConfigs[1]`

**Line 1579**: `numOpsInChain = 2` (`%4, %5`)

**Line 1580**: `firstOp = %4`

**Line 1585**: `bool consumesFromDispatch = chainToDispatchProducer.count(1) > 0` → **TRUE**
- Chain1 consumes from `%dispatch`

**Line 1587**: `producerDispatchOp = %dispatch`

**Line 1589**: `if (consumesFromDispatch)` → **TRUE**
- **Line 1590-1593**: Debug: "Chain 1 consumes from dispatch_d2m - solving for optimal input layout"

**Line 1596**: `ShardSolver shardSolver = l1ChainConfig.resolveWithSolver(...)`
- **Parameter**: `solveForOptimalFirstOpInput = TRUE` ← **KEY DIFFERENCE**

#### Inside `ShardSolver::preprocessFirstOp()` with `solveForOptimalFirstOpInput = TRUE`:

**Line 413**: `Operation *firstOp = %4`

**Line 461**: `if (solveForOptimalFirstOpInput)` → **TRUE** ← **ENTER DISPATCH CONSUMER LOGIC**

**Line 467-477**: Find which operand comes from DispatchD2MOp
- Iterate through `%4->getOperands()`
- `%4->getOperand(0) = %dispatch->getResult(0)`
- `%dispatch->getResult(0).getDefiningOp() = %dispatch`
- `isa<DispatchD2MOp>(%dispatch)` → **TRUE**
- `dispatchOperandIdx = 0`

**Line 485**: `Value dispatchOperand = %4->getOperand(0)` → `%dispatch->getResult(0)`

**Line 486**: `inputTensorType = RankedTensorType(64x256xbf16, #layout)`

**Line 491**: `candidateInputLayouts = getShardedLayoutsForTensorTypeAndScalarType(...)`
- Gets all possible L1 sharded layouts for `64x256xbf16` tensor
- **Result**: ~17 candidate layouts (different sharding patterns: 2x2, 2x4, 4x2, etc.)

**Line 507**: `for (const TTNNLayoutAttr &candidateInput : candidateInputLayouts)`
- Iterate through each candidate input layout

**Line 512**: `inputLayouts[0] = candidateInput` (set dispatch operand to candidate)

**Line 514**: `for (size_t i = 0; i < firstOpConfigs.size(); ++i)`
- Iterate through all legal output configs for `%4`

**Line 527-530**: **KEY VALIDATION**:
```cpp
op_constraint_validation::ValidationResult validationResult =
    op_constraint_validation::validateOperation(
        %4,                    // Operation
        inputLayouts,           // [candidateInput, %arg2 layout]
        OpConfig(outputLayout, opSpecificAttrs)  // Desired output config
    );
```

This calls OpModel to check: **"Can %4 accept this input layout and produce this output config?"**

**Line 532**: `if (validationResult.isError())` → Skip if invalid

**Line 536**: `actualLayout = validationResult.actualOutputLayout`
- OpModel returns the actual output layout it would produce

**Line 537**: `if (actualLayout == layoutForComparison)` → Check if matches desired
- If matches, this is a valid (input, output) combination

**Line 538**: `validCombinations.emplace_back(candidateInput, i)`
- Store valid combination

**After loop**: `validCombinations` contains all valid (input_layout, output_config_idx) pairs

**Line 547**: `if (!validCombinations.empty())` → **TRUE** (found valid combinations)

**Line 550**: `auto [bestInput, bestOutputIdx] = validCombinations[0]`
- Currently picks first valid (TODO: add scoring)
- **Example**: `bestInput = #ttnn_layout4` (2x2 block_sharded)
- `bestOutputIdx = 0`

**Line 551**: `resolvedFirstOpInputLayout = bestInput`
- **Stores**: `resolvedFirstOpInputLayout = #ttnn_layout4`

**Line 552**: `firstOpBitset->set(bestOutputIdx)`
- Sets bitset for `%4` to only allow config at index 0

**Line 559**: `return true` → Success

**Result**: Found optimal input layout `#ttnn_layout4` (2x2 block_sharded) for `%4`

#### Continue ShardSolver::resolve():

**Line 122**: `for (const auto &shardSpec : *shardSpecs)` → Process `%4` and `%5`

**For `%4`**:
- Bitset already constrained by `preprocessFirstOp()`
- Process edges, further constrain if needed

**For `%5`**:
- Process normally (not first op, so no special dispatch logic)
- Constrain based on `%4`'s output layout

**Result**: ShardSolver finds valid configs for both `%4` and `%5`

**Line 1607**: `pickOpShardConfigs(shardSolver, l1ChainConfig)`
- Picks configs for `%4` and `%5` based on core usage

**Line 1609**: `resolvedShardSolution = shardSolver.finish()`
- Returns: `{selectedOpConfig: {%4: config1, %5: config2}, memReconfigEntryMap: {...}}`

**Line 1610**: `l1ChainConfig.complete(...)`
- **State**: Chain1 completed

**Line 1615**: `if (consumesFromDispatch && producerDispatchOp)` → **TRUE**
- **Line 1616**: `optimalInput = shardSolver.getResolvedFirstOpInputLayout()`
  - Returns: `#ttnn_layout4` (2x2 block_sharded)
- **Line 1618**: `dispatchOutputLayouts[%dispatch] = optimalInput`
  - **State**: `dispatchOutputLayouts[%dispatch] = #ttnn_layout4`

**Line 1630**: `lastOp = %5`

**Line 1631-1643**: Check if `%5` feeds `DispatchD2MOp` → **FALSE**

**Line 1658**: `else if (!resolvedShardSolution.selectedOpConfig[%5].outputLayout.hasDRAMBufferType())`
- Output is L1 sharded → **TRUE**
- **Line 1661**: `l1ChainConfig.spillLocation = SpillLocation::DRAM`
- Chain1 output will be spilled to DRAM

---

## Phase 5: Compute DispatchD2M Configs

**Line 1680**: `dispatchD2MConfigs = computeDispatchD2MConfigs(...)`

### Inside `computeDispatchD2MConfigs()`:

1. For each `DispatchD2MOp`:
   - Find producer chain (Chain0)
   - Get producer chain's output layout → `#ttnn_layout3` (2x8 block_sharded)
   - Compute L1 budget: `usableL1CacheSize - producer_output_size - consumer_input_size`
   - **Result**: `dispatchD2MConfigs[%dispatch] = {inputLayout: #ttnn_layout3, outputLayout: (unset), l1Budget: 1463296}`

**Line 1686**: `for (const auto &[dispatchOp, layout] : dispatchOutputLayouts)`
- Override output layout with consumer's optimal input
- **Line 1688**: `dispatchD2MConfigs[%dispatch].outputLayout = #ttnn_layout4`
  - **Final State**: `dispatchD2MConfigs[%dispatch] = {inputLayout: #ttnn_layout3, outputLayout: #ttnn_layout4, l1Budget: 1463296}`

---

## Key Data Structures Throughout

### After buildL1Chains():
```cpp
l1ChainConfigs = [
  Chain0: {
    state: Built,
    opL1MemSpecs: [{op: %0}],
    spillLocation: None  // Feeds dispatch, keep in L1
  },
  Chain1: {
    state: Built,
    opL1MemSpecs: [{op: %4}, {op: %5}],
    spillLocation: DRAM  // Will be set later
  }
]

schedule[func] = [%0, %dispatch, %4, %5]

dispatchRelationships = [{
  dispatchOp: %dispatch,
  producerChainIdx: 0,
  consumerChainIdx: 1
}]

chainToDispatchProducer = {
  1: %dispatch  // Chain1 consumes from dispatch
}
```

### After solveChains():
```cpp
l1ChainConfigs = [
  Chain0: {
    state: Completed,
    opL1MemSpecs: [{op: %0, config: {...}}],
    spillLocation: None
  },
  Chain1: {
    state: Completed,
    opL1MemSpecs: [{op: %4, config: {...}}, {op: %5, config: {...}}],
    spillLocation: DRAM
  }
]

dispatchOutputLayouts = {
  %dispatch: #ttnn_layout4  // Consumer's optimal input
}

dispatchD2MConfigs = {
  %dispatch: {
    inputLayout: #ttnn_layout3,   // Producer's output
    outputLayout: #ttnn_layout4,  // Consumer's optimal input
    l1Budget: 1463296
  }
}
```

---

## Summary of Flow

1. **buildL1Chains()**: Creates chains by following dataflow, breaks at `DispatchD2MOp`
   - Chain0: `{%0}` (producer)
   - Chain1: `{%4, %5}` (consumer)

2. **buildDispatchChainRelationships()**: Links chains through dispatch ops

3. **computeChainSolvingOrder()**: Topological sort ensures producers solved before consumers

4. **solveChains()**:
   - Chain0: Normal solving (interleaved input → sharded output)
   - Chain1: **Special logic** - solves for optimal input layout using OpModel validation
   - Stores optimal input as dispatch output layout

5. **computeDispatchD2MConfigs()**: Computes L1 budgets and sets layouts

The key insight: **ShardSolver avoids chicken-and-egg by using OpModel to validate (input, output) pairs independently**, then propagates constraints forward through the chain.
