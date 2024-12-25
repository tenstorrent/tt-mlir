// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/ShardSolver.h"
#include "ttmlir/Dialect/TTNN/Analysis/L1ChainConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include <mlir/Interfaces/DestinationStyleOpInterface.h>
#include <mlir/Support/LLVM.h>
#include <unordered_set>

namespace mlir::tt::ttnn {

ShardSolver::Bitset ShardSolver::kBitsetAll = ~kBitsetNone;

ShardSolver::ShardSolver(
    const llvm::DenseMap<Operation *, std::vector<TTNNLayoutAttr>>
        &legalLayouts,
    const std::vector<OpL1MemSpec> &shardSpecs,
    const llvm::DenseSet<Operation *> &shardedOps,
    const unsigned usableL1CacheSize,
    const std::unordered_set<Edge> &overrideReshardEdges,
    std::function<bool(Operation *, TTNNLayoutAttr const &, Operation *,
                       TTNNLayoutAttr const &)>
        customCheckShardCompatible)
    : legalLayouts(&legalLayouts), shardSpecs(&shardSpecs),
      shardedOps(&shardedOps), usableL1CacheSize(usableL1CacheSize),
      memReconfigEdges(overrideReshardEdges),
      customCheckShardCompatible(customCheckShardCompatible) {
  pathSets.reserve(shardSpecs.size());
  pathSetIds.reserve(shardSpecs.size());
  bitsets.reserve(shardedOps.size());
  bitsetIds.reserve(shardedOps.size());

  // Cache DeviceAttr.
  //
  deviceAttr = getCurrentScopeDevice(shardSpecs.front().op);

  // Populate operandOpEdges and userOpEdges.
  //
  for (const auto shardSpec : shardSpecs) {
    Operation *op = shardSpec.op;
    for (size_t operandIndex = 0; operandIndex < op->getNumOperands();
         operandIndex++) {
      Value operand = op->getOperand(operandIndex);
      Operation *operandOp = operand.getDefiningOp();
      if (operandOp && shardedOps.count(operandOp) > 0) {
        operandOpEdges[op].emplace_back(Edge(operandOp, op, operandIndex));
        userOpEdges[operandOp].emplace_back(Edge(operandOp, op, operandIndex));
      }
    }
  }

  // Resolve shard chain.
  //
  resolve();
}

void ShardSolver::reset() {
  pathSets.clear();
  pathSetIds.clear();
  bitsets.clear();
  bitsetIds.clear();
}

bool ShardSolver::resolveStep() {
  OperationPathsProcessor opProcessor;
  bitsets.reserve(shardedOps->size());
  bitsetIds.reserve(shardedOps->size());
  selectedOpLayout.reserve(shardedOps->size());
  bool reshardInserted = false;

  // We need special handling for the first op in the chain.
  //
  preprocessFirstOp();

  for (const auto shardSpec : *shardSpecs) {
    Operation *consumerOp = shardSpec.op;
    Bitset *consumerBitset = getOrInsertBitset(consumerOp, kBitsetAll);
    std::vector<TTNNLayoutAttr> const &consumerLayouts =
        getLegalLayouts(consumerOp);

    for (Edge edge : operandOpEdges[consumerOp]) {

      bool reshardOnEdge = memReconfigEdges.count(edge) > 0;

      Operation *producerOp = edge.producerOp;
      Bitset *producerBitset = getOrInsertBitset(producerOp, kBitsetAll);
      std::vector<TTNNLayoutAttr> const &producerLayouts =
          getLegalLayouts(producerOp);

      assert(not(consumerLayouts.empty() && producerLayouts.empty()));

      PathSet::Paths paths;
      Bitset edgeProducerBitset = kBitsetNone;
      Bitset edgeConsumerBitset = kBitsetNone;
      std::uint64_t producer_count =
          std::min(kNumBitsetBits, std::max(1lu, producerLayouts.size()));
      std::uint64_t consumer_count =
          std::min(kNumBitsetBits, std::max(1lu, consumerLayouts.size()));
      for (std::uint64_t producerId = 0; producerId < producer_count;
           ++producerId) {
        // If the producer cannot accomodate this path, continue.
        // Also if this is not the TTNNLayoutAttr we selected, continue.
        //
        if (!producerBitset->test(producerId)) {
          continue;
        }

        for (std::uint64_t consumerId = 0; consumerId < consumer_count;
             ++consumerId) {

          // If the consumer cannot accomodate this path, continue.
          //
          if (!consumerBitset->test(consumerId)) {
            continue;
          }

          // TODO(nobradovic):
          // Update checkShardCompatible with op type, other input
          // spec(weight).
          //
          bool validShardPair =
              reshardOnEdge ||
              checkShardCompatible(producerOp, producerLayouts[producerId],
                                   consumerOp, consumerLayouts[consumerId]);

          if (validShardPair) {
            assert(producerId <=
                   std::numeric_limits<decltype(Path::producerId)>::max());
            assert(consumerId <=
                   std::numeric_limits<decltype(Path::consumerId)>::max());
            paths.push_back(Path(producerId, consumerId));
            edgeProducerBitset.set(producerId);
            edgeConsumerBitset.set(consumerId);
          }
        }
      }

      if (paths.empty() || ((*producerBitset & edgeProducerBitset) == 0) ||
          ((*consumerBitset & edgeConsumerBitset) == 0)) {

        // No valid paths found for this edge, mark it for resharding.
        //
        insertReshard(edge);
        reshardInserted = true;
        *consumerBitset = kBitsetAll;
      }

      if (!isSubset(*producerBitset, edgeProducerBitset) && !reshardInserted) {
        opProcessor.addOp(producerOp);
      }

      *producerBitset &= edgeProducerBitset;
      *consumerBitset &= edgeConsumerBitset;
      assert(pathSetIds.find(edge) == pathSetIds.end());
      PathSetId pathSetId = static_cast<PathSetId>(pathSets.size());
      pathSets.emplace_back(bitsetIds[producerOp], bitsetIds[consumerOp],
                            producerOp, consumerOp, paths);
      pathSetIds.emplace(edge, pathSetId);
    }

    if (!reshardInserted) {
      opProcessor.process(this);
    }
  }

  if (reshardInserted) {
    return false;
  }

  for (const auto shardSpec : *shardSpecs) {
    Operation *op = shardSpec.op;

    // No need to expand root as we are calling for all ops anyway.
    //
    updateSolver(op, false /* expand_root */);
  }

  return true;
}

bool ShardSolver::supportsInterleavedInputShardedOutput(Operation *op) {
  // TODO(nobradovic,mbezulj): Add check whether this op type can have sharded
  // output from interleaved inputs. For now assuming it can.
  //
  return true;
}

// We need to check if first op requires sharded inputs and if so, insert
// reshard edge, then invalidate all sharding options which would go above L1
// size limits.
//
void ShardSolver::preprocessFirstOp() {
  Operation *firstOp = shardSpecs->front().op;
  if (supportsInterleavedInputShardedOutput(firstOp) &&
      memReconfigEdges.count(
          Edge(firstOp->getOperand(0).getDefiningOp(), firstOp, 0)) == 0) {
    return;
  }

  // Insert reshard edge for the first op to start the chain.
  //
  Edge shardChainInputEdge =
      Edge(firstOp->getOperand(0).getDefiningOp(), firstOp, 0 /*operandIndex*/);

  if (memReconfigEdges.count(shardChainInputEdge) == 0) {
    insertReshard(shardChainInputEdge);
  }

  Bitset *firstOpBitset = getOrInsertBitset(firstOp, kBitsetAll);
  std::vector<TTNNLayoutAttr> const &firstOpLayouts = getLegalLayouts(firstOp);
  Operation *operandOp = firstOp->getOperand(0).getDefiningOp();

  RankedTensorType firstOpInputTensorType =
      mlir::cast<RankedTensorType>(operandOp->getResult(0).getType());
  TTNNLayoutAttr firstOpInputLayout =
      mlir::cast<TTNNLayoutAttr>(firstOpInputTensorType.getEncoding());
  constexpr float tensorL1UsageCap = 0.8;

  for (size_t i = 0; i < firstOpLayouts.size(); ++i) {
    if (!firstOpBitset->test(i)) {
      continue;
    }

    TTNNLayoutAttr firstOpLayout = firstOpLayouts[i];
    assert(firstOpLayout.hasShardedL1TensorMemoryLayout());

    TTNNLayoutAttr firstOpInputShardedLayout =
        firstOpInputLayout
            .withBufferType(firstOp->getContext(),
                            firstOpLayout.getBufferType())
            .withMemoryLayout(firstOp->getContext(),
                              firstOpLayout.getMemLayout())
            .withGrid(firstOp->getContext(), firstOpInputTensorType,
                      firstOpLayout.getGrid());

    uint64_t firstInputL1Usage = firstOpInputShardedLayout.getTensorSizeInBytes(
        firstOpInputTensorType.getShape(), deviceAttr);
    uint64_t firstOpL1OutputUsage = firstOpLayout.getTensorSizeInBytes(
        mlir::cast<RankedTensorType>(firstOp->getResult(0).getType())
            .getShape(),
        deviceAttr);

    if ((firstInputL1Usage + firstOpL1OutputUsage) >=
        tensorL1UsageCap * usableL1CacheSize) {
      firstOpBitset->reset(i);
    }
  }
}

void ShardSolver::insertReshard(const Edge &edge) {
  // Same edge should not be resharded twice!
  //
  assert(memReconfigEdges.count(edge) == 0);
  memReconfigEdges.insert(edge);
}

void ShardSolver::resolve() {

  const int max_retry_step = shardedOps->size() + 1;
  int retry_step = 1;
  bool resolved = false;

  do {
    // Reset ShardSolver to default state.
    //
    reset();

    // Try to resolve shard chain. Retry if not resolved(resharding).
    //
    resolved = resolveStep();
    retry_step++;
  } while (!resolved && retry_step <= max_retry_step);

  assert(resolved);
}

ShardSolver::PathSet *ShardSolver::getPathSetPt(const Edge &edge) {
  if (pathSetIds.count(edge) > 0) {
    return &pathSets[pathSetIds.at(edge)];
  }

  return nullptr;
}

SmallVector<ShardSolver::PathSet *>
ShardSolver::getOperandPathSetsPts(Operation *op) {
  SmallVector<PathSet *> operandPathSets;
  for (auto edge : operandOpEdges[op]) {
    PathSet *el = getPathSetPt(edge);
    if (nullptr != el) {
      operandPathSets.push_back(el);
    }
  }

  return operandPathSets;
}

SmallVector<ShardSolver::PathSet *>
ShardSolver::getUserPathSetsPts(Operation *op) {
  SmallVector<PathSet *> userPathSets;
  for (auto edge : userOpEdges[op]) {
    PathSet *el = getPathSetPt(edge);
    if (nullptr != el) {
      userPathSets.push_back(el);
    }
  }

  return userPathSets;
}

void ShardSolver::addOperandsAndUsers(Operation *op,
                                      std::vector<Operation *> &needsUpdate,
                                      Operation *ignoreOp) {

  for (auto operand : op->getOperands()) {
    Operation *opOperand = operand.getDefiningOp();
    if (opOperand == nullptr ||
        !llvm::isa<mlir::DestinationStyleOpInterface>(opOperand) ||
        opOperand == ignoreOp || shardedOps->count(opOperand) == 0) {
      continue;
    }

    needsUpdate.push_back(opOperand);
  }

  for (Operation *opUser : op->getUsers()) {
    if (opUser == nullptr || opUser == ignoreOp ||
        shardedOps->count(opUser) == 0) {
      continue;
    }

    needsUpdate.push_back(opUser);
  }
}

void ShardSolver::handleNoPathsLeftOnUpdate(bool invokedBySet) {
  // We ended-up in a situation without valid solution due to circular
  // dependency.
  //
  assert(invokedBySet);

  // Invoking resolve again will use resharding to resolve the issue.
  //
  return resolve();
}

void ShardSolver::updateSolver(Operation *root, bool expand_root,
                               bool invokedBySet) {
  std::vector<Operation *> needsUpdate = {root};

  if (expand_root) {
    auto operandPathSets = getOperandPathSetsPts(root);
    auto userPathSets = getUserPathSetsPts(root);

    for (auto *path_set : operandPathSets) {
      path_set->update(bitsets);
    }

    for (auto *path_set : userPathSets) {
      path_set->update(bitsets);
    }

    // When op bitsets are updated(set of valid op layouts), we need to update
    // paths for all operands and users.
    //
    addOperandsAndUsers(root, needsUpdate);
  }

  // Iterate through the ops that need to be updated and update their operand
  // and user path sets.
  while (not needsUpdate.empty()) {
    auto *op = needsUpdate.back();

    // Get path sets for incoming edges
    auto operandPathSets = getOperandPathSetsPts(op);
    // Get path sets for outgoing edges
    auto userPathSets = getUserPathSetsPts(op);

    bool edge_changed = false;

    std::vector<bool> producersChanged(operandPathSets.size());
    for (size_t i = 0; i < operandPathSets.size(); i++) {
      auto *operandPathSet = operandPathSets[i];
      producersChanged[i] = operandPathSet->update(bitsets);

      if (operandPathSet->empty(bitsets)) {
        return handleNoPathsLeftOnUpdate(invokedBySet);
      }
    }

    std::vector<bool> consumers_changed(userPathSets.size());
    for (size_t i = 0; i < userPathSets.size(); i++) {
      auto *userPathSet = userPathSets[i];
      consumers_changed[i] = userPathSet->update(bitsets);

      if (userPathSet->empty(bitsets)) {
        return handleNoPathsLeftOnUpdate(invokedBySet);
      }
    }

    // If any of the paths between producer and this consumer changed, we need
    // to visit producer op and add its operands and users to the needsUpdate
    // list.
    for (size_t i = 0; i < producersChanged.size(); i++) {
      if (producersChanged[i]) {
        Operation *producerOp = operandPathSets[i]->getProducerOp();
        needsUpdate.push_back(producerOp);
        addOperandsAndUsers(producerOp, needsUpdate, op);

        edge_changed = true;
      }
    }

    // If any of the paths between this producer and consumer changed, we need
    // to visit consumer op and add its operands and users to the needsUpdate
    // list.
    for (size_t i = 0; i < consumers_changed.size(); i++) {
      if (consumers_changed[i]) {
        Operation *consumerOp = userPathSets[i]->getConsumerOp();
        needsUpdate.push_back(consumerOp);
        addOperandsAndUsers(consumerOp, needsUpdate, op);

        edge_changed = true;
      }
    }

    if (not edge_changed) {
      needsUpdate.pop_back();
    }
  }
}

ShardSolver::Bitset *ShardSolver::getBitset(Operation *op) {
  return &bitsets[bitsetIds.at(op)];
}

ShardSolver::Bitset const *ShardSolver::getBitset(Operation *op) const {
  return &bitsets[bitsetIds.at(op)];
}

ShardSolver::Bitset *ShardSolver::getOrInsertBitset(Operation *op,
                                                    const Bitset &init) {
  auto match = bitsetIds.find(op);
  if (match == bitsetIds.end()) {
    BitsetId bitset_id = bitsets.size();
    bitsetIds.insert({op, bitset_id});
    auto *tmp = bitsets.data();
    bitsets.push_back(init);

    // Bitsets reallocated, pointers invalid.
    //
    assert(tmp == bitsets.data());
    return &bitsets.back();
  }

  return &bitsets[match->second];
}

// Returns vector of legal LayoutAttrs for passed in op.
//
const std::vector<TTNNLayoutAttr> &
ShardSolver::getLegalLayouts(Operation *op) const {
  static std::vector<TTNNLayoutAttr> nullLayouts;

  const auto legalIt = legalLayouts->find(op);

  if (legalIt != legalLayouts->end()) {
    return legalIt->second;
  }

  return nullLayouts;
}

ShardSolver::RemainingLayoutAttrs ShardSolver::at(Operation *op) const {
  auto layouts = RemainingLayoutAttrs(getLegalLayouts(op), *getBitset(op));
  assert(layouts.begin() != layouts.end());
  return layouts;
}

void ShardSolver::set(Operation *op, TTNNLayoutAttr const &layout) {
  assert(selectedOpLayout.count(op) == 0);

  selectedOpLayout[op] = layout;

  auto const &layouts = getLegalLayouts(op);
  assert(!layouts.empty());
  size_t selection = layouts.size();
  for (size_t i = 0; i < layouts.size(); ++i) {
    if (layouts[i] == layout) {
      selection = i;
      break;
    }
  }

  Bitset *op_bitset = getBitset(op);

  assert(selection != layouts.size());
  assert((*op_bitset)[selection]);

  op_bitset->reset();
  op_bitset->set(selection);

  updateSolver(op, true /*expand_root*/, true /*invokedBySet*/);
}

bool ShardSolver::checkShardCompatible(
    Operation *producerOp, TTNNLayoutAttr const &producerLayout,
    Operation *consumerOp, TTNNLayoutAttr const &consumerLayout) const {

  // Custom(test) hook for shard compatibility check.
  //
  if (customCheckShardCompatible) {
    return customCheckShardCompatible(producerOp, producerLayout, consumerOp,
                                      consumerLayout);
  }

  // TEMP : Dummy mock implementation, will be replaced.
  //

  if (OpModel backend = dyn_cast<OpModel>(consumerOp)) {

    auto deviceAttr = mlir::tt::getCurrentScopeDevice(producerOp);
    assert(deviceAttr);
    auto workerGrid = deviceAttr.getWorkerGrid();

    // map consumer operands to DRAM interleave or provided producerLayout
    // only one operand can be mapped to producerLayout, it's picked as first
    // operand matching producerOp output shape.

    uint32_t numOperands = consumerOp->getNumOperands();
    // some ops have multiple operands; and some ops have output also an
    // operand. TBD if there is a more robust way to get real number of inputs

    // TODO(odjuricic): cast to DPSop?
    numOperands = (numOperands > 1) ? numOperands - 1 : numOperands;
    std::vector<TTNNLayoutAttr> inputLayouts;

    auto inputUnderCheck =
        mlir::cast<RankedTensorType>(producerOp->getResult(0).getType());
    bool inputUnderCheckFound = false;

    for (uint32_t i = 0; i < numOperands; i++) {
      auto operand = consumerOp->getOperand(i);
      auto input = mlir::cast<RankedTensorType>(operand.getType());

      if ((inputUnderCheckFound == false) &&
          (inputUnderCheck.getShape() == input.getShape())) {
        // this is the input we are checking compatibility for
        inputUnderCheckFound = true;
        inputLayouts.push_back(producerLayout);
      } else {
        // this is the other input that we DRAM interleave

        // what if it is tilized already?
        auto elementType =
            TileType::get(consumerOp->getContext(), input.getElementType());

        auto layout = TTNNLayoutAttr::get(
            consumerOp->getContext(), input.getShape(), elementType,
            BufferType::DRAM, workerGrid,
            TensorMemoryLayoutAttr::get(consumerOp->getContext(),
                                        TensorMemoryLayout::Interleaved));
        inputLayouts.push_back(layout);
      }
    }

    auto [legal, l1Usage, errorMsg] =
        backend.getOpConstraints(inputLayouts, consumerLayout);

    constexpr bool debug = false;
    if (false == legal) {
      // early exit
      if (debug) {
        llvm::errs() << "OpModel constraints failed: ";
        llvm::errs() << producerOp->getName() << "->" << consumerOp->getName()
                     << " :: " << errorMsg.value() << "\n";
        producerLayout.dump();
        consumerLayout.dump();
      }
      return false;
    }
    if (debug) {
      llvm::errs() << "OpModel constraints valid. ";
      llvm::errs() << producerOp->getName() << "->" << consumerOp->getName()
                   << "\n";
      producerLayout.dump();
      consumerLayout.dump();
    }
  }

  // May need to fetch other inputs for consumerOp(weights/join node).
  //

  // Need to plug shard checking API.
  //

  // Need to plug API for L1 usage.
  //
  // Calculate L1 tensor memory usage based on :
  // currentOp output tensor shard spec, nextOp exec and nextOp output
  // tensor.
  //
  assert(producerLayout.hasShardedL1TensorMemoryLayout() &&
         consumerLayout.hasShardedL1TensorMemoryLayout());

  // Perform L1 usage check only if deviceAttr is available.
  //
  if (deviceAttr) {
    RankedTensorType producerTensorType =
        mlir::cast<RankedTensorType>(producerOp->getResult(0).getType());
    uint64_t producerL1OutputUsage = producerLayout.getTensorSizeInBytes(
        producerTensorType.getShape(), deviceAttr);

    RankedTensorType consumerTensorType =
        mlir::cast<RankedTensorType>(consumerOp->getResult(0).getType());
    uint64_t consumerL1OutputUsage = consumerLayout.getTensorSizeInBytes(
        consumerTensorType.getShape(), deviceAttr);
    // Figure out this const based on exec data, but will be replaced
    // with API.
    //
    constexpr float tensorL1UsageCap = 0.8;
    bool l1UsageValid = (producerL1OutputUsage + consumerL1OutputUsage) <
                        tensorL1UsageCap * usableL1CacheSize;

    if (!l1UsageValid) {
      return false;
    }
  }

  return true;
}

// Preprocess ShardSolver search space to make a helper structure which links op
// layout choices to global max core usage.
// Example:
// Lets assume simple case where layouts at same index are compatible for input
// graph provided below. Tupples represent layout core
// usage (Layout0GridVolume, Layout1GridVolume, Layout2GridVolume).
//
//    Op0 ----- (4, 8, 2)
//     |
//    Op1 ----- (8, 4, 2)
//    / \
//   /   \
//  Op2  Op3 -- (4, 4, 2) (4, 4, 2)
//   \   /
//    \ /
//    Op4 ----- (2, 1, 1)
//     |
//    Op5 ----- (2, 1, 1)
//
// Here is how structure looks after preprocessing is complete:
//
//    Op0 ----- (24, 22, 10)
//     |
//    Op1 ----- (20, 14, 8)
//    / \
//   /   \
//  Op2  Op3 -- (6, 5, 3) (6, 5, 3)
//   \   /
//    \ /
//    Op4 ----- (4, 2, 2)
//     |
//    Op5 ----- (2, 1, 1)
//
// Global max of 24 core usage is achieved by selecting layout[0] for each Op.
//
// Returns map of op to vector of max core usage for each layout.
llvm::DenseMap<Operation *, SmallVector<float, 64>>
ShardSolver::produceMaxCoreUsage() {
  using Paths = llvm::SmallVector<Path, 16>;
  llvm::DenseMap<Operation *, SmallVector<float, 64>> accCoreUsage(
      shardedOps->size());

  // Start from the tail of the chain and build up the max core usage(schedule
  // in backwards).
  //
  for (auto shardSpec = shardSpecs->rbegin(); shardSpec != shardSpecs->rend();
       ++shardSpec) {
    Operation *op = shardSpec->op;
    std::vector<TTNNLayoutAttr> const &layouts = getLegalLayouts(op);
    assert(!layouts.empty());

    // Find the layout that leads to the max core usage.
    // Start with grid volume of current op.
    //
    for (size_t i = 0; i < layouts.size(); ++i) {
      TTNNLayoutAttr const &layout = layouts[i];
      uint64_t coreUsage = layout.getGrid().getGridVolume();
      accCoreUsage[op].push_back(coreUsage);
    }

    // Add core usage of current op users via live path connections.
    //
    SmallVector<ShardSolver::PathSet *> userPathSets = getUserPathSetsPts(op);
    for (size_t i = 0; i < userPathSets.size(); ++i) {
      ShardSolver::PathSet *pathSet = userPathSets[i];
      const Paths &paths = pathSet->getPaths();
      SmallVector<uint64_t, 64> maxCoreUsage(layouts.size(), 0);
      Operation *consumerOp = pathSet->getConsumerOp();
      size_t consumerInChainOperandSize =
          getOperandPathSetsPts(consumerOp).size();
      uint64_t consumerCoreUsage = 0;
      for (auto const &path : paths) {
        assert(bitsets[bitsetIds[op]].test(path.producerId));
        assert(bitsets[bitsetIds[consumerOp]].test(path.consumerId));
        consumerCoreUsage = accCoreUsage[consumerOp][path.consumerId];
        if (consumerCoreUsage > maxCoreUsage[path.producerId]) {
          maxCoreUsage[path.producerId] = consumerCoreUsage;
        }
      }

      for (size_t i = 0; i < layouts.size(); ++i) {
        // Add max core usage of consumer ops to current op layout.
        // We divide by consumerInChainOperandSize to normalize the core usage
        // based on forking factor(so that cores are not counted more than
        // once).
        //
        // Incorrect results will be produced in case chain consists of joins
        // without previous forks, ie - chain having multiple input ops. In that
        // case total sum of used cores would be a sum of maxCoreUsage generated
        // by all input ops. This is currently not needed for making a
        // decision on layout choice for maximizing core usage.
        //
        accCoreUsage[op][i] += static_cast<float>(maxCoreUsage[i]) /
                               static_cast<float>(consumerInChainOperandSize);
      }
    }
  }

  return accCoreUsage;
}

// Returns ShardSolverSolution.
//
ShardSolverSolution ShardSolver::finish() const {
  assert(selectedOpLayout.size() == shardedOps->size());
  return ShardSolverSolution(selectedOpLayout, memReconfigEdges);
}
} // namespace mlir::tt::ttnn
