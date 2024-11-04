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
    const llvm::DenseMap<Operation *, std::vector<tt::LayoutAttr>>
        &legalLayouts,
    const std::vector<OpL1MemSpec> &shardSpecs,
    const llvm::DenseSet<Operation *> &shardedOps,
    const unsigned usableL1CacheSize,
    const std::unordered_set<Edge> &overrideReshardEdges)
    : legalLayouts(&legalLayouts), shardSpecs(&shardSpecs),
      shardedOps(&shardedOps), usableL1CacheSize(usableL1CacheSize) {
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

  // Insert override resharding edges
  //
  for (const Edge &edge : overrideReshardEdges) {
    insertReshard(edge);
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
    std::vector<tt::LayoutAttr> const &consumerLayouts =
        getLegalLayouts(consumerOp);

    for (Edge edge : operandOpEdges[consumerOp]) {

      bool reshardOnEdge = memReconfigEdges.count(edge) > 0;

      Operation *producerOp = edge.producerOp;
      Bitset *producerBitset = getOrInsertBitset(producerOp, kBitsetAll);
      std::vector<tt::LayoutAttr> const &producerLayouts =
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
        // Also if this is not the tt::LayoutAttr we selected, continue.
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

// We need to check if first op requires sharded inputs and if so, insert
// reshard edge, then invalidate all sharding options which would go above L1
// size limits.
//
void ShardSolver::preprocessFirstOp() {
  // TODO(nobradovic): Add check whether this op type can have sharded output
  // from interleaved inputs. For now assuming it can not.
  //
  // Insert reshard edge for the first op to start the chain.
  //
  Operation *firstOp = shardSpecs->front().op;
  Edge shardChainInputEdge =
      Edge(firstOp->getOperand(0).getDefiningOp(), firstOp, 0 /*operandIndex*/);

  if (memReconfigEdges.count(shardChainInputEdge) == 0) {
    insertReshard(shardChainInputEdge);
  }

  Bitset *firstOpBitset = getOrInsertBitset(firstOp, kBitsetAll);
  std::vector<tt::LayoutAttr> const &firstOpLayouts = getLegalLayouts(firstOp);
  Operation *operandOp = firstOp->getOperand(0).getDefiningOp();

  RankedTensorType firstOpInputTensorType =
      mlir::cast<RankedTensorType>(operandOp->getResult(0).getType());
  tt::LayoutAttr firstOpInputLayout =
      mlir::cast<tt::LayoutAttr>(firstOpInputTensorType.getEncoding());
  constexpr float tensorL1UsageCap = 0.8;

  for (size_t i = 0; i < firstOpLayouts.size(); ++i) {
    if (!firstOpBitset->test(i)) {
      continue;
    }

    tt::LayoutAttr firstOpLayout = firstOpLayouts[i];
    assert(firstOpLayout.hasShardedL1TensorMemoryLayout());

    tt::LayoutAttr firstOpInputShardedLayout =
        firstOpInputLayout
            .withMemorySpace(firstOp->getContext(),
                             firstOpLayout.getMemorySpace())
            .withMemoryLayout(firstOp->getContext(),
                              firstOpLayout.getMemLayout())
            .withGrid(firstOp->getContext(), firstOpInputTensorType,
                      firstOpLayout.getGrid());

    uint64_t firstInputL1Usage = deviceAttr.getLayoutSizeBytes(
        firstOpInputTensorType.getShape(), firstOpInputShardedLayout,
        firstOpInputShardedLayout.getMemorySpace());
    uint64_t firstOpL1OutputUsage = deviceAttr.getLayoutSizeBytes(
        mlir::cast<RankedTensorType>(firstOp->getResult(0).getType())
            .getShape(),
        firstOpLayout, firstOpLayout.getMemorySpace());

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
const std::vector<tt::LayoutAttr> &
ShardSolver::getLegalLayouts(Operation *op) const {
  static std::vector<tt::LayoutAttr> nullLayouts;

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

void ShardSolver::set(Operation *op, tt::LayoutAttr const &layout) {
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
    Operation *producerOp, tt::LayoutAttr const &producerLayout,
    Operation *consumerOp, tt::LayoutAttr const &consumerLayout) const {

  // TEMP : Dummy mock implementation, will be replaced.
  //

  if (TTNNOpBackend backend = dyn_cast<TTNNOpBackend>(consumerOp)) {
    if (false ==
        backend.isOpLegal(std::vector{producerLayout}, consumerLayout)) {
      return false;
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
  RankedTensorType producerTensorType =
      mlir::cast<RankedTensorType>(producerOp->getResult(0).getType());
  uint64_t producerL1OutputUsage = deviceAttr.getLayoutSizeBytes(
      producerTensorType.getShape(), producerLayout,
      producerLayout.getMemorySpace());

  RankedTensorType consumerTensorType =
      mlir::cast<RankedTensorType>(consumerOp->getResult(0).getType());
  uint64_t consumerL1OutputUsage = deviceAttr.getLayoutSizeBytes(
      consumerTensorType.getShape(), consumerLayout,
      consumerLayout.getMemorySpace());
  // Figure out this const based on exec data, but will be replaced
  // with API.
  //
  constexpr float tensorL1UsageCap = 0.8;
  bool l1UsageValid = (producerL1OutputUsage + consumerL1OutputUsage) <
                      tensorL1UsageCap * usableL1CacheSize;

  if (!l1UsageValid) {
    return false;
  }

  // Shard compat assumption. Try to keep same shard layout.
  //
  if (producerLayout.getMemLayout() != consumerLayout.getMemLayout()) {
    return false;
  }

  return true;
}

// Returns ShardSolverSolution.
//
ShardSolverSolution const ShardSolver::finish() {
  assert(selectedOpLayout.size() == shardedOps->size());
  return ShardSolverSolution(selectedOpLayout, memReconfigEdges);
}
} // namespace mlir::tt::ttnn
