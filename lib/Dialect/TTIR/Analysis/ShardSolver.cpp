// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Analysis/ShardSolver.h"
#include "ttmlir/Dialect/TTIR/Analysis/ShardChainConfig.h"
#include <mlir/Interfaces/DestinationStyleOpInterface.h>

namespace mlir::tt::ttir {

ShardSolver::Bitset ShardSolver::kBitsetAll = ~kBitsetNone;

ShardSolver::ShardSolver(
    const llvm::DenseMap<Operation *, std::vector<LayoutAttr>> &legalGrids,
    const std::vector<ShardSpec> &shardSpecs,
    const llvm::DenseSet<Operation *> &shardedOps,
    const unsigned usableL1CacheSize)
    : legalGrids(&legalGrids), shardSpecs(&shardSpecs), shardedOps(&shardedOps),
      usableL1CacheSize(usableL1CacheSize) {
  pathSets.reserve(shardSpecs.size());
  pathSetIds.reserve(shardSpecs.size());
  bitsets.reserve(shardedOps.size());
  bitsetIds.reserve(shardedOps.size());

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

  for (const auto shardSpec : *shardSpecs) {
    Operation *consumerOp = shardSpec.op;
    Bitset *consumerBitset = getOrInsertBitset(consumerOp, kBitsetAll);
    std::vector<LayoutAttr> const &consumerGrids = getLegalGrids(consumerOp);

    for (Edge edge : operandOpEdges[consumerOp]) {

      bool reshardOnEdge = reshardedEdges.count(edge) > 0;

      Operation *producerOp = edge.producerOp;
      Bitset *producerBitset = getOrInsertBitset(producerOp, kBitsetAll);
      std::vector<LayoutAttr> const &producerGrids = getLegalGrids(producerOp);

      assert(not(consumerGrids.empty() && producerGrids.empty()));

      PathSet::Paths paths;
      Bitset edgeProducerBitset = kBitsetNone;
      Bitset edgeConsumerBitset = kBitsetNone;
      std::uint64_t producer_count =
          std::min(kNumBitsetBits, std::max(1lu, producerGrids.size()));
      std::uint64_t consumer_count =
          std::min(kNumBitsetBits, std::max(1lu, consumerGrids.size()));
      for (std::uint64_t producerId = 0; producerId < producer_count;
           ++producerId) {
        // If the producer cannot accomodate this path, continue.
        // Also if this is not the LayoutAttr we selected, continue.
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
              checkShardCompatible(producerOp, producerGrids[producerId],
                                   consumerOp, consumerGrids[consumerId]);

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

        // No valid paths found for this edge, lets try self-cutting if enabled.
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

void ShardSolver::insertReshard(const Edge &edge) {
  // Same edge should not be resharded twice!
  //
  assert(reshardedEdges.count(edge) == 0);
  reshardedEdges.insert(edge);
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

    // When op bitsets are updated(set of valid op grids), we need to update
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
const std::vector<LayoutAttr> &ShardSolver::getLegalGrids(Operation *op) const {
  static std::vector<LayoutAttr> nullGrids;

  const auto legalIt = legalGrids->find(op);

  if (legalIt != legalGrids->end()) {
    return legalIt->second;
  }

  return nullGrids;
}

ShardSolver::RemainingLayoutAttrs ShardSolver::at(Operation *op) const {
  auto grids = RemainingLayoutAttrs(getLegalGrids(op), *getBitset(op));
  assert(grids.begin() != grids.end());
  return grids;
}

void ShardSolver::set(Operation *op, LayoutAttr const &grid) {
  assert(selectedOpLayout.count(op) == 0);

  selectedOpLayout[op] = grid;

  auto const &grids = getLegalGrids(op);
  assert(!grids.empty());
  size_t selection = grids.size();
  for (size_t i = 0; i < grids.size(); ++i) {
    if (grids[i] == grid) {
      selection = i;
      break;
    }
  }

  Bitset *op_bitset = getBitset(op);

  assert(selection != grids.size());
  assert((*op_bitset)[selection]);

  op_bitset->reset();
  op_bitset->set(selection);

  updateSolver(op, true /*expand_root*/, true /*invokedBySet*/);
}

bool ShardSolver::checkShardCompatible(const Operation *producerOp,
                                       LayoutAttr const &producerLayout,
                                       const Operation *consumerOp,
                                       LayoutAttr const &consumerLayout) const {

  // TEMP : Dummy mock implementation, will be replaced.
  //

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
  llvm::SmallVector<int64_t> producerOpShardShape =
      producerLayout.getShardShape(false /*convertTileToScalar*/);
  uint64_t l1InputUsage = producerLayout.getElementSizeBytes();
  for (int64_t dim : producerOpShardShape) {
    l1InputUsage *= dim;
  }

  llvm::SmallVector<int64_t> consumerOpShardShape =
      consumerLayout.getShardShape(false /*convertTileToScalar*/);
  uint64_t l1OutputUsage = consumerLayout.getElementSizeBytes();
  for (int64_t dim : consumerOpShardShape) {
    l1OutputUsage *= dim;
  }

  // Figure out this const based on exec data, but will be replaced
  // with API.
  //
  constexpr float tensorL1UsageCap = 0.8;
  bool l1UsageValid =
      (l1InputUsage + l1OutputUsage) < tensorL1UsageCap * usableL1CacheSize;

  return l1UsageValid;
}

// Returns ShardSolverSolution.
//
ShardSolverSolution const ShardSolver::finish() {
  return ShardSolverSolution(selectedOpLayout, reshardedEdges);
}
} // namespace mlir::tt::ttir
