// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/ShardSolver.h"
#include "ttmlir/Dialect/TTNN/Analysis/L1ChainConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include <llvm/Support/Casting.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/Interfaces/DestinationStyleOpInterface.h>
#include <mlir/Support/LLVM.h>
#include <unordered_set>

namespace mlir::tt::ttnn {

ShardSolver::Bitset ShardSolver::kBitsetAll = ~kBitsetNone;
constexpr bool DEBUG = false;

ShardSolver::ShardSolver(
    const llvm::DenseMap<Operation *, std::vector<TTNNLayoutAttr>>
        &legalLayouts,
    const std::vector<OpL1MemSpec> &shardSpecs,
    const llvm::DenseSet<Operation *> &shardedOps,
    const unsigned usableL1CacheSize,
    const std::unordered_set<Edge> &overrideReshardEdges,
    std::function<bool(Value, TTNNLayoutAttr const &, Operation *,
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
  if (!preprocessFirstOp()) {
    return false;
  }

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
      std::unordered_map<std::string, int> errorCount;
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
          if (reshardOnEdge) {
            // TODO(odjuricic): This should read from results of previous
            // resolve instead of accepting all.
            //
            assert(producerId <=
                   std::numeric_limits<decltype(Path::producerId)>::max());
            assert(consumerId <=
                   std::numeric_limits<decltype(Path::consumerId)>::max());
            paths.push_back(Path(producerId, consumerId));
            edgeProducerBitset.set(producerId);
            edgeConsumerBitset.set(consumerId);
            continue;
          }

          llvm::Expected<bool> shardCompatible = checkShardCompatible(
              producerOp->getResult(0), producerLayouts[producerId], consumerOp,
              consumerLayouts[consumerId]);

          if (shardCompatible && shardCompatible.get()) {
            assert(producerId <=
                   std::numeric_limits<decltype(Path::producerId)>::max());
            assert(consumerId <=
                   std::numeric_limits<decltype(Path::consumerId)>::max());
            paths.push_back(Path(producerId, consumerId));
            edgeProducerBitset.set(producerId);
            edgeConsumerBitset.set(consumerId);
          } else {
            std::string error = llvm::toString(shardCompatible.takeError());
            if (DEBUG) {
              llvm::errs() << "Error: " << error << "\n";
              if (!errorCount.count(error)) {
                errorCount.insert({error, 0});
              }
              errorCount[error]++;
            }
          }
        }
      }

      if (paths.empty() || ((*producerBitset & edgeProducerBitset) == 0) ||
          ((*consumerBitset & edgeConsumerBitset) == 0)) {

        if (DEBUG) {
          // Print error counts.
          //
          for (const auto &[error, count] : errorCount) {
            llvm::errs() << "Count: " << count << " Error: " << error << "\n";
          }
        }

        // No valid paths found for this edge, mark it for resharding.
        //
        if (!insertReshard(edge)) {
          return false;
        }
        reshardInserted = true;
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
    if (!updateSolver(op, false /* expand_root */)) {
      return false;
    }
  }

  return true;
}

bool ShardSolver::supportsInterleavedInputShardedOutput(
    Operation *op, TTNNLayoutAttr outputLayout) {
  TTNNLayoutAttr inputLayout = mlir::cast<TTNNLayoutAttr>(
      mlir::cast<RankedTensorType>(op->getOperand(0).getType()).getEncoding());

  inputLayout =
      inputLayout.withBufferType(op->getContext(), BufferType::DRAM)
          .withMemoryLayout(op->getContext(), TensorMemoryLayout::Interleaved);

  llvm::Expected<bool> shardCompatible =
      checkShardCompatible(op->getOperand(0), inputLayout, op, outputLayout);

  if (!shardCompatible) {
    llvm::consumeError(shardCompatible.takeError());
    return false;
  }
  return shardCompatible.get();
}

// We need to check if first op requires sharded inputs and if so, insert
// reshard edge, then invalidate all sharding options which would go above L1
// size limits.
//
bool ShardSolver::preprocessFirstOp() {
  Operation *firstOp = shardSpecs->front().op;

  if (memReconfigEdges.count(
          Edge(firstOp->getOperand(0).getDefiningOp(), firstOp, 0)) > 0) {
    return true;
  }

  Bitset *firstOpBitset = getOrInsertBitset(firstOp, kBitsetAll);
  std::vector<TTNNLayoutAttr> const &firstOpLayouts = getLegalLayouts(firstOp);

  RankedTensorType firstOpInputTensorType =
      mlir::cast<RankedTensorType>(firstOp->getOperand(0).getType());
  TTNNLayoutAttr firstOpInputLayout =
      mlir::cast<TTNNLayoutAttr>(firstOpInputTensorType.getEncoding());
  constexpr float tensorL1UsageCap = 0.8;

  bool hasValidLayout = false;

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
    } else if (not supportsInterleavedInputShardedOutput(firstOp,
                                                         firstOpLayouts[i])) {
      firstOpBitset->reset(i);
    } else {
      hasValidLayout = true;
    }
  }

  if (!hasValidLayout) {
    // Insert reshard edge for the first op to start the chain.
    Edge shardChainInputEdge = Edge(firstOp->getOperand(0).getDefiningOp(),
                                    firstOp, 0 /*operandIndex*/);

    return insertReshard(shardChainInputEdge);
  }

  return true;
}

bool ShardSolver::insertReshard(const Edge &edge) {
  // Same edge should not be resharded twice!
  //
  assert(memReconfigEdges.count(edge) == 0);

  Operation *consumerOp = edge.consumerOp;
  Bitset *consumerBitset = getOrInsertBitset(consumerOp, kBitsetAll);
  *consumerBitset = kBitsetNone;

  std::vector<TTNNLayoutAttr> const &consumerLayouts =
      getLegalLayouts(consumerOp);
  // TODO(odjuricic): This needs to be raplaced with all possible layouts for
  // the input tensor instead of the producer op, as these are not always the
  // same. Related: #2219
  std::vector<TTNNLayoutAttr> const &producerLayouts =
      getLegalLayouts(edge.producerOp);

  // For all legal outputs, check if there is at least one valid input.
  //
  bool validLayoutExists = false;
  for (size_t i = 0; i < consumerLayouts.size(); ++i) {
    TTNNLayoutAttr outputLayout = consumerLayouts[i];

    for (TTNNLayoutAttr producerLayout : producerLayouts) {
      llvm::Expected<bool> shardCompatible =
          checkShardCompatible(consumerOp->getOperand(edge.operandIndex),
                               producerLayout, consumerOp, outputLayout);

      if (shardCompatible && shardCompatible.get()) {
        consumerBitset->set(i);
        validLayoutExists = true;
        break;
      }
      llvm::consumeError(shardCompatible.takeError());
    }
  }

  if (not validLayoutExists) {
    consumerOp->emitWarning()
        << "No valid output layout found for resharded input!";

    if (DEBUG) {
      llvm::errs() << "Op location" << consumerOp->getLoc() << "\n";

      if (edge.producerOp) {
        llvm::errs() << "Producer op" << *edge.producerOp << "\n";
      } else {
        llvm::errs() << "Producer op is null\n";
      }

      llvm::errs() << "Producer layouts: " << producerLayouts.size() << "\n";
      for (auto layout : producerLayouts) {
        llvm::errs() << "\t" << layout << "\n";
      }
      llvm::errs() << "Consumer layouts: " << consumerLayouts.size() << "\n";
      for (auto layout : consumerLayouts) {
        llvm::errs() << "\t" << layout << "\n";
      }
    }

    setEarlyExit(true);
    return false;
  }

  memReconfigEdges.insert(edge);
  return true;
}

bool ShardSolver::resolve() {

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
    if (shouldEarlyExit()) {
      assert(!resolved);
      return false;
    }
    retry_step++;
  } while (!resolved && retry_step <= max_retry_step);

  return resolved;
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

bool ShardSolver::handleNoPathsLeftOnUpdate(bool invokedBySet) {
  // We ended-up in a situation without valid solution due to circular
  // dependency.
  //
  assert(invokedBySet);

  // Invoking resolve again will use resharding to resolve the issue.
  //
  return resolve();
}

bool ShardSolver::updateSolver(Operation *root, bool expand_root,
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

  return true;
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

  bool updateSuccessful =
      updateSolver(op, true /*expand_root*/, true /*invokedBySet*/);
  assert(updateSuccessful && "Failed to update solver after setting layout");
}

llvm::Expected<bool> ShardSolver::checkShardCompatible(
    Value producerOperand, TTNNLayoutAttr const &producerLayout,
    Operation *consumerOp, TTNNLayoutAttr const &consumerLayout) const {

  // Custom(test) hook for shard compatibility check.
  //
  if (customCheckShardCompatible) {
    return customCheckShardCompatible(producerOperand, producerLayout,
                                      consumerOp, consumerLayout);
  }
  // Figure out this const based on exec data, but will be replaced
  // with API.
  //
  constexpr float tensorL1UsageCap = 0.8;

  if (OpModel backend = dyn_cast<OpModel>(consumerOp)) {
    // Constraints are implemented for this op.
    //
    auto deviceAttr = mlir::tt::getCurrentScopeDevice(consumerOp);
    assert(deviceAttr);
    auto workerGrid = deviceAttr.getWorkerGrid();

    // Map consumer operands to DRAM interleave or provided producerLayout
    // only one operand can be mapped to producerLayout, it's picked as first
    // operand matching producerOp output shape.

    uint32_t numOperands = consumerOp->getNumOperands();
    // Discard DPS operand since it's not used in runtime.
    // TODO(odjuricic,#2088): Remove once fix this on MLIR / runtime side.
    if (llvm::isa<DestinationStyleOpInterface>(consumerOp)) {
      numOperands = numOperands - 1;
    }

    std::vector<TTNNLayoutAttr> inputLayouts;

    bool inputUnderCheckFound = false;
    for (uint32_t i = 0; i < numOperands; i++) {
      auto operand = consumerOp->getOperand(i);

      if (operand == producerOperand) {
        // This is the input we are checking compatibility for.

        inputLayouts.push_back(producerLayout);
        inputUnderCheckFound = true;
        continue;
      }

      RankedTensorType input = mlir::cast<RankedTensorType>(operand.getType());

      // TODO(odjuricic): Hardcode other operands to TILE DRAM INTERLEAVED for
      // now. This will change once fork joins are supported.
      Type elementType = input.getElementType();
      if (!llvm::isa<TileType>(elementType)) {
        elementType =
            TileType::get(consumerOp->getContext(), input.getElementType());
      }

      inputLayouts.push_back(TTNNLayoutAttr::get(
          consumerOp->getContext(), input.getShape(), elementType,
          BufferType::DRAM, workerGrid,
          TensorMemoryLayoutAttr::get(consumerOp->getContext(),
                                      TensorMemoryLayout::Interleaved)));
    }

    assert(inputUnderCheckFound && "Input under check not found");

    llvm::Expected<std::tuple<size_t, size_t, size_t>> l1UsageExp =
        backend.getOpConstraints(inputLayouts, consumerLayout);

    if (!l1UsageExp) {
      // early exit
      if (DEBUG) {
        llvm::errs() << "OpModel constraints failed: ";
        llvm::errs() << producerOperand.getLoc() << "->"
                     << consumerOp->getName()
                     << " :: " << llvm::toString(l1UsageExp.takeError())
                     << "\n";
        producerLayout.dump();
        consumerLayout.dump();
      }

      return l1UsageExp.takeError();
    }
    auto [cBUsagePeak, tensorUsage, outputTensorUsage] = l1UsageExp.get();

    if (DEBUG) {
      llvm::errs() << "OpModel constraints valid. ";
      llvm::errs() << producerOperand.getLoc() << "->" << consumerOp->getName()
                   << "\n";
      producerLayout.dump();
      consumerLayout.dump();
      llvm::errs() << "L1 usage: " << cBUsagePeak << ", " << tensorUsage << ", "
                   << outputTensorUsage << "\n";
    }

    RankedTensorType producerTensorType =
        mlir::cast<RankedTensorType>(producerOperand.getType());
    uint64_t producerL1OutputUsage = producerLayout.getTensorSizeInBytes(
        producerTensorType.getShape(), deviceAttr);

    bool l1UsageValid = (producerL1OutputUsage + outputTensorUsage +
                         cBUsagePeak) < tensorL1UsageCap * usableL1CacheSize;

    if (!l1UsageValid) {
      return llvm::createStringError("Not enough L1 memory");
    }

  } else {
    // Constraints are not implemented for this op. Use fallback.
    // Shard compat assumption. Try to keep same shard layout.
    //

    // TODO(odjurcic,#2265) Put this fallback under a flag.

    if (producerLayout.getMemLayout() != consumerLayout.getMemLayout()) {
      return llvm::createStringError("FALLBACK: tensor memory layout mismatch");
    }

    uint64_t producerL1OutputUsage = 0;
    if (producerLayout.hasL1BufferType()) {
      RankedTensorType producerTensorType =
          mlir::cast<RankedTensorType>(producerOperand.getType());
      producerL1OutputUsage = producerLayout.getTensorSizeInBytes(
          producerTensorType.getShape(), deviceAttr);
    }

    uint64_t consumerL1OutputUsage = 0;
    if (consumerLayout.hasL1BufferType()) {
      RankedTensorType consumerTensorType =
          mlir::cast<RankedTensorType>(consumerOp->getResult(0).getType());
      consumerL1OutputUsage = consumerLayout.getTensorSizeInBytes(
          consumerTensorType.getShape(), deviceAttr);
    }

    bool l1UsageValid = (producerL1OutputUsage + consumerL1OutputUsage) <
                        tensorL1UsageCap * usableL1CacheSize;
    if (!l1UsageValid) {
      return llvm::createStringError("FALLBACK: Not enough L1 memory");
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
