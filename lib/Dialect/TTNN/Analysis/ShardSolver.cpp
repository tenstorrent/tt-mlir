// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/ShardSolver.h"
#include "ttmlir/Dialect/TTNN/Analysis/L1ChainConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <utility>
#include <vector>

namespace mlir::tt::ttnn {

ShardSolver::Bitset ShardSolver::kBitsetAll = ~kBitsetNone;

ShardSolver::ShardSolver(
    const TensorTypeLayoutsMap *tensorTypePossibleLayouts,
    const llvm::DenseMap<Operation *, std::vector<OpConfig>> &legalConfigs,
    const std::vector<OpL1MemSpec> &shardSpecs,
    const llvm::DenseSet<Operation *> &shardedOps,
    const unsigned usableL1CacheSize,
    const llvm::DenseSet<Edge> &overrideReshardEdges,
    std::function<bool(Value, TTNNLayoutAttr, Operation *, OpConfig)>
        customCheckShardCompatible)
    : tensorTypePossibleLayouts(tensorTypePossibleLayouts),
      legalConfigs(&legalConfigs), shardSpecs(&shardSpecs),
      shardedOps(&shardedOps), usableL1CacheSize(usableL1CacheSize),
      memReconfigEdges(overrideReshardEdges),
      customCheckShardCompatible(customCheckShardCompatible) {
  pathSets.reserve(shardSpecs.size());
  pathSetIds.reserve(shardSpecs.size());
  bitsets.reserve(shardedOps.size());
  bitsetIds.reserve(shardedOps.size());

  // Cache DeviceAttr.
  //
  deviceAttr = lookupDevice(shardSpecs.front().op);

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
  selectedOpConfig.reserve(shardedOps->size());
  bool reshardInserted = false;

  // We need special handling for the first op in the chain.
  //
  if (!preprocessFirstOp()) {
    TTMLIR_TRACE(ttmlir::LogComponent::Optimizer, "{}",
                 "Preprocessing first op failed, aborting.");
    return false;
  }

  for (const auto shardSpec : *shardSpecs) {
    TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                 "Resolving constraints for: {}", shardSpec.op->getName());

    Operation *consumerOp = shardSpec.op;
    Bitset *consumerBitset = getOrInsertBitset(consumerOp, kBitsetAll);
    const std::vector<OpConfig> &consumerConfigs = getLegalConfigs(consumerOp);

    for (Edge edge : operandOpEdges[consumerOp]) {

      bool reshardOnEdge =
          memReconfigEdges.count(edge) > 0 || memReconfigMap.count(edge) > 0;

      if (reshardOnEdge) {
        TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                     "Found resharding on edge {}", edge);
      }

      Operation *producerOp = edge.producerOp;
      Bitset *producerBitset = getOrInsertBitset(producerOp, kBitsetAll);
      const std::vector<OpConfig> &producerConfigs =
          getLegalConfigs(producerOp);

      assert(not(consumerConfigs.empty() && producerConfigs.empty()));

      PathSet::Paths paths;
      std::unordered_map<std::string, int> errorCount;
      Bitset edgeProducerBitset = kBitsetNone;
      Bitset edgeConsumerBitset = kBitsetNone;
      std::uint64_t producer_count =
          std::min(kNumBitsetBits, std::max(1lu, producerConfigs.size()));
      std::uint64_t consumer_count =
          std::min(kNumBitsetBits, std::max(1lu, consumerConfigs.size()));
      for (std::uint64_t producerId = 0; producerId < producer_count;
           ++producerId) {
        // If the producer cannot accomodate this path, continue.
        // Also if this is not the OpConfig we selected, continue.
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

          llvm::Expected<bool> shardCompatible =
              checkShardCompatible(producerOp->getResult(0),
                                   producerConfigs[producerId].outputLayout,
                                   consumerOp, consumerConfigs[consumerId]);

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
            TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                         "Shard not compabitle, error: {}", error);
            errorCount[error]++;
          }
        }
      }

      if (paths.empty() || ((*producerBitset & edgeProducerBitset) == 0) ||
          ((*consumerBitset & edgeConsumerBitset) == 0)) {

        if (llvm::DebugFlag) {
          std::string errorStr;
          for (const auto &[error, count] : errorCount) {
            errorStr += llvm::formatv("  Count: {} Error: {}", count, error);
          }
          TTMLIR_TRACE(ttmlir::LogComponent::Optimizer, "Error counts: {}",
                       errorStr);
        }

        TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                     "No valid paths found on edge {}", edge);

        // No valid paths found for this edge, mark it for resharding.
        //
        if (!insertReshard(edge)) {
          return false;
        }
        reshardInserted = true;
      }

      if (!isSubset(*producerBitset, edgeProducerBitset) && !reshardInserted) {
        TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                     "Producer bitset is not a subset of edge producer bitset, "
                     "adding op {} to processor",
                     producerOp->getName());
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
      TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                   "Failed to update solver for op {}, returning false",
                   op->getName());
      return false;
    }
  }

  TTMLIR_TRACE(ttmlir::LogComponent::Optimizer, "{}",
               "ShardSolver::resolveStep: returning true");

  return true;
}

bool ShardSolver::supportsInterleavedInputShardedOutput(Operation *op,
                                                        OpConfig outputConfig) {
  TTNNLayoutAttr inputLayout = mlir::cast<TTNNLayoutAttr>(
      mlir::cast<RankedTensorType>(op->getOperand(0).getType()).getEncoding());

  TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
               "Checking if interleaved to sharded is possible for op : {}",
               op->getName());

  inputLayout = inputLayout.withBufferType(BufferType::DRAM)
                    .withMemoryLayout(TensorMemoryLayout::Interleaved);

  llvm::Expected<bool> shardCompatible =
      checkShardCompatible(op->getOperand(0), inputLayout, op, outputConfig);

  if (!shardCompatible) {
    llvm::consumeError(shardCompatible.takeError());
    return false;
  }
  return shardCompatible.get();
}

// We need to check if first op requires sharded inputs and if so, insert
// reshard edge.
//
bool ShardSolver::preprocessFirstOp() {
  Operation *firstOp = shardSpecs->front().op;

  const Edge firstOpEdge =
      Edge(firstOp->getOperand(0).getDefiningOp(), firstOp, /*operandIndex*/ 0);
  const bool reshardOnEdgeExists = memReconfigEdges.count(firstOpEdge) > 0 ||
                                   memReconfigMap.count(firstOpEdge) > 0;
  if (reshardOnEdgeExists) {
    return true;
  }

  Bitset *firstOpBitset = getOrInsertBitset(firstOp, kBitsetAll);
  const std::vector<OpConfig> &firstOpConfigs = getLegalConfigs(firstOp);

  bool hasValidConfig = false;
  for (size_t i = 0; i < firstOpConfigs.size(); ++i) {
    if (!firstOpBitset->test(i)) {
      continue;
    }

    TTNNLayoutAttr firstOpLayout = firstOpConfigs[i].outputLayout;
    assert(firstOpLayout.hasShardedL1TensorMemoryLayout());

    if (!supportsInterleavedInputShardedOutput(firstOp, firstOpConfigs[i])) {
      TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                   "Interleaved to sharded not possible for config idx {}", i);
      // Invalidate this config.
      firstOpBitset->reset(i);
      continue;
    }

    hasValidConfig = true;
  }

  if (hasValidConfig) {
    TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                 "First op {} has valid interleaved to sharded config, no need "
                 "to insert reshard",
                 firstOp->getName());
    return true;
  }

  // None of the configs are valid, so we need to insert a reshard op.
  Edge shardChainInputEdge =
      Edge(firstOp->getOperand(0).getDefiningOp(), firstOp, 0 /*operandIndex*/);

  TTMLIR_DEBUG(
      ttmlir::LogComponent::Optimizer,
      "Interleaved to sharded is not possible, trying reshard for first op {}",
      firstOp->getName());

  return insertReshard(shardChainInputEdge);
}

bool ShardSolver::insertReshard(const Edge &edge) {
  // Same edge should not be resharded twice!
  assert(memReconfigMap.count(edge) == 0);

  Operation *consumerOp = edge.consumerOp;
  Bitset *consumerBitset = getOrInsertBitset(consumerOp, kBitsetAll);
  *consumerBitset = kBitsetNone;

  const std::vector<OpConfig> &consumerConfigs = getLegalConfigs(consumerOp);
  std::vector<OpConfig> producerConfigs;

  auto inputTensor = mlir::cast<RankedTensorType>(
      consumerOp->getOperand(edge.operandIndex).getType());

  Type scalarElementType = mlir::cast<TTNNLayoutAttr>(inputTensor.getEncoding())
                               .getScalarElementType();

  // Get sharded layouts using the helper function
  std::vector<TTNNLayoutAttr> layouts =
      getShardedLayoutsForTensorTypeAndScalarType(
          *tensorTypePossibleLayouts, inputTensor, scalarElementType);

  for (TTNNLayoutAttr layout : layouts) {
    producerConfigs.emplace_back(layout);
  }

  std::unordered_map<std::string,
                     std::vector<std::pair<TTNNLayoutAttr, TTNNLayoutAttr>>>
      errorCount;

  // For all legal outputs, check if there is at least one valid input.
  //
  MemReconfigEntry memReconfigEntry;
  bool validConfigExists = false;
  for (size_t i = 0; i < consumerConfigs.size(); ++i) {
    OpConfig outputConfig = consumerConfigs[i];

    for (OpConfig producerConfig : producerConfigs) {
      llvm::Expected<bool> shardCompatible = checkShardCompatible(
          consumerOp->getOperand(edge.operandIndex),
          producerConfig.outputLayout, consumerOp, outputConfig);

      if (shardCompatible && shardCompatible.get()) {
        consumerBitset->set(i);
        validConfigExists = true;

        if (memReconfigEntry.reshardOutputConfigMap.find(i) ==
            memReconfigEntry.reshardOutputConfigMap.end()) {
          memReconfigEntry.reshardOutputConfigMap[i] =
              llvm::SmallVector<OpConfig>();
        }
        TTMLIR_TRACE(
            ttmlir::LogComponent::Optimizer,
            "Resharding found valid config for edge: {}, producer layout: {}",
            edge, producerConfig.outputLayout);

        memReconfigEntry.reshardOutputConfigMap[i].push_back(
            producerConfig.outputLayout);

        // Breaking as soon as we find one valid config. In future, we might
        // want to keep all valid configs for optimal resharding.
        break;
      }

      if (llvm::DebugFlag) {
        std::string errorMsg = ttmlir::utils::firstNLines(
            llvm::toString(shardCompatible.takeError()), 4);
        errorCount[errorMsg].push_back(
            {producerConfig.outputLayout, outputConfig.outputLayout});
      } else {
        llvm::consumeError(shardCompatible.takeError());
      }
    }
  }

  if (!validConfigExists) {
    consumerOp->emitWarning()
        << "No valid output config found for resharded input!";

    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "Resharding failed for edge: {}", edge);
    for (auto &config : producerConfigs) {
      (void)config;
      TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer, "\t{}",
                   config.outputLayout);
    }
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer, "Consumer layouts: {}",
                 consumerConfigs.size());
    for (auto config : consumerConfigs) {
      (void)config;
      TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer, "\t{}",
                   config.outputLayout);
    }

    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer, "Error count: {}",
                 errorCount.size());
    for (const auto &[error, layouts] : errorCount) {
      TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer, "Count: {} Error: {}",
                   layouts.size(), error);
      for (const auto &[producerLayout, consumerLayout] : layouts) {
        (void)producerLayout;
        (void)consumerLayout;
        TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                     "Producer layout: {}\nConsumer layout: {}", producerLayout,
                     consumerLayout);
      }
    }

    earlyExit = true;
    return false;
  }

  TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
               "Found viable resharding on edge: {}, entry {}", edge,
               memReconfigEntry);

  memReconfigMap[edge] = memReconfigEntry;

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

    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "Resolving shard chain, attempt: {}", retry_step);

    resolved = resolveStep();
    if (earlyExit) {
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

    // When op bitsets are updated(set of valid op configs), we need to update
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

const ShardSolver::Bitset *ShardSolver::getBitset(Operation *op) const {
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

// Returns vector of legal OpConfigs for passed in op.
//
const std::vector<OpConfig> &ShardSolver::getLegalConfigs(Operation *op) const {
  static std::vector<OpConfig> nullConfigs;

  const auto legalIt = legalConfigs->find(op);

  if (legalIt != legalConfigs->end()) {
    return legalIt->second;
  }

  return nullConfigs;
}

ShardSolver::RemainingConfigAttrs ShardSolver::at(Operation *op) const {
  auto configs = RemainingConfigAttrs(getLegalConfigs(op), *getBitset(op));
  assert(configs.begin() != configs.end());
  return configs;
}

void ShardSolver::set(Operation *op, const OpConfig &config) {
  assert(selectedOpConfig.count(op) == 0);

  selectedOpConfig[op] = config;

  const std::vector<OpConfig> &configs = getLegalConfigs(op);
  assert(!configs.empty());
  size_t selection = configs.size();
  for (size_t i = 0; i < configs.size(); ++i) {
    if (configs[i] == config) {
      selection = i;
      break;
    }
  }

  Bitset *op_bitset = getBitset(op);

  assert(selection != configs.size());
  assert((*op_bitset)[selection]);

  op_bitset->reset();
  op_bitset->set(selection);

  // Check if there exists an edge from a producer to an operand of this op.
  // If so, check if this edge is eligible for reshard.
  for (int64_t operandIdx = 0; operandIdx < op->getNumOperands();
       ++operandIdx) {
    Value operand = op->getOperand(operandIdx);
    Operation *producerOp = operand.getDefiningOp();

    auto it = memReconfigMap.find(Edge(producerOp, op, operandIdx));

    if (it != memReconfigMap.end()) {
      // Found edge in reshard map, now let's just save the index
      // corresponding to the op config selection.
      assert(it->second.reshardOutputConfigMap.find(selection) !=
             it->second.reshardOutputConfigMap.end());
      it->second.setSelectedReshardOutputConfigBitIndex(selection);
    }
  }

  bool updateSuccessful =
      updateSolver(op, true /*expand_root*/, true /*invokedBySet*/);
  assert(updateSuccessful && "Failed to update solver after setting config");
}

llvm::Expected<bool> ShardSolver::checkShardCompatible(
    Value producerOperand, const TTNNLayoutAttr &producerLayout,
    Operation *consumerOp, const OpConfig &consumerConfig) const {

  // Custom(test) hook for shard compatibility check.
  //
  if (customCheckShardCompatible) {
    return customCheckShardCompatible(producerOperand, producerLayout,
                                      consumerOp, consumerConfig);
  }
  // Figure out this const based on exec data, but will be replaced
  // with API.
  //
  constexpr float tensorL1UsageCap = 0.8;

  if (OpModel backend = dyn_cast<OpModel>(consumerOp)) {
    // Constraints are implemented for this op.
    //
    auto deviceAttr = mlir::tt::lookupDevice(consumerOp);
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

      if (mlir::isa<TypedValue<mlir::tt::ttnn::DeviceType>>(operand)) {
        // Skip device type operand.
        continue;
      }

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

    llvm::Expected<
        std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
        l1UsageExp = backend.getOpConstraints(inputLayouts, consumerConfig);

    if (!l1UsageExp) {
      llvm::Error error = l1UsageExp.takeError();

      // early exit
      TTMLIR_DEBUG(
          ttmlir::LogComponent::Optimizer,
          "OpModel constraints failed: {0}->{1} :: {2}, \nproducerLayout: {3}, "
          "\nconsumerLayout: {4}",
          producerOperand.getLoc(), consumerOp->getName(),
          ttmlir::utils::firstNLines(llvm::toStringWithoutConsuming(error), 4),
          producerLayout, consumerConfig.outputLayout);

      return error;
    }
    auto [cBUsagePeak, tensorUsage, outputTensorUsage, outputLayout] =
        l1UsageExp.get();

    uint64_t producerL1OutputUsage = producerLayout.getShardSizeInBytes();

    bool l1UsageValid = (producerL1OutputUsage + outputTensorUsage +
                         cBUsagePeak) < tensorL1UsageCap * usableL1CacheSize;

    if (!l1UsageValid) {
      TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                   "OpModel constraints failed: {0}->{1} :: {2}",
                   producerOperand.getLoc(), consumerOp->getName(),
                   "Not enough L1 memory");
      return llvm::createStringError("Not enough L1 memory");
    }

    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "OpModel constraints valid. {0}->{1}\n{2}\n{3}\nL1 usage: "
                 "{4}, {5}, {6}\n=== End of debug dump ===",
                 producerOperand.getLoc(), consumerOp->getName(),
                 producerLayout, consumerConfig.outputLayout, cBUsagePeak,
                 tensorUsage, outputTensorUsage);

  } else {
    // Constraints are not implemented for this op. Use fallback.
    // Shard compat assumption. Try to keep same shard layout.
    //

    // TODO(odjurcic,#2265) Put this fallback under a flag.

    if (producerLayout.getMemLayout() !=
        consumerConfig.outputLayout.getMemLayout()) {
      return llvm::createStringError("FALLBACK: tensor memory layout mismatch");
    }

    uint64_t producerL1OutputUsage = 0;
    if (producerLayout.hasL1BufferType()) {
      producerL1OutputUsage = producerLayout.getShardSizeInBytes();
    }

    uint64_t consumerL1OutputUsage = 0;
    if (consumerConfig.outputLayout.hasL1BufferType()) {
      consumerL1OutputUsage = consumerConfig.outputLayout.getShardSizeInBytes();
    }

    bool l1UsageValid = (producerL1OutputUsage + consumerL1OutputUsage) <
                        tensorL1UsageCap * usableL1CacheSize;
    if (!l1UsageValid) {
      return llvm::createStringError("FALLBACK: Not enough L1 memory");
    }
  }

  return true;
}

// Preprocess ShardSolver search space to make a helper structure which links
// op config choices to global max core usage. Example: Lets assume simple
// case where configs at same index are compatible for input graph provided
// below. Tupples represent grid core usage (Config0GridVolume,
// Config1GridVolume, Config2GridVolume).
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
// Global max of 24 core usage is achieved by selecting config[0] for each Op.
//
// Returns map of op to vector of max core usage for each config.
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
    const std::vector<OpConfig> &configs = getLegalConfigs(op);
    assert(!configs.empty());

    // Find the config that leads to the max core usage.
    // Start with grid volume of current op.
    //
    for (size_t i = 0; i < configs.size(); ++i) {
      const OpConfig &config = configs[i];
      uint64_t coreUsage = config.outputLayout.getGrid().getGridVolume();
      accCoreUsage[op].push_back(coreUsage);
    }

    // Add core usage of current op users via live path connections.
    //
    SmallVector<ShardSolver::PathSet *> userPathSets = getUserPathSetsPts(op);
    for (size_t i = 0; i < userPathSets.size(); ++i) {
      ShardSolver::PathSet *pathSet = userPathSets[i];
      const Paths &paths = pathSet->getPaths();
      SmallVector<uint64_t, 64> maxCoreUsage(configs.size(), 0);
      Operation *consumerOp = pathSet->getConsumerOp();
      size_t consumerInChainOperandSize =
          getOperandPathSetsPts(consumerOp).size();
      uint64_t consumerCoreUsage = 0;
      for (const auto &path : paths) {
        assert(bitsets[bitsetIds[op]].test(path.producerId));
        assert(bitsets[bitsetIds[consumerOp]].test(path.consumerId));
        consumerCoreUsage = accCoreUsage[consumerOp][path.consumerId];
        if (consumerCoreUsage > maxCoreUsage[path.producerId]) {
          maxCoreUsage[path.producerId] = consumerCoreUsage;
        }
      }

      for (size_t i = 0; i < configs.size(); ++i) {
        // Add max core usage of consumer ops to current op config.
        // We divide by consumerInChainOperandSize to normalize the core usage
        // based on forking factor(so that cores are not counted more than
        // once).
        //
        // Incorrect results will be produced in case chain consists of joins
        // without previous forks, ie - chain having multiple input ops. In
        // that case total sum of used cores would be a sum of maxCoreUsage
        // generated by all input ops. This is currently not needed for making
        // a decision on config choice for maximizing core usage.
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
  assert(selectedOpConfig.size() == shardedOps->size());
  return ShardSolverSolution(selectedOpConfig, memReconfigMap);
}

} // namespace mlir::tt::ttnn
