// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/ShardSolver.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Analysis/L1ChainConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/MemReconfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
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
#include "llvm/Support/ErrorHandling.h"
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
    std::function<llvm::Expected<TTNNLayoutAttr>(Value, TTNNLayoutAttr,
                                                 Operation *, OpConfig)>
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

  // We need special handling for the first op in the chain.
  //
  if (!preprocessFirstOp()) {
    TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                 "Preprocessing first op failed, aborting.");
    return false;
  }

  for (const auto shardSpec : *shardSpecs) {
    TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                 "Resolving constraints for: {}", shardSpec.op->getName());

    Operation *consumerOp = shardSpec.op;
    Bitset *consumerBitset = getOrInsertBitset(consumerOp, kBitsetAll);
    const std::vector<OpConfig> &consumerConfigs = getLegalConfigs(consumerOp);

    // For now, we don't change op-specific attributes in this analysis so we
    // can check that all consumer configs have the same op-specific attribute.
    for (const OpConfig &config : consumerConfigs) {
      if (config.opSpecificAttr != consumerConfigs.begin()->opSpecificAttr) {
        llvm::report_fatal_error("[TTNN Optimizer] All consumer configs must "
                                 "have the same op-specific attribute");
      }
    }

    OpConfig consumerConfigNoLayout = OpConfig(
        /*outputLayout=*/nullptr, consumerConfigs.begin()->opSpecificAttr);

    auto edges = operandOpEdges.find(consumerOp);
    if (edges == operandOpEdges.end()) {
      continue;
    }

    for (const Edge &edge : edges->second) {
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
      std::uint64_t producerCount =
          std::min(kNumBitsetBits, std::max(1lu, producerConfigs.size()));
      std::uint64_t consumerCount =
          std::min(kNumBitsetBits, std::max(1lu, consumerConfigs.size()));

      // reshardOnEdge can only happen if an override exists for the edge. This
      // is because we have only one resolve step per chain in the current
      // implementation. So if optimizer decided to insert a reshard edge, we
      // won't visit that edge again in the same resolve step.
      if (reshardOnEdge) {
        const MemReconfigEntry &memReconfigEntry = memReconfigMap.at(edge);
        for (const auto &[configBitIndex, configs] :
             memReconfigEntry.reshardOutputConfigMap) {
          // Since we have a reshard entry on the edge, we will save path such
          // that all producer configs are valid with any of chosen consumer
          // config.
          for (std::size_t producerId = 0; producerId < producerCount;
               ++producerId) {
            if (!producerBitset->test(producerId)) {
              continue;
            }
            paths.push_back(Path(producerId, configBitIndex));
            edgeProducerBitset.set(producerId);
          }
          edgeConsumerBitset.set(configBitIndex);
        }
      } else {
        for (std::uint64_t producerId = 0; producerId < producerCount;
             ++producerId) {
          // TODO(rpavlovicTT) After we inserted reshard in
          // preprocessFirstOp we dont need to try every producerId here, right?

          // If the producer cannot accomodate this path, continue.
          // Also if this is not the OpConfig we selected, continue.
          if (!producerBitset->test(producerId)) {
            continue;
          }

          llvm::Expected<TTNNLayoutAttr> shardCompatible =
              checkShardCompatible(producerOp->getResult(0),
                                   producerConfigs[producerId].outputLayout,
                                   consumerOp, consumerConfigNoLayout);

          if (shardCompatible) {
            assert(producerId <=
                   std::numeric_limits<decltype(Path::producerId)>::max());
            TTNNLayoutAttr consumerLayout = shardCompatible.get();

            // Go through all consumer configs and find the one that matches
            // the consumer layout. This search can be optimized in the future.
            for (std::size_t consumerId = 0; consumerId < consumerCount;
                 ++consumerId) {

              if (consumerConfigs[consumerId].outputLayout != consumerLayout) {
                TTMLIR_TRACE(
                    ttmlir::LogComponent::Optimizer,
                    "OpName: {} Generated consumer layout {} does not match "
                    "backend returned layout {}",
                    consumerOp->getName(),
                    consumerConfigs[consumerId].outputLayout, consumerLayout);
                continue;
              }

              assert(consumerId <=
                     std::numeric_limits<decltype(Path::consumerId)>::max());
              TTMLIR_TRACE(
                  ttmlir::LogComponent::Optimizer,
                  "Backend chose valid consumer layout {}, consumerId {}",
                  consumerLayout, consumerId);
              paths.push_back(Path(producerId, consumerId));
              edgeProducerBitset.set(producerId);
              edgeConsumerBitset.set(consumerId);
              break;
            }
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

        // Try resharding, if it fails return false as we cannot resolve. In the
        // long term, we should find a way to fall back to DRAM and not abort
        // sharding totally.
        if (!insertReshard(edge)) {
          return false;
        }

        auto reshardEntry = memReconfigMap.find(edge);
        assert(reshardEntry != memReconfigMap.end());

        // Set all producer configs as valid because we will reshard in between
        // ops. Consumer valid config is read from reshardEntry.
        edgeProducerBitset = *producerBitset;
        for (const auto &[configBitIndex, configs] :
             reshardEntry->second.reshardOutputConfigMap) {
          edgeConsumerBitset.set(configBitIndex);
          for (std::uint64_t producerId = 0; producerId < producerCount;
               ++producerId) {
            if (!edgeProducerBitset.test(producerId)) {
              continue;
            }
            paths.push_back(Path(producerId, configBitIndex));
          }
        }
      }

      if (!isSubset(*producerBitset, edgeProducerBitset)) {
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
    } // end for edges

    opProcessor.process(this);
  } // end for ops

  for (const auto shardSpec : *shardSpecs) {
    Operation *op = shardSpec.op;

    // No need to expand root as we are calling for all ops anyway.
    bool updateSuccess = updateSolver(op, false /* expand_root */);
    assert(updateSuccess && "Failed to update solver");
  }

  TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
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

  llvm::Expected<TTNNLayoutAttr> shardCompatible =
      checkShardCompatible(op->getOperand(0), inputLayout, op, outputConfig);

  if (!shardCompatible) {
    llvm::consumeError(shardCompatible.takeError());
    return false;
  }

  return true;
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

    // TODO(rpavlovicTT) this is bad as we are hardcoding this layout, while it
    // could be overriden.
    // https://github.com/tenstorrent/tt-mlir/issues/3749
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

  auto inputTensor = mlir::cast<RankedTensorType>(
      consumerOp->getOperand(edge.operandIndex).getType());

  Type scalarElementType = mlir::cast<TTNNLayoutAttr>(inputTensor.getEncoding())
                               .getScalarElementType();

  // Get sharded input layouts using the helper function
  std::vector<TTNNLayoutAttr> inputLayouts =
      getShardedLayoutsForTensorTypeAndScalarType(
          *tensorTypePossibleLayouts, inputTensor, scalarElementType);

  std::unordered_map<std::string, std::vector<TTNNLayoutAttr>> errorCount;

  // For all legal outputs, check if there is at least one valid input.
  //
  MemReconfigEntry memReconfigEntry;
  bool validConfigExists = false;

  // Check that all consumer configs have the same op-specific attribute
  for (const OpConfig &config : consumerConfigs) {
    if (config.opSpecificAttr != consumerConfigs.begin()->opSpecificAttr) {
      llvm::report_fatal_error("[TTNN Optimizer] All consumer configs must "
                               "have the same op-specific attribute");
    }
  }
  OpConfig consumerConfigNoLayout = OpConfig(
      /*outputLayout=*/nullptr, consumerConfigs.begin()->opSpecificAttr);

  for (const TTNNLayoutAttr &inputLayout : inputLayouts) {
    llvm::Expected<TTNNLayoutAttr> shardCompatible =
        checkShardCompatible(consumerOp->getOperand(edge.operandIndex),
                             inputLayout, consumerOp, consumerConfigNoLayout);

    if (shardCompatible) {
      TTNNLayoutAttr consumerLayout = shardCompatible.get();

      size_t consumerConfigIdx;
      for (consumerConfigIdx = 0; consumerConfigIdx < consumerConfigs.size();
           ++consumerConfigIdx) {
        if (consumerConfigs[consumerConfigIdx].outputLayout == consumerLayout) {
          break;
        }
      }

      if (consumerConfigIdx == consumerConfigs.size()) {
        // Once we enable row-major, this case should not occur, at that point
        // we can add a fatal error.
        TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                     "Did not find consumer config {} among generated configs",
                     consumerLayout);
        continue;
      }

      consumerBitset->set(consumerConfigIdx);
      validConfigExists = true;

      if (memReconfigEntry.reshardOutputConfigMap.find(consumerConfigIdx) ==
          memReconfigEntry.reshardOutputConfigMap.end()) {
        memReconfigEntry.reshardOutputConfigMap[consumerConfigIdx] =
            llvm::SmallVector<OpConfig>();
      }
      TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                   "Resharding found valid config for edge: {}, producer "
                   "layout: {}, consumer idx: {}",
                   edge, inputLayout, consumerConfigIdx);

      memReconfigEntry.reshardOutputConfigMap[consumerConfigIdx].push_back(
          inputLayout);

      // Breaking as soon as we find one valid config. In future, we might
      // want to keep all valid configs for optimal resharding
      break;
    }

    if (llvm::DebugFlag) {
      std::string errorMsg = ttmlir::utils::firstNLines(
          llvm::toString(shardCompatible.takeError()), 4);
      errorCount[errorMsg].push_back(inputLayout);
    } else {
      llvm::consumeError(shardCompatible.takeError());
    }
  }

  if (!validConfigExists) {
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "Resharding failed for edge: {}", edge);
    for ([[maybe_unused]] auto &layout : inputLayouts) {
      TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer, "\t{}", layout);
    }
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer, "Consumer layouts: {}",
                 consumerConfigs.size());
    for ([[maybe_unused]] auto config : consumerConfigs) {
      TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer, "\t{}",
                   config.outputLayout);
    }

    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer, "Error count: {}",
                 errorCount.size());
    for (const auto &[error, layouts] : errorCount) {
      TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer, "Count: {} Error: {}",
                   layouts.size(), error);
      for ([[maybe_unused]] const auto &layout : layouts) {
        TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer, "\t{}", layout);
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
  reset();

  bool resolved = resolveStep();
  if (earlyExit) {
    assert(!resolved);
    return false;
  }

  assert(resolved);
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

  llvm::llvm_unreachable_internal("Optimizer should not reach here");
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

llvm::Expected<TTNNLayoutAttr> ShardSolver::checkShardCompatible(
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

  OpModel backend = mlir::dyn_cast<OpModel>(consumerOp);
  if (!backend) {
    // This function should not be called for ops without backend constraints.
    llvm::report_fatal_error(
        ("Backend constraints are not implemented for op " +
         consumerOp->getName().getStringRef()));
  }

  // Constraints are implemented for this op.
  //
  auto deviceAttr = mlir::tt::lookupDevice(consumerOp);
  assert(deviceAttr);

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

    auto layout = mlir::cast<TTNNLayoutAttr>(input.getEncoding());

    assert(layout && "Input operand must have a layout");
    inputLayouts.push_back(layout);
  }

  assert(inputUnderCheckFound && "Input under check not found");

  llvm::Expected<op_model::ttnn::OpConstraints> l1UsageExp =
      backend.getOpConstraints(inputLayouts, consumerConfig);

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

  if (consumerConfig.outputLayout &&
      outputLayout != consumerConfig.outputLayout) {
    std::string message = "Output layout mismatch: backend returned layout "
                          "doesn't match requested consumer layout";
    TTMLIR_TRACE(ttmlir::LogComponent::Optimizer, "{}", message);
    return llvm::createStringError("[Optimizer] " + message);
  }

  uint64_t producerL1OutputUsage = producerLayout.getShardSizeInBytes();

  bool l1UsageValid = (producerL1OutputUsage + outputTensorUsage +
                       cBUsagePeak) < tensorL1UsageCap * usableL1CacheSize;

  if (!l1UsageValid) {
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "Not enough L1 memory. OpModel constraints failed: {0}->{1} "
                 "\n producerLayout: {2}, outputLayout: {3}, l1Usage: {4}, "
                 "producerL1OutputUsage: {5}, "
                 "outputTensorUsage: {6}, cBUsagePeak: {7}",
                 producerOperand.getLoc(), consumerOp->getName(),
                 producerLayout, outputLayout,
                 cBUsagePeak + outputTensorUsage + producerL1OutputUsage,
                 producerL1OutputUsage, outputTensorUsage, cBUsagePeak);
    return llvm::createStringError("Not enough L1 memory");
  }

  TTMLIR_DEBUG(
      ttmlir::LogComponent::Optimizer,
      "OpModel constraints valid. Producer: {0} -> Consumer: {1}\n"
      "ProducerLayout: {2}\nOutputLayout: {3}\n"
      "L1 usage: cBUsagePeak: {4}, tensorUsage: {5}, outputTensorUsage: {6}, "
      "producerL1OutputUsage: {7}, totalL1Usage: {8}\n"
      "=== End of debug dump ===",
      producerOperand.getLoc(), consumerOp->getName(), producerLayout,
      outputLayout, cBUsagePeak, tensorUsage, outputTensorUsage,
      producerL1OutputUsage,
      cBUsagePeak + outputTensorUsage + producerL1OutputUsage);

  return outputLayout;
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
