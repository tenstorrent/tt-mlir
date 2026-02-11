// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/ShardSolver.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Analysis/L1ChainConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/MemReconfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfigAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/PassOverrides.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <unordered_map>
#include <utility>
#include <vector>

namespace mlir::tt::ttnn {

// Helper to check if matmul/linear op should use IgnorePhysicalLayout.
// When activation is present, we need full layout because the internal unary op
// for activation cannot handle partial memory configs (crashes in
// validate_shard_spec).
// TODO(tt-metal#34500): Remove activation check once tt-metal handles partial
// memory configs in fused activations.
static bool shouldUseIgnorePhysicalLayout(Operation *op) {
  if (auto matmulOp = mlir::dyn_cast<ttnn::MatmulOp>(op)) {
    return !matmulOp.getActivation().has_value();
  }
  if (auto linearOp = mlir::dyn_cast<ttnn::LinearOp>(op)) {
    return !linearOp.getActivation().has_value();
  }
  return false;
}

ShardSolver::Bitset ShardSolver::kBitsetAll = ~kBitsetNone;

ShardSolver::ShardSolver(
    const TensorTypeLayoutsMap *tensorTypePossibleLayouts,
    const llvm::DenseMap<Operation *, std::vector<OpConfig>> &legalConfigs,
    const std::vector<OpL1MemSpec> &shardSpecs,
    const llvm::DenseSet<Operation *> &shardedOps,
    const llvm::DenseSet<Edge> &overrideReshardEdges,
    const llvm::StringMap<OutputLayoutOverrideParams> &overrideOutputLayout,
    std::function<llvm::Expected<TTNNLayoutAttr>(Value, TTNNLayoutAttr,
                                                 Operation *, OpConfig)>
        customCheckShardCompatible)
    : tensorTypePossibleLayouts(tensorTypePossibleLayouts),
      legalConfigs(&legalConfigs), shardSpecs(&shardSpecs),
      shardedOps(&shardedOps), memReconfigEdges(overrideReshardEdges),
      overrideOutputLayout(overrideOutputLayout),
      customCheckShardCompatible(customCheckShardCompatible) {
  pathSets.reserve(shardSpecs.size());
  pathSetIds.reserve(shardSpecs.size());
  bitsets.reserve(shardedOps.size());
  bitsetIds.reserve(shardedOps.size());

  // Cache DeviceAttr.
  //
  deviceAttr = ttcore::lookupDevice(shardSpecs.front().op);

  // Populate operandOpEdges and userOpEdges.
  //
  for (const auto &shardSpec : shardSpecs) {
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

  for (const auto &shardSpec : *shardSpecs) {
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "Resolving constraints for: {}", shardSpec.op->getName());

    Operation *consumerOp = shardSpec.op;
    Bitset *consumerBitset = getOrInsertBitset(consumerOp, kBitsetAll);
    const std::vector<OpConfig> &consumerConfigs = getLegalConfigs(consumerOp);

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
        llvm::SmallVector<OpConfig> testConfigs =
            optimizer_utils::getUniqueTestConfigs(
                consumerConfigs, shouldUseIgnorePhysicalLayout(consumerOp));

        // Extract input layouts template once
        std::vector<TTNNLayoutAttr> inputLayouts =
            utils::extractInputLayouts(consumerOp);

        for (std::uint64_t producerId = 0; producerId < producerCount;
             ++producerId) {
          // TODO(rpavlovicTT) After we inserted reshard in
          // preprocessFirstOp we dont need to try every producerId here, right?

          // If the producer cannot accomodate this path, continue.
          // Also if this is not the OpConfig we selected, continue.
          if (!producerBitset->test(producerId)) {
            continue;
          }

          TTNNLayoutAttr inputLayout = producerConfigs[producerId].outputLayout;

          // Try custom checker first.
          if (tryCustomShardCompatible(
                  edge, consumerOp, inputLayout, consumerConfigs, producerId,
                  edgeProducerBitset, edgeConsumerBitset, paths)) {
            continue;
          }

          inputLayouts[edge.operandIndex] = inputLayout;
          std::vector<op_constraint_validation::ValidationResult> results =
              op_constraint_validation::validateWithMultipleAttributes(
                  consumerOp, inputLayouts, testConfigs,
                  /*referenceConfigs*/ consumerConfigs);

          for (std::size_t i = 0; i < results.size(); ++i) {
            const auto &result = results[i];
            if (result.isSuccess()) {
              // For elementwise binary ops with sharded input, reject configs
              // that shrink the core count. Implicit resharding in these ops
              // causes hangs. See: github.com/tenstorrent/tt-metal/issues/34765
              if (inputLayout.hasShardedL1TensorMemoryLayout() &&
                  result.firstActualOutputLayout
                      .hasShardedL1TensorMemoryLayout() &&
                  llvm::isa<ttnn::AddOp, ttnn::MultiplyOp, ttnn::MinimumOp>(
                      consumerOp)) {
                int64_t inputCores = inputLayout.getGrid().getGridVolume();
                int64_t outputCores =
                    result.firstActualOutputLayout.getGrid().getGridVolume();
                if (outputCores < inputCores) {
                  TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                               "Rejecting {} config: elementwise binary op "
                               "cannot shrink cores from {} to {}",
                               consumerOp->getName(), inputCores, outputCores);
                  continue;
                }
              }

              TTMLIR_TRACE(
                  ttmlir::LogComponent::Optimizer,
                  "Backend chose valid consumer layout {}, consumerId {}",
                  result.firstActualOutputLayout, result.configIndex);
              edgeProducerBitset.set(producerId);
              edgeConsumerBitset.set(result.configIndex);
              paths.push_back(Path(
                  producerId, static_cast<std::uint64_t>(result.configIndex)));
            } else {
              TTMLIR_TRACE(
                  ttmlir::LogComponent::Optimizer,
                  "Producer -> consumer sharding not compatible, error: {}\n\t "
                  "producer layout: {} \n\t consumer layout: {}",
                  result.errorMessage, inputLayout,
                  testConfigs[i].outputLayout);
              errorCount[result.errorMessage]++;
            }
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

  for (const auto &shardSpec : *shardSpecs) {
    Operation *op = shardSpec.op;

    // No need to expand root as we are calling for all ops anyway.
    bool updateSuccess = updateSolver(op, false /* expand_root */);
    assert(updateSuccess && "Failed to update solver");
  }

  TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
               "ShardSolver::resolveStep: returning true");

  return true;
}

bool ShardSolver::tryCustomShardCompatible(
    const Edge &edge, Operation *consumerOp, TTNNLayoutAttr inputLayout,
    const std::vector<OpConfig> &consumerConfigs, std::uint64_t producerId,
    Bitset &edgeProducerBitset, Bitset &edgeConsumerBitset,
    PathSet::Paths &paths) {

  if (!customCheckShardCompatible) {
    return false;
  }

  // Use custom checker - just call it once with empty config
  llvm::Expected<TTNNLayoutAttr> customResult =
      customCheckShardCompatible(consumerOp->getOperand(edge.operandIndex),
                                 inputLayout, consumerOp, OpConfig());

  if (customResult) {
    // Find matching config in consumerConfigs
    for (size_t j = 0; j < consumerConfigs.size(); ++j) {
      if (consumerConfigs[j].outputLayout == customResult.get()) {
        edgeProducerBitset.set(producerId);
        edgeConsumerBitset.set(j);
        paths.push_back(Path(producerId, j));
        break;
      }
    }
  } else {
    TTMLIR_TRACE(ttmlir::LogComponent::Optimizer, "Custom checker failed: {}",
                 llvm::toString(customResult.takeError()));
  }

  return true; // Custom checker was used
}

llvm::Expected<TTNNLayoutAttr>
ShardSolver::supportsInterleavedInputShardedOutput(Operation *op,
                                                   OpConfig outputConfig,
                                                   bool rowMajorInputOverride) {

  RankedTensorType tensorType =
      mlir::cast<RankedTensorType>(op->getOperand(0).getType());
  TTNNLayoutAttr inputLayout =
      mlir::cast<TTNNLayoutAttr>(tensorType.getEncoding());
  llvm::ArrayRef<int64_t> tensorShape = tensorType.getShape();

  inputLayout = inputLayout.withBufferType(BufferType::DRAM)
                    .withMemoryLayout(TensorMemoryLayout::Interleaved);

  if (rowMajorInputOverride) {
    inputLayout = inputLayout.withLayout(Layout::RowMajor, tensorShape);
  }

  if (customCheckShardCompatible) {
    return customCheckShardCompatible(op->getOperand(0), inputLayout, op,
                                      outputConfig);
  }

  std::vector<TTNNLayoutAttr> inputLayouts = utils::extractInputLayouts(op);
  inputLayouts[0] = inputLayout;

  op_constraint_validation::ValidationResult validationResult =
      op_constraint_validation::validateOperation(op, inputLayouts,
                                                  outputConfig);

  if (validationResult.isError()) {
    return llvm::createStringError(validationResult.errorMessage);
  }

  if (!validationResult.firstActualOutputLayout
           .hasShardedL1TensorMemoryLayout()) {
    return llvm::createStringError(
        "Interleaved to sharded not supported - backend did not return sharded "
        "layout");
  }

  return validationResult.firstActualOutputLayout;
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

  Operation *preFirstOp = firstOp->getOperand(0).getDefiningOp();
  bool rowMajorInputOverride = false;
  if (preFirstOp && isa<NameLoc>(preFirstOp->getLoc())) {
    StringRef opLocName = mlir::cast<NameLoc>(preFirstOp->getLoc()).getName();
    auto opOutputOverride = overrideOutputLayout.find(opLocName);
    if (opOutputOverride != overrideOutputLayout.end() &&
        opOutputOverride->getValue().memoryLayout.has_value() &&
        opOutputOverride->getValue().memoryLayout.value() == Layout::RowMajor) {
      rowMajorInputOverride = true;
      TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                   "[preprocessing first op {}] Row-major input override found "
                   "for first op in chain",
                   firstOp->getName());
    }
  }

  Bitset *firstOpBitset = getOrInsertBitset(firstOp, kBitsetAll);
  const std::vector<OpConfig> &firstOpConfigs = getLegalConfigs(firstOp);

  firstOpBitset->reset();

  // None of the configs are valid, so we need to insert a reshard op.
  Edge shardChainInputEdge =
      Edge(firstOp->getOperand(0).getDefiningOp(), firstOp, 0 /*operandIndex*/);

  if (mlir::isa<ttnn::Conv2dOp, ttnn::ConvTranspose2dOp>(firstOp)) {
    TTMLIR_TRACE(
        ttmlir::LogComponent::Optimizer,
        "[preprocessing first op {}] First op is Conv2d/ConvTranspose2d, "
        "inserting reshard unconditionally",
        firstOp->getName());
    return insertReshard(shardChainInputEdge);
  }

  for (size_t i = 0; i < firstOpConfigs.size(); ++i) {
    TTNNLayoutAttr firstOpLayout = firstOpConfigs[i].outputLayout;
    assert(firstOpLayout.hasShardedL1TensorMemoryLayout());

    TTNNLayoutAttr layoutForComparison = firstOpLayout;

    if (shouldUseIgnorePhysicalLayout(firstOp)) {
      firstOpLayout = firstOpLayout.withIgnorePhysicalLayout(true);
    }

    TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                 "[preprocessing first op {}] Checking interleaved to sharded "
                 "for config idx {} "
                 "\n\tlayout: {} ",
                 firstOp->getName(), i, firstOpLayout);

    llvm::Expected<TTNNLayoutAttr> result =
        supportsInterleavedInputShardedOutput(
            firstOp, OpConfig(firstOpLayout, firstOpConfigs[i].opSpecificAttrs),
            rowMajorInputOverride);

    if (!result) {
      std::string errorStr = llvm::toString(result.takeError());
      TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                   "[preprocessing first op {}] Interleaved to sharded not "
                   "possible for config idx {} "
                   "\n\tlayout: {} \n\terror: {}",
                   firstOp->getName(), i, firstOpLayout, errorStr);
      continue;
    }

    TTNNLayoutAttr actualLayout = result.get();
    if (actualLayout == layoutForComparison) {
      TTMLIR_TRACE(
          ttmlir::LogComponent::Optimizer,
          "[preprocessing first op {}] Backend actual layout matches "
          "config layout, marking config idx {} as valid \n\t layout {}",
          firstOp->getName(), i, firstOpLayout);
      firstOpBitset->set(i);
    } else {
      TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                   "[preprocessing first op {}] Backend actual layout does not "
                   "match given output config layout\n\t actual layout: {}\n\t "
                   "expected layout: {}",
                   firstOp->getName(), actualLayout, layoutForComparison);
    }
  }

  if (firstOpBitset->any()) {
    TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                 "[preprocessing first op {}] Has valid interleaved to sharded "
                 "config, no need "
                 "to insert reshard, bitset: {}",
                 firstOp->getName(), firstOpBitset->to_string());
    return true;
  }

  TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
               "[preprocessing first op {}] Interleaved to sharded is not "
               "possible, trying reshard",
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

  TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
               "insertReshard: consumerConfigs for {} (count: {})",
               consumerOp->getName(), consumerConfigs.size());
  for (size_t i = 0; i < consumerConfigs.size(); ++i) {
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer, "  consumerConfig[{}]: {}", i,
                 consumerConfigs[i].outputLayout);
  }

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

  llvm::SmallVector<OpConfig> testConfigs =
      optimizer_utils::getUniqueTestConfigs(
          consumerConfigs, shouldUseIgnorePhysicalLayout(consumerOp));

  // Extract and set input layouts for validation
  std::vector<TTNNLayoutAttr> consumerInputOperandLayouts =
      utils::extractInputLayouts(consumerOp);

  // Try each input layout to find compatible consumer configs
  for (const TTNNLayoutAttr &inputLayout : inputLayouts) {
    consumerInputOperandLayouts[edge.operandIndex] = inputLayout;
    std::vector<op_constraint_validation::ValidationResult> results =
        op_constraint_validation::validateWithMultipleAttributes(
            consumerOp, consumerInputOperandLayouts, testConfigs,
            /*referenceConfigs*/ consumerConfigs);

    for (const auto &result : results) {
      if (result.isSuccess()) {
        std::size_t consumerConfigIdx = result.configIndex;
        consumerBitset->set(consumerConfigIdx);

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
      } else if (llvm::DebugFlag) {
        std::string errorMsg =
            ttmlir::utils::firstNLines(result.errorMessage, 4);
        errorCount[errorMsg].push_back(inputLayout);
      }
    }
  }

  if (memReconfigEntry.reshardOutputConfigMap.empty()) {
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "Resharding failed for edge: {}", edge);
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer, "=== Debug start ===");
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer, "Input layouts: {}",
                 inputLayouts.size());
    for ([[maybe_unused]] auto &layout : inputLayouts) {
      TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer, "\t{}", layout);
    }
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer, "Consumer layouts: {}",
                 consumerConfigs.size());
    for ([[maybe_unused]] auto config : consumerConfigs) {
      TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer, "\t{}",
                   config.outputLayout);
    }

    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "Error count: {} types of errors", errorCount.size());
    for (const auto &[error, layouts] : errorCount) {
      TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer, "Count: {} Error: {}",
                   layouts.size(), error);
      for ([[maybe_unused]] const auto &layout : layouts) {
        TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer, "\t{}", layout);
      }
    }
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer, "=== End of debug dump ===");

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
