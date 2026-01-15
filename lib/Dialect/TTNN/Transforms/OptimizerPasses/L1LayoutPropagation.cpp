// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Analysis/LegalOpConfigAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/LegalOpLayoutAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/LegalTensorLayoutAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfigAttrs.h"
#include "ttmlir/Dialect/TTNN/Analysis/ScalarDataTypeAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/TensorLayouts.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/PassOverrides.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

// TODO: Implement look-ahead optimization for input resharding.
//
// Current limitation: The greedy approach optimizes each operation locally
// without considering how the input layout choice affects downstream ops.
//
// Example problem with conv2d:
// - When resharding inputs, we pick the layout with highest core usage (e.g.,
//   64 cores with block_sharded)
// - But the downstream conv2d with that input might only achieve 32 cores BS
// - Whereas picking a different input (e.g., 49 cores height_sharded) would
//   allow conv2d to use 49 cores HS, which is faster overall
//
// The old optimizer (ShardSolver) handles this by considering the entire graph.
//
// Potential solutions:
// 1. Look-ahead: When selecting input reshard layouts, also validate what
//    output layouts the consuming op can achieve with each input option, and
//    optimize for the op's output rather than the reshard's core count.
// 2. Op-specific heuristics: For conv2d inputs, prefer HS layouts even if
//    they use fewer cores for the reshard itself.
// 3. Two-pass approach: First pass determines compatible layout pairs for
//    producer-consumer edges, second pass picks the globally optimal config.

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_TTNNL1LAYOUTPROPAGATION
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

class TTNNL1LayoutPropagation
    : public impl::TTNNL1LayoutPropagationBase<TTNNL1LayoutPropagation> {
public:
  using impl::TTNNL1LayoutPropagationBase<
      TTNNL1LayoutPropagation>::TTNNL1LayoutPropagationBase;

  void runOnOperation() final {
#ifndef TTMLIR_ENABLE_OPMODEL
    llvm::llvm_unreachable_internal(
        "TTNNL1LayoutPropagation pass requires OpModel support to be enabled.");
#else
    op_model::ScopedSingletonDeviceGuard deviceGuard;

    ModuleOp moduleOp = getOperation();

    // Get the max grid size from the system description.
    deviceGrid = ttcore::lookupDevice(moduleOp).getWorkerGrid();

    // Step 1: Run ScalarDataTypeAnalysis to collect all scalar types.
    llvm::StringMap<OutputLayoutOverrideParams> emptyOverrides;
    ScalarDataTypeAnalysis scalarDataTypeAnalysis =
        getAnalysis<ScalarDataTypeAnalysis>();
    scalarDataTypeAnalysis.init(ScalarDataTypeAnalysisInput(&emptyOverrides));
    auto scalarTypes = scalarDataTypeAnalysis.getResult();

    // Step 2: Run LegalTensorLayoutAnalysis to generate layouts for all tensor
    // types.
    LegalTensorLayoutAnalysis legalTensorLayoutAnalysis =
        getAnalysis<LegalTensorLayoutAnalysis>();
    legalTensorLayoutAnalysis.init(LegalTensorLayoutAnalysisInput(
        deviceGrid, &scalarTypes, /*rowMajorAllowed=*/false));
    TensorTypeLayoutsMap tensorTypePossibleLayouts =
        legalTensorLayoutAnalysis.getResult();

    // Step 3: Process each function.
    moduleOp->walk([&](func::FuncOp funcOp) {
      // Skip const-eval functions.
      if (ttmlir::utils::isConstEvalFunc(funcOp)) {
        return;
      }

      processFunction(funcOp, tensorTypePossibleLayouts);
    });
#endif
  }

private:
  ttcore::GridAttr deviceGrid;

  // Create a DRAM interleaved layout from an existing layout.
  TTNNLayoutAttr createDRAMInterleavedLayout(TTNNLayoutAttr baseLayout) {
    return baseLayout.withBufferType(BufferType::DRAM)
        .withMemoryLayout(TensorMemoryLayout::Interleaved);
  }

  // Create a MemoryConfigAttr from a TTNNLayoutAttr.
  MemoryConfigAttr createMemoryConfigFromLayout(TTNNLayoutAttr layout) {
    MLIRContext *ctx = layout.getContext();

    TensorMemoryLayoutAttr tensorMemoryLayout =
        TensorMemoryLayoutAttr::get(ctx, layout.getMemLayout().getValue());
    BufferTypeAttr bufferType = BufferTypeAttr::get(ctx, layout.getBufferType());

    // Create ShardSpec if the layout is sharded.
    std::optional<ShardSpecAttr> shardSpec =
        utils::createShardSpecIfNeeded(layout, deviceGrid);

    return MemoryConfigAttr::get(ctx, tensorMemoryLayout, bufferType, shardSpec);
  }

  // Create an L1 sharded layout for an input tensor based on the target output
  // layout's grid. This is used when we need to reshard inputs before an op.
  TTNNLayoutAttr createL1ShardedInputLayout(RankedTensorType inputTensorType,
                                            TTNNLayoutAttr inputLayout,
                                            TTNNLayoutAttr targetOutputLayout) {
    // Use the same grid and memory layout as the target output.
    return inputLayout.withBufferType(BufferType::L1)
        .withMemoryLayout(targetOutputLayout.getMemLayout().getValue())
        .withGrid(inputTensorType.getShape(), targetOutputLayout.getGrid());
  }

  // Insert a to_memory_config op to reshard an input tensor.
  // Returns the new value to use as the op's input.
  Value insertInputReshard(OpBuilder &builder, Operation *op, Value input,
                           TTNNLayoutAttr newLayout) {
    auto tensorType = cast<RankedTensorType>(input.getType());
    auto newType = RankedTensorType::get(tensorType.getShape(),
                                         tensorType.getElementType(), newLayout);

    MemoryConfigAttr memConfig = createMemoryConfigFromLayout(newLayout);

    builder.setInsertionPoint(op);
    auto reshardOp = builder.create<ToMemoryConfigOp>(op->getLoc(), newType,
                                                      input, memConfig);

    TTMLIR_TRACE(ttmlir::LogComponent::L1Optimizer,
                 "[L1LayoutPropagation] Inserted input reshard before {}: {} -> "
                 "{}",
                 op->getName(), input.getType(), newType);

    return reshardOp.getResult();
  }

  // Check if Matmul/Linear should use ignorePhysicalLayout.
  // Returns false if activation is present (workaround for tt-metal#34500).
  bool shouldUseIgnorePhysicalLayout(Operation *op) {
    if (auto matmulOp = dyn_cast<MatmulOp>(op)) {
      return !matmulOp.getActivation().has_value();
    }
    if (auto linearOp = dyn_cast<LinearOp>(op)) {
      return !linearOp.getActivation().has_value();
    }
    return false;
  }

  // Check if ALL tensor inputs are DRAM interleaved (not L1 sharded).
  // This indicates the op is "first in L1 chain" and needs explicit output
  // layout to transition from interleaved to sharded.
  bool allInputsAreInterleaved(Operation *op) {
    bool hasTensorInput = false;

    for (Value operand : op->getOperands()) {
      auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
      if (!tensorType) {
        continue;
      }

      hasTensorInput = true;

      // Block arguments (function parameters) are always interleaved - OK.
      if (isa<BlockArgument>(operand)) {
        continue;
      }

      auto layout = dyn_cast_or_null<TTNNLayoutAttr>(tensorType.getEncoding());
      if (layout && layout.hasShardedL1TensorMemoryLayout()) {
        // Found an L1 sharded input - not all interleaved.
        return false;
      }
    }

    // Return true only if we have tensor inputs and all are interleaved.
    return hasTensorInput;
  }

  // Get input layouts for validation from the current IR state.
  // This respects layouts applied by earlier ops in the pass and preserves
  // system_memory layouts (e.g., Conv2d weights that must stay on host).
  std::vector<TTNNLayoutAttr> getInputLayoutsForValidation(Operation *op) {
    return utils::extractInputLayouts(op);
  }

  // Get all legal configs for an operation, split into L1 sharded and DRAM.
  struct LegalConfigs {
    std::vector<OpConfig> l1Sharded;
    std::vector<OpConfig> dramInterleaved;
  };

  LegalConfigs getLegalConfigs(Operation *op,
                               const TensorTypeLayoutsMap &tensorTypePossibleLayouts) {
    LegalConfigs result;

    auto tensorType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!tensorType) {
      return result;
    }

    auto tensorLayouts = tensorTypePossibleLayouts.find(tensorType);
    if (tensorLayouts == tensorTypePossibleLayouts.end()) {
      return result;
    }

    // Run legal layout analysis.
    llvm::StringMap<OutputLayoutOverrideParams> emptyOverrides;
    LegalOpLayoutAnalysis legalOpLayoutAnalysis =
        getChildAnalysis<LegalOpLayoutAnalysis>(op);
    legalOpLayoutAnalysis.init(LegalOpLayoutAnalysisInput(
        const_cast<TensorTypeLayoutsForScalarType *>(
            &tensorLayouts->getSecond()),
        /*maxShardedConfigs=*/64, &emptyOverrides,
        /*rowMajorEnabled=*/false));

    // Run legal config analysis to add op-specific configs.
    llvm::StringMap<Conv2dConfigOverrideParams> emptyConv2dOverrides;
    LegalOpConfigAnalysis legalOpConfigAnalysis =
        getChildAnalysis<LegalOpConfigAnalysis>(op);
    legalOpConfigAnalysis.init(LegalOpConfigAnalysisInput(
        legalOpLayoutAnalysis.getResult(), &emptyConv2dOverrides));

    // Split configs into L1 sharded and DRAM interleaved.
    for (const auto &config : legalOpConfigAnalysis.getResult()) {
      if (config.outputLayout &&
          config.outputLayout.hasShardedL1TensorMemoryLayout()) {
        result.l1Sharded.push_back(config);
      } else if (config.outputLayout &&
                 config.outputLayout.getBufferType() == BufferType::DRAM) {
        result.dramInterleaved.push_back(config);
      }
    }

    return result;
  }

  // Update an operation's result type with a new layout.
  void updateOpResultLayout(Operation *op, TTNNLayoutAttr newLayout) {
    if (op->getNumResults() == 0) {
      return;
    }

    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultType) {
      return;
    }

    auto newType = RankedTensorType::get(resultType.getShape(),
                                         resultType.getElementType(), newLayout);
    op->getResult(0).setType(newType);
  }

  // Update an operation's attributes based on the validated config and output
  // layout. Sets conv2d config but does NOT set shard_layout - rely on output
  // layout only for sharding decisions.
  void updateOpSpecificAttrs(Operation *op, const OpConfig &config,
                             TTNNLayoutAttr outputLayout) {
    if (auto *conv2dAttrs =
            std::get_if<Conv2dAttrs>(&config.opSpecificAttrs)) {
      // Update Conv2d/ConvTranspose2d config attribute.
      // Do NOT set shard_layout - rely on output layout only.
      Conv2dConfigAttr conv2dConfig = conv2dAttrs->conv2dConfig.has_value()
                                          ? conv2dAttrs->conv2dConfig.value()
                                          : Conv2dConfigAttr::get(op->getContext());

      if (auto conv2dOp = dyn_cast<Conv2dOp>(op)) {
        conv2dOp.setConv2dConfigAttr(conv2dConfig);
        if (conv2dAttrs->deviceComputeKernelConfig.has_value()) {
          conv2dOp.setComputeConfigAttr(
              conv2dAttrs->deviceComputeKernelConfig.value());
        }
      } else if (auto convTranspose2dOp = dyn_cast<ConvTranspose2dOp>(op)) {
        convTranspose2dOp.setConv2dConfigAttr(conv2dConfig);
        if (conv2dAttrs->deviceComputeKernelConfig.has_value()) {
          convTranspose2dOp.setComputeConfigAttr(
              conv2dAttrs->deviceComputeKernelConfig.value());
        }
      }
    } else if (auto *matmulAttrs =
                   std::get_if<MatmulAttrs>(&config.opSpecificAttrs)) {
      // Update Matmul/Linear program config.
      if (auto matmulOp = dyn_cast<MatmulOp>(op)) {
        if (matmulAttrs->matmulProgramConfig.has_value()) {
          matmulOp.setMatmulProgramConfigAttr(
              matmulAttrs->matmulProgramConfig.value());
        }
      } else if (auto linearOp = dyn_cast<LinearOp>(op)) {
        if (matmulAttrs->matmulProgramConfig.has_value()) {
          linearOp.setMatmulProgramConfigAttr(
              matmulAttrs->matmulProgramConfig.value());
        }
      }
    }
  }

  // Helper to get sharding priority: HS > WS > BS (lower is better).
  static int getMemLayoutPriority(TensorMemoryLayout layout) {
    switch (layout) {
    case TensorMemoryLayout::HeightSharded:
      return 0;
    case TensorMemoryLayout::WidthSharded:
      return 1;
    case TensorMemoryLayout::BlockSharded:
      return 2;
    default:
      return 3;
    }
  }

  // Helper struct for comparing candidates by core usage and memory layout.
  // Criteria: 1) Higher core usage wins, 2) On tie, prefer HS > WS > BS.
  struct CandidateScore {
    uint64_t coreUsage = 0;
    int memLayoutPriority = 3;

    static CandidateScore fromLayout(TTNNLayoutAttr layout) {
      return {layout.getGrid().getGridVolume(),
              getMemLayoutPriority(layout.getMemLayout().getValue())};
    }

    bool isBetterThan(const CandidateScore &other) const {
      if (coreUsage > other.coreUsage) {
        return true;
      }
      return coreUsage == other.coreUsage &&
             memLayoutPriority < other.memLayoutPriority;
    }
  };

  // Try to get L1 sharded output for an op with interleaved inputs.
  // This is for "first in chain" ops that need to transition from interleaved
  // to sharded. If direct DRAM->L1 doesn't work, try resharding inputs first.
  bool tryInterleavedToSharded(
      Operation *op, const std::vector<TTNNLayoutAttr> &inputLayouts,
      const std::vector<OpConfig> &l1Configs,
      const TensorTypeLayoutsMap &tensorTypePossibleLayouts,
      const llvm::DenseSet<Value> &activationValues) {
    std::optional<std::pair<CandidateScore, std::pair<TTNNLayoutAttr, OpConfig>>>
        best;

    // For first-in-chain ops, we need to specify the output layout explicitly
    // so the backend knows to produce L1 sharded output.
    for (const OpConfig &config : l1Configs) {

      OpConfig testConfig = config;
      if (shouldUseIgnorePhysicalLayout(op)) {
        testConfig.outputLayout =
            testConfig.outputLayout.withIgnorePhysicalLayout(true);
      }

      auto result = op_constraint_validation::validateOperation(
          op, inputLayouts, testConfig);

      if (result.isSuccess() &&
          result.actualOutputLayout.hasShardedL1TensorMemoryLayout()) {
        CandidateScore score = CandidateScore::fromLayout(result.actualOutputLayout);
        if (!best || score.isBetterThan(best->first)) {
          best = {score, {result.actualOutputLayout, testConfig}};
        }
      }
    }

    if (best) {
      updateOpResultLayout(op, best->second.first);
      updateOpSpecificAttrs(op, best->second.second, best->second.first);

      TTMLIR_TRACE(ttmlir::LogComponent::L1Optimizer,
                   "[L1LayoutPropagation] Op {} (first in chain): "
                   "interleaved -> L1 sharded (direct, cores={}, memLayout={})",
                   op->getName(), best->first.coreUsage,
                   best->first.memLayoutPriority);
      return true;
    }

    // If direct conversion failed for all configs, try with L1 sharded inputs.
    // This is needed for ops like conv2d that require L1 input for L1 output.
    // Use getUniqueTestConfigs to let the backend pick the output layout.
    return tryWithReshardedInputs(op, inputLayouts, l1Configs,
                                  tensorTypePossibleLayouts, activationValues);
  }

  // Precompute which values are on the activation path.
  // Rule: If ANY tensor input to an op is an activation, the output is an
  // activation. Function args with argument_type=Input are activations.
  llvm::DenseSet<Value> computeActivationValues(func::FuncOp funcOp) {
    llvm::DenseSet<Value> activationValues;

    // Step 1: Mark function arguments based on argument_type attribute.
    for (BlockArgument arg : funcOp.getArguments()) {
      if (auto typeAttr = funcOp.getArgAttrOfType<ttcore::ArgumentTypeAttr>(
              arg.getArgNumber(), ttcore::ArgumentTypeAttr::name)) {
        if (typeAttr.getValue() == ttcore::ArgumentType::Input) {
          activationValues.insert(arg);
        }
      }
      // If no attribute, assume it's a parameter (not activation).
    }

    // Step 2: Propagate through ops in topological order.
    // If ANY tensor input is an activation, the output is an activation.
    funcOp.walk([&](Operation *op) {
      // Skip non-TTNN ops.
      if (!isa<TTNNDialect>(op->getDialect())) {
        return;
      }

      // Skip ops without results.
      if (op->getNumResults() == 0) {
        return;
      }

      // Check if any tensor operand is an activation.
      bool hasActivationInput = false;
      for (Value operand : op->getOperands()) {
        if (!isa<RankedTensorType>(operand.getType())) {
          continue;
        }
        if (activationValues.contains(operand)) {
          hasActivationInput = true;
          break;
        }
      }

      // If any input is activation, output is activation.
      if (hasActivationInput) {
        for (Value result : op->getResults()) {
          if (isa<RankedTensorType>(result.getType())) {
            activationValues.insert(result);
          }
        }
      }
    });

    return activationValues;
  }

  // Check if a value is on the activation path using precomputed set.
  bool isActivationValue(Value value,
                         const llvm::DenseSet<Value> &activationValues) {
    return activationValues.contains(value);
  }

  // Try to shard an op by first resharding its activation inputs to L1.
  // Gets all legal sharded layouts for each activation operand and tries
  // all combinations. Uses getUniqueTestConfigs to let the backend pick the
  // output layout. Returns true if successful and inserts the necessary
  // to_memory_config ops.
  bool tryWithReshardedInputs(
      Operation *op, const std::vector<TTNNLayoutAttr> &originalInputLayouts,
      const std::vector<OpConfig> &l1Configs,
      const TensorTypeLayoutsMap &tensorTypePossibleLayouts,
      const llvm::DenseSet<Value> &activationValues) {
    // TODO(tt-metal#35145): Reduction ops internally call reshape which crashes
    // when given certain sharded input layouts. Skip input resharding for these
    // ops until the tt-metal bug is fixed.
    if (isa<SumOp, MeanOp, MaxOp, MinOp>(op)) {
      return false;
    }

    // Collect activation operand info: index, tensor type, and possible layouts.
    struct ActivationOperandInfo {
      unsigned inputIdx;
      Value operand;
      RankedTensorType tensorType;
      std::vector<TTNNLayoutAttr> possibleLayouts;
    };
    llvm::SmallVector<ActivationOperandInfo> activationOperands;

    unsigned inputIdx = 0;
    for (Value operand : op->getOperands()) {
      auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
      if (!tensorType) {
        continue;
      }
      if (inputIdx >= originalInputLayouts.size()) {
        break;
      }

      // Check if this operand is an activation tensor (not weights/bias).
      // Use precomputed activation values from the dataflow analysis.
      if (isActivationValue(operand, activationValues)) {
        // Get scalar element type from current layout.
        Type scalarElementType =
            originalInputLayouts[inputIdx].getScalarElementType();

        // Get all legal sharded layouts for this input tensor.
        std::vector<TTNNLayoutAttr> layouts =
            getShardedLayoutsForTensorTypeAndScalarType(
                tensorTypePossibleLayouts, tensorType, scalarElementType);

        if (!layouts.empty()) {
          activationOperands.push_back(
              {inputIdx, operand, tensorType, std::move(layouts)});
        }
      }
      inputIdx++;
    }

    if (activationOperands.empty()) {
      return false;
    }

    TTMLIR_TRACE(ttmlir::LogComponent::L1Optimizer,
                 "[L1LayoutPropagation] Op {}: trying input reshard with {} "
                 "activation operands",
                 op->getName(), activationOperands.size());

    // Use getUniqueTestConfigs to let the backend pick the output layout.
    llvm::SmallVector<OpConfig> testConfigs =
        optimizer_utils::getUniqueTestConfigs(l1Configs,
                                              shouldUseIgnorePhysicalLayout(op));

    // Candidate data: inputLayouts, outputLayout, configIndex.
    struct CandidateData {
      std::vector<TTNNLayoutAttr> inputLayouts;
      TTNNLayoutAttr outputLayout;
      size_t configIndex;
    };
    std::optional<std::pair<CandidateScore, CandidateData>> best;

    // Try all combinations of input layouts using iterative Cartesian product.
    std::vector<size_t> indices(activationOperands.size(), 0);

    while (true) {
      // Build the input layouts for this combination.
      std::vector<TTNNLayoutAttr> testInputLayouts = originalInputLayouts;
      for (size_t i = 0; i < activationOperands.size(); ++i) {
        const auto &info = activationOperands[i];
        testInputLayouts[info.inputIdx] = info.possibleLayouts[indices[i]];
      }

      // Validate with all test configs for this input layout combination.
      std::vector<op_constraint_validation::ValidationResult> results =
          op_constraint_validation::validateWithMultipleAttributes(
              op, testInputLayouts, testConfigs, /*referenceConfigs=*/{});

      for (const auto &result : results) {
        if (result.isSuccess() &&
            result.actualOutputLayout.hasShardedL1TensorMemoryLayout()) {
          CandidateScore score =
              CandidateScore::fromLayout(result.actualOutputLayout);
          if (!best || score.isBetterThan(best->first)) {
            best = {score, {testInputLayouts, result.actualOutputLayout,
                            result.configIndex}};
          }
        }
      }

      // Move to next combination (increment indices like a multi-digit counter).
      size_t pos = 0;
      while (pos < activationOperands.size()) {
        indices[pos]++;
        if (indices[pos] < activationOperands[pos].possibleLayouts.size()) {
          break;
        }
        indices[pos] = 0;
        pos++;
      }
      if (pos >= activationOperands.size()) {
        break;
      }
    }

    if (!best) {
      return false;
    }

    // Apply the best candidate.
    OpBuilder builder(op);
    for (const auto &info : activationOperands) {
      TTNNLayoutAttr newLayout = best->second.inputLayouts[info.inputIdx];
      Value reshardedInput =
          insertInputReshard(builder, op, info.operand, newLayout);
      op->setOperand(info.inputIdx, reshardedInput);
    }

    updateOpResultLayout(op, best->second.outputLayout);
    updateOpSpecificAttrs(op, testConfigs[best->second.configIndex],
                          best->second.outputLayout);

    TTMLIR_TRACE(ttmlir::LogComponent::L1Optimizer,
                 "[L1LayoutPropagation] Op {} (first in chain): "
                 "interleaved -> L1 sharded (with input reshard, cores={}, "
                 "memLayout={})",
                 op->getName(), best->first.coreUsage,
                 best->first.memLayoutPriority);
    return true;
  }

  // Try to get L1 sharded output for an op with L1 sharded inputs.
  // Collects all valid configs and selects the best one based on core usage
  // (higher is better) and memory layout priority (HS > WS > BS).
  bool tryShardedToSharded(Operation *op,
                           const std::vector<TTNNLayoutAttr> &inputLayouts,
                           const std::vector<OpConfig> &l1Configs) {
    std::optional<std::pair<CandidateScore, std::pair<TTNNLayoutAttr, OpConfig>>>
        best;

    llvm::SmallVector<OpConfig> testConfigs =
        optimizer_utils::getUniqueTestConfigs(l1Configs,
                                              shouldUseIgnorePhysicalLayout(op));

    for (const OpConfig &config : testConfigs) {
      if (isa<Conv2dOp>(op)) {
        TTMLIR_DEBUG(ttmlir::LogComponent::L1Optimizer,
                     "[L1LayoutPropagation] tryShardedToSharded Conv2d: "
                     "outputLayout={}, numInputs={}",
                     config.outputLayout, inputLayouts.size());
        for (size_t i = 0; i < inputLayouts.size(); ++i) {
          TTMLIR_DEBUG(ttmlir::LogComponent::L1Optimizer,
                       "  input[{}]: {}", i, inputLayouts[i]);
        }
        if (auto *conv2dAttrs =
                std::get_if<Conv2dAttrs>(&config.opSpecificAttrs)) {
          TTMLIR_DEBUG(ttmlir::LogComponent::L1Optimizer,
                       "  Conv2dConfig: {}, DeviceComputeKernelConfig: {}",
                       conv2dAttrs->conv2dConfig,
                       conv2dAttrs->deviceComputeKernelConfig);
        } else {
          TTMLIR_DEBUG(ttmlir::LogComponent::L1Optimizer,
                       "  WARNING: No Conv2dAttrs in config!");
        }
      }

      auto result = op_constraint_validation::validateOperation(
          op, inputLayouts, config);

      if (result.isSuccess() &&
          result.actualOutputLayout.hasShardedL1TensorMemoryLayout()) {
        CandidateScore score =
            CandidateScore::fromLayout(result.actualOutputLayout);
        if (!best || score.isBetterThan(best->first)) {
          best = {score, {result.actualOutputLayout, config}};
        }
      }
    }

    if (best) {
      updateOpResultLayout(op, best->second.first);
      updateOpSpecificAttrs(op, best->second.second, best->second.first);

      TTMLIR_TRACE(ttmlir::LogComponent::L1Optimizer,
                   "[L1LayoutPropagation] Op {} (in chain): L1 sharded -> L1 "
                   "sharded (cores={}, memLayout={})",
                   op->getName(), best->first.coreUsage,
                   best->first.memLayoutPriority);
      return true;
    }
    return false;
  }

  // Check if an operation's result is directly returned from the function.
  // We should not apply L1 sharding to such ops to avoid type mismatches.
  bool isResultReturnedFromFunction(Operation *op) {
    if (op->getNumResults() == 0) {
      return false;
    }
    Value result = op->getResult(0);
    for (Operation *user : result.getUsers()) {
      if (isa<func::ReturnOp>(user)) {
        return true;
      }
    }
    return false;
  }

  // Apply a DRAM interleaved config to an op (fallback path).
  void applyDRAMConfig(Operation *op, const OpConfig &config) {
    updateOpSpecificAttrs(op, config, config.outputLayout);

    // For Conv2d, also set Conv2dSliceConfigAttr if not set.
    if (auto conv2dOp = dyn_cast<Conv2dOp>(op)) {
      if (!conv2dOp.getConv2dSliceConfigAttr()) {
        conv2dOp.setConv2dSliceConfigAttr(Conv2dSliceConfigAttr::get(
            conv2dOp.getContext(), Conv2dSliceType::L1Full, 0));
      }
    }

    TTMLIR_TRACE(ttmlir::LogComponent::L1Optimizer,
                 "[L1LayoutPropagation] Op {}: applied DRAM config",
                 op->getName());
  }

  // Process a single operation - try to get L1 sharded output.
  void processOp(Operation *op,
                 const TensorTypeLayoutsMap &tensorTypePossibleLayouts,
                 const llvm::DenseSet<Value> &activationValues) {
    if (!LegalOpLayoutAnalysis::isValidAnalysisTarget(op)) {
      return;
    }

    // Skip layout-related ops that don't support constraint validation.
    if (isa<ToLayoutOp, ToMemoryConfigOp, ToDTypeOp, GetDeviceOp>(op)) {
      return;
    }

    // Ops that should not be sharded - assign DRAM interleaved directly.
    // ReshapeOp: sharding can cause shard spec mismatches with subsequent ops.
    // PermuteOp: kernel doesn't properly support sharded tensor accessors.
    if (isa<ReshapeOp, PermuteOp>(op)) {
      LegalConfigs configs = getLegalConfigs(op, tensorTypePossibleLayouts);
      if (!configs.dramInterleaved.empty()) {
        applyDRAMConfig(op, configs.dramInterleaved[0]);
      }
      return;
    }

    // Skip ops whose results are directly returned from function.
    // Function return types must match the signature (DRAM interleaved).
    if (isResultReturnedFromFunction(op)) {
      TTMLIR_TRACE(ttmlir::LogComponent::L1Optimizer,
                   "[L1LayoutPropagation] Op {}: skipping (result returned "
                   "from function)",
                   op->getName());
      return;
    }

    // Get all legal configs (L1 sharded and DRAM).
    LegalConfigs configs = getLegalConfigs(op, tensorTypePossibleLayouts);

    // Get input layouts for validation.
    std::vector<TTNNLayoutAttr> inputLayouts =
        getInputLayoutsForValidation(op);

    bool success = false;

    // Try L1 sharded configs first.
    if (!configs.l1Sharded.empty()) {
      bool isFirstInChain = allInputsAreInterleaved(op);

      if (isFirstInChain) {
        // First in chain: need explicit output layout to transition
        // interleaved -> sharded.
        success = tryInterleavedToSharded(op, inputLayouts, configs.l1Sharded,
                                          tensorTypePossibleLayouts,
                                          activationValues);
      } else {
        // Middle of chain: inputs are already L1 sharded.
        success = tryShardedToSharded(op, inputLayouts, configs.l1Sharded);
      }
    }

    // Fall back to DRAM config if L1 sharding failed.
    if (!success) {
      if (!configs.dramInterleaved.empty()) {
        applyDRAMConfig(op, configs.dramInterleaved[0]);
      } else {
        TTMLIR_TRACE(ttmlir::LogComponent::L1Optimizer,
                     "[L1LayoutPropagation] Op {}: no configs available",
                     op->getName());
      }
    }
  }

  // Process a function by walking ops in topological order.
  void processFunction(func::FuncOp funcOp,
                       const TensorTypeLayoutsMap &tensorTypePossibleLayouts) {
    // Precompute which values are activations vs parameters.
    // This allows us to only reshard activation tensors, not weights/bias.
    llvm::DenseSet<Value> activationValues = computeActivationValues(funcOp);

    TTMLIR_TRACE(ttmlir::LogComponent::L1Optimizer,
                 "[L1LayoutPropagation] Function {}: {} activation values "
                 "identified",
                 funcOp.getName(), activationValues.size());

    // Walk operations in topological order (which is the default for MLIR).
    funcOp.walk([&](Operation *op) {
      // Skip non-TTNN ops.
      if (!isa<TTNNDialect>(op->getDialect())) {
        return;
      }

      processOp(op, tensorTypePossibleLayouts, activationValues);
    });
  }
};

} // namespace mlir::tt::ttnn
