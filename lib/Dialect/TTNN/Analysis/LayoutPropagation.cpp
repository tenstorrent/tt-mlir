// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/LayoutPropagation.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Analysis/LegalOpLayoutAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpModelStrategy.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"

#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>

namespace mlir::tt::ttnn {

LayoutPropagation::LayoutPropagation(
    func::FuncOp func, ttcore::GridAttr deviceGrid,
    const llvm::DenseMap<Operation *, std::vector<OpConfig>> &legalConfigs)
    : func(func), deviceGrid(deviceGrid), legalConfigs(legalConfigs) {}

void LayoutPropagation::run() {
  TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
               "LayoutPropagation::run() starting for func {0}",
               func.getName());

  size_t opIndex = 0;
  // Forward pass: propagate layouts in topological (IR) order.
  func->walk([&](Operation *op) {
    if (!LegalOpLayoutAnalysis::isValidAnalysisTarget(op)) {
      return;
    }
    // Skip ops that don't implement the OpModel interface (e.g.,
    // ttcore.load_cached). These ops cannot be validated by the backend.
    if (!mlir::dyn_cast<OpModel>(op)) {
      return;
    }
    // Skip ToLayoutOp -- these are inserted by earlier passes and their
    // layouts should be preserved, not re-decided by layout propagation.
    if (isa<ToLayoutOp>(op)) {
      return;
    }
    if (!legalConfigs.count(op)) {
      return;
    }
    // Skip ops whose operands all derive from constant/parameter arguments.
    // These ops (e.g., BFP8 typecast on weights) will be re-hoisted into
    // const_eval functions. Promoting their output to L1 would cause the
    // const_eval to return L1 tensors that starve other ops of L1 budget.
    bool allFromConstEval =
        op->getNumOperands() > 0 &&
        llvm::all_of(op->getOperands(), [](Value operand) {
          return ttcore::valueTracesToConstantArgs(operand);
        });
    if (allFromConstEval) {
      TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
                   "[op {0}] Skipping {1} @{2}: all operands from const_eval",
                   opIndex, op->getName(), op->getLoc());
      return;
    }
    // Skip ops whose output feeds directly into func.return.
    // Function outputs must stay in DRAM -- the caller expects DRAM tensors,
    // and promoting to L1 wastes budget (Pass 2 would spill them anyway).
    bool feedsReturn = llvm::any_of(
        op->getResult(0).getUsers(),
        [](Operation *user) { return isa<func::ReturnOp>(user); });
    if (feedsReturn) {
      TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
                   "[op {0}] Skipping {1} @{2}: feeds func.return",
                   opIndex, op->getName(), op->getLoc());
      return;
    }

    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "[op {0}] Processing {1} @{2}, legalConfigs={3}",
                 opIndex, op->getName(), op->getLoc(),
                 legalConfigs.find(op)->second.size());

    beamState[op] = processOp(op);

    if (!beamState[op].empty()) {
      const auto &best = beamState[op][0];
      TTMLIR_DEBUG(
          ttmlir::LogComponent::GreedyOptimizer,
          "[op {0}] -> chosen: bufType={1}, memLayout={2}, "
          "coreCount={3}, isSharded={4}, isL1={5}, reshard={6}",
          opIndex, best.config.outputLayout.getBufferType(),
          best.config.outputLayout.getMemLayout(), best.score.coreCount,
          best.score.isSharded, best.score.isL1, best.score.requiresReshard);
    }
    ++opIndex;
  });

  TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
               "LayoutPropagation: processed {0} ops, applying to IR",
               opIndex);
  // Apply resolved configs to IR.
  applyToIR();
}

llvm::SmallVector<BeamCandidate>
LayoutPropagation::processOp(Operation *op) {
  // Step 1: Build input candidate sets (one set per operand).
  std::vector<std::vector<InputCandidate>> inputCandidateSets =
      getInputCandidateSets(op);

  // Step 2: Get output hints.
  auto it = legalConfigs.find(op);
  assert(it != legalConfigs.end());
  const std::vector<OpConfig> &configs = it->second;
  OutputHints outputHints = getOutputHints(op, configs);

  // Log search space dimensions.
  size_t crossProductSize = outputHints.hints.size();
  for (const auto &ics : inputCandidateSets) {
    crossProductSize *= ics.size();
  }
  TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
               "  processOp {0}: inputSets={1}, outputHints={2}, "
               "crossProduct={3}",
               op->getName(), inputCandidateSets.size(),
               outputHints.hints.size(), crossProductSize);

  // Step 3: Cross-product evaluation.
  // For greedy (K=1), each operand typically has 1-3 candidates, and output
  // hints are small (1-3 for most ops). Total candidates are manageable.
  llvm::SmallVector<BeamCandidate> candidates;

  // Build cross-product of input candidates. For ops with many operands,
  // limit the explosion by only iterating meaningful combinations.
  // For most TTNN ops: 1-2 tensor operands.

  // Collect per-operand candidates into a flat structure for iteration.
  // We iterate the cross-product using a simple recursive approach for
  // arbitrary number of operands.
  size_t numOperandSets = inputCandidateSets.size();

  // If no tensor operands (e.g., constant-like ops), just try output hints.
  if (numOperandSets == 0) {
    for (const auto &hint : outputHints.hints) {
      auto result =
          op_constraint_validation::validateOperation(op, {}, hint);
      if (result.isSuccess()) {
        bool anyReshard = false;
        BeamCandidate candidate;
        candidate.config = OpConfig(result.actualOutputLayout,
                                    hint.opSpecificAttrs);
        candidate.score = scoreCandidate(op, hint, result, anyReshard);
        candidate.validationResult = result;
        candidates.push_back(std::move(candidate));
      }
    }
  } else {
    // Iterate cross-product of input candidates using index vector.
    llvm::SmallVector<size_t> indices(numOperandSets, 0);
    bool done = false;

    while (!done) {
      // Build the current input combination.
      std::vector<TTNNLayoutAttr> inputLayouts;
      llvm::SmallVector<size_t> producerCandidateIndices;
      llvm::DenseMap<size_t, TTNNLayoutAttr> reshardLayouts;
      bool anyReshard = false;

      inputLayouts.reserve(numOperandSets);
      producerCandidateIndices.reserve(numOperandSets);

      for (size_t i = 0; i < numOperandSets; ++i) {
        const InputCandidate &ic = inputCandidateSets[i][indices[i]];
        inputLayouts.push_back(ic.layout);
        producerCandidateIndices.push_back(ic.producerCandidateIndex);
        if (ic.isReshard) {
          anyReshard = true;
          reshardLayouts[i] = ic.layout;
        }
      }

      // Try each output hint with this input combination.
      for (const auto &hint : outputHints.hints) {
        auto result = op_constraint_validation::validateOperation(
            op, inputLayouts, hint);
        if (result.isSuccess()) {
          BeamCandidate candidate;
          candidate.config = OpConfig(result.actualOutputLayout,
                                      hint.opSpecificAttrs);
          candidate.score = scoreCandidate(op, hint, result, anyReshard);
          candidate.validationResult = result;
          candidate.inputLayouts = inputLayouts;
          candidate.producerCandidateIndices = producerCandidateIndices;
          candidate.reshardLayouts = reshardLayouts;
          candidates.push_back(std::move(candidate));
        }
      }

      // Advance the index vector (odometer-style).
      for (int i = static_cast<int>(numOperandSets) - 1; i >= 0; --i) {
        ++indices[i];
        if (indices[i] < inputCandidateSets[i].size()) {
          break;
        }
        indices[i] = 0;
        if (i == 0) {
          done = true;
        }
      }
    }
  }

  TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
               "  processOp {0}: {1} valid candidates from cross-product",
               op->getName(), candidates.size());

  // Step 4: Sort by score descending, keep top-K.
  std::sort(candidates.begin(), candidates.end(),
            [](const BeamCandidate &a, const BeamCandidate &b) {
              return a.score > b.score;
            });

  if (candidates.size() > beamWidth) {
    candidates.resize(beamWidth);
  }

  // Fallback: if no valid candidate found, use DRAM interleaved.
  if (candidates.empty()) {
    TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
                 "No valid candidate for op {0} @{1}, falling back to DRAM "
                 "interleaved.",
                 op->getName(), op->getLoc());

    TTNNLayoutAttr dramLayout = getDRAMInterleavedFallback(op);
    if (dramLayout) {
      BeamCandidate fallback;
      fallback.config = OpConfig(dramLayout);
      fallback.score = LayoutScore(); // Lowest possible score.
      candidates.push_back(std::move(fallback));
    }
  }

  return candidates;
}

std::vector<std::vector<LayoutPropagation::InputCandidate>>
LayoutPropagation::getInputCandidateSets(Operation *op) {
  std::vector<std::vector<InputCandidate>> result;

  for (auto operand : op->getOperands()) {
    auto tensorType = mlir::dyn_cast<RankedTensorType>(operand.getType());
    if (!tensorType) {
      continue;
    }
    auto currentLayout =
        mlir::dyn_cast_or_null<TTNNLayoutAttr>(tensorType.getEncoding());
    if (!currentLayout) {
      continue;
    }

    std::vector<InputCandidate> candidatesForOperand;

    // Get the producer op's resolved layout from beam state.
    Operation *producerOp = operand.getDefiningOp();
    if (producerOp && beamState.count(producerOp)) {
      const auto &producerBeam = beamState[producerOp];
      for (size_t k = 0; k < producerBeam.size() && k < beamWidth; ++k) {
        InputCandidate ic;
        ic.layout = producerBeam[k].config.outputLayout;
        ic.producerCandidateIndex = k;
        ic.isReshard = false;
        if (ic.layout) {
          candidatesForOperand.push_back(ic);
        }
      }
    }

    // If no producer in beam (func arg or unresolved), use current layout.
    if (candidatesForOperand.empty()) {
      InputCandidate ic;
      ic.layout = currentLayout;
      ic.producerCandidateIndex = 0;
      ic.isReshard = false;
      candidatesForOperand.push_back(ic);
    }

    // Add reshard candidates if applicable.
    // Skip reshards for operands derived from constant/parameter arguments.
    // These will be re-hoisted into const_eval — L1 reshards would make the
    // const_eval return L1, occupying L1 for the lifetime of the tensor.
    bool isFromConstEvalChain = ttcore::valueTracesToConstantArgs(operand);
    if (shouldExploreReshards(op) && !isFromConstEvalChain) {
      std::vector<TTNNLayoutAttr> reshardCandidates =
          generateReshardCandidates(tensorType, currentLayout);
      for (const auto &reshardLayout : reshardCandidates) {
        // Only add if different from already-present candidates.
        bool alreadyPresent = false;
        for (const auto &existing : candidatesForOperand) {
          if (existing.layout == reshardLayout) {
            alreadyPresent = true;
            break;
          }
        }
        if (!alreadyPresent) {
          InputCandidate ic;
          ic.layout = reshardLayout;
          // Back-point to the first producer candidate (greedy: always 0).
          ic.producerCandidateIndex = 0;
          ic.isReshard = true;
          candidatesForOperand.push_back(ic);
        }
      }
    }

    TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
                 "  operand {0}: {1} candidates (fromProducer={2}, "
                 "reshards={3})",
                 result.size(), candidatesForOperand.size(),
                 producerOp && beamState.count(producerOp)
                     ? beamState[producerOp].size()
                     : 0,
                 candidatesForOperand.size() -
                     (producerOp && beamState.count(producerOp)
                          ? std::min(beamState[producerOp].size(), beamWidth)
                          : 1));
    result.push_back(std::move(candidatesForOperand));
  }

  return result;
}

std::vector<TTNNLayoutAttr>
LayoutPropagation::generateReshardCandidates(
    RankedTensorType tensorType, TTNNLayoutAttr currentLayout) {
  // Don't generate interleaved reshard candidates. Resharding from sharded to
  // interleaved (DRAM or L1) almost always hurts performance — the consumer op
  // ends up on a slow kernel path (e.g. in0:l1_interleaved matmul is ~5x
  // slower than in0:dram_interleaved). Sharded-to-sharded reshards will be
  // added here once op-type-aware candidate generation is implemented.
  return {};
}

TTNNLayoutAttr LayoutPropagation::getDRAMInterleavedFallback(Operation *op) {
  if (op->getNumResults() == 0) {
    return nullptr;
  }
  auto tensorType =
      mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!tensorType) {
    return nullptr;
  }
  auto currentLayout =
      mlir::dyn_cast_or_null<TTNNLayoutAttr>(tensorType.getEncoding());
  if (!currentLayout) {
    return nullptr;
  }
  return currentLayout.withBufferType(BufferType::DRAM)
      .withMemoryLayout(TensorMemoryLayout::Interleaved);
}

//===----------------------------------------------------------------------===//
// IR Transformation
//===----------------------------------------------------------------------===//

void LayoutPropagation::applyToIR() {
  TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
               "applyToIR: applying configs for {0} ops in beam state",
               beamState.size());
  // First pass: apply op configs (result types, DPS operands, op-specific
  // attrs, L1 usage annotation).
  func->walk([&](Operation *op) {
    if (!beamState.count(op)) {
      return;
    }

    const auto &beam = beamState[op];
    if (beam.empty()) {
      return;
    }

    // For greedy (K=1), always use index 0 (best candidate).
    applyOpConfig(op, beam[0]);
  });

  // Second pass: insert reshard ops for edges that require memory
  // reconfiguration.
  func->walk([&](Operation *op) {
    if (!beamState.count(op)) {
      return;
    }

    const auto &beam = beamState[op];
    if (beam.empty()) {
      return;
    }

    const BeamCandidate &chosen = beam[0];
    for (const auto &[operandIdx, reshardLayout] : chosen.reshardLayouts) {
      insertReshardOp(op, operandIdx, reshardLayout);
    }
  });

  // Fixup: disable deallocate_activation for conv2d/conv_transpose2d ops
  // whose input has multiple users, preventing use-after-free. This mirrors
  // the tryDisableDeallocateActivation logic in DFShardingPolicy.cpp.
  func->walk([&](Operation *op) {
    auto disableDeallocIfMultiUser = [](auto convOp) {
      auto config = convOp.getConv2dConfigAttr();
      if (!config || !config.getDeallocateActivation() ||
          !config.getDeallocateActivation().getValue()) {
        return;
      }
      Value input = convOp.getInput();
      if (!input.hasOneUse()) {
        convOp.setConv2dConfigAttr(
            config.withDeallocateActivation(false));
        TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                     "Disabled deallocate_activation for conv2d with "
                     "multi-use input: {}",
                     ttmlir::opToString(convOp));
      }
    };

    if (auto conv2d = dyn_cast<ttnn::Conv2dOp>(op)) {
      disableDeallocIfMultiUser(conv2d);
    } else if (auto convT = dyn_cast<ttnn::ConvTranspose2dOp>(op)) {
      disableDeallocIfMultiUser(convT);
    }
  });

  // Third pass: update function return types.
  updateFunctionReturnTypes();
}

void LayoutPropagation::applyOpConfig(Operation *op,
                                       const BeamCandidate &candidate) {
  TTNNLayoutAttr chosenLayout = candidate.config.outputLayout;
  if (!chosenLayout) {
    return;
  }

  RankedTensorType tensorType =
      mlir::cast<RankedTensorType>(op->getResult(0).getType());
  llvm::ArrayRef<int64_t> tensorShape = tensorType.getShape();

  // Preserve quantized element types.
  Type originalElementType = tensorType.getElementType();
  Type newElementType = originalElementType;
  if (!mlir::isa<mlir::quant::QuantizedType>(originalElementType)) {
    newElementType = chosenLayout.getScalarElementType();
  }

  RankedTensorType newTensorType =
      RankedTensorType::get(tensorShape, newElementType, chosenLayout);

  // Update layout attribute for ops that have layout interface.
  if (auto opWithLayoutIF = mlir::dyn_cast<TTNNLayoutOpInterface>(op)) {
    opWithLayoutIF.setLayoutAttr(
        LayoutAttr::get(op->getContext(), chosenLayout.getLayout()));
  }

  // Update result type.
  op->getResult(0).setType(newTensorType);

  // Update output data type attribute.
  if (auto dtypeOp = mlir::dyn_cast<TTNNDtypeOpInterface>(op)) {
    ttcore::DataTypeAttr newDataTypeAttr = ttcore::DataTypeAttr::get(
        op->getContext(), chosenLayout.getDataType());
    dtypeOp.setDtypeAttr(newDataTypeAttr);
  }

  // Update DPS operand (EmptyOp).
  if (isa<mlir::DestinationStyleOpInterface>(op)) {
    BufferType bufferType = chosenLayout.getBufferType();
    TensorMemoryLayoutAttr tensorMemoryLayoutAttr =
        chosenLayout.getMemLayout();

    op->getOperands().back().setType(newTensorType);
    EmptyOp emptyOp =
        mlir::cast<EmptyOp>(op->getOperands().back().getDefiningOp());

    emptyOp.setDtype(chosenLayout.getDataType());
    if (chosenLayout.isTiled()) {
      emptyOp.setLayout(ttnn::Layout::Tile);
    } else {
      emptyOp.setLayout(ttnn::Layout::RowMajor);
    }

    emptyOp.setMemoryConfigAttr(ttnn::MemoryConfigAttr::get(
        op->getContext(), tensorMemoryLayoutAttr,
        BufferTypeAttr::get(op->getContext(), bufferType),
        utils::createShardSpecIfNeeded(chosenLayout, deviceGrid)));
  }
  // Handle existing ToLayoutOp memory config alignment.
  else if (isa<ttnn::ToLayoutOp>(op)) {
    ttnn::ToLayoutOp toLayoutOp = llvm::cast<ttnn::ToLayoutOp>(op);
    toLayoutOp.setMemoryConfigAttr(ttnn::MemoryConfigAttr::get(
        op->getContext(), chosenLayout.getMemLayout(),
        ttnn::BufferTypeAttr::get(op->getContext(),
                                  chosenLayout.getBufferType()),
        utils::createShardSpecIfNeeded(chosenLayout, deviceGrid)));
  }

  // Set op-specific configurations (Conv2d, Matmul).
  llvm::TypeSwitch<Operation *, void>(op)
      .Case<ttnn::Conv2dOp>([&](ttnn::Conv2dOp convOp) {
        if (std::holds_alternative<ttnn::Conv2dAttrs>(
                candidate.config.opSpecificAttrs)) {
          ttnn::Conv2dAttrs conv2dAttrs =
              std::get<ttnn::Conv2dAttrs>(candidate.config.opSpecificAttrs);
          if (conv2dAttrs.conv2dConfig.has_value()) {
            convOp.setConv2dConfigAttr(conv2dAttrs.conv2dConfig.value());
          }
          if (conv2dAttrs.deviceComputeKernelConfig.has_value()) {
            convOp.setComputeConfigAttr(
                conv2dAttrs.deviceComputeKernelConfig.value());
          }
        }
      })
      .Case<ttnn::ConvTranspose2dOp>(
          [&](ttnn::ConvTranspose2dOp convOp) {
            if (std::holds_alternative<ttnn::Conv2dAttrs>(
                    candidate.config.opSpecificAttrs)) {
              ttnn::Conv2dAttrs conv2dAttrs =
                  std::get<ttnn::Conv2dAttrs>(
                      candidate.config.opSpecificAttrs);
              if (conv2dAttrs.conv2dConfig.has_value()) {
                convOp.setConv2dConfigAttr(
                    conv2dAttrs.conv2dConfig.value());
              }
            }
          })
      .Case<ttnn::MatmulOp, ttnn::LinearOp>([&](auto matmulOp) {
        if (std::holds_alternative<ttnn::MatmulAttrs>(
                candidate.config.opSpecificAttrs)) {
          ttnn::MatmulAttrs matmulAttrs = std::get<ttnn::MatmulAttrs>(
              candidate.config.opSpecificAttrs);
          if (matmulAttrs.matmulProgramConfig.has_value()) {
            auto programConfig = matmulAttrs.matmulProgramConfig.value();
            matmulOp.setMatmulProgramConfigAttr(programConfig);
            // Workaround for tt-metal issue #35060.
            bool hasFusedActivation =
                llvm::TypeSwitch<mlir::Attribute, bool>(programConfig)
                    .template Case<
                        MatmulMultiCoreReuseMultiCastProgramConfigAttr,
                        MatmulMultiCoreReuseMultiCast1DProgramConfigAttr,
                        MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr>(
                        [](auto config) {
                          return config.getFusedActivation() != nullptr;
                        })
                    .Default([](mlir::Attribute) { return false; });
            if (hasFusedActivation) {
              matmulOp.removeActivationAttr();
            }
          }
        }
      })
      .Default([](Operation *) {});

  // Attach L1 usage annotation for Pass 2 (spill management).
  if (chosenLayout.hasL1BufferType() &&
      candidate.validationResult.isSuccess() &&
      candidate.validationResult.outputL1Usage > 0) {
    OpBuilder builder(op->getContext());
    op->setAttr("ttnn.output_l1_usage",
                builder.getI64IntegerAttr(
                    candidate.validationResult.outputL1Usage));
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "L1 annotation: {0} @{1} -> outputL1Usage={2} bytes, "
                 "bufType={3}, memLayout={4}",
                 op->getName(), op->getLoc(),
                 candidate.validationResult.outputL1Usage,
                 chosenLayout.getBufferType(),
                 chosenLayout.getMemLayout());
  }
}

void LayoutPropagation::insertReshardOp(Operation *consumerOp,
                                         size_t operandIndex,
                                         TTNNLayoutAttr reshardLayout) {
  Value operand = consumerOp->getOperand(operandIndex);
  auto producerTensorType =
      mlir::cast<RankedTensorType>(operand.getType());

  // Skip if the memory config transition would be a no-op.
  // Check buffer type, memory layout, and (for sharded layouts) grid.
  if (auto producerLayout = mlir::dyn_cast_or_null<TTNNLayoutAttr>(
          producerTensorType.getEncoding())) {
    bool sameBufferType =
        producerLayout.getBufferType() == reshardLayout.getBufferType();
    bool sameMemLayout =
        producerLayout.getMemLayout() == reshardLayout.getMemLayout();

    if (sameBufferType && sameMemLayout) {
      // For sharded layouts, also require matching grids.
      bool bothSharded =
          isShardedMemoryLayout(producerLayout.getMemLayout().getValue()) &&
          isShardedMemoryLayout(reshardLayout.getMemLayout().getValue());
      if (!bothSharded || producerLayout.getGrid() == reshardLayout.getGrid()) {
        return;
      }
    }
  }

  // Build the output layout by taking the producer's current layout and
  // applying the target buffer type, memory layout, and grid.
  TTNNLayoutAttr producerLayout =
      utils::getLayoutAttrFromTensor(producerTensorType);
  TTNNLayoutAttr outputLayout =
      producerLayout.withBufferType(reshardLayout.getBufferType())
          .withMemoryLayout(reshardLayout.getMemLayout())
          .withGrid(producerTensorType.getShape(), reshardLayout.getGrid())
          .withShardShape(reshardLayout.getScalarShardShape());
  RankedTensorType newTensorType =
      utils::RankedTensorTypeFactory::create(producerTensorType, outputLayout);

  MemoryConfigAttr outputMemConfigAttr = MemoryConfigAttr::get(
      consumerOp->getContext(), reshardLayout.getMemLayout(),
      BufferTypeAttr::get(consumerOp->getContext(),
                          reshardLayout.getBufferType()),
      utils::createShardSpecIfNeeded(reshardLayout, deviceGrid));

  OpBuilder builder(consumerOp);
  Location loc = ttmlir::utils::appendLocationSuffix(consumerOp->getLoc(),
                                                     "_mem_reconfig");

  ToMemoryConfigOp memoryReconfigOp = builder.create<ToMemoryConfigOp>(
      loc, newTensorType, operand, outputMemConfigAttr);

  consumerOp->setOperand(operandIndex, memoryReconfigOp->getResult(0));

  TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
               "Inserted memory reconfig op: {0}", memoryReconfigOp);
}

void LayoutPropagation::updateFunctionReturnTypes() {
  SmallVector<Type> funcResultTypes;

  func->walk([&](Operation *op) {
    if (op->getNumResults() == 0) {
      if (auto funcReturn = dyn_cast<func::ReturnOp>(op)) {
        funcResultTypes.append(funcReturn.getOperandTypes().begin(),
                               funcReturn.getOperandTypes().end());
      }
    }
  });

  FunctionType funcType = func.getFunctionType();
  FunctionType newFuncType = FunctionType::get(
      func.getContext(), funcType.getInputs(), funcResultTypes);
  func.setType(newFuncType);
}

} // namespace mlir::tt::ttnn
