// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNTraits.h"
#include "ttmlir/Dialect/TTNN/Interfaces/OpModelError.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Support/Logger.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

#include <limits>

namespace mlir::tt::ttnn {

namespace op_constraint_validation {

llvm::StringRef validationStatusToString(ValidationStatus status) {
  switch (status) {
  case ValidationStatus::Success:
    return "Success";
  case ValidationStatus::NotImplemented:
    return "NotImplemented";
  case ValidationStatus::MetalBackendError:
    return "MetalBackendError";
  case ValidationStatus::UnmatchedReferenceConfig:
    return "UnmatchedReferenceConfig";
  case ValidationStatus::OutOfMemoryError:
    return "OutOfMemoryError";
  }
  return "Unknown";
}

static bool tensorBackedCBFits(TTNNLayoutAttr layout, uint64_t rows,
                               uint64_t cols) {
  if (!layout) {
    return true;
  }
  auto memLayout = layout.getMemLayoutOpt();
  if (!memLayout || !isShardedMemoryLayout(*memLayout)) {
    return true;
  }

  uint64_t elementSize = layout.getElementSizeBytes();
  if (rows != 0 && cols > std::numeric_limits<uint64_t>::max() / rows) {
    return false;
  }
  uint64_t elements = rows * cols;
  if (elementSize != 0 &&
      elements > std::numeric_limits<uint64_t>::max() / elementSize) {
    return false;
  }
  return elements * elementSize <= layout.getShardSizeInBytes();
}

std::optional<std::string>
getMatmulPreflightError(llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
                        const OpConfig &config, Operation *contextOp) {
  TTNNLayoutAttr output = config.outputLayout;
  // TTNN matmul kernels produce tiled outputs.  Passing only a row-major
  // sharded MemoryConfig makes the backend combine its tiled page layout with
  // a scalar shard shape (for example 1024 / 5 -> 204 rows), which TT-Metal
  // rejects because the physical shard is not tile aligned.  Check this before
  // the optional program config: the auto-picker path has the same constraint.
  if (output && output.hasShardedTensorMemoryLayout() && !output.isTiled()) {
    return "matmul sharded output requires a tiled layout";
  }
  for (size_t i = 0; i < inputLayouts.size(); ++i) {
    TTNNLayoutAttr input = inputLayouts[i];
    if (input && input.hasShardedTensorMemoryLayout() && !input.isTiled()) {
      return "matmul sharded input " + std::to_string(i) +
             " requires a tiled layout";
    }
  }

  std::optional<Attribute> programConfig;
  if (const auto *attrs =
          std::get_if<MatmulAttrs>(&config.opSpecificAttrs)) {
    if (attrs->matmulProgramConfig && *attrs->matmulProgramConfig) {
      programConfig = *attrs->matmulProgramConfig;
    }
  }
  // Match TTNNOpModelInterface::unpackMatmulAttrs: when the candidate does not
  // override the program config, the backend still receives the config already
  // attached to the original Matmul/Linear op.
  if (!programConfig && contextOp) {
    programConfig =
        llvm::TypeSwitch<Operation *, std::optional<Attribute>>(contextOp)
            .Case<MatmulOp, LinearOp>([](auto op) {
              auto attr = op.getMatmulProgramConfig();
              return attr ? std::optional<Attribute>(attr) : std::nullopt;
            })
            .Default([](Operation *) { return std::nullopt; });
  }

  if (!programConfig) {
    if (output && output.hasShardedTensorMemoryLayout()) {
      return "matmul sharded output requires an explicit program config";
    }
    for (TTNNLayoutAttr input : inputLayouts) {
      if (input && input.hasShardedTensorMemoryLayout()) {
        return "matmul sharded input requires an explicit program config";
      }
    }
    return std::nullopt;
  }

  Attribute programConfigAttr = *programConfig;
  uint64_t perCoreM = 0;
  uint64_t perCoreN = 0;
  bool is1D = false;
  bool movesIn0 = false;
  if (auto attr = dyn_cast<MatmulMultiCoreReuseMultiCastProgramConfigAttr>(
          programConfigAttr)) {
    perCoreM = attr.getPerCoreM();
    perCoreN = attr.getPerCoreN();
  } else if (auto attr =
                 dyn_cast<MatmulMultiCoreReuseMultiCast1DProgramConfigAttr>(
                     programConfigAttr)) {
    perCoreM = attr.getPerCoreM();
    perCoreN = attr.getPerCoreN();
    is1D = true;
    movesIn0 = attr.getMcastIn0() || attr.getGatherIn0();
  } else {
    return std::nullopt;
  }

  if (output && output.hasShardedTensorMemoryLayout() &&
      output.getIgnorePhysicalLayout()) {
    return "matmul explicit program config requires a physical sharded output "
           "layout";
  }
  if (!tensorBackedCBFits(output, perCoreM, perCoreN)) {
    return "matmul output circular buffer exceeds its tensor shard";
  }

  if (!inputLayouts.empty() && inputLayouts[0]) {
    TTNNLayoutAttr in0 = inputLayouts[0];
    auto memLayout = in0.getMemLayoutOpt();
    if (memLayout && isShardedMemoryLayout(*memLayout)) {
      auto shardShape = in0.getShardShape();
      if (shardShape.size() != 2 ||
          !tensorBackedCBFits(in0, perCoreM, shardShape[1])) {
        return "matmul input 0 circular buffer exceeds its tensor shard";
      }
    }
  }

  if (inputLayouts.size() > 1 && inputLayouts[1]) {
    TTNNLayoutAttr in1 = inputLayouts[1];
    auto memLayout = in1.getMemLayoutOpt();
    if (memLayout && isShardedMemoryLayout(*memLayout)) {
      auto shardShape = in1.getShardShape();
      if (shardShape.size() != 2 ||
          !tensorBackedCBFits(in1, shardShape[0], perCoreN)) {
        return "matmul input 1 circular buffer exceeds its tensor shard";
      }
    }
  }

  if (is1D) {
    auto outputMemLayout = output ? output.getMemLayoutOpt() : std::nullopt;
    TensorMemoryLayout directionalLayout =
        movesIn0 ? TensorMemoryLayout::WidthSharded
                 : TensorMemoryLayout::HeightSharded;
    if (!outputMemLayout ||
        (*outputMemLayout != directionalLayout &&
         *outputMemLayout != TensorMemoryLayout::BlockSharded)) {
      return movesIn0
                 ? "matmul 1D row output must be WidthSharded or BlockSharded"
                 : "matmul 1D column output must be HeightSharded or "
                   "BlockSharded";
    }
    CoreRangeSetAttr ranges = output.getCoreRangeSet();
    auto bbox = ranges ? ranges.getBoundingBox() : std::nullopt;
    if (!bbox) {
      return "matmul 1D sharded output has no physical core range";
    }
    if (movesIn0 &&
        bbox->getStartCoord().getY() != bbox->getEndCoord().getY()) {
      return "matmul 1D output is not a physical row";
    }
    if (!movesIn0 &&
        bbox->getStartCoord().getX() != bbox->getEndCoord().getX()) {
      return "matmul 1D output is not a physical column";
    }
  }

  return std::nullopt;
}

static std::optional<std::string>
getRmsNormPreflightError(llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
                         const OpConfig &config, RMSNormOp op) {
  if (inputLayouts.empty() || !inputLayouts[0]) {
    return std::nullopt;
  }

  TTNNLayoutAttr input = inputLayouts[0];
  auto inputMemLayout = input.getMemLayoutOpt();
  if (!inputMemLayout || !isShardedMemoryLayout(*inputMemLayout)) {
    return std::nullopt;
  }

  if (op.getResidualInputTensor()) {
    size_t residualIdx = 1;
    residualIdx += op.getWeight() ? 1 : 0;
    residualIdx += op.getBias() ? 1 : 0;
    if (residualIdx >= inputLayouts.size() ||
        inputLayouts[residualIdx] != input) {
      return "rms_norm sharded input and residual require identical physical "
             "layouts";
    }
  }

  if (config.outputLayout && config.outputLayout != input) {
    return "rms_norm sharded input and output require identical physical "
           "layouts";
  }
  return std::nullopt;
}

static ValidationResult
validateConstraints(Operation *op, llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
                    const OpConfig &config, uint64_t additionalL1Usage);

//----------- Public API implementations ----------

ValidationResult validateOperation(Operation *op,
                                   llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
                                   const OpConfig &config,
                                   uint64_t additionalL1Usage) {
  return validateConstraints(op, inputLayouts, config, additionalL1Usage);
}

std::vector<ValidationResult>
validateWithMultipleAttributes(Operation *op,
                               llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
                               llvm::ArrayRef<OpConfig> opConfigs,
                               llvm::ArrayRef<OpConfig> referenceConfigs) {

  std::vector<ValidationResult> results;
  for (const auto &testConfig : opConfigs) {
    // 1. Call core constraint checking.
    ValidationResult constraintResult = validateConstraints(
        op, inputLayouts, testConfig, /*additionalL1Usage=*/0);

    // If not supported, backend error, or validation error - add to results
    // and continue (don't fail early, collect all results)
    if (!constraintResult.isSuccess()) {
      results.push_back(constraintResult);
      continue;
    }

    // TODO(bmalesevic, #7108): propagate all output layouts once multi-output
    // matching is supported.
    const auto firstActualOutputLayout =
        constraintResult.checkAndGetFirstActualOutputLayout();

    // 2. Search referenceConfigs for matching (outputLayout + opSpecificAttr).
    if (!referenceConfigs.empty()) {
      bool foundMatch = false;
      for (size_t i = 0; i < referenceConfigs.size(); ++i) {
        if (referenceConfigs[i].outputLayout == firstActualOutputLayout &&
            referenceConfigs[i].opSpecificAttrs == testConfig.opSpecificAttrs) {
          results.push_back(
              ValidationResult::success(i, firstActualOutputLayout));
          foundMatch = true;
          break;
        }
      }

      if (!foundMatch) {
        results.push_back(ValidationResult::unmatchedReferenceConfig(
            "No matching reference config found"));
      }
    } else {
      // No reference configs to search - consider validation success as match.
      results.push_back(ValidationResult::success(0, firstActualOutputLayout));
    }
  }

  return results;
}

// ----------- Shared L1 budget check ----------

ValidationResult
checkConstraintsResult(Operation *contextOp,
                       llvm::Expected<op_model::OpConstraints> constraints,
                       uint64_t additionalL1Usage) {
  if (!constraints) {
    ValidationResult result;
    llvm::handleAllErrors(
        constraints.takeError(),
        [&](ttnn::detail::OpNotSupportedError &notSupportedErr) {
          result = ValidationResult::notImplemented(notSupportedErr.message());
        },
        [&](llvm::ErrorInfoBase &otherErr) {
          std::string errorMsg = otherErr.message();
          TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
                       "OpModel constraints failed: {}",
                       ttmlir::utils::firstNLines(errorMsg, 8));
          result = ValidationResult::metalBackendError(
              ttmlir::utils::firstNLines(errorMsg, 8));
        });
    return result;
  }

  auto [cbPeakUsage, l1BuffersPeakUsage, overallPeakL1Usage,
        outputTensorUsagePerCore, outputLayouts] = constraints.get();

  uint64_t effectiveL1Limit = utils::getUsableL1PerCore(contextOp);
  uint64_t totalL1Usage = overallPeakL1Usage + additionalL1Usage;

  if (totalL1Usage > effectiveL1Limit) {
    TTMLIR_DEBUG(
        ttmlir::LogComponent::OpValidation,
        "Not enough L1 memory. "
        "totalL1Usage: {} [overallPeakL1Usage={}, additionalL1Usage={}]"
        " [cbPeakUsage={}, l1BuffersPeakUsage={}] limit: {}",
        totalL1Usage, overallPeakL1Usage, additionalL1Usage, cbPeakUsage,
        l1BuffersPeakUsage, effectiveL1Limit);
    return ValidationResult::outOfMemoryError("Not enough L1 memory");
  }

  TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
               "OpModel constraints valid. FirstOutputLayout: {}\n"
               "L1 usage: overallPeakL1Usage={}, cbPeakUsage={}, "
               "l1BuffersPeakUsage={}, outputTensorUsagePerCore={}",
               outputLayouts.empty() ? nullptr : outputLayouts[0],
               overallPeakL1Usage, cbPeakUsage, l1BuffersPeakUsage,
               outputTensorUsagePerCore);

  return ValidationResult::success(0, outputLayouts, outputTensorUsagePerCore,
                                   cbPeakUsage);
}

// ----------- Core constraint validation implementation ----------

static ValidationResult
validateConstraints(Operation *op, llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
                    const OpConfig &config, uint64_t additionalL1Usage) {

  if (isa<MatmulOp, LinearOp>(op)) {
    if (auto error = getMatmulPreflightError(inputLayouts, config, op)) {
      return ValidationResult::metalBackendError(*error);
    }
  }
  if (isa<RMSNormOp>(op)) {
    if (auto error = getRmsNormPreflightError(
            inputLayouts, config, cast<RMSNormOp>(op))) {
      return ValidationResult::metalBackendError(*error);
    }
  }

  // Check that operation supports OpModel interface.
  auto backend = mlir::dyn_cast<OpModel>(op);
  if (!backend) {
    // Ops marked with the OpModelExempt trait deliberately do not implement
    // the OpModel interface (e.g. CCL/multi-device, trace, generic, or other
    // ops without a metal-side definition). The optimizer relies on observing
    // a NotImplemented result for such ops so it can fall back gracefully
    // (e.g. evict L1 state) instead of treating the op as analyzable.
    if (op->hasTrait<OpModelExempt>()) {
      return ValidationResult::notImplemented(
          (llvm::Twine("OpModel interface not implemented for op ") +
           op->getName().getStringRef() + " (OpModelExempt)")
              .str());
    }
    llvm::reportFatalInternalError(llvm::Twine("Backend constraints are not "
                                               "implemented for op ")
                                       .concat(op->getName().getStringRef()));
  }

  // Constraints are implemented for this op.
  auto deviceAttr = ttcore::lookupDevice(op);
  if (!deviceAttr) {
    llvm::reportFatalInternalError(
        llvm::Twine("No device attribute found for operation ")
            .concat(op->getName().getStringRef()));
  }

  TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
               "About to call getOpConstraints for {} with {} input layouts, "
               "additionalL1={}",
               ttmlir::opToString(op), inputLayouts.size(), additionalL1Usage);

  for (size_t i = 0; i < inputLayouts.size(); ++i) {
    TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
                 "Input layout {}: {}, getLayout()={}, dtype={}", i,
                 inputLayouts[i], static_cast<int>(inputLayouts[i].getLayout()),
                 static_cast<int>(inputLayouts[i].getDataType()));
  }
  TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation, "Output config {}", config);

  llvm::Expected<ttnn::op_model::OpConstraints> l1UsageExp =
      backend.getOpConstraints(inputLayouts, config);

  return checkConstraintsResult(op, std::move(l1UsageExp), additionalL1Usage);
}

} // namespace op_constraint_validation
} // namespace mlir::tt::ttnn
