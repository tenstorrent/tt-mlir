// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/LegalOpConfigAnalysis.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTCore/Utils/CoreRangeSet.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfigAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Support/Logger.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <vector>

namespace mlir::tt::ttnn {

static inline int64_t divUp(int64_t a, int64_t b) { return (a + b - 1) / b; }

// Find the largest divisor of 'value' that is <= 'maxDivisor'.
static inline int64_t largestDivisorUpTo(int64_t value, int64_t maxDivisor) {
  for (int64_t d = std::min(value, maxDivisor); d >= 1; --d) {
    if (value % d == 0) {
      return d;
    }
  }
  return 1;
}

// Compute the bounding box grid dimensions from a layout's shard grid.
static std::pair<int64_t, int64_t>
getPhysicalGridDimensions(TTNNLayoutAttr layout) {
  ttcore::GridAttr shardGrid = layout.getGrid();
  AffineMap mapping = shardGrid.getMapping();

  auto coreRanges =
      ttcore::utils::toCoreRangeSet(shardGrid.getShape(), mapping);

  int64_t maxX = 0;
  int64_t maxY = 0;
  for (const auto &[loc, size] : coreRanges) {
    // loc is [x, y] per toCoreRangeSet convention
    maxX = std::max(maxX, static_cast<int64_t>(loc[0] + size[0]));
    maxY = std::max(maxY, static_cast<int64_t>(loc[1] + size[1]));
  }

  return {maxX, maxY};
}

// Convert activation string to UnaryWithParamAttr.
// Returns nullptr if activation is not set or not recognized.
static UnaryWithParamAttr
getActivationAttr(MLIRContext *ctx, std::optional<StringRef> activation) {
  if (!activation.has_value() || activation->empty()) {
    return nullptr;
  }
  auto unaryOpType = symbolizeUnaryOpType(*activation);
  if (!unaryOpType.has_value()) {
    return nullptr;
  }
  return UnaryWithParamAttr::get(ctx, *unaryOpType,
                                 llvm::ArrayRef<FloatAttr>{});
}

// Matmul Program Config Constraints (from matmul_op.cpp):
// --------------------------------------------------------
// Non-zero constraints:
//   - in0_block_w != 0
//   - out_subblock_h != 0
//   - out_subblock_w != 0
//   - out_block_h != 0
//   - out_block_w != 0
//   - per_core_M != 0
//   - per_core_N != 0
//
// Divisibility constraints:
//   - Kt % in0_block_w == 0
//   - per_core_M % out_subblock_h == 0
//   - per_core_N % out_subblock_w == 0
//   - per_core_M % out_block_h == 0
//   - per_core_N % out_block_w == 0
//   - out_block_h % out_subblock_h == 0
//   - out_block_w % out_subblock_w == 0
//
// Register constraints:
//   - out_subblock_w * out_subblock_h <= available_reg_count (typically 8)
//
// L1 memory constraints:
//   - Circular buffers for input/output tiles must fit in L1 memory.
//   - out_block_h * out_block_w determines output CB size per core.
//   - in0_block_w * out_block_h determines in0 CB size per core.
//
// TODO(rpavlovicTT): Currently we set out_block_h = per_core_M and out_block_w
// = per_core_N, which may exceed L1 capacity for large tensors. A follow-up
// improvement is to generate multiple configs with different out_block_h/w
// values (divisors of per_core_M/N) and use OpModel validation to find a config
// that fits in L1. This would enable handling larger matmuls by trading off
// reuse for memory.
//
// Generate MatmulMultiCoreReuseMultiCast1DProgramConfig for width/height
// sharded output.
static mlir::Attribute
generateMatmul1DProgramConfig(MLIRContext *ctx, int64_t Mt, int64_t Nt,
                              int64_t Kt, TTNNLayoutAttr outputLayout,
                              TensorMemoryLayout outputMemLayout,
                              UnaryWithParamAttr fusedActivation) {
  auto [gridX, gridY] = getPhysicalGridDimensions(outputLayout);
  int64_t numCores = gridX * gridY;

  bool mcastIn0 = (outputMemLayout == TensorMemoryLayout::WidthSharded);
  int64_t perCoreM, perCoreN;

  if (mcastIn0) {
    perCoreM = Mt;
    perCoreN = divUp(Nt, numCores);
  } else {
    perCoreM = divUp(Mt, numCores);
    perCoreN = Nt;
  }

  constexpr int64_t kLargeNtThreshold = 128;
  int64_t in0BlockW;
  if (!mcastIn0) {
    in0BlockW = Kt;
  } else {
    if (Nt > kLargeNtThreshold) {
      in0BlockW = (Kt % 2 == 0) ? 2 : 1;
    } else {
      if (Kt % 8 == 0) {
        in0BlockW = 8;
      } else if (Kt % 4 == 0) {
        in0BlockW = 4;
      } else if (Kt % 2 == 0) {
        in0BlockW = 2;
      } else {
        in0BlockW = 1;
      }
    }
  }

  int64_t outBlockH = perCoreM;
  int64_t outBlockW = perCoreN;
  int64_t outSubblockH = 1;
  // out_subblock_w must divide out_block_w (== perCoreN) evenly.
  // See matmul_op.cpp constraints: out_block_w % out_subblock_w == 0.
  int64_t outSubblockW = largestDivisorUpTo(outBlockW, 8);

  auto gridAttr = CoreCoordAttr::get(ctx, gridX, gridY);
  auto hopCoresAttr = CoreRangeSetAttr::get(ctx, {});

  return MatmulMultiCoreReuseMultiCast1DProgramConfigAttr::get(
      ctx, gridAttr, static_cast<uint64_t>(in0BlockW),
      static_cast<uint64_t>(outSubblockH), static_cast<uint64_t>(outSubblockW),
      static_cast<uint64_t>(outBlockH), static_cast<uint64_t>(outBlockW),
      static_cast<uint64_t>(perCoreM), static_cast<uint64_t>(perCoreN),
      /*fuse_batch=*/true, /*fusedActivation=*/fusedActivation, mcastIn0,
      /*gather_in0=*/false, hopCoresAttr, /*num_global_cb_receivers=*/0,
      /*untilize_out=*/false);
}

// Generate MatmulMultiCoreReuseMultiCastProgramConfig for block sharded output.
static mlir::Attribute
generateMatmul2DProgramConfig(MLIRContext *ctx, int64_t Mt, int64_t Nt,
                              int64_t Kt, TTNNLayoutAttr outputLayout,
                              UnaryWithParamAttr fusedActivation) {
  auto [gridX, gridY] = getPhysicalGridDimensions(outputLayout);

  int64_t perCoreM = divUp(Mt, gridY);
  int64_t perCoreN = divUp(Nt, gridX);

  int64_t in0BlockW = (Kt % 2 == 0) ? 2 : 1;
  int64_t outSubblockH = 1;

  // out_subblock_w must divide out_block_w (== perCoreN) evenly.
  // See matmul_op.cpp constraints: out_block_w % out_subblock_w == 0.
  int64_t outSubblockW = largestDivisorUpTo(perCoreN, 8);
  int64_t outBlockH = perCoreM;
  int64_t outBlockW = perCoreN;

  auto gridAttr = CoreCoordAttr::get(ctx, gridX, gridY);

  return MatmulMultiCoreReuseMultiCastProgramConfigAttr::get(
      ctx, gridAttr, static_cast<uint64_t>(in0BlockW),
      static_cast<uint64_t>(outSubblockH), static_cast<uint64_t>(outSubblockW),
      static_cast<uint64_t>(outBlockH), static_cast<uint64_t>(outBlockW),
      static_cast<uint64_t>(perCoreM), static_cast<uint64_t>(perCoreN),
      /*transpose_mcast=*/false, /*fusedActivation=*/fusedActivation,
      /*fuse_batch=*/true);
}

// Generate matmul program config for an op with given output layout.
// Returns nullopt if output is not sharded.
[[maybe_unused]] static std::optional<mlir::Attribute>
generateMatmulProgramConfigForOp(Operation *op, TTNNLayoutAttr outputLayout) {
  if (!outputLayout || !outputLayout.hasShardedL1TensorMemoryLayout()) {
    return std::nullopt;
  }

  TensorMemoryLayout outputMemLayout = outputLayout.getMemLayout().getValue();
  if (outputMemLayout != TensorMemoryLayout::WidthSharded &&
      outputMemLayout != TensorMemoryLayout::HeightSharded &&
      outputMemLayout != TensorMemoryLayout::BlockSharded) {
    return std::nullopt;
  }

  auto resultType =
      mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultType) {
    return std::nullopt;
  }
  llvm::ArrayRef<int64_t> outShape = resultType.getShape();
  if (outShape.size() < 2) {
    return std::nullopt;
  }

  // Get input A shape and activation from the op.
  auto [inputA, activation] =
      llvm::TypeSwitch<Operation *, std::pair<Value, std::optional<StringRef>>>(
          op)
          .Case<ttnn::MatmulOp, ttnn::LinearOp>([](auto matmulOp) {
            std::optional<StringRef> act;
            if (auto actAttr = matmulOp.getActivationAttr()) {
              act = actAttr.getValue();
            }
            return std::make_pair(matmulOp.getA(), act);
          })
          .Default([](Operation *) {
            return std::make_pair(nullptr, std::optional<StringRef>{});
          });

  if (!inputA) {
    return std::nullopt;
  }

  auto inputAType = mlir::dyn_cast<RankedTensorType>(inputA.getType());
  if (!inputAType) {
    return std::nullopt;
  }
  llvm::ArrayRef<int64_t> aShape = inputAType.getShape();
  if (aShape.size() < 2) {
    return std::nullopt;
  }

  int64_t M = outShape[outShape.size() - 2];
  int64_t N = outShape[outShape.size() - 1];
  int64_t K = aShape[aShape.size() - 1];
  int64_t Mt = divUp(M, TILE_HEIGHT);
  int64_t Nt = divUp(N, TILE_WIDTH);
  int64_t Kt = divUp(K, TILE_WIDTH);

  MLIRContext *ctx = op->getContext();
  UnaryWithParamAttr fusedActivation = getActivationAttr(ctx, activation);

  if (outputMemLayout == TensorMemoryLayout::BlockSharded) {
    return generateMatmul2DProgramConfig(ctx, Mt, Nt, Kt, outputLayout,
                                         fusedActivation);
  }

  return generateMatmul1DProgramConfig(ctx, Mt, Nt, Kt, outputLayout,
                                       outputMemLayout, fusedActivation);
}

static bool isOpEnabledForAnalysis(Operation *op) {
  // Enable only for specific ops.
  if (llvm::isa<ttnn::Conv2dOp, ttnn::ConvTranspose2dOp, ttnn::MatmulOp,
                ttnn::LinearOp>(op)) {
    return true;
  }

  return false;
}

template <typename ConvOpT>
static void
applyConv2dConfigOverrides(ConvOpT op,
                           const Conv2dConfigOverrideParams &overrides,
                           std::vector<OpConfig> &analysisResult) {
  // Apply conv2d config overrides to all legal (layout) configurations of
  // current op.

  // If conv2d config is not set get default conv2d config.
  Conv2dConfigAttr conv2dConfigAttr = op.getConv2dConfigAttr();
  if (!conv2dConfigAttr) {
    TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                 "Conv2d config not set, using empty default config");
    conv2dConfigAttr = Conv2dConfigAttr::get(op.getContext());
  }
  TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
               "Conv2d config before overrides: {}", conv2dConfigAttr);
  TTMLIR_TRACE(ttmlir::LogComponent::Optimizer, "Overrides: {}", overrides);

  if (overrides.weightsDtype.has_value()) {
    conv2dConfigAttr =
        conv2dConfigAttr.withWeightsDtype(*overrides.weightsDtype);
  }

  if (overrides.activation.has_value()) {
    conv2dConfigAttr = conv2dConfigAttr.withActivation(*overrides.activation);
  }

  if (overrides.deallocateActivation.has_value()) {
    conv2dConfigAttr = conv2dConfigAttr.withDeallocateActivation(
        *overrides.deallocateActivation);
  }

  if (overrides.reallocateHaloOutput.has_value()) {
    conv2dConfigAttr = conv2dConfigAttr.withReallocateHaloOutput(
        *overrides.reallocateHaloOutput);
  }

  if (overrides.actBlockHOverride.has_value()) {
    conv2dConfigAttr =
        conv2dConfigAttr.withActBlockHOverride(*overrides.actBlockHOverride);
  }

  if (overrides.actBlockWDiv.has_value()) {
    conv2dConfigAttr =
        conv2dConfigAttr.withActBlockWDiv(*overrides.actBlockWDiv);
  }

  if (overrides.reshardIfNotOptimal.has_value()) {
    conv2dConfigAttr = conv2dConfigAttr.withReshardIfNotOptimal(
        *overrides.reshardIfNotOptimal);
  }

  if (overrides.overrideShardingConfig.has_value()) {
    conv2dConfigAttr = conv2dConfigAttr.withOverrideShardingConfig(
        *overrides.overrideShardingConfig);
  }

  if (overrides.shardLayout.has_value()) {
    conv2dConfigAttr = conv2dConfigAttr.withShardLayout(*overrides.shardLayout);
  }

  if (overrides.coreGrid.has_value()) {
    conv2dConfigAttr = conv2dConfigAttr.withCoreGrid(*overrides.coreGrid);
  }

  if (overrides.transposeShards.has_value()) {
    conv2dConfigAttr =
        conv2dConfigAttr.withTransposeShards(*overrides.transposeShards);
  }

  if (overrides.outputLayout.has_value()) {
    conv2dConfigAttr =
        conv2dConfigAttr.withOutputLayout(*overrides.outputLayout);
  }

  if (overrides.enableActDoubleBuffer.has_value()) {
    conv2dConfigAttr = conv2dConfigAttr.withEnableActDoubleBuffer(
        *overrides.enableActDoubleBuffer);
  }

  if (overrides.enableWeightsDoubleBuffer.has_value()) {
    conv2dConfigAttr = conv2dConfigAttr.withEnableWeightsDoubleBuffer(
        *overrides.enableWeightsDoubleBuffer);
  }

  TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
               "Conv2d config after overrides: {}", conv2dConfigAttr);

  // Set overriden conv2d config for all OpConfigs.
  for (OpConfig &opConfig : analysisResult) {
    assert(opConfig.isAttrUninitialized() &&
           "OpConfig should not have a config set before applying overrides");
    opConfig.opSpecificAttrs = Conv2dAttrs{conv2dConfigAttr, std::nullopt};
  }
}

bool LegalOpConfigAnalysis::applyOverrides() {
  // For now, easiest way to initialize analysisResult is to copy the legal
  // configs here. Proper solution is that init() method is overridden in child
  // classes.
  analysisResult = analysisInput.legalConfigs;

  if (!isOpEnabledForAnalysis(op)) {
    return true;
  }

  if (!analysisInput.conv2dConfigOverrides) {
    return false;
  }

  return llvm::TypeSwitch<Operation *, bool>(op)
      .Case<ttnn::Conv2dOp, ttnn::ConvTranspose2dOp>([&](auto convOp) {
        Conv2dConfigOverrideParams conv2dConfigOverrides;
        if (!isa<NameLoc>(op->getLoc())) {
          return false;
        }
        StringRef opLocName = mlir::cast<NameLoc>(op->getLoc()).getName();
        auto overrideConv2dIt =
            analysisInput.conv2dConfigOverrides->find(opLocName);
        if (overrideConv2dIt != analysisInput.conv2dConfigOverrides->end()) {
          conv2dConfigOverrides = overrideConv2dIt->getValue();
        }
        applyConv2dConfigOverrides(convOp, conv2dConfigOverrides,
                                   analysisResult);

        // Conv2d config overrides were applied, return true if all config
        // parameters were overridden, therefore no need to search for legal
        // configs.
        return conv2dConfigOverrides.fullConfigOverride();
      })
      .Case<ttnn::MatmulOp, ttnn::LinearOp>([](auto) {
        // Matmul/Linear ops don't use conv2d config overrides.
        return false;
      })
      .Default([](Operation *op) {
        llvm::llvm_unreachable_internal("Unsupported op type");
        return false;
      });
}

void LegalOpConfigAnalysis::fillOpSpecificAttrs() {
  llvm::TypeSwitch<Operation *>(op)
      .Case<ttnn::Conv2dOp, ttnn::ConvTranspose2dOp>([&](auto convOp) {
        assert(!analysisResult.empty() &&
               "Analysis result should not be empty after applying overrides");
        TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                     "Filling op specific attrs for conv2d op {}, starting "
                     "with {} configs",
                     convOp, analysisResult.size());

        // It's possible that no conv2d config was applied in
        // applyConv2dConfigOverrides (e.g. when op does not have loc assigned)
        // so base config is empty.
        Conv2dConfigAttr conv2dConfigAttrBase =
            analysisResult.begin()->isAttrUninitialized()
                ? (convOp.getConv2dConfigAttr()
                       ? convOp.getConv2dConfigAttr()
                       : Conv2dConfigAttr::get(op->getContext()))
                : std::get<Conv2dAttrs>(analysisResult.begin()->opSpecificAttrs)
                      .conv2dConfig.value();

        // If weights dtype is not set, set it to the weight tensor dtype.
        if (!conv2dConfigAttrBase.getWeightsDtype().has_value()) {
          conv2dConfigAttrBase = conv2dConfigAttrBase.withWeightsDtype(
              ttcore::elementTypeToDataType(
                  convOp.getWeight().getType().getElementType()));
        }

        TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                     "Op {} Base conv2d config: {}", convOp.getLoc(),
                     conv2dConfigAttrBase);

        auto filterOut = [](const Conv2dConfigAttr &config) {
          //
          // Combinations that are invalid:
          // 1. reshard_if_not_optimal = true and shard_layout is not set.
          //
          return (config.hasReshardIfNotOptimal() &&
                  config.getReshardIfNotOptimal().getValue() &&
                  !config.hasShardLayout());
        };

        Conv2dConfigGenerator configGenerator(&convOp, conv2dConfigAttrBase,
                                              searchSpace, filterOut);

        std::vector<OpConfig> newLegalConfigs;
        auto addConfigs = [&](const Conv2dConfigAttr &configAttr) {
          for (const OpConfig &existingOpConfig : analysisResult) {
            // Create a new OpConfig pairing the existing layout with the new
            // conv config.
            newLegalConfigs.emplace_back(existingOpConfig.outputLayout,
                                         Conv2dAttrs{configAttr, std::nullopt});
          }
        };

        if (configGenerator.searchDone()) {
          // If search is done before any configs are generated, we will just
          // put base config in all possible layouts. This way we are ensuring
          // dtype and weights_dtype will be set.
          addConfigs(conv2dConfigAttrBase);
        } else {
          // Otherwise, generate all possible configs and add them to the
          // result.
          while (Conv2dConfigAttr configAttr =
                     configGenerator.getNextConfig()) {
            addConfigs(configAttr);
          }
        }

        analysisResult = std::move(newLegalConfigs);
        TTMLIR_TRACE(
            ttmlir::LogComponent::Optimizer,
            "Filled op specific attrs for conv2d op {}, ending with {} configs",
            convOp, analysisResult.size());
      })
      .Case<ttnn::MatmulOp, ttnn::LinearOp>([&](auto matmulOp) {
        // Generate matmul program config for each output layout.
        for (OpConfig &opConfig : analysisResult) {
          auto programConfig =
              generateMatmulProgramConfigForOp(op, opConfig.outputLayout);
          if (programConfig.has_value()) {
            opConfig.opSpecificAttrs = MatmulAttrs{programConfig.value()};
          }
          TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                       "Filled op specific attrs for matmul/linear op {}, "
                       "\n\t op config: {}",
                       ttmlir::opToString(matmulOp), opConfig);
        }
      })
      .Default([](Operation *op) -> void {
        op->emitError("Unsupported op type");
        llvm::llvm_unreachable_internal("Unsupported op type");
      });
}

void LegalOpConfigAnalysis::analysisImplementation() {
  if (!isOpEnabledForAnalysis(op)) {
    return;
  }

  fillOpSpecificAttrs();

  if (analysisResult.empty()) {
    op->emitError("No legal config found for the operation");
    llvm::llvm_unreachable_internal("No legal config found for the operation");
  }
}

} // namespace mlir::tt::ttnn
