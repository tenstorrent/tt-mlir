// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/MatmulRules.h"
#include "ttmlir/Dialect/TTNN/Analysis/MatmulProgramConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/LayoutFilterUtils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Interfaces/TTNNTensorSpecInterface.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::tt::ttnn {

// Prefill Mt threshold: shapes with Mt > this are "prefill-like" and should
// avoid width-sharded output (too many tiles in M to mcast efficiently).
static constexpr int64_t kPrefillMtThreshold = 4;

static inline int64_t divUp(int64_t a, int64_t b) { return (a + b - 1) / b; }

// Deduplicate legalConfigs by (bufferType, memLayout) pair.
// Returns one representative TTNNLayoutAttr per unique pair, with
// ignorePhysicalLayout=true so the backend decides the physical grid.
static std::vector<TTNNLayoutAttr>
dedupByMemoryLayout(const std::vector<OpConfig> &configs) {
  llvm::DenseSet<std::pair<unsigned, unsigned>> seen;
  std::vector<TTNNLayoutAttr> result;

  for (const auto &cfg : configs) {
    if (!cfg.outputLayout) {
      continue;
    }
    auto memLayout = cfg.outputLayout.getMemLayout();
    if (!memLayout) {
      continue;
    }
    auto key = std::make_pair(static_cast<unsigned>(cfg.outputLayout.getBufferType()),
                              static_cast<unsigned>(memLayout.getValue()));
    if (seen.insert(key).second) {
      TTNNLayoutAttr layout = cfg.outputLayout;
      result.push_back(layout.withIgnorePhysicalLayout(true));
    }
  }

  return result;
}

// Extract matmul shape info (Mt, Nt, Kt, fuseBatch, activation) from op.
// Returns false if shapes cannot be extracted.
struct MatmulShapeInfo {
  int64_t Mt;
  int64_t Nt;
  int64_t Kt;
  bool fuseBatch;
  std::optional<StringRef> activation;
};

static std::optional<MatmulShapeInfo> extractMatmulShapes(Operation *op) {
  auto resultType =
      mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultType) {
    return std::nullopt;
  }
  llvm::ArrayRef<int64_t> outShape = resultType.getShape();
  if (outShape.size() < 2) {
    return std::nullopt;
  }

  // Get input A, input B, and activation from the concrete op type.
  auto [inputA, inputB, activation] =
      llvm::TypeSwitch<Operation *,
                       std::tuple<Value, Value, std::optional<StringRef>>>(op)
          .Case<ttnn::MatmulOp, ttnn::LinearOp>([](auto matmulOp) {
            std::optional<StringRef> act;
            if (auto actAttr = matmulOp.getActivationAttr()) {
              act = actAttr.getValue();
            }
            return std::make_tuple(matmulOp.getA(), matmulOp.getB(), act);
          })
          .Default([](Operation *) {
            return std::make_tuple(nullptr, nullptr,
                                   std::optional<StringRef>{});
          });

  if (!inputA || !inputB) {
    return std::nullopt;
  }

  auto inputAType = mlir::dyn_cast<RankedTensorType>(inputA.getType());
  auto inputBType = mlir::dyn_cast<RankedTensorType>(inputB.getType());
  if (!inputAType || !inputBType) {
    return std::nullopt;
  }
  llvm::ArrayRef<int64_t> aShape = inputAType.getShape();
  llvm::ArrayRef<int64_t> bShape = inputBType.getShape();
  if (aShape.size() < 2 || bShape.size() < 2) {
    return std::nullopt;
  }

  // fuse_batch can only be true when all batch dims of B are 1.
  bool fuseBatch = true;
  for (size_t i = 0; i < bShape.size() - 2; ++i) {
    if (bShape[i] != 1) {
      fuseBatch = false;
      break;
    }
  }

  int64_t M = outShape[outShape.size() - 2];
  int64_t N = outShape[outShape.size() - 1];
  int64_t K = aShape[aShape.size() - 1];
  int64_t Mt = divUp(M, TILE_HEIGHT);
  int64_t Nt = divUp(N, TILE_WIDTH);
  int64_t Kt = divUp(K, TILE_WIDTH);

  return MatmulShapeInfo{Mt, Nt, Kt, fuseBatch, activation};
}

LayoutFilterFn MatmulRuleBook::getInputLayoutFilter() const {
  // Reject width-sharded inputs for matmul/linear: accuracy issues observed
  // with width-sharded activation tensors feeding into matmul.
  return layout_filter_utils::rejectWidthSharded;
}

bool MatmulRuleBook::isValidOutputHintForInputs(
    const OpConfig &hint,
    llvm::ArrayRef<TTNNLayoutAttr> inputLayouts) const {
  if (!std::holds_alternative<MatmulAttrs>(hint.opSpecificAttrs)) {
    return true;
  }
  const auto &matmulAttrs = std::get<MatmulAttrs>(hint.opSpecificAttrs);
  if (!matmulAttrs.matmulProgramConfig.has_value()) {
    return true;
  }

  auto programConfig = matmulAttrs.matmulProgramConfig.value();

  if (inputLayouts.size() < 2) {
    return true;
  }

  // Helper to safely extract TensorMemoryLayout from a layout attribute.
  auto getMemLayoutVal =
      [](TTNNLayoutAttr layout) -> std::optional<TensorMemoryLayout> {
    if (!layout) {
      return std::nullopt;
    }
    auto ml = layout.getMemLayout();
    if (!ml) {
      return std::nullopt;
    }
    return ml.getValue();
  };

  auto in0Mem = getMemLayoutVal(inputLayouts[0]);

  return llvm::TypeSwitch<mlir::Attribute, bool>(programConfig)
      .Case<MatmulMultiCoreReuseMultiCast1DProgramConfigAttr>(
          [&](auto config) {
            // mcast_in0=true broadcasts input A: incompatible with
            // height-sharded input A. mcast_in0=false broadcasts input B:
            // incompatible with width-sharded input A.
            if (config.getMcastIn0()) {
              if (in0Mem == TensorMemoryLayout::HeightSharded) {
                return false;
              }
            } else {
              if (in0Mem == TensorMemoryLayout::WidthSharded) {
                return false;
              }
            }
            return true;
          })
      .Case<MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr>(
          [&](auto) {
            // DRAM-sharded config requires input B to live in DRAM.
            if (inputLayouts[1] &&
                inputLayouts[1].getBufferType() != BufferType::DRAM) {
              return false;
            }
            return true;
          })
      .Case<MatmulMultiCoreReuseMultiCastProgramConfigAttr>(
          [&](auto) {
            // 2D block-sharded config: width-sharded input A is incompatible
            // with the row-multicast pattern.
            if (in0Mem == TensorMemoryLayout::WidthSharded) {
              return false;
            }
            return true;
          })
      .Default([](mlir::Attribute) { return true; });
}

OutputHints MatmulRuleBook::getOutputHints(
    Operation *op, const std::vector<OpConfig> &legalConfigs) const {

  // Extract shape information. On failure, return a NULL hint fallback.
  auto shapeInfo = extractMatmulShapes(op);
  if (!shapeInfo) {
    return OutputHints{{OpConfig(TTNNLayoutAttr())}, {}};
  }

  MLIRContext *ctx = op->getContext();
  UnaryWithParamAttr fusedActivation =
      ttnn::utils::getActivationAttr(ctx, shapeInfo->activation);

  // Get compute kernel config from the op for max subblock size.
  DeviceComputeKernelConfigAttr computeConfig = nullptr;
  if (auto computeConfigOp =
          dyn_cast<TTNNComputeKernelConfigOpInterface>(op)) {
    computeConfig = computeConfigOp.getComputeConfigAttr();
  }
  int64_t maxSubblockSize = getMaxSubblockSize(computeConfig);

  // Deduplicate by (bufferType, memLayout).
  auto uniqueLayouts = dedupByMemoryLayout(legalConfigs);

  std::vector<OpConfig> hints;
  for (const auto &layout : uniqueLayouts) {
    auto bufferType = layout.getBufferType();
    auto memLayout = layout.getMemLayout();
    TensorMemoryLayout memLayoutVal = memLayout.getValue();

    // Skip L1 interleaved: worst of both worlds for matmul.
    if (bufferType == BufferType::L1 &&
        memLayoutVal == TensorMemoryLayout::Interleaved) {
      continue;
    }

    // Skip width-sharded for prefill-like shapes (large Mt).
    if (memLayoutVal == TensorMemoryLayout::WidthSharded &&
        shapeInfo->Mt > kPrefillMtThreshold) {
      continue;
    }

    // Skip height-sharded for decode-like shapes (Mt <= 1).
    if (memLayoutVal == TensorMemoryLayout::HeightSharded &&
        shapeInfo->Mt <= 1) {
      continue;
    }

    if (memLayoutVal == TensorMemoryLayout::WidthSharded) {
      auto config = generateMatmul1DWidthConfig(
          ctx, shapeInfo->Mt, shapeInfo->Nt, shapeInfo->Kt, layout,
          fusedActivation, maxSubblockSize, shapeInfo->fuseBatch);
      MatmulAttrs attrs{config, computeConfig};
      hints.push_back(OpConfig(layout, std::move(attrs)));

      // For decode shapes (Mt==1), also try DRAM-sharded config.
      if (shapeInfo->Mt == 1) {
        auto dramConfig = generateMatmulDRAMShardedConfig(
            ctx, shapeInfo->Mt, shapeInfo->Nt, shapeInfo->Kt, layout,
            fusedActivation, maxSubblockSize, shapeInfo->fuseBatch);
        if (dramConfig) {
          MatmulAttrs dramAttrs{dramConfig, computeConfig};
          hints.push_back(OpConfig(layout, std::move(dramAttrs)));
        }
      }
    } else if (memLayoutVal == TensorMemoryLayout::HeightSharded) {
      auto config = generateMatmul1DHeightConfig(
          ctx, shapeInfo->Mt, shapeInfo->Nt, shapeInfo->Kt, layout,
          fusedActivation, maxSubblockSize, shapeInfo->fuseBatch);
      MatmulAttrs attrs{config, computeConfig};
      hints.push_back(OpConfig(layout, std::move(attrs)));
    } else if (memLayoutVal == TensorMemoryLayout::BlockSharded) {
      auto config = generateMatmul2DConfig(
          ctx, shapeInfo->Mt, shapeInfo->Nt, shapeInfo->Kt, layout,
          fusedActivation, maxSubblockSize, shapeInfo->fuseBatch);
      MatmulAttrs attrs{config, computeConfig};
      hints.push_back(OpConfig(layout, std::move(attrs)));
    } else {
      // DRAM interleaved: push with no op-specific attrs.
      hints.push_back(OpConfig(layout));
    }
  }

  // Always include a NULL hint fallback as the last entry.
  hints.push_back(OpConfig(TTNNLayoutAttr()));

  return OutputHints{hints, {}};
}

void MatmulRuleBook::applyOpSpecificAttrs(
    Operation *op, const BeamCandidate &candidate) const {
  auto matmulOp = dyn_cast<MatmulOp>(op);
  auto linearOp = dyn_cast<LinearOp>(op);
  if (!matmulOp && !linearOp) {
    return;
  }

  if (!std::holds_alternative<MatmulAttrs>(
          candidate.configHint.opSpecificAttrs)) {
    return;
  }
  MatmulAttrs matmulAttrs =
      std::get<MatmulAttrs>(candidate.configHint.opSpecificAttrs);
  if (!matmulAttrs.matmulProgramConfig.has_value()) {
    return;
  }
  auto programConfig = matmulAttrs.matmulProgramConfig.value();

  auto setConfigAndFixup = [&](auto concreteOp) {
    concreteOp.setMatmulProgramConfigAttr(programConfig);
    // Workaround for tt-metal issue #35060: if the program config carries a
    // fused activation, remove the op-level activation attr to prevent
    // double application.
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
      concreteOp.removeActivationAttr();
    }
  };

  if (matmulOp) {
    setConfigAndFixup(matmulOp);
  } else {
    setConfigAndFixup(linearOp);
  }
}

} // namespace mlir::tt::ttnn
