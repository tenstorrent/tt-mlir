// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/MatmulProgramConfig.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "llvm/ADT/TypeSwitch.h"

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
  auto [gridX, gridY] = utils::getPhysicalGridDimensions(outputLayout);
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
  auto [gridX, gridY] = utils::getPhysicalGridDimensions(outputLayout);

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

std::optional<mlir::Attribute>
generateMatmulProgramConfig(Operation *op, TTNNLayoutAttr outputLayout) {
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
  UnaryWithParamAttr fusedActivation =
      ttnn::utils::getActivationAttr(ctx, activation);

  if (outputMemLayout == TensorMemoryLayout::BlockSharded) {
    return generateMatmul2DProgramConfig(ctx, Mt, Nt, Kt, outputLayout,
                                         fusedActivation);
  }

  return generateMatmul1DProgramConfig(ctx, Mt, Nt, Kt, outputLayout,
                                       outputMemLayout, fusedActivation);
}

} // namespace mlir::tt::ttnn
