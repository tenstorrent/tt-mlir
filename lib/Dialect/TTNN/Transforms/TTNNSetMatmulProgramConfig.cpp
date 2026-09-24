// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Support/Logger.h"

#include "llvm/ADT/SmallVector.h"

#include <array>
#include <optional>
#include <utility>

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNSETMATMULPROGRAMCONFIG
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

// Mirrors tt-metal's matmul program config selection
// (matmul_program_config.cpp).
constexpr int64_t kTileDim = 32;
constexpr int64_t kNarrowShapeRatioThreshold = 8;
constexpr int64_t kMcastInputBufferingDepth = 2;
constexpr int64_t kFp32TileBytes = 4096;
constexpr int64_t kBf16TileBytes = 2048;

// tt-metal's SUBBLOCK_HW_CHOICES, as (out_subblock_h, out_subblock_w).
constexpr std::array<std::pair<int64_t, int64_t>, 20> kSubblockChoices = {{
    {4, 2}, {2, 4}, {8, 1}, {1, 8}, {7, 1}, {1, 7}, {3, 2},
    {2, 3}, {6, 1}, {1, 6}, {5, 1}, {1, 5}, {2, 2}, {4, 1},
    {1, 4}, {3, 1}, {1, 3}, {2, 1}, {1, 2}, {1, 1},
}};

enum class MatmulKernel { Mcast2D, Mcast1DIn0, Mcast1DIn1 };

struct Blocking {
  int64_t in0BlockW = 0;
  int64_t outBlockH = 0;
  int64_t outBlockW = 0;

  bool valid() const { return in0BlockW > 0; }
};

// Everything the heuristics need about one matmul, in tiles.
struct MatmulProblem {
  MatmulKernel kernel;
  int64_t m;
  int64_t n;
  int64_t kt;
  int64_t perCoreM;
  int64_t perCoreN;
  int64_t gridX;
  int64_t gridY;
  bool fuseBatch;
  int64_t aTileBytes;
  int64_t bTileBytes;
  int64_t outTileBytes;
  int64_t intermTileBytes;
  int64_t biasTileBytes;
  int64_t maxSubblockArea;
};

int64_t divUp(int64_t a, int64_t b) { return (a + b - 1) / b; }

llvm::SmallVector<int64_t> divisorsUpTo(int64_t value, int64_t upper) {
  llvm::SmallVector<int64_t> divisors;
  for (int64_t d = 1; d <= std::min(value, upper); ++d) {
    if (value % d == 0) {
      divisors.push_back(d);
    }
  }
  return divisors;
}

int64_t batchVolume(llvm::ArrayRef<int64_t> shape) {
  int64_t volume = 1;
  for (int64_t dim : shape.drop_back(2)) {
    volume *= dim;
  }
  return volume;
}

// Returns the layout of a DRAM-interleaved, 32x32-tiled device tensor, or
// null for anything else.
TTNNLayoutAttr getDramInterleavedTiledLayout(Value value) {
  auto type = mlir::dyn_cast<RankedTensorType>(value.getType());
  if (!type) {
    return nullptr;
  }
  auto layout = mlir::dyn_cast_if_present<TTNNLayoutAttr>(type.getEncoding());
  if (!layout || !layout.isTiled() || !layout.getMemLayout() ||
      !layout.hasInterleavedDRAMTensorMemoryLayout()) {
    return nullptr;
  }
  auto tile = mlir::cast<ttcore::TileType>(layout.getElementType());
  if (tile.getHeight() != kTileDim || tile.getWidth() != kTileDim) {
    return nullptr;
  }
  return layout;
}

int64_t biasTileBytes(Value bias) {
  if (!bias) {
    return 0;
  }
  auto type = mlir::cast<RankedTensorType>(bias.getType());
  if (auto layout =
          mlir::dyn_cast_if_present<TTNNLayoutAttr>(type.getEncoding())) {
    if (layout.isTiled()) {
      return layout.getElementSizeBytes();
    }
    return ttcore::TileType::get(layout.getScalarElementType()).getSizeBytes();
  }
  return kBf16TileBytes;
}

// is_narrow_shape plus the wide/tall split of
// create_simple_matmul_program_config for all-DRAM-interleaved operands.
MatmulKernel route(int64_t mt, int64_t nt) {
  const int64_t height = mt * kTileDim;
  const int64_t width = nt * kTileDim;
  const int64_t ratio = std::max(height, width) / std::min(height, width);
  const bool narrow = ratio > kNarrowShapeRatioThreshold ||
                      height <= kTileDim || width <= kTileDim;
  if (!narrow) {
    return MatmulKernel::Mcast2D;
  }
  return width > height ? MatmulKernel::Mcast1DIn0 : MatmulKernel::Mcast1DIn1;
}

// Circular buffer bytes of the mcast factories for one block choice.
int64_t cbBytes(const MatmulProblem &p, int64_t outBlockH, int64_t outBlockW,
                int64_t in0BlockW) {
  const int64_t depth =
      p.kt / in0BlockW > 1 ? kMcastInputBufferingDepth : int64_t{1};
  const int64_t area = outBlockH * outBlockW;
  return depth * outBlockH * in0BlockW * p.aTileBytes +
         depth * outBlockW * in0BlockW * p.bTileBytes +
         area * (p.outTileBytes + p.intermTileBytes) +
         outBlockW * p.biasTileBytes;
}

// in0_block_w candidates: divisors of Kt, keeping at least two K-blocks so the
// input CBs stay double buffered.
llvm::SmallVector<int64_t> in0BlockWCandidates(const MatmulProblem &p,
                                               int64_t maxIn0BlockW) {
  llvm::SmallVector<int64_t> candidates;
  for (int64_t bw : divisorsUpTo(p.kt, maxIn0BlockW)) {
    if (p.kt == 1 || p.kt / bw >= 2) {
      candidates.push_back(bw);
    }
  }
  return candidates;
}

// 2D mcast: maximise in0_block_w * out_block_h * out_block_w, ties towards the
// larger in0_block_w.
Blocking pick2D(const MatmulProblem &p, int64_t maxIn0BlockW,
                int64_t l1Budget) {
  Blocking best;
  int64_t bestVolume = 0;
  for (int64_t bw : in0BlockWCandidates(p, maxIn0BlockW)) {
    for (int64_t h : divisorsUpTo(p.perCoreM, p.perCoreM)) {
      for (int64_t w : divisorsUpTo(p.perCoreN, p.perCoreN)) {
        if (cbBytes(p, h, w, bw) > l1Budget) {
          continue;
        }
        const int64_t volume = bw * h * w;
        if (volume > bestVolume ||
            (volume == bestVolume && bw > best.in0BlockW)) {
          best = Blocking{bw, h, w};
          bestVolume = volume;
        }
      }
    }
  }
  return best;
}

// 1D mcast: keep the output block whole along the multicast direction's
// partner dimension (out_block_w = per_core_N for mcast_in0, out_block_h =
// per_core_M for mcast_in1), allow the other one to be halved once when that
// dimension is at least splitMinDim, and take the largest in0_block_w that
// fits; ties towards the larger free block dimension.
Blocking pick1D(const MatmulProblem &p, int64_t maxIn0BlockW, int64_t l1Budget,
                int64_t splitMinDim) {
  const bool mcastIn0 = p.kernel == MatmulKernel::Mcast1DIn0;
  const int64_t full = mcastIn0 ? p.perCoreM : p.perCoreN;
  const int64_t dim = mcastIn0 ? p.m : p.n;
  llvm::SmallVector<int64_t> freeChoices = {full};
  if (dim >= splitMinDim && full % 2 == 0) {
    freeChoices.push_back(full / 2);
  }

  Blocking best;
  int64_t bestFree = 0;
  for (int64_t bw : in0BlockWCandidates(p, maxIn0BlockW)) {
    for (int64_t freeDim : freeChoices) {
      const int64_t h = mcastIn0 ? freeDim : p.perCoreM;
      const int64_t w = mcastIn0 ? p.perCoreN : freeDim;
      if (cbBytes(p, h, w, bw) > l1Budget) {
        continue;
      }
      if (bw > best.in0BlockW || (bw == best.in0BlockW && freeDim > bestFree)) {
        best = Blocking{bw, h, w};
        bestFree = freeDim;
      }
    }
  }
  return best;
}

// get_matmul_subblock_params without the sharded-output constraints.
std::pair<int64_t, int64_t> pickSubblock(int64_t outBlockH, int64_t outBlockW,
                                         int64_t maxArea) {
  for (auto [h, w] : kSubblockChoices) {
    if (h * w <= maxArea && outBlockH % h == 0 && outBlockW % w == 0) {
      return {h, w};
    }
  }
  return {1, 1};
}

template <typename MatmulOpTy>
std::optional<MatmulProblem> analyze(MatmulOpTy op, Value bias) {
  if (op.getMatmulProgramConfigAttr() || op.getTransposeA()) {
    return std::nullopt;
  }

  // Without a compute_config the runtime passes none, and tt-metal then picks
  // a lower default math fidelity whenever a program config is present.
  DeviceComputeKernelConfigAttr computeConfig = op.getComputeConfigAttr();
  if (!computeConfig) {
    return std::nullopt;
  }

  TTNNLayoutAttr aLayout = getDramInterleavedTiledLayout(op.getA());
  TTNNLayoutAttr bLayout = getDramInterleavedTiledLayout(op.getB());
  TTNNLayoutAttr outLayout = getDramInterleavedTiledLayout(op.getResult());
  if (!aLayout || !bLayout || !outLayout) {
    return std::nullopt;
  }

  llvm::ArrayRef<int64_t> aShape = op.getA().getType().getShape();
  llvm::ArrayRef<int64_t> bShape = op.getB().getType().getShape();
  if (aShape.size() < 2 || bShape.size() < 2) {
    return std::nullopt;
  }

  const int64_t m = aShape[aShape.size() - 2];
  const int64_t k = aShape.back();
  const int64_t n =
      op.getTransposeB() ? bShape[bShape.size() - 2] : bShape.back();
  const int64_t batchA = batchVolume(aShape);
  const int64_t batchB = batchVolume(bShape);
  // A batch-broadcast A takes tt-metal's dedicated in0-reuse path.
  if (batchB > 1 && (batchA != batchB || aShape.size() != bShape.size())) {
    return std::nullopt;
  }

  ttcore::DeviceAttr device = ttcore::lookupDevice(op);
  if (!device) {
    return std::nullopt;
  }
  llvm::ArrayRef<int64_t> workerGrid = device.getWorkerGrid().getShape();
  if (workerGrid.size() != 2) {
    return std::nullopt;
  }

  const bool fp32DestAccEn = computeConfig.getFp32DestAccEn() &&
                             computeConfig.getFp32DestAccEn().getValue();
  const bool dstFullSyncEn = computeConfig.getDstFullSyncEn() &&
                             computeConfig.getDstFullSyncEn().getValue();
  const int64_t outTileBytes = outLayout.getElementSizeBytes();

  MatmulProblem p;
  p.m = m;
  p.n = n;
  p.kt = divUp(k, kTileDim);
  p.gridX = workerGrid[1];
  p.gridY = workerGrid[0];
  p.aTileBytes = aLayout.getElementSizeBytes();
  p.bTileBytes = bLayout.getElementSizeBytes();
  p.outTileBytes = outTileBytes;
  p.intermTileBytes =
      fp32DestAccEn ? kFp32TileBytes : std::max(kBf16TileBytes, outTileBytes);
  p.biasTileBytes = biasTileBytes(bias);
  p.maxSubblockArea = (dstFullSyncEn ? 16 : 8) / (fp32DestAccEn ? 2 : 1);

  const int64_t mt = divUp(m, kTileDim);
  const int64_t nt = divUp(n, kTileDim);
  p.kernel = route(mt, nt);
  const int64_t numCores = p.gridX * p.gridY;
  if (p.kernel == MatmulKernel::Mcast2D) {
    if (p.gridX < 2 || p.gridY < 2) {
      return std::nullopt;
    }
    p.fuseBatch = batchB == 1;
    const int64_t mtTotal = p.fuseBatch ? batchA * mt : mt;
    p.perCoreM = divUp(mtTotal, p.gridY);
    p.perCoreN = divUp(nt, p.gridX);
  } else if (p.kernel == MatmulKernel::Mcast1DIn0) {
    p.fuseBatch = false;
    p.perCoreM = mt;
    p.perCoreN = divUp(nt, numCores);
  } else {
    p.fuseBatch = false;
    p.perCoreM = divUp(mt, numCores);
    p.perCoreN = nt;
  }
  return p;
}

Attribute buildConfig(MLIRContext *context, const MatmulProblem &p,
                      const Blocking &b) {
  auto [subblockH, subblockW] =
      pickSubblock(b.outBlockH, b.outBlockW, p.maxSubblockArea);
  auto grid = CoreCoordAttr::get(context, p.gridX, p.gridY);
  if (p.kernel == MatmulKernel::Mcast2D) {
    return MatmulMultiCoreReuseMultiCastProgramConfigAttr::get(
        context, grid, b.in0BlockW, subblockH, subblockW, b.outBlockH,
        b.outBlockW, p.perCoreM, p.perCoreN, /*transpose_mcast=*/false,
        /*fused_activation=*/nullptr, p.fuseBatch);
  }
  return MatmulMultiCoreReuseMultiCast1DProgramConfigAttr::get(
      context, grid, b.in0BlockW, subblockH, subblockW, b.outBlockH,
      b.outBlockW, p.perCoreM, p.perCoreN, p.fuseBatch,
      /*fused_activation=*/nullptr,
      /*mcast_in0=*/p.kernel == MatmulKernel::Mcast1DIn0,
      /*gather_in0=*/false, CoreRangeSetAttr::get(context, {}),
      /*num_global_cb_receivers=*/0, /*untilize_out=*/false);
}

} // namespace

class TTNNSetMatmulProgramConfig
    : public impl::TTNNSetMatmulProgramConfigBase<TTNNSetMatmulProgramConfig> {
public:
  using impl::TTNNSetMatmulProgramConfigBase<
      TTNNSetMatmulProgramConfig>::TTNNSetMatmulProgramConfigBase;

  void runOnOperation() final {
    getOperation()->walk([&](Operation *op) {
      if (auto matmulOp = dyn_cast<MatmulOp>(op)) {
        setConfig(matmulOp, /*bias=*/nullptr);
      } else if (auto linearOp = dyn_cast<LinearOp>(op)) {
        setConfig(linearOp, linearOp.getBias());
      }
    });
  }

private:
  template <typename MatmulOpTy>
  void setConfig(MatmulOpTy op, Value bias) {
    std::optional<MatmulProblem> problem = analyze(op, bias);
    if (!problem) {
      return;
    }

    const int64_t l1Budget = std::min<int64_t>(
        l1BudgetBytes, ttcore::getOpChipDescAttr(op).getUsableL1Size());
    Blocking blocking =
        problem->kernel == MatmulKernel::Mcast2D
            ? pick2D(*problem, maxIn0BlockW, l1Budget)
            : pick1D(*problem, maxIn0BlockW, l1Budget, splitMinDim);
    if (!blocking.valid()) {
      return;
    }

    Attribute config = buildConfig(op.getContext(), *problem, blocking);
    TTMLIR_DEBUG(ttmlir::LogComponent::General,
                 "TTNNSetMatmulProgramConfig - {0}: {1}",
                 op->getName().getStringRef(), config);
    op.setMatmulProgramConfigAttr(config);
  }
};

} // namespace mlir::tt::ttnn
