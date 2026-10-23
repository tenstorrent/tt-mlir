// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// TTNNConvTransposeToPixelShuffle
// ===============================
//
// A 2x2 / stride-2 transposed convolution without padding never overlaps its
// output pixels:
//
//   out[2h+dy, 2w+dx, co] = sum_ci x[h, w, ci] * Wt[ci, co, dy, dx]
//
// so it is a 1x1 matmul C_in -> 4*C_out whose output channel (dy*2+dx)*C_out
// + co holds the tap (dy, dx), followed by a pixel shuffle that moves channel
// block (dy, dx) of pixel (h, w) to pixel (2h+dy, 2w+dx). tt-metal runs the
// transposed conv as a dense 2x2 conv over the zero-inserted input (4x the
// useful FLOPs plus a halo over a 2x larger image); the matmul multiplies no
// zero and the shuffle is pure data movement. See the pass description in
// Passes.td.
//
// clang-format off
//   before:  %y = ttnn.conv_transpose2d(%x, %Wt)   kernel 2x2, stride 2   [1,1,H*W,Cin] -> [1,1,4*H*W,Cout]
//   after:   %x1 = ttnn.to_memory_config(%x)        tile, L1 height-sharded, [32*rt rows, Cin] per core
//            %W1 = reshape(permute(%Wt, [0,2,3,1]), [Cin, 4*Cout])   (const-eval)
//            %m  = ttnn.matmul(%x1, %W1)            1-D mcast config, untilize_out: row-major, same shards, 4*Cout wide
//            %r  = ttnn.reshape(%m, [1,H,W,4*Cout])
//            %p  = ttnn.pixel_shuffle(%r, 2, spatial_major, channels_last)   row-major, L1 height-sharded [1,2H,2W,Cout]
//            %f  = ttnn.reshape(%p, [1,1,4*H*W,Cout])
//            %y  = ttnn.to_tensor_spec(%f)          the layout the search gave the transposed conv's result
// clang-format on

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/BFPDtypeParser.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <optional>

namespace mlir::tt::ttnn {

// Discardable unit attribute on a matmul whose compute config the fidelity
// pass must not change.
static constexpr llvm::StringLiteral kKeepComputeConfigAttr =
    "ttnn.keep_compute_config";

// Shuffle outputs larger than this (bf16 bytes) are written straight onto the
// search's height-sharded grid instead of the matmul grid (see the grid
// choice below).
static constexpr int64_t kLargeOutputBytes = 8ll * 1024 * 1024;
#define GEN_PASS_DEF_TTNNCONVTRANSPOSETOPIXELSHUFFLE
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

// Kill switch (comma list in TTMLIR_OPT_DISABLE).
static bool rewriteDisabled() {
  static const bool disabled = [] {
    const char *env = std::getenv("TTMLIR_OPT_DISABLE");
    return env && llvm::StringRef(env).contains("convt_pixel_shuffle");
  }();
  return disabled;
}

static bool probeEnabled() {
  static const bool enabled = std::getenv("TTMLIR_CONVT_PROBE") != nullptr;
  return enabled;
}

static bool allEqual(llvm::ArrayRef<int32_t> values, int32_t v) {
  return llvm::all_of(values, [v](int32_t x) { return x == v; });
}

static int64_t largestDivisorUpTo(int64_t value, int64_t maxDivisor) {
  for (int64_t d = std::min(value, maxDivisor); d >= 1; --d) {
    if (value % d == 0) {
      return d;
    }
  }
  return 1;
}

struct Match {
  ConvTranspose2dOp op;
  int64_t cin = 0, cout = 0, h = 0, w = 0;
};

static std::optional<Match> matchOp(ConvTranspose2dOp op) {
  auto kernel = op.getKernelSize();
  auto stride = op.getStride();
  if (kernel.size() != 2 || kernel[0] != 2 || kernel[1] != 2 ||
      stride.size() != 2 || stride[0] != 2 || stride[1] != 2 ||
      !allEqual(op.getPadding(), 0) || !allEqual(op.getOutputPadding(), 0) ||
      !allEqual(op.getDilation(), 1) || op.getGroups() != 1 ||
      op.getBatchSize() != 1 || op.getBias()) {
    return std::nullopt;
  }
  auto inTy = mlir::dyn_cast<RankedTensorType>(op.getInput().getType());
  auto outTy = mlir::dyn_cast<RankedTensorType>(op.getResult().getType());
  auto wTy = mlir::dyn_cast<RankedTensorType>(op.getWeight().getType());
  if (!inTy || !outTy || !wTy || inTy.getRank() != 4 || outTy.getRank() != 4 ||
      wTy.getRank() != 4) {
    return std::nullopt;
  }
  Match m;
  m.op = op;
  m.cin = op.getInChannels();
  m.cout = op.getOutChannels();
  m.h = op.getInputHeight();
  m.w = op.getInputWidth();
  const int64_t pixels = m.h * m.w;
  // Flattened NHWC activations: [1, 1, H*W, Cin] -> [1, 1, 4*H*W, Cout].
  if (inTy.getShape()[0] != 1 || inTy.getShape()[1] != 1 ||
      inTy.getShape()[2] != pixels || inTy.getShape()[3] != m.cin ||
      outTy.getShape()[2] != 4 * pixels || outTy.getShape()[3] != m.cout ||
      wTy.getShape()[0] != m.cin || wTy.getShape()[1] != m.cout ||
      wTy.getShape()[2] != 2 || wTy.getShape()[3] != 2 || m.cin % 32 != 0 ||
      m.cout % 32 != 0) {
    return std::nullopt;
  }
  auto inLayout = mlir::dyn_cast_or_null<TTNNLayoutAttr>(inTy.getEncoding());
  auto outLayout = mlir::dyn_cast_or_null<TTNNLayoutAttr>(outTy.getEncoding());
  if (!inLayout || !outLayout || !inLayout.isDeviceBufferType()) {
    return std::nullopt;
  }
  return m;
}

class TTNNConvTransposeToPixelShuffle
    : public impl::TTNNConvTransposeToPixelShuffleBase<
          TTNNConvTransposeToPixelShuffle> {
public:
  using impl::TTNNConvTransposeToPixelShuffleBase<
      TTNNConvTransposeToPixelShuffle>::TTNNConvTransposeToPixelShuffleBase;

  void runOnOperation() final {
    if (rewriteDisabled()) {
      return;
    }
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    ttcore::DeviceAttr device = ttcore::lookupDevice(module);
    llvm::ArrayRef<int64_t> workerGrid = device.getWorkerGrid().getShape();
    const int64_t workerCores = workerGrid[0] * workerGrid[1];

    llvm::SmallVector<Match> matches;
    module.walk([&](ConvTranspose2dOp op) {
      if (auto m = matchOp(op)) {
        matches.push_back(*m);
      }
    });

    for (const Match &m : matches) {
      ConvTranspose2dOp op = m.op;
      // The chain takes the transposed conv's place in the schedule. (Placing
      // it right after its input's producer, ahead of the skip-path ops that
      // feed the same residual add, changed the L1 allocation order enough
      // that the halo of the following 3x3 conv clashed with a resident
      // buffer at run time, although the spill simulation accepted it.)
      OpBuilder b(op);
      Location loc = op.getLoc();
      auto inTy = mlir::cast<RankedTensorType>(op.getInput().getType());
      auto inLayout = mlir::cast<TTNNLayoutAttr>(inTy.getEncoding());
      Type scalarTy = inTy.getElementType();
      const int64_t pixels = m.h * m.w, outPixels = 4 * pixels;
      const int64_t kNeck = 4 * m.cout;

      // Height-sharded layout on numCores cores, rowsPerCore rows each, with
      // the canonical core placement.
      // Height-sharded L1 layout on `numCores` cores with exactly `rowsPerCore`
      // rows per shard. The builder derives the shard from rows / cores, which
      // for a row-major tensor is not tile aligned when the rows do not divide
      // evenly (46080 rows on 63 cores give 732-row shards; the tiled layout
      // the search chose has 736). The shard is set explicitly so the
      // row-major and tiled layouts of a stage share the same shards and the
      // tilize between them runs in place.
      auto hsLayout = [&](llvm::ArrayRef<int64_t> shape, int64_t numCores,
                          int64_t rowsPerCore, Layout layout,
                          ttcore::DataType dtype) {
        Type elementType = utils::getElementType(ctx, layout, dtype);
        TTNNLayoutAttr derived =
            TTNNLayoutAttr::Builder(inLayout, shape)
                .setBufferType(BufferType::L1)
                .setMemoryLayout(TensorMemoryLayout::HeightSharded)
                .setElementType(elementType)
                .setGridShape({numCores, 1})
                .setCoreRangeSet(CoreRangeSetAttr())
                .buildWithCanonicalCorePlacement(device);
        llvm::SmallVector<int64_t> shardShape{rowsPerCore, shape.back()};
        MemRefType memref = ttcore::buildMemRef<BufferType, BufferTypeAttr>(
            ctx, shardShape, elementType, BufferType::L1);
        return TTNNLayoutAttr::get(ctx, derived.getLinear(),
                                   derived.getGridShape(), memref,
                                   derived.getMemLayout(),
                                   derived.getTensorMesh(),
                                   derived.getIgnorePhysicalLayout(),
                                   derived.getCoreRangeSet());
      };
      auto typed = [&](llvm::ArrayRef<int64_t> shape, TTNNLayoutAttr layout) {
        return RankedTensorType::get(shape, scalarTy, layout);
      };

      // Grid of the matmul: whole tile rows per core, the fewest tile rows
      // that fit the pixels on the worker grid.
      const int64_t rowsTiles =
          llvm::divideCeil(pixels, workerCores * 32);
      const int64_t rowsPerCore = rowsTiles * 32;
      const int64_t numCores = llvm::divideCeil(pixels, rowsPerCore);
      // Grid of the shuffled output. On the matmul's grid with four times
      // the rows per core every write of the shuffle kernel is core local
      // (input pixel p of a core becomes output pixels of that core's own
      // shard); the search's result layout is then reached by a sharded
      // tilize in place plus an L1 reshard of the tiled tensor. On the
      // search's own height-sharded grid the writes are mostly remote (about
      // 3x slower per byte) but no reshard follows. Measured on a 17.7 MB
      // output: 17 + 7 + 72 us (local + tilize + reshard) against 54 + 7 us;
      // on a 4.4 MB output 7 + 4 + 7 against 18 + 4. The reshard grows
      // faster than the remote-write penalty, so the search's grid wins for
      // large outputs.
      auto outTy = mlir::cast<RankedTensorType>(op.getResult().getType());
      auto outLayout = mlir::cast<TTNNLayoutAttr>(outTy.getEncoding());
      int64_t outRowsPerCore = 4 * rowsPerCore;
      int64_t outCores = numCores;
      const int64_t outBytes = outPixels * m.cout * 2;
      if (outBytes > kLargeOutputBytes &&
          outLayout.hasShardedL1TensorMemoryLayout() &&
          outLayout.getMemLayout().getValue() ==
              TensorMemoryLayout::HeightSharded) {
        llvm::SmallVector<int64_t> shard = outLayout.getScalarShardShape();
        if (shard.size() == 2 && shard[0] % 32 == 0 && shard[1] == m.cout) {
          outRowsPerCore = shard[0];
          outCores = llvm::divideCeil(outPixels, outRowsPerCore);
        }
      }

      ttcore::DataType inDtype = inLayout.getDataType();

      // 1. Input: tile, height-sharded [rowsPerCore, Cin].
      TTNNLayoutAttr x1Layout =
          hsLayout(inTy.getShape(), numCores, rowsPerCore, Layout::Tile, inDtype);
      Value x = op.getInput();
      if (inLayout != x1Layout) {
        x = b.create<ToTensorSpecOp>(loc, typed(inTy.getShape(), x1Layout), x)
                .getResult();
      }

      // 2. Weight [Cin, Cout, 2, 2] -> [Cin, 2, 2, Cout] -> [Cin, 4*Cout]:
      //    row-major bf16 in DRAM for the permute (a device permute of a
      //    4-D row-major tensor), back to the conv weight dtype and a tiled
      //    DRAM tensor; const-eval hoists the chain, it runs once.
      Value w = op.getWeight();
      auto wTy = mlir::cast<RankedTensorType>(w.getType());
      auto wLayout = mlir::cast<TTNNLayoutAttr>(wTy.getEncoding());
      // The tiled matmul weight is stored in block-float: with `mcast_in0 =
      // false` one core reads the whole [Cin, 4*Cout] weight from DRAM and
      // multicasts it, a floor independent of the stage size, and halving the
      // bytes halves that floor (the weight is const-eval'd, so the typecast
      // runs once).
      ttcore::DataType wDtype = ttcore::DataType::BFP_BFloat8;
      auto dramRm = [&](llvm::ArrayRef<int64_t> shape, ttcore::DataType dt) {
        return TTNNLayoutAttr::Builder(wLayout, shape)
            .setBufferType(BufferType::DRAM)
            .setMemoryLayout(TensorMemoryLayout::Interleaved)
            .setElementType(utils::getElementType(ctx, Layout::RowMajor, dt))
            .setGridShape({1, 1})
            .setCoreRangeSet(CoreRangeSetAttr())
            .build();
      };
      Type bf16Scalar = b.getBF16Type();
      Value wRm =
          b.create<ToTensorSpecOp>(
               loc,
               RankedTensorType::get(wTy.getShape(), bf16Scalar,
                                     dramRm(wTy.getShape(),
                                            ttcore::DataType::BFloat16)),
               w)
              .getResult();
      llvm::SmallVector<int64_t> permShape{m.cin, 2, 2, m.cout};
      Value wPerm =
          b.create<PermuteOp>(
               loc,
               RankedTensorType::get(
                   permShape, bf16Scalar,
                   dramRm(permShape, ttcore::DataType::BFloat16)),
               wRm, b.getDenseI64ArrayAttr({0, 2, 3, 1}),
               /*pad_value=*/nullptr)
              .getResult();
      llvm::SmallVector<int64_t> w1Shape{m.cin, kNeck};
      Value w1 = b.create<ReshapeOp>(
                      loc,
                      RankedTensorType::get(
                          w1Shape, bf16Scalar,
                          dramRm(w1Shape, ttcore::DataType::BFloat16)),
                      wPerm,
                      b.getI32ArrayAttr({static_cast<int32_t>(m.cin),
                                         static_cast<int32_t>(kNeck)}))
                     .getResult();
      // to device, tiled, the conv weight dtype
      TTNNLayoutAttr w1Layout =
          TTNNLayoutAttr::Builder(wLayout, w1Shape)
              .setBufferType(BufferType::DRAM)
              .setMemoryLayout(TensorMemoryLayout::Interleaved)
              .setElementType(utils::getElementType(ctx, Layout::Tile, wDtype))
              .setGridShape({1, 1})
              .setCoreRangeSet(CoreRangeSetAttr())
              .build();
      Value w1Dev = b.create<ToTensorSpecOp>(
                         loc, RankedTensorType::get(w1Shape, scalarTy, w1Layout),
                         w1)
                        .getResult();

      // 3. Matmul with row-major output on the same shards.
      llvm::SmallVector<int64_t> mmShape{1, 1, pixels, kNeck};
      TTNNLayoutAttr mmLayout = hsLayout(mmShape, numCores, rowsPerCore, Layout::RowMajor,
                                         ttcore::DataType::BFloat16);
      RankedTensorType mmTy = RankedTensorType::get(mmShape, bf16Scalar, mmLayout);
      const int64_t perCoreN = kNeck / 32, kt = m.cin / 32;
      auto [gridX, gridY] = utils::getPhysicalGridDimensions(mmLayout);
      auto programConfig = MatmulMultiCoreReuseMultiCast1DProgramConfigAttr::get(
          ctx, CoreCoordAttr::get(ctx, gridX, gridY),
          /*in0_block_w=*/static_cast<uint64_t>(kt),
          /*out_subblock_h=*/1,
          /*out_subblock_w=*/
          static_cast<uint64_t>(largestDivisorUpTo(perCoreN, 8)),
          /*out_block_h=*/static_cast<uint64_t>(rowsTiles),
          /*out_block_w=*/static_cast<uint64_t>(perCoreN),
          /*per_core_m=*/static_cast<uint64_t>(rowsTiles),
          /*per_core_n=*/static_cast<uint64_t>(perCoreN),
          /*fuse_batch=*/true, /*fused_activation=*/UnaryWithParamAttr{},
          /*mcast_in0=*/false, /*gather_in0=*/false,
          CoreRangeSetAttr::get(ctx, {}),
          /*num_global_cb_receivers=*/0, /*untilize_out=*/true);
      // K is one block (in0_block_w = Cin / 32), so the accumulation stays
      // in DEST for the whole dot product; bf16 DEST holds 8 tiles, which
      // the 8-wide sub-blocks need (fp32 DEST holds 4 and would also halve
      // the sub-block). The later fidelity pass leaves this config alone.
      DeviceComputeKernelConfigAttr mmCompute =
          (op.getComputeConfigAttr() ? op.getComputeConfigAttr()
                                     : DeviceComputeKernelConfigAttr::get(ctx))
              .withMathFidelity(MathFidelity::HiFi2)
              .withFp32DestAccEn(false);
      auto mmOp = b.create<MatmulOp>(loc, mmTy, x, w1Dev, /*transpose_a=*/false,
                                     /*transpose_b=*/false, programConfig,
                                     /*activation=*/StringAttr{}, mmCompute);
      mmOp->setAttr(kKeepComputeConfigAttr, b.getUnitAttr());
      Value mm = mmOp.getResult();

      // 4. View [1, H, W, 4*Cout] on the same shards.
      llvm::SmallVector<int64_t> nhwcShape{1, m.h, m.w, kNeck};
      TTNNLayoutAttr nhwcLayout = hsLayout(nhwcShape, numCores, rowsPerCore, Layout::RowMajor,
                                           ttcore::DataType::BFloat16);
      Value nhwc = b.create<ReshapeOp>(
                        loc, RankedTensorType::get(nhwcShape, bf16Scalar, nhwcLayout),
                        mm,
                        b.getI32ArrayAttr({1, static_cast<int32_t>(m.h),
                                           static_cast<int32_t>(m.w),
                                           static_cast<int32_t>(kNeck)}))
                       .getResult();

      // 5. Pixel shuffle: [1, 2H, 2W, Cout] row-major height-sharded.
      llvm::SmallVector<int64_t> psShape{1, 2 * m.h, 2 * m.w, m.cout};
      TTNNLayoutAttr psLayout = hsLayout(psShape, outCores, outRowsPerCore, Layout::RowMajor,
                                         ttcore::DataType::BFloat16);
      Value ps = b.create<PixelShuffleOp>(
                      loc, RankedTensorType::get(psShape, bf16Scalar, psLayout),
                      nhwc, b.getUI32IntegerAttr(2),
                      PixelUnshuffleChannelOrderAttr::get(
                          ctx, PixelUnshuffleChannelOrder::SpatialMajor),
                      MemoryConfigAttr::get(psLayout), b.getBoolAttr(true),
                      b.getUI32IntegerAttr(static_cast<uint32_t>(m.cout)))
                     .getResult();

      // 6. Flatten view [1, 1, 4*H*W, Cout] on the same shards.
      llvm::SmallVector<int64_t> flatShape{1, 1, outPixels, m.cout};
      TTNNLayoutAttr flatLayout = hsLayout(flatShape, outCores, outRowsPerCore, Layout::RowMajor,
                                           ttcore::DataType::BFloat16);
      Value flat = b.create<ReshapeOp>(
                        loc, RankedTensorType::get(flatShape, bf16Scalar, flatLayout),
                        ps,
                        b.getI32ArrayAttr({1, 1, static_cast<int32_t>(outPixels),
                                           static_cast<int32_t>(m.cout)}))
                       .getResult();

      // 7. Back to the layout the search gave the transposed conv's result.
      Value result = flat;
      if (flatLayout != outLayout) {
        result = b.create<ToTensorSpecOp>(loc, outTy, flat).getResult();
      }

      if (probeEnabled()) {
        llvm::errs() << "[CONVT_PROBE] " << m.h << "x" << m.w << " " << m.cin
                     << "->" << m.cout << ": matmul on " << numCores
                     << " cores x " << rowsPerCore << " rows, shuffle on "
                     << outCores << " cores x " << outRowsPerCore
                     << " rows, result layout " << outLayout << "\n";
      }

      op.getResult().replaceAllUsesWith(result);
      op.erase();
    }
  }
};

} // namespace
} // namespace mlir::tt::ttnn
