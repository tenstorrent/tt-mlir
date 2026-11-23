// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// TTNNPixelUnshuffleChannelsLast
//
// Second half of the space-to-depth stem (the first half is
// TTIRStridedConvSpaceToDepthOpt, which emits a channels_last pixel_unshuffle
// feeding the packed stride-1 conv).
//
// After layout assignment the graph is
//
//   %pu = ttnn.pixel_unshuffle(%x) {channels_last, padded_channels = Cp}
//                                         [N, H/r, W/r, Cp]  ROW_MAJOR
//   ... layout ops (to_layout / to_memory_config) ...
//   %flat = ttnn.reshape(...)             [1, 1, N*H/r*W/r, Cp]
//   ... layout ops ...
//   %y = ttnn.conv2d(%flat, ...)          height-sharded result
//
// ttnn::pixel_unshuffle(channels_last) writes each core's shard of the NHWC
// tensor directly, but only into a HEIGHT_SHARDED L1 memory config, and the
// layout passes know nothing about that: they tilize the result and re-shard
// it for the conv. Once the optimizer has fixed the conv's sharding, this pass
//
//   1. gives %pu (and the reshape, a view) the conv's height-shard grid in
//      ROW_MAJOR: shard [N*H/r*W/r / ncores, Cp] on the conv's cores, and
//      feeds the conv that reshape directly, erasing the layout ops between;
//   2. sets the conv2d blocking for that row-major input on the stem conv:
//      act_block_h = one output row when that divides the per-core rows,
//      double-buffered activations, no halo re-allocation;
//   3. tunes a following stride-1 conv that consumes the stem conv's
//      height-sharded result unchanged (same rows per core). With a TILE input
//      and a kernel taller than one row it gets full_inner_dim: the whole
//      kh x kw window is one inner-dim block, so the fp32 partial sums are not
//      spilled to L1 and reloaded per kernel row, with the largest act_block_h
//      whose double-buffered activation block fits an L1 budget. Otherwise it
//      gets the stem conv's blocking;
//   4. rebuilds an NHWC -> NCHW output tail of the last tuned conv
//
//        reshape [N,H,W,C] -> permute [0,3,1,2] -> layout ops -> return
//
//      as
//
//        reshape [N,H,W,C]                    view, whole image rows per core
//        transpose(2,3)   -> [N,H,C,W]        WH tile transpose on each shard
//        to_layout ROW_MAJOR, same shards     in-place sharded untilize
//        transpose(1,2)   -> [N,C,H,W] RM     H<->C on whole W-sticks, written
//                                             straight to the output memory
//
//      The tile permute would move the C -> 32 tile padding over the NOC into
//      L1 interleaved, then the result is copied to DRAM and untilized again on
//      readback; the rebuilt tail keeps every step but the last on its own
//      core and returns the output already ROW_MAJOR.
//
// If the consumer conv is not height-sharded over the same pixels (e.g. the
// optimizer did not run), step 1 cannot apply. pixel_unshuffle is then
// height-sharded on the device's worker grid and converted to the layout its
// users already expect, and steps 2-4 are skipped.

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ttnn-pixel-unshuffle-channels-last"

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_TTNNPIXELUNSHUFFLECHANNELSLAST
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

static constexpr int64_t kTileDim = 32;

static TTNNLayoutAttr getLayout(Value v) {
  auto ty = mlir::dyn_cast<RankedTensorType>(v.getType());
  return ty ? mlir::dyn_cast_or_null<TTNNLayoutAttr>(ty.getEncoding())
            : nullptr;
}

// Layout-only ops the layout passes may put between pixel_unshuffle and the
// conv. They change placement, never values.
static bool isLayoutOnly(Operation *op) {
  return mlir::isa<ToLayoutOp, ToMemoryConfigOp>(op);
}

// The ttnn.conv2d an op's result reaches through layout-only ops and at most
// one reshape, recording that reshape. Every op on the way must have one use.
static Conv2dOp findConsumerConv(Value v, ReshapeOp &reshape,
                                 SmallVectorImpl<Operation *> &chain) {
  while (v.hasOneUse()) {
    Operation *user = *v.getUsers().begin();
    if (auto conv = mlir::dyn_cast<Conv2dOp>(user)) {
      return conv.getInput() == v ? conv : nullptr;
    }
    if (auto r = mlir::dyn_cast<ReshapeOp>(user)) {
      if (reshape) {
        return nullptr;
      }
      reshape = r;
    } else if (!isLayoutOnly(user)) {
      return nullptr;
    }
    chain.push_back(user);
    v = user->getResult(0);
  }
  return nullptr;
}

// act_block_h for a height-sharded conv with `rowsPerCore` input rows per
// core and output width `outW`: one output row when that is tile aligned and
// divides the per-core rows. Smaller blocks re-read the halo rows of the
// kernel window once per block; past one output row the kernel time stops
// improving while the activation buffer keeps growing.
static std::optional<uint32_t> pickActBlockH(int64_t rowsPerCore,
                                             int64_t outW) {
  if (outW <= 0 || outW % kTileDim != 0 || rowsPerCore % outW != 0) {
    return std::nullopt;
  }
  return static_cast<uint32_t>(outW);
}

// L1 budget for the double-buffered activation block of a full_inner_dim
// conv. The block is [act_block_h, kh*kw*C] tiles, so it grows with the whole
// kernel window; past roughly a fifth of L1 it collides with the conv's other
// resident buffers (input shard, halo, weights, output).
static constexpr int64_t kFullInnerDimActBudgetBytes = 256 * 1024;

// act_block_h for a height-sharded full_inner_dim conv: the largest tile-row
// count that divides the per-core tile rows and keeps the double-buffered
// [act_block_h, kh*kw*C] activation block within the budget. Only for a TILE
// input (a row-major input is reader-bound and does not gain from it) and a
// kernel taller than one row (otherwise there is nothing to un-slice).
static std::optional<uint32_t> pickFullInnerDimActBlockH(Conv2dOp conv,
                                                         int64_t rowsPerCore) {
  TTNNLayoutAttr in = getLayout(conv.getInput());
  ArrayRef<int32_t> k = conv.getKernelSize();
  if (!in || !in.isTiled() || k.size() != 2 || k[0] <= 1 ||
      rowsPerCore % kTileDim != 0) {
    return std::nullopt;
  }
  const int64_t paddedC = llvm::alignTo(conv.getInChannels(), kTileDim);
  const int64_t actWTiles = llvm::divideCeil(paddedC * k[0] * k[1], kTileDim);
  // getElementSizeBytes() of a tiled layout is the size of one tile.
  const int64_t tileBytes = in.getElementSizeBytes();
  const int64_t tileRows = rowsPerCore / kTileDim;
  int64_t best = 0;
  for (int64_t d = 1; d <= tileRows; ++d) {
    if (tileRows % d == 0 &&
        2 * d * actWTiles * tileBytes <= kFullInnerDimActBudgetBytes) {
      best = d;
    }
  }
  if (best == 0) {
    return std::nullopt;
  }
  return static_cast<uint32_t>(best * kTileDim);
}

static void tuneConvConfig(Conv2dOp conv, int64_t rowsPerCore,
                           bool allowFullInnerDim) {
  Conv2dConfigAttr cfg = conv.getConv2dConfigAttr();
  if (!cfg) {
    cfg = Conv2dConfigAttr::get(conv.getContext());
  }
  cfg = cfg.withDeallocateActivation(true)
            .withReallocateHaloOutput(false)
            .withEnableActDoubleBuffer(true)
            .withShardLayout(TensorMemoryLayout::HeightSharded);
  std::optional<uint32_t> fullInnerDimBlockH =
      allowFullInnerDim ? pickFullInnerDimActBlockH(conv, rowsPerCore)
                        : std::nullopt;
  if (fullInnerDimBlockH) {
    cfg = cfg.withFullInnerDim(true).withActBlockHOverride(*fullInnerDimBlockH);
  } else if (auto blockH = pickActBlockH(rowsPerCore, conv.getInputWidth())) {
    cfg = cfg.withActBlockHOverride(*blockH);
  }
  conv.setConv2dConfigAttr(cfg);
}

// Step 4: rebuild `conv -> reshape [N,H,W,C] -> permute [0,3,1,2] -> layout
// ops -> return` into the row-wise tail (see the file comment). The conv
// result must be TILE height-sharded in L1 with whole image rows per core and
// a tile-aligned W, so every op but the last transpose stays on its own core.
static LogicalResult rewriteNchwOutputTail(Conv2dOp conv, int64_t rowsPerCore) {
  Value y = conv.getResult();
  TTNNLayoutAttr yLayout = getLayout(y);
  if (!y.hasOneUse() || !yLayout || !yLayout.isTiled() ||
      !yLayout.hasShardedL1TensorMemoryLayout() ||
      yLayout.getMemLayoutOpt() != TensorMemoryLayout::HeightSharded) {
    return failure();
  }
  auto reshape = mlir::dyn_cast<ReshapeOp>(*y.getUsers().begin());
  if (!reshape || !reshape.getResult().hasOneUse()) {
    return failure();
  }
  // The reshape must stay a view of conv's shards.
  auto nhwcTy = mlir::cast<RankedTensorType>(reshape.getType());
  TTNNLayoutAttr nhwcLayout = getLayout(reshape.getResult());
  if (nhwcTy.getRank() != 4 || !nhwcLayout || !nhwcLayout.isTiled() ||
      nhwcLayout.getMemLayoutOpt() != TensorMemoryLayout::HeightSharded ||
      nhwcLayout.getGridShape() != yLayout.getGridShape() ||
      nhwcLayout.getCoreRangeSet() != yLayout.getCoreRangeSet()) {
    return failure();
  }
  const int64_t n = nhwcTy.getDimSize(0), h = nhwcTy.getDimSize(1),
                w = nhwcTy.getDimSize(2), c = nhwcTy.getDimSize(3);
  if (w % kTileDim != 0 || rowsPerCore % w != 0 ||
      n * h * w != mlir::cast<RankedTensorType>(y.getType()).getDimSize(2)) {
    return failure();
  }
  auto permute =
      mlir::dyn_cast<PermuteOp>(*reshape.getResult().getUsers().begin());
  if (!permute || permute.getPermutation() != ArrayRef<int64_t>{0, 3, 1, 2}) {
    return failure();
  }
  SmallVector<Operation *> chain{permute};
  Value out = permute.getResult();
  while (out.hasOneUse() && isLayoutOnly(*out.getUsers().begin())) {
    chain.push_back(*out.getUsers().begin());
    out = chain.back()->getResult(0);
  }
  if (!out.hasOneUse()) {
    return failure();
  }
  auto ret = mlir::dyn_cast<func::ReturnOp>(*out.getUsers().begin());
  TTNNLayoutAttr outLayout = getLayout(out);
  if (!ret || !outLayout || !outLayout.isDeviceBufferType() ||
      outLayout.hasShardedTensorMemoryLayout()) {
    return failure();
  }

  MLIRContext *ctx = conv.getContext();
  Type elemTy = nhwcTy.getElementType();

  // [N,H,C,W] TILE on conv's shards: C is padded to a tile per image row, so
  // each core holds (rowsPerCore / W) * align(C, 32) rows of W.
  SmallVector<int64_t> nhcw{n, h, c, w};
  TTNNLayoutAttr tileLayout =
      TTNNLayoutAttr::Builder(ctx, nhcw, elemTy)
          .setLayout(Layout::Tile)
          .setBufferType(BufferType::L1)
          .setMemoryLayout(TensorMemoryLayout::HeightSharded)
          .setGridShape(yLayout.getGridShape())
          .setCoreRangeSet(yLayout.getCoreRangeSet())
          .build();
  SmallVector<int64_t> shard = tileLayout.getScalarShardShape();
  if (shard.size() != 2 ||
      shard[0] != static_cast<int64_t>((rowsPerCore / w) *
                                       llvm::alignTo(c, kTileDim)) ||
      shard[1] != w) {
    return failure();
  }
  // The same shards ROW_MAJOR. The untilize keeps the tile-padded shard height
  // (ttnn rejects the unpadded one), so this layout is derived from the tiled
  // one rather than built from the logical shape.
  TTNNLayoutAttr rmLayout = TTNNLayoutAttr::get(
      ctx, tileLayout.getLinear(), tileLayout.getGridShape(),
      ttcore::buildMemRef<BufferType, BufferTypeAttr>(ctx, shard, elemTy,
                                                      BufferType::L1),
      tileLayout.getMemLayout(), tileLayout.getTensorMesh(),
      tileLayout.getIgnorePhysicalLayout(), tileLayout.getCoreRangeSet());
  // The model output, ROW_MAJOR in the memory the old tail ended in.
  auto outTy = mlir::cast<RankedTensorType>(out.getType());
  TTNNLayoutAttr outRmLayout =
      TTNNLayoutAttr::Builder(ctx, outTy.getShape(), elemTy)
          .setLayout(Layout::RowMajor)
          .setBufferType(outLayout.getBufferType())
          .setMemoryLayout(outLayout.getMemLayout())
          .setGridShape(outLayout.getGridShape())
          .build();

  OpBuilder b(ret);
  Location loc = permute.getLoc();
  auto t1 = b.create<TransposeOp>(
      loc, RankedTensorType::get(nhcw, elemTy, tileLayout), reshape.getResult(),
      b.getSI32IntegerAttr(2), b.getSI32IntegerAttr(3));
  auto t2 = b.create<ToLayoutOp>(
      loc, RankedTensorType::get(nhcw, elemTy, rmLayout), t1.getResult());
  auto t3 = b.create<TransposeOp>(
      loc, RankedTensorType::get(outTy.getShape(), elemTy, outRmLayout),
      t2.getResult(), b.getSI32IntegerAttr(1), b.getSI32IntegerAttr(2));
  out.replaceAllUsesWith(t3.getResult());
  for (Operation *op : llvm::reverse(chain)) {
    if (op->use_empty()) {
      op->erase();
    }
  }

  // The function now returns the ROW_MAJOR tensor.
  auto func = ret->getParentOfType<func::FuncOp>();
  func.setType(
      FunctionType::get(ctx, func.getArgumentTypes(), ret.getOperandTypes()));
  return success();
}

class TTNNPixelUnshuffleChannelsLast
    : public impl::TTNNPixelUnshuffleChannelsLastBase<
          TTNNPixelUnshuffleChannelsLast> {
public:
  using impl::TTNNPixelUnshuffleChannelsLastBase<
      TTNNPixelUnshuffleChannelsLast>::TTNNPixelUnshuffleChannelsLastBase;

  void runOnOperation() final {
    SmallVector<PixelUnshuffleOp> ops;
    getOperation()->walk([&](PixelUnshuffleOp op) {
      if (op.getChannelsLast()) {
        ops.push_back(op);
      }
    });
    for (PixelUnshuffleOp op : ops) {
      if (succeeded(placeOnConvGrid(op))) {
        continue;
      }
      if (failed(placeOnWorkerGrid(op))) {
        // The op cannot run unsharded: leaving it would fail at runtime.
        op.emitOpError("channels_last pixel_unshuffle could not be "
                       "height-sharded in L1");
        signalPassFailure();
        return;
      }
    }
  }

private:
  // Steps 1-4 (see the file comment).
  LogicalResult placeOnConvGrid(PixelUnshuffleOp pu) {
    ReshapeOp oldReshape;
    SmallVector<Operation *> chain;
    Conv2dOp conv = findConsumerConv(pu.getResult(), oldReshape, chain);
    if (!conv || !oldReshape) {
      LLVM_DEBUG(llvm::dbgs() << "no pixel_unshuffle -> reshape -> conv2d\n");
      return failure();
    }

    auto puTy = mlir::cast<RankedTensorType>(pu.getResult().getType());
    auto flatTy = mlir::cast<RankedTensorType>(oldReshape.getType());
    ArrayRef<int64_t> puShape = puTy.getShape(); // [N, Ho, Wo, Cp]
    const int64_t pixels = puShape[0] * puShape[1] * puShape[2];
    const int64_t cp = puShape[3];
    if (flatTy.getRank() != 4 || flatTy.getDimSize(2) != pixels ||
        flatTy.getDimSize(3) != cp) {
      return failure();
    }

    // The conv's output sharding is the grid its input must be sharded on:
    // a stride-1 conv with 'same' output keeps one output row per input row.
    TTNNLayoutAttr convOut = getLayout(conv.getResult());
    auto convResTy = mlir::cast<RankedTensorType>(conv.getResult().getType());
    if (!convOut || !convOut.hasShardedL1TensorMemoryLayout() ||
        convOut.getMemLayoutOpt() != TensorMemoryLayout::HeightSharded ||
        convResTy.getDimSize(2) != pixels) {
      LLVM_DEBUG(llvm::dbgs() << "conv2d output is not height-sharded over "
                                 "the same pixels\n");
      return failure();
    }
    ArrayRef<int64_t> grid = convOut.getGridShape();
    const int64_t numCores = grid[0] * grid[1];
    if (numCores <= 0 || pixels % numCores != 0) {
      return failure();
    }
    const int64_t rowsPerCore = pixels / numCores;
    if (convOut.getScalarShardShape()[0] != rowsPerCore) {
      return failure();
    }

    // ROW_MAJOR, HEIGHT_SHARDED in L1, shard [rowsPerCore, Cp] on the conv's
    // cores. The 4D and the flattened tensor collapse to the same 2D shape,
    // so the reshape between them stays a view.
    auto rmSharded = [&](ArrayRef<int64_t> shape, Type elemTy) {
      return TTNNLayoutAttr::Builder(&getContext(), shape, elemTy)
          .setLayout(Layout::RowMajor)
          .setBufferType(BufferType::L1)
          .setMemoryLayout(TensorMemoryLayout::HeightSharded)
          .setGridShape(grid)
          .setCoreRangeSet(convOut.getCoreRangeSet())
          .build();
    };
    Type elemTy = puTy.getElementType();
    TTNNLayoutAttr puLayout = rmSharded(puShape, elemTy);
    TTNNLayoutAttr flatLayout = rmSharded(flatTy.getShape(), elemTy);

    pu.getResult().setType(RankedTensorType::get(puShape, elemTy, puLayout));
    pu.setMemoryConfigAttr(MemoryConfigAttr::get(puLayout));

    OpBuilder b(conv);
    auto newReshape = b.create<ReshapeOp>(
        oldReshape.getLoc(),
        RankedTensorType::get(flatTy.getShape(), elemTy, flatLayout),
        pu.getResult(), oldReshape.getShapeAttr());
    conv.getInputMutable().assign(newReshape.getResult());

    // Erase the now-dead layout chain, last op first.
    for (Operation *op : llvm::reverse(chain)) {
      if (op->use_empty()) {
        op->erase();
      }
    }

    // The stem conv reads a ROW_MAJOR input, which full_inner_dim does not
    // speed up (reader-bound), so it keeps the sliced inner dim.
    tuneConvConfig(conv, rowsPerCore, /*allowFullInnerDim=*/false);

    // A stride-1 conv reading the stem conv's height-sharded result as is has
    // the same rows per core and benefits from the same blocking.
    Conv2dOp last = conv;
    Value y = conv.getResult();
    if (y.hasOneUse()) {
      if (auto next = mlir::dyn_cast<Conv2dOp>(*y.getUsers().begin())) {
        ArrayRef<int32_t> stride = next.getStride();
        auto nextResTy =
            mlir::cast<RankedTensorType>(next.getResult().getType());
        if (next.getInput() == y &&
            llvm::all_of(stride, [](int32_t s) { return s == 1; }) &&
            nextResTy.getDimSize(2) == pixels) {
          tuneConvConfig(next, rowsPerCore, /*allowFullInnerDim=*/true);
          last = next;
        }
      }
    }

    if (succeeded(rewriteNchwOutputTail(last, rowsPerCore))) {
      LLVM_DEBUG(llvm::dbgs() << "rebuilt the NCHW output tail\n");
    }
    return success();
  }

  // Fallback: height-shard pixel_unshuffle's output on the worker grid (the
  // largest core count that divides the pixels) and convert it back to the
  // layout its users were built against. The layout decomposition pass turns
  // that to_layout into the actual memory ops.
  LogicalResult placeOnWorkerGrid(PixelUnshuffleOp pu) {
    auto puTy = mlir::cast<RankedTensorType>(pu.getResult().getType());
    TTNNLayoutAttr oldLayout = getLayout(pu.getResult());
    if (!oldLayout || !puTy.hasStaticShape()) {
      return failure();
    }
    ArrayRef<int64_t> shape = puTy.getShape(); // [N, Ho, Wo, Cp]
    const int64_t pixels = shape[0] * shape[1] * shape[2];
    const int64_t cp = shape[3];
    Type elemTy = puTy.getElementType();
    if (pixels <= 0 || !elemTy.isIntOrFloat()) {
      return failure();
    }

    ttcore::DeviceAttr device = ttcore::lookupDevice(pu);
    const int64_t maxCores =
        ttmlir::utils::volume(device.getWorkerGrid().getShape());
    int64_t numCores = std::min(maxCores, pixels);
    while (numCores > 1 && pixels % numCores != 0) {
      --numCores;
    }
    // The shard must leave room in L1 for the ops around it.
    const uint64_t shardBytes = static_cast<uint64_t>(pixels / numCores) * cp *
                                (elemTy.getIntOrFloatBitWidth() / 8);
    const uint64_t l1Bytes = ttcore::getOpChipDescAttr(pu).getUsableL1Size();
    if (shardBytes > l1Bytes / 2) {
      LLVM_DEBUG(llvm::dbgs() << "pixel_unshuffle shard of " << shardBytes
                              << " B does not fit L1\n");
      return failure();
    }

    SmallVector<int64_t> gridShape{numCores, 1};
    TTNNLayoutAttr shardedLayout =
        TTNNLayoutAttr::Builder(&getContext(), shape, elemTy)
            .setLayout(Layout::RowMajor)
            .setBufferType(BufferType::L1)
            .setMemoryLayout(TensorMemoryLayout::HeightSharded)
            .setGridShape(gridShape)
            .buildWithCanonicalCorePlacement(device);

    pu.getResult().setType(RankedTensorType::get(shape, elemTy, shardedLayout));
    pu.setMemoryConfigAttr(MemoryConfigAttr::get(shardedLayout));

    OpBuilder b(pu.getContext());
    b.setInsertionPointAfter(pu);
    auto toOld = b.create<ToLayoutOp>(
        pu.getLoc(), RankedTensorType::get(shape, elemTy, oldLayout),
        pu.getResult());
    pu.getResult().replaceAllUsesExcept(toOld.getResult(), toOld);
    return success();
  }
};

} // namespace
} // namespace mlir::tt::ttnn
