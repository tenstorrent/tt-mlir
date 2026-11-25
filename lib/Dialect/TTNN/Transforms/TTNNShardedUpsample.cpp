// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// TTNNShardedUpsample
//
// A decoder upsample stage between two convs lowers to
//
//   %y = conv2d(...)                        [1,1,H*W,C]   TILE, L1 height
//   sharded %r = reshape(%y)                        [1,H,W,C]
//   ... to_memory_config (DRAM) / to_layout (untilize in DRAM) ...
//   %u = upsample(%r') {2x2}                [1,2H,2W,C]   ROW_MAJOR, DRAM
//   ... to_layout (tilize in DRAM) / reshape / to_memory_config ...
//   %z = conv2d(%u', ...)                   input [1,1,4*H*W,C]
//
// so the full-resolution tensor crosses DRAM twice and is tilized and
// untilized at full resolution. This pass rebuilds the stage in L1:
//
//   %a = to_layout(%y)       ROW_MAJOR, height sharded on %y's cores, shard
//                            [S, C] with S = %y's tile-padded shard height
//   %b = reshape(%a)         [1,H,W,C]   view
//   %c = upsample(%b)        ROW_MAJOR, height sharded on the same cores,
//                            shard [4*S, C]
//   %d = reshape(%c)         [1,1,4*H*W,C] view, the next conv's input
//
// The shard heights line up 1:4 on the same cores: no DRAM round trip and no
// full-resolution tilize / untilize. Runs after the optimizer and before
// layout decomposition.

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ttnn-sharded-upsample"

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_TTNNSHARDEDUPSAMPLE
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

static TTNNLayoutAttr getLayout(Value v) {
  auto ty = mlir::dyn_cast<RankedTensorType>(v.getType());
  return ty ? mlir::dyn_cast_or_null<TTNNLayoutAttr>(ty.getEncoding())
            : nullptr;
}

static bool isMovement(Operation *op) {
  return mlir::isa<ToLayoutOp, ToMemoryConfigOp, ReshapeOp>(op);
}

// A ROW_MAJOR L1 height-sharded type of `shape` on `like`'s grid and cores,
// with an explicit [shardRows, C] shard.
static RankedTensorType rmHeightSharded(MLIRContext *ctx,
                                        ArrayRef<int64_t> shape, Type elemTy,
                                        TTNNLayoutAttr like,
                                        int64_t shardRows) {
  // The builder gives the linear map of a ROW_MAJOR tensor of this shape;
  // the grid, shard and cores are then set to match `like`.
  TTNNLayoutAttr base = TTNNLayoutAttr::Builder(ctx, shape, elemTy)
                            .setLayout(Layout::RowMajor)
                            .setBufferType(BufferType::L1)
                            .setMemoryLayout(TensorMemoryLayout::Interleaved)
                            .build();
  MemRefType likeMemref = like.getMemref();
  MemRefType memref =
      MemRefType::get({shardRows, shape.back()}, elemTy,
                      MemRefLayoutAttrInterface{}, likeMemref.getMemorySpace());
  auto layout = TTNNLayoutAttr::get(
      ctx, base.getLinear(), like.getGridShape(), memref, like.getMemLayout(),
      like.getTensorMesh(), /*ignorePhysicalLayout=*/false,
      like.getCoreRangeSet());
  return RankedTensorType::get(shape, elemTy, layout);
}

static LogicalResult rewrite(UpsampleOp up) {
  // 2x2 only: the 1:4 shard mapping depends on it.
  if (auto dense = mlir::dyn_cast<DenseI32ArrayAttr>(up.getScaleFactor())) {
    if (dense.asArrayRef() != ArrayRef<int32_t>{2, 2}) {
      return failure();
    }
  } else if (mlir::cast<IntegerAttr>(up.getScaleFactor()).getInt() != 2) {
    return failure();
  }

  // Back to the producing conv through single-use movement ops (the last
  // step may be a multi-use reshape view of the conv output).
  SmallVector<Operation *> back;
  Value v = up.getInput();
  while (Operation *def = v.getDefiningOp()) {
    if (!isMovement(def) || !v.hasOneUse()) {
      break;
    }
    back.push_back(def);
    v = def->getOperand(0);
  }
  if (auto view = v.getDefiningOp<ReshapeOp>()) {
    if (view.getInput().getDefiningOp<Conv2dOp>()) {
      v = view.getInput();
    }
  }
  auto producer = v.getDefiningOp<Conv2dOp>();
  if (!producer || back.empty()) {
    return failure();
  }
  Value y = producer.getResult();
  TTNNLayoutAttr yLayout = getLayout(y);
  auto yTy = mlir::cast<RankedTensorType>(y.getType());
  if (!yLayout || !yLayout.isTiled() || !yTy.hasStaticShape() ||
      yTy.getRank() != 4 || yLayout.getBufferType() != BufferType::L1 ||
      !yLayout.getMemLayout() ||
      yLayout.getMemLayout().getValue() != TensorMemoryLayout::HeightSharded ||
      !yLayout.getCoreRangeSet()) {
    return failure();
  }

  // Forward to the consuming conv's input through single-use movement ops.
  SmallVector<Operation *> fwd;
  Value w = up.getResult();
  Conv2dOp consumer = nullptr;
  while (w.hasOneUse()) {
    Operation *user = *w.getUsers().begin();
    if (isMovement(user)) {
      fwd.push_back(user);
      w = user->getResult(0);
      continue;
    }
    consumer = mlir::dyn_cast<Conv2dOp>(user);
    break;
  }
  if (!consumer || fwd.empty() || consumer.getInput() != w) {
    return failure();
  }

  auto inTy = mlir::cast<RankedTensorType>(up.getInput().getType());
  auto upTy = mlir::cast<RankedTensorType>(up.getResult().getType());
  auto wTy = mlir::cast<RankedTensorType>(w.getType());
  const int64_t H = inTy.getDimSize(1), W = inTy.getDimSize(2),
                C = inTy.getDimSize(3);
  if (inTy.getDimSize(0) != 1 || yTy.getDimSize(0) != 1 ||
      yTy.getDimSize(1) != 1 || yTy.getDimSize(2) != H * W ||
      yTy.getDimSize(3) != C) {
    return failure();
  }
  if (wTy.getRank() != 4 || wTy.getDimSize(2) != 4 * H * W ||
      wTy.getDimSize(3) != C) {
    return failure();
  }

  // Tile-padded shard height of the conv output.
  auto tileTy = mlir::dyn_cast<ttcore::TileType>(yLayout.getElementType());
  if (!tileTy) {
    return failure();
  }
  const int64_t shardRows =
      yLayout.getMemref().getShape()[0] * tileTy.getHeight();
  const int64_t numCores =
      yLayout.getGridShape()[0] * yLayout.getGridShape()[1];
  if (shardRows * numCores < H * W) {
    return failure();
  }

  MLIRContext *ctx = up.getContext();
  Type elemTy = yTy.getElementType();
  OpBuilder b(up);
  Location loc = up.getLoc();
  auto i32s = [&](ArrayRef<int64_t> s) {
    SmallVector<int32_t> r(s.begin(), s.end());
    return b.getI32ArrayAttr(r);
  };

  Value a =
      b.create<ToLayoutOp>(
           loc,
           rmHeightSharded(ctx, yTy.getShape(), elemTy, yLayout, shardRows), y)
          .getResult();
  SmallVector<int64_t> nhwc{1, H, W, C};
  Value bView =
      b.create<ReshapeOp>(
           loc, rmHeightSharded(ctx, nhwc, elemTy, yLayout, shardRows), a,
           i32s(nhwc))
          .getResult();
  Value c =
      b.create<UpsampleOp>(loc,
                           rmHeightSharded(ctx, upTy.getShape(), elemTy,
                                           yLayout, 4 * shardRows),
                           bView, up.getScaleFactorAttr(), up.getModeAttr())
          .getResult();
  Value d = b.create<ReshapeOp>(loc,
                                rmHeightSharded(ctx, wTy.getShape(), elemTy,
                                                yLayout, 4 * shardRows),
                                c, i32s(wTy.getShape()))
                .getResult();

  consumer.getInputMutable().assign(d);
  for (Operation *op : llvm::reverse(fwd)) {
    op->erase();
  }
  up.erase();
  for (Operation *op : back) {
    if (op->use_empty()) {
      op->erase();
    }
  }
  // The multi-use reshape view of the conv output may now be dead too.
  for (Operation *user : llvm::make_early_inc_range(y.getUsers())) {
    if (mlir::isa<ReshapeOp>(user) && user->use_empty()) {
      user->erase();
    }
  }
  return success();
}

// A channel-padded upsample stage
//
//   %p = pad(%x) {last dim C -> Cp, zeros}   TILE pad (a masked FillPad pass)
//   ... to_memory_config (DRAM) / to_layout (untilize) ...
//   %u = upsample(%p') {2x2}                 ROW_MAJOR, DRAM
//   ... to_layout (tilize) / reshape ...
//   conv2d(%u', ...)                         Cp input channels
//
// is rebuilt row-major in L1: untilize %x, pad the row-major sticks, upsample
// and hand the conv a reshape view of the result. No DRAM round trip and no
// tilize of the full-resolution tensor.
static RankedTensorType rmL1Interleaved(MLIRContext *ctx,
                                        ArrayRef<int64_t> shape, Type elemTy,
                                        ArrayRef<int64_t> grid) {
  return RankedTensorType::get(
      shape, elemTy,
      TTNNLayoutAttr::Builder(ctx, shape, elemTy)
          .setLayout(Layout::RowMajor)
          .setBufferType(BufferType::L1)
          .setMemoryLayout(TensorMemoryLayout::Interleaved)
          .setGridShape(grid)
          .build());
}

static LogicalResult rewritePadded(UpsampleOp up) {
  if (auto dense = mlir::dyn_cast<DenseI32ArrayAttr>(up.getScaleFactor())) {
    if (dense.asArrayRef() != ArrayRef<int32_t>{2, 2}) {
      return failure();
    }
  } else if (mlir::cast<IntegerAttr>(up.getScaleFactor()).getInt() != 2) {
    return failure();
  }

  // Back to the pad through single-use layout ops.
  SmallVector<Operation *> back;
  Value v = up.getInput();
  while (Operation *def = v.getDefiningOp()) {
    if (!mlir::isa<ToLayoutOp, ToMemoryConfigOp>(def) || !v.hasOneUse()) {
      break;
    }
    back.push_back(def);
    v = def->getOperand(0);
  }
  auto pad = v.getDefiningOp<PadOp>();
  if (!pad || !pad.getResult().hasOneUse() ||
      pad.getValue().convertToFloat() != 0.0f) {
    return failure();
  }
  ArrayRef<int32_t> padding = pad.getPadding();
  auto padTy = mlir::cast<RankedTensorType>(pad.getResult().getType());
  if (padTy.getRank() != 4 || padding.size() != 8 || !padTy.hasStaticShape()) {
    return failure();
  }
  for (size_t i = 0; i + 1 < padding.size(); ++i) {
    if (padding[i] != 0) {
      return failure(); // only a trailing pad of the channels
    }
  }
  Value x = pad.getInput();
  TTNNLayoutAttr xLayout = getLayout(x);
  if (!xLayout || !xLayout.isTiled()) {
    return failure();
  }

  // Forward to the consuming conv's input through single-use movement ops.
  SmallVector<Operation *> fwd;
  Value w = up.getResult();
  Conv2dOp consumer = nullptr;
  while (w.hasOneUse()) {
    Operation *user = *w.getUsers().begin();
    if (isMovement(user)) {
      fwd.push_back(user);
      w = user->getResult(0);
      continue;
    }
    consumer = mlir::dyn_cast<Conv2dOp>(user);
    break;
  }
  if (!consumer || fwd.empty() || consumer.getInput() != w) {
    return failure();
  }

  MLIRContext *ctx = up.getContext();
  auto xTy = mlir::cast<RankedTensorType>(x.getType());
  auto upTy = mlir::cast<RankedTensorType>(up.getResult().getType());
  auto wTy = mlir::cast<RankedTensorType>(w.getType());
  Type elemTy = xTy.getElementType();
  SmallVector<int64_t> grid(
      ttcore::lookupDevice(up).getWorkerGrid().getShape());

  OpBuilder b(up);
  Location loc = up.getLoc();
  Value a = b.create<ToLayoutOp>(
                 loc, rmL1Interleaved(ctx, xTy.getShape(), elemTy, grid), x)
                .getResult();
  Value p =
      b.create<PadOp>(loc, rmL1Interleaved(ctx, padTy.getShape(), elemTy, grid),
                      a, pad.getPaddingAttr(), pad.getValueAttr(),
                      pad.getUseMulticoreAttr())
          .getResult();
  Value u = b.create<UpsampleOp>(
                 loc, rmL1Interleaved(ctx, upTy.getShape(), elemTy, grid), p,
                 up.getScaleFactorAttr(), up.getModeAttr())
                .getResult();
  SmallVector<int32_t> wShape(wTy.getShape().begin(), wTy.getShape().end());
  Value r = b.create<ReshapeOp>(
                 loc, rmL1Interleaved(ctx, wTy.getShape(), elemTy, grid), u,
                 b.getI32ArrayAttr(wShape))
                .getResult();

  consumer.getInputMutable().assign(r);
  for (Operation *op : llvm::reverse(fwd)) {
    op->erase();
  }
  up.erase();
  for (Operation *op : back) {
    op->erase();
  }
  pad.erase();
  return success();
}

class TTNNShardedUpsample
    : public impl::TTNNShardedUpsampleBase<TTNNShardedUpsample> {
public:
  using impl::TTNNShardedUpsampleBase<
      TTNNShardedUpsample>::TTNNShardedUpsampleBase;

  void runOnOperation() final {
    SmallVector<UpsampleOp> ups;
    getOperation()->walk([&](UpsampleOp op) { ups.push_back(op); });
    for (UpsampleOp op : ups) {
      if (succeeded(rewrite(op))) {
        LLVM_DEBUG(llvm::dbgs() << "upsample stage kept sharded in L1\n");
      } else if (succeeded(rewritePadded(op))) {
        LLVM_DEBUG(llvm::dbgs() << "padded upsample stage kept in L1\n");
      }
    }
  }
};

} // namespace
} // namespace mlir::tt::ttnn
