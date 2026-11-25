// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// TTNNRowMajorConcat
//
// A channel concat of height-sharded TILE inputs whose widths are not tile
// multiples (e.g. 16 + 16 channels) makes ttnn untilize both inputs, concat
// row-major and re-tilize the result, after which the consuming conv's halo
// untilizes it again. When the concat's only user is a conv, this pass emits
//
//   %a' = to_layout(%a)   ROW_MAJOR, height sharded on %a's cores
//   %b' = to_layout(%b)   ROW_MAJOR, height sharded on %b's cores
//   %c  = concat(%a', %b') {dim = 3}   ROW_MAJOR, height sharded
//   conv2d(%c, ...)                    reads the row-major shards directly
//
// Row-major shards keep the inputs' tile-padded shard height so every core
// holds the same rows as before. Runs after the optimizer and before layout
// decomposition.

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ttnn-row-major-concat"

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_TTNNROWMAJORCONCAT
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

constexpr int64_t kTileWidth = 32;

static TTNNLayoutAttr getLayout(Value v) {
  auto ty = mlir::dyn_cast<RankedTensorType>(v.getType());
  return ty ? mlir::dyn_cast_or_null<TTNNLayoutAttr>(ty.getEncoding())
            : nullptr;
}

static bool isTiledHeightShardedL1(TTNNLayoutAttr layout) {
  return layout && layout.isTiled() &&
         layout.getBufferType() == BufferType::L1 && layout.getMemLayout() &&
         layout.getMemLayout().getValue() ==
             TensorMemoryLayout::HeightSharded &&
         layout.getCoreRangeSet();
}

// ROW_MAJOR L1 height-sharded type of `shape` on `like`'s grid and cores
// with `like`'s tile-padded [rows, cols] shard. Keeping the TILE shard shape
// (rather than C-wide) makes the untilize an in-place, shard-by-shard one;
// a C-wide shard would be re-derived from the logical shape (not tile-padded)
// by the layout decomposition.
static RankedTensorType rmHeightSharded(MLIRContext *ctx,
                                        ArrayRef<int64_t> shape, Type elemTy,
                                        TTNNLayoutAttr like) {
  auto tile = mlir::cast<ttcore::TileType>(like.getElementType());
  const int64_t shardRows = like.getMemref().getShape()[0] * tile.getHeight();
  const int64_t shardCols = like.getMemref().getShape()[1] * tile.getWidth();
  TTNNLayoutAttr base = TTNNLayoutAttr::Builder(ctx, shape, elemTy)
                            .setLayout(Layout::RowMajor)
                            .setBufferType(BufferType::L1)
                            .setMemoryLayout(TensorMemoryLayout::Interleaved)
                            .build();
  MemRefType memref = MemRefType::get({shardRows, shardCols}, elemTy,
                                      MemRefLayoutAttrInterface{},
                                      like.getMemref().getMemorySpace());
  return RankedTensorType::get(
      shape, elemTy,
      TTNNLayoutAttr::get(ctx, base.getLinear(), like.getGridShape(), memref,
                          like.getMemLayout(), like.getTensorMesh(),
                          /*ignorePhysicalLayout=*/false,
                          like.getCoreRangeSet()));
}

static int64_t shardRowsOf(TTNNLayoutAttr layout) {
  auto tile = mlir::cast<ttcore::TileType>(layout.getElementType());
  return layout.getMemref().getShape()[0] * tile.getHeight();
}

static LogicalResult rewrite(ConcatOp concat) {
  auto outTy = mlir::cast<RankedTensorType>(concat.getResult().getType());
  const int64_t rank = outTy.getRank();
  int64_t dim = concat.getDim();
  if (dim < 0) {
    dim += rank;
  }
  if (rank != 4 || dim != 3 || !outTy.hasStaticShape() ||
      !concat.getResult().hasOneUse()) {
    return failure();
  }
  auto conv = mlir::dyn_cast<Conv2dOp>(*concat.getResult().getUsers().begin());
  if (!conv || conv.getInput() != concat.getResult()) {
    return failure();
  }
  TTNNLayoutAttr outLayout = getLayout(concat.getResult());
  if (!isTiledHeightShardedL1(outLayout)) {
    return failure();
  }
  const int64_t shardRows = shardRowsOf(outLayout);
  bool anyUnaligned = false;
  for (Value in : concat.getInputs()) {
    TTNNLayoutAttr l = getLayout(in);
    if (!isTiledHeightShardedL1(l) || shardRowsOf(l) != shardRows ||
        l.getCoreRangeSet() != outLayout.getCoreRangeSet()) {
      return failure();
    }
    anyUnaligned |=
        mlir::cast<RankedTensorType>(in.getType()).getDimSize(3) % kTileWidth !=
        0;
  }
  if (!anyUnaligned) {
    return failure(); // tile-aligned channels concat as tiles already
  }

  MLIRContext *ctx = concat.getContext();
  Type elemTy = outTy.getElementType();
  OpBuilder b(concat);
  Location loc = concat.getLoc();
  SmallVector<Value> rmInputs;
  for (Value in : concat.getInputs()) {
    auto inTy = mlir::cast<RankedTensorType>(in.getType());
    rmInputs.push_back(
        b.create<ToLayoutOp>(
             loc, rmHeightSharded(ctx, inTy.getShape(), elemTy, getLayout(in)),
             in)
            .getResult());
  }
  auto rmConcat = b.create<ConcatOp>(
      loc, rmHeightSharded(ctx, outTy.getShape(), elemTy, outLayout), rmInputs,
      concat.getDimAttr());
  conv.getInputMutable().assign(rmConcat.getResult());
  concat.erase();
  return success();
}

class TTNNRowMajorConcat
    : public impl::TTNNRowMajorConcatBase<TTNNRowMajorConcat> {
public:
  using impl::TTNNRowMajorConcatBase<
      TTNNRowMajorConcat>::TTNNRowMajorConcatBase;

  void runOnOperation() final {
    SmallVector<ConcatOp> concats;
    getOperation()->walk([&](ConcatOp op) { concats.push_back(op); });
    for (ConcatOp op : concats) {
      if (succeeded(rewrite(op))) {
        LLVM_DEBUG(llvm::dbgs() << "row-major sharded concat into conv\n");
      }
    }
  }
};

} // namespace
} // namespace mlir::tt::ttnn
