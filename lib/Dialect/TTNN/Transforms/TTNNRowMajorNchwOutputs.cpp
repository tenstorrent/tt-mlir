// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// TTNNRowMajorNchwOutputs
//
// A model output produced in NHWC by a sharded TILE op (typically a conv2d)
// and returned as NCHW lowers to
//
//   %y   = producer                         [1,1,N*H*W,C]  TILE, L1 sharded
//   %r   = ttnn.reshape(%y)                 [N,H,W,C]      TILE (copy when W
//                                                          is not tile aligned)
//   %p   = ttnn.permute(%r) {0,3,1,2}       [N,C,H,W]      TILE (C padded to a
//                                                          tile in every row)
//   ... to_memory_config / to_layout ...    TILE DRAM
//   return                                  (the runtime untilizes on readback)
//
// For a narrow C this moves several times the useful bytes: the reshape is a
// real copy, the permute works on tiles padded to 32 channels, and the output
// is untilized once more on readback. This pass rebuilds the chain row-major:
//
//   %u   = ttnn.to_layout(%y)               [1,1,N*H*W,C]  ROW_MAJOR, L1
//                                                          interleaved (one
//                                                          sharded untilize)
//   %r'  = ttnn.reshape(%u)                 [N,H,W,C]      view
//   %p'  = ttnn.permute(%r') {0,3,1,2}      [N,C,H,W]      ROW_MAJOR, written
//                                                          to the output memory
//   return %p'
//
// and updates the function result type, so the output is already row-major
// when it is read back. Runs after the optimizer and before trace hoisting and
// layout decomposition.

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ttnn-row-major-nchw-outputs"

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_TTNNROWMAJORNCHWOUTPUTS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

static TTNNLayoutAttr getLayout(Value v) {
  auto ty = mlir::dyn_cast<RankedTensorType>(v.getType());
  return ty ? mlir::dyn_cast_or_null<TTNNLayoutAttr>(ty.getEncoding())
            : nullptr;
}

static bool isLayoutOnly(Operation *op) {
  return mlir::isa<ToLayoutOp, ToMemoryConfigOp>(op);
}

// Rebuilds one `producer -> reshape -> permute{0,3,1,2} -> layout ops ->
// return` chain (see the file comment). Fails, changing nothing, when the
// chain does not have that exact shape.
static LogicalResult rewriteOutput(PermuteOp permute) {
  if (permute.getPermutation() != ArrayRef<int64_t>{0, 3, 1, 2}) {
    return failure();
  }
  auto reshape = permute.getInput().getDefiningOp<ReshapeOp>();
  if (!reshape || !reshape.getResult().hasOneUse()) {
    return failure();
  }
  Value y = reshape.getInput();
  TTNNLayoutAttr yLayout = getLayout(y);
  auto yTy = mlir::cast<RankedTensorType>(y.getType());
  auto nhwcTy = mlir::cast<RankedTensorType>(reshape.getType());
  if (!yLayout || !yLayout.isTiled() ||
      !yLayout.hasShardedL1TensorMemoryLayout() || nhwcTy.getRank() != 4 ||
      !yTy.hasStaticShape() || !nhwcTy.hasStaticShape()) {
    return failure();
  }

  // Follow the permute's single-use layout-only chain to a return.
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

  MLIRContext *ctx = permute.getContext();
  Type elemTy = yTy.getElementType();
  ttcore::DeviceAttr device = ttcore::lookupDevice(permute);
  SmallVector<int64_t> workerGrid(device.getWorkerGrid().getShape());

  auto rmL1 = [&](ArrayRef<int64_t> shape) {
    return TTNNLayoutAttr::Builder(ctx, shape, elemTy)
        .setLayout(Layout::RowMajor)
        .setBufferType(BufferType::L1)
        .setMemoryLayout(TensorMemoryLayout::Interleaved)
        .setGridShape(workerGrid)
        .build();
  };
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
  auto untilized = b.create<ToLayoutOp>(
      loc, RankedTensorType::get(yTy.getShape(), elemTy, rmL1(yTy.getShape())),
      y);
  auto view = b.create<ReshapeOp>(
      loc,
      RankedTensorType::get(nhwcTy.getShape(), elemTy, rmL1(nhwcTy.getShape())),
      untilized.getResult(), reshape.getShapeAttr());
  auto nchw = b.create<PermuteOp>(
      loc, RankedTensorType::get(outTy.getShape(), elemTy, outRmLayout),
      view.getResult(), permute.getPermutationAttr(),
      permute.getPadValueAttr());

  out.replaceAllUsesWith(nchw.getResult());
  for (Operation *op : llvm::reverse(chain)) {
    if (op->use_empty()) {
      op->erase();
    }
  }
  if (reshape.getResult().use_empty()) {
    reshape.erase();
  }

  auto func = ret->getParentOfType<func::FuncOp>();
  func.setType(
      FunctionType::get(ctx, func.getArgumentTypes(), ret.getOperandTypes()));
  return success();
}

// One channel-range output of a sliced head (see rewriteHead).
struct HeadOutput {
  int64_t c0;
  int64_t c1;
  SmallVector<Operation *> ops; // the branch, producer side first
  Value out;                    // the value returned
};

// Matches `y -> layout ops -> slice_static (channels c0:c1 of the last dim)
// -> [reshape to [1,H,W,c]] -> permute{0,3,1,2} -> layout ops -> return`.
static std::optional<HeadOutput> matchHeadOutput(Operation *user, Value y,
                                                 int64_t &H, int64_t &W) {
  auto yTy = mlir::cast<RankedTensorType>(y.getType());
  HeadOutput br;
  Operation *op = user;
  Value v = y;
  while (isLayoutOnly(op)) {
    br.ops.push_back(op);
    v = op->getResult(0);
    if (!v.hasOneUse()) {
      return std::nullopt;
    }
    op = *v.getUsers().begin();
  }
  auto slice = mlir::dyn_cast<SliceStaticOp>(op);
  if (!slice || !slice.getResult().hasOneUse()) {
    return std::nullopt;
  }
  auto ints = [](ArrayAttr a) {
    SmallVector<int64_t> r;
    for (Attribute x : a) {
      r.push_back(mlir::cast<IntegerAttr>(x).getInt());
    }
    return r;
  };
  SmallVector<int64_t> b = ints(slice.getBegins()), e = ints(slice.getEnds()),
                       s = ints(slice.getStep());
  if (b.size() != 4) {
    return std::nullopt;
  }
  for (int64_t d = 0; d < 4; ++d) {
    if (s[d] != 1 || (d < 3 && (b[d] != 0 || e[d] != yTy.getDimSize(d)))) {
      return std::nullopt;
    }
  }
  br.c0 = b[3];
  br.c1 = e[3];
  br.ops.push_back(slice);
  op = *slice.getResult().getUsers().begin();

  // [1,1,H*W,c] -> [1,H,W,c]; without a reshape the view is [1,1,H*W,c].
  int64_t h = yTy.getDimSize(1), w = yTy.getDimSize(2);
  if (auto reshape = mlir::dyn_cast<ReshapeOp>(op)) {
    auto rTy = mlir::cast<RankedTensorType>(reshape.getType());
    if (!reshape.getResult().hasOneUse() || rTy.getRank() != 4 ||
        rTy.getDimSize(0) != 1 ||
        rTy.getDimSize(1) * rTy.getDimSize(2) != h * w) {
      return std::nullopt;
    }
    h = rTy.getDimSize(1);
    w = rTy.getDimSize(2);
    br.ops.push_back(reshape);
    op = *reshape.getResult().getUsers().begin();
  }
  auto permute = mlir::dyn_cast<PermuteOp>(op);
  if (!permute || permute.getPermutation() != ArrayRef<int64_t>{0, 3, 1, 2}) {
    return std::nullopt;
  }
  if ((H >= 0 && H != h) || (W >= 0 && W != w)) {
    return std::nullopt;
  }
  H = h;
  W = w;
  br.ops.push_back(permute);
  Value out = permute.getResult();
  while (out.hasOneUse() && isLayoutOnly(*out.getUsers().begin())) {
    br.ops.push_back(*out.getUsers().begin());
    out = br.ops.back()->getResult(0);
  }
  TTNNLayoutAttr outLayout = getLayout(out);
  if (!out.hasOneUse() || !mlir::isa<func::ReturnOp>(*out.getUsers().begin()) ||
      !outLayout || !outLayout.isDeviceBufferType() ||
      outLayout.hasShardedTensorMemoryLayout()) {
    return std::nullopt;
  }
  br.out = out;
  return br;
}

// A multi-output head: one sharded TILE conv output [1,1,H*W,C] cut into
// channel ranges, each returned as NCHW through its own interleave copy, TILE
// slice, reshape, permute and output copy. Rebuilt as one sharded untilize, one
// row-major view + permute of all C channels to NCHW in L1, and one row-major
// slice per output written straight into the output memory. Branches that are
// not model outputs (e.g. a range feeding another conv) keep using the conv.
static LogicalResult rewriteHead(Value y) {
  TTNNLayoutAttr yLayout = getLayout(y);
  auto yTy = mlir::dyn_cast<RankedTensorType>(y.getType());
  if (!yLayout || !yTy || !yLayout.isTiled() ||
      !yLayout.hasShardedL1TensorMemoryLayout() || yTy.getRank() != 4 ||
      !yTy.hasStaticShape() || yTy.getDimSize(0) != 1 ||
      yTy.getDimSize(1) != 1) {
    return failure();
  }
  int64_t H = -1, W = -1;
  SmallVector<HeadOutput> outs;
  for (Operation *user : y.getUsers()) {
    if (auto br = matchHeadOutput(user, y, H, W)) {
      outs.push_back(std::move(*br));
    }
  }
  if (outs.size() < 2) {
    return failure(); // a single unsliced output is rewriteOutput's case
  }
  llvm::sort(outs, [](const HeadOutput &a, const HeadOutput &b) {
    return a.c0 < b.c0;
  });

  MLIRContext *ctx = y.getContext();
  Type elemTy = yTy.getElementType();
  const int64_t C = yTy.getDimSize(3);
  SmallVector<int64_t> workerGrid(
      ttcore::lookupDevice(y.getParentBlock()->getParentOp())
          .getWorkerGrid()
          .getShape());
  auto rmL1 = [&](ArrayRef<int64_t> shape) {
    return RankedTensorType::get(
        shape, elemTy,
        TTNNLayoutAttr::Builder(ctx, shape, elemTy)
            .setLayout(Layout::RowMajor)
            .setBufferType(BufferType::L1)
            .setMemoryLayout(TensorMemoryLayout::Interleaved)
            .setGridShape(workerGrid)
            .build());
  };

  // Build before the first branch op so every output can use the result.
  Operation *first = outs.front().ops.front();
  for (const HeadOutput &br : outs) {
    if (br.ops.front()->isBeforeInBlock(first)) {
      first = br.ops.front();
    }
  }
  auto func = first->getParentOfType<func::FuncOp>();
  OpBuilder b(first);
  Location loc = first->getLoc();
  Value cur = b.create<ToLayoutOp>(loc, rmL1(yTy.getShape()), y).getResult();
  if (H != 1) {
    SmallVector<int64_t> nhwc{1, H, W, C};
    cur = b.create<ReshapeOp>(loc, rmL1(nhwc), cur,
                              b.getI32ArrayAttr({1, static_cast<int32_t>(H),
                                                 static_cast<int32_t>(W),
                                                 static_cast<int32_t>(C)}))
              .getResult();
  }
  auto permute =
      mlir::cast<PermuteOp>(*llvm::find_if(outs.front().ops, [](Operation *op) {
        return mlir::isa<PermuteOp>(op);
      }));
  Value nchw = b.create<PermuteOp>(loc, rmL1({1, C, H, W}), cur,
                                   permute.getPermutationAttr(),
                                   permute.getPadValueAttr())
                   .getResult();

  for (HeadOutput &br : outs) {
    auto outTy = mlir::cast<RankedTensorType>(br.out.getType());
    TTNNLayoutAttr outLayout = getLayout(br.out);
    TTNNLayoutAttr rm = TTNNLayoutAttr::Builder(ctx, outTy.getShape(), elemTy)
                            .setLayout(Layout::RowMajor)
                            .setBufferType(outLayout.getBufferType())
                            .setMemoryLayout(outLayout.getMemLayout())
                            .setGridShape(outLayout.getGridShape())
                            .build();
    auto slice = b.create<SliceStaticOp>(
        loc, RankedTensorType::get(outTy.getShape(), elemTy, rm), nchw,
        b.getI32ArrayAttr({0, static_cast<int32_t>(br.c0), 0, 0}),
        b.getI32ArrayAttr({1, static_cast<int32_t>(br.c1),
                           static_cast<int32_t>(H), static_cast<int32_t>(W)}),
        b.getI32ArrayAttr({1, 1, 1, 1}));
    br.out.replaceAllUsesWith(slice.getResult());
  }
  // Erase only once every slice exists: the builder inserts before `first`.
  for (HeadOutput &br : outs) {
    for (Operation *op : llvm::reverse(br.ops)) {
      op->erase();
    }
  }

  func::ReturnOp ret = nullptr;
  func.walk([&](func::ReturnOp r) { ret = r; });
  func.setType(
      FunctionType::get(ctx, func.getArgumentTypes(), ret.getOperandTypes()));
  return success();
}

class TTNNRowMajorNchwOutputs
    : public impl::TTNNRowMajorNchwOutputsBase<TTNNRowMajorNchwOutputs> {
public:
  using impl::TTNNRowMajorNchwOutputsBase<
      TTNNRowMajorNchwOutputs>::TTNNRowMajorNchwOutputsBase;

  void runOnOperation() final {
    // Candidate head values: anything with several uses (rewriteHead checks
    // the layout and the branch shapes).
    SmallVector<Value> candidates;
    getOperation()->walk([&](func::FuncOp func) {
      for (BlockArgument arg : func.getArguments()) {
        candidates.push_back(arg);
      }
    });
    getOperation()->walk([&](Operation *op) {
      for (Value r : op->getResults()) {
        if (!r.hasOneUse() && !r.use_empty()) {
          candidates.push_back(r);
        }
      }
    });
    for (Value y : candidates) {
      if (succeeded(rewriteHead(y))) {
        LLVM_DEBUG(llvm::dbgs() << "rebuilt a sliced row-major NCHW head\n");
      }
    }

    SmallVector<PermuteOp> permutes;
    getOperation()->walk([&](PermuteOp op) { permutes.push_back(op); });
    for (PermuteOp op : permutes) {
      if (succeeded(rewriteOutput(op))) {
        LLVM_DEBUG(llvm::dbgs() << "rebuilt a row-major NCHW output\n");
      }
    }
  }
};

} // namespace
} // namespace mlir::tt::ttnn
