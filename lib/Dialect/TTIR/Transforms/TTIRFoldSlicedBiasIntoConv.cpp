// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// TTIRFoldSlicedBiasIntoConv
//
// A multi-output head is a bias-less conv2d whose output channels are cut
// into several ranges, each with its own bias:
//
//   %y  = ttir.conv2d(%x, %w)                       NHWC, no bias
//   %t  = transpose / permute ...                   e.g. NHWC -> NCHW
//   %s0 = ttir.index(%t) {dim = C, 0:c0}    %o0 = ttir.add(%s0, %b0 [1,c0,1,1])
//   %s1 = ttir.index(%t) {dim = C, c0:c1}   %o1 = ttir.add(%s1, %b1
//   [1,c1-c0,1,1])
//   ...
//
// Each branch re-reads the conv output and runs its own broadcast add. When
// the ranges partition all output channels and every bias is a parameter or
// constant, the biases are the conv's bias, cut the same way. This pass builds
// it once (reshape each bias to [1,1,1,c_i] and concatenate in channel order;
// const-eval folds that out of the forward graph), gives it to the conv and
// drops the adds. The conv then adds the bias in its fp32 accumulator, which is
// at least as accurate as the separate bf16 add.

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRFOLDSLICEDBIASINTOCONV
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

struct Branch {
  int64_t begin;
  int64_t end;
  Operation *slice;
  AddOp add;
  Value bias;
  bool inner;           // feeds more compute, not (only) a model output
  int64_t newBegin = 0; // position after the channel reorder
};

static int64_t normalize(int64_t dim, int64_t rank) {
  return dim < 0 ? dim + rank : dim;
}

// The channel range [begin, end) a slice/index op takes along `axis`, if it is
// a unit-step slice of that axis only.
static std::optional<std::pair<int64_t, int64_t>>
channelRange(Operation *op, int64_t axis, ArrayRef<int64_t> shape) {
  if (auto index = mlir::dyn_cast<IndexOp>(op)) {
    if (normalize(index.getDim(), shape.size()) != axis ||
        index.getStep() != 1) {
      return std::nullopt;
    }
    return std::make_pair<int64_t, int64_t>(index.getBegin(), index.getEnd());
  }
  if (auto slice = mlir::dyn_cast<SliceStaticOp>(op)) {
    auto ints = [](ArrayAttr a) {
      SmallVector<int64_t> v;
      for (Attribute x : a) {
        v.push_back(mlir::cast<IntegerAttr>(x).getInt());
      }
      return v;
    };
    SmallVector<int64_t> b = ints(slice.getBegins()), e = ints(slice.getEnds()),
                         s = ints(slice.getStep());
    if (b.size() != shape.size()) {
      return std::nullopt;
    }
    for (size_t d = 0; d < shape.size(); ++d) {
      if (s[d] != 1) {
        return std::nullopt;
      }
      if (static_cast<int64_t>(d) != axis && (b[d] != 0 || e[d] != shape[d])) {
        return std::nullopt;
      }
    }
    return std::make_pair(b[axis], e[axis]);
  }
  return std::nullopt;
}

// The add's other operand, if it is a parameter/constant bias that is 1 on
// every dim except `axis`, where it is `width`.
static Value constBias(AddOp add, Value sliced, int64_t axis, int64_t width) {
  Value other = add.getLhs() == sliced ? add.getRhs() : add.getLhs();
  if (other == sliced) {
    return nullptr;
  }
  auto ty = mlir::dyn_cast<RankedTensorType>(other.getType());
  if (!ty || !ty.hasStaticShape() ||
      ty.getRank() !=
          static_cast<int64_t>(
              mlir::cast<RankedTensorType>(sliced.getType()).getRank())) {
    return nullptr;
  }
  for (int64_t d = 0; d < ty.getRank(); ++d) {
    if (ty.getDimSize(d) != (d == axis ? width : 1)) {
      return nullptr;
    }
  }
  auto arg = mlir::dyn_cast<BlockArgument>(other);
  auto func = arg ? mlir::dyn_cast<func::FuncOp>(arg.getOwner()->getParentOp())
                  : nullptr;
  if (!func || !ttcore::isConstOrParamArg(arg, func)) {
    return nullptr;
  }
  return other;
}

static void process(Conv2dOp conv) {
  if (conv.getBias()) {
    return;
  }
  auto convTy = mlir::cast<RankedTensorType>(conv.getResult().getType());
  if (convTy.getRank() != 4 || !convTy.hasStaticShape()) {
    return;
  }
  // Output channel axis of the conv (NHWC unless stated otherwise).
  int64_t axis = 3;
  if (auto cd = conv->getAttrOfType<IntegerAttr>("channel_dim")) {
    axis = cd.getInt();
  }
  const int64_t channels = convTy.getDimSize(axis);

  // Follow single-use transposes / permutes, tracking the channel axis.
  Value v = conv.getResult();
  while (v.hasOneUse()) {
    Operation *user = *v.getUsers().begin();
    if (auto t = mlir::dyn_cast<TransposeOp>(user)) {
      int64_t d0 = normalize(t.getDim0(), 4), d1 = normalize(t.getDim1(), 4);
      axis = axis == d0 ? d1 : axis == d1 ? d0 : axis;
    } else if (auto p = mlir::dyn_cast<PermuteOp>(user)) {
      ArrayRef<int64_t> perm = p.getPermutation();
      axis = std::find(perm.begin(), perm.end(), axis) - perm.begin();
    } else {
      break;
    }
    v = user->getResult(0);
  }
  auto vTy = mlir::cast<RankedTensorType>(v.getType());
  if (v.hasOneUse() || vTy.getDimSize(axis) != channels) {
    return; // a single consumer is not a sliced head
  }

  SmallVector<Branch> branches;
  for (Operation *user : v.getUsers()) {
    auto range = channelRange(user, axis, vTy.getShape());
    if (!range || !user->getResult(0).hasOneUse()) {
      return;
    }
    auto add = mlir::dyn_cast<AddOp>(*user->getResult(0).getUsers().begin());
    if (!add) {
      return;
    }
    Value bias =
        constBias(add, user->getResult(0), axis, range->second - range->first);
    if (!bias || add.getResult().getType() != user->getResult(0).getType()) {
      return;
    }
    bool inner = llvm::any_of(add.getResult().getUsers(), [](Operation *u) {
      return !mlir::isa<func::ReturnOp>(u);
    });
    branches.push_back({range->first, range->second, user, add, bias, inner});
  }
  llvm::sort(branches, [](const Branch &a, const Branch &b) {
    return a.begin < b.begin;
  });
  int64_t next = 0;
  for (const Branch &br : branches) {
    if (br.begin != next) {
      return; // gap or overlap
    }
    next = br.end;
  }
  if (next != channels) {
    return;
  }

  OpBuilder b(conv);
  Location loc = conv.getLoc();
  Type elemTy = convTy.getElementType();

  // An inner range that is a whole number of tiles wide but starts off a tile
  // boundary (e.g. channels 46:142) can only be cut out of the TILE conv
  // output by an untilize / slice / tilize round trip. Reorder the conv's
  // output channels (weight rows and bias) so such ranges come first and are
  // tile aligned; the other ranges follow in their original order.
  constexpr int64_t kTile = 32;
  auto misaligned = [&](const Branch &br) {
    return br.inner && (br.end - br.begin) % kTile == 0 &&
           br.begin % kTile != 0;
  };
  SmallVector<Branch *> order;
  for (Branch &br : branches) {
    if (misaligned(br)) {
      order.push_back(&br);
    }
  }
  Value weight = conv.getWeight();
  auto weightArg = mlir::dyn_cast<BlockArgument>(weight);
  auto func = conv->getParentOfType<func::FuncOp>();
  bool reorder =
      !order.empty() && weightArg && ttcore::isConstOrParamArg(weightArg, func);
  if (!reorder) {
    order.clear();
  }
  for (Branch &br : branches) {
    if (!reorder || !misaligned(br)) {
      order.push_back(&br);
    }
  }
  int64_t pos = 0;
  for (Branch *br : order) {
    br->newBegin = pos;
    pos += br->end - br->begin;
  }

  if (reorder) {
    auto wTy = mlir::cast<RankedTensorType>(weight.getType());
    SmallVector<int64_t> wShape(wTy.getShape());
    SmallVector<Value> rows;
    for (Branch *br : order) {
      SmallVector<int32_t> begins(wShape.size(), 0), ends,
          step(wShape.size(), 1);
      for (int64_t d : wShape) {
        ends.push_back(static_cast<int32_t>(d));
      }
      begins[0] = static_cast<int32_t>(br->begin);
      ends[0] = static_cast<int32_t>(br->end);
      SmallVector<int64_t> rowShape(wShape);
      rowShape[0] = br->end - br->begin;
      rows.push_back(b.create<SliceStaticOp>(
          loc, RankedTensorType::get(rowShape, wTy.getElementType()), weight,
          b.getI32ArrayAttr(begins), b.getI32ArrayAttr(ends),
          b.getI32ArrayAttr(step)));
    }
    conv.getWeightMutable().assign(
        b.create<ConcatOp>(loc, wTy, rows, b.getSI32IntegerAttr(0))
            .getResult());

    // Point every branch at its channels' new position.
    for (Branch *br : order) {
      int64_t n0 = br->newBegin, n1 = n0 + (br->end - br->begin);
      if (auto index = mlir::dyn_cast<IndexOp>(br->slice)) {
        index.setBeginAttr(b.getI32IntegerAttr(n0));
        index.setEndAttr(b.getI32IntegerAttr(n1));
      } else {
        auto slice = mlir::cast<SliceStaticOp>(br->slice);
        SmallVector<int32_t> begins, ends;
        for (Attribute x : slice.getBegins()) {
          begins.push_back(mlir::cast<IntegerAttr>(x).getInt());
        }
        for (Attribute x : slice.getEnds()) {
          ends.push_back(mlir::cast<IntegerAttr>(x).getInt());
        }
        begins[axis] = n0;
        ends[axis] = n1;
        slice.setBeginsAttr(b.getI32ArrayAttr(begins));
        slice.setEndsAttr(b.getI32ArrayAttr(ends));
      }
    }
  }

  // Conv bias [1,1,1,C] in (possibly reordered) output-channel order.
  SmallVector<Value> parts;
  for (Branch *brp : order) {
    const Branch &br = *brp;
    int64_t width = br.end - br.begin;
    SmallVector<int64_t> shape{1, 1, 1, width};
    parts.push_back(b.create<ReshapeOp>(
        loc, RankedTensorType::get(shape, elemTy), br.bias,
        b.getI32ArrayAttr({1, 1, 1, static_cast<int32_t>(width)})));
  }
  Value bias =
      parts.size() == 1
          ? parts.front()
          : b.create<ConcatOp>(
                 loc, RankedTensorType::get({1, 1, 1, channels}, elemTy), parts,
                 b.getSI32IntegerAttr(3))
                .getResult();
  conv.getBiasMutable().assign(bias);

  for (Branch &br : branches) {
    br.add.getResult().replaceAllUsesWith(br.slice->getResult(0));
    br.add.erase();
  }
}

class TTIRFoldSlicedBiasIntoConv
    : public impl::TTIRFoldSlicedBiasIntoConvBase<TTIRFoldSlicedBiasIntoConv> {
public:
  using impl::TTIRFoldSlicedBiasIntoConvBase<
      TTIRFoldSlicedBiasIntoConv>::TTIRFoldSlicedBiasIntoConvBase;

  void runOnOperation() final {
    SmallVector<Conv2dOp> convs;
    getOperation()->walk([&](Conv2dOp op) { convs.push_back(op); });
    for (Conv2dOp conv : convs) {
      process(conv);
    }
  }
};

} // namespace
} // namespace mlir::tt::ttir
