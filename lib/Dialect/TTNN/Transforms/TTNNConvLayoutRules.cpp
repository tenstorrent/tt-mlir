// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Two post-optimizer conv rules from per-conv sweeps:
//
// TTNNRowMajorConvInputs
//   A large height-sharded conv with a >= 3x3 kernel whose input is a
//   height-sharded TILE tensor with a channel count that is not a tile
//   multiple (e.g. 16) is cheaper when it reads its input row-major: the input
//   is untilized once on its own shards and the conv's halo reads row-major
//   sticks. Below ~30k input pixels the extra untilize costs more than it
//   saves, so smaller convs are left alone.
//
// TTNNConvConfigRules (blocking / buffering only; math unchanged)
//   R1 block-sharded conv:                      full_inner_dim, act + weights
//                                               double buffering
//   R2 height-sharded conv, input <= 4096 px:   act + weights double buffering
//   R3 height-sharded 3x3/s1 conv, 30k-100k px, full_inner_dim,
//      <= 32 input channels:                    act_block_h_override = 128,
//                                               act double buffering

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_TTNNROWMAJORCONVINPUTS
#define GEN_PASS_DEF_TTNNCONVCONFIGRULES
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

constexpr int64_t kTileWidth = 32;

static TTNNLayoutAttr getLayout(Value v) {
  auto ty = mlir::dyn_cast<RankedTensorType>(v.getType());
  return ty ? mlir::dyn_cast_or_null<TTNNLayoutAttr>(ty.getEncoding())
            : nullptr;
}

static bool isHeightShardedL1(TTNNLayoutAttr layout) {
  return layout && layout.getBufferType() == BufferType::L1 &&
         layout.getMemLayout() &&
         layout.getMemLayout().getValue() ==
             TensorMemoryLayout::HeightSharded &&
         layout.getCoreRangeSet();
}

static std::optional<TensorMemoryLayout> convShardLayout(Conv2dOp conv) {
  if (Conv2dConfigAttr cfg = conv.getConv2dConfigAttr()) {
    if (cfg.getShardLayout()) {
      return *cfg.getShardLayout();
    }
  }
  if (TTNNLayoutAttr out = getLayout(conv.getResult())) {
    if (out.getMemLayout()) {
      return out.getMemLayout().getValue();
    }
  }
  return std::nullopt;
}

// ---- RM_IN ---------------------------------------------------------------

// Below this many input pixels the extra untilize costs more than the
// narrower halo / inner dim saves (sweeps: gains on the 96x416 and 192x832
// stages only).
constexpr int64_t kRowMajorInputMinPixels = 30000;

static LogicalResult rowMajorInput(Conv2dOp conv,
                                   llvm::DenseMap<Value, Value> &untilized) {
  ArrayRef<int32_t> k = conv.getKernelSize();
  const int64_t px = static_cast<int64_t>(conv.getInputHeight()) *
                     conv.getInputWidth() * conv.getBatchSize();
  if (k.size() != 2 || k[0] < 3 || k[1] < 3 || px < kRowMajorInputMinPixels ||
      convShardLayout(conv) != TensorMemoryLayout::HeightSharded) {
    return failure();
  }
  Value x = conv.getInput();
  TTNNLayoutAttr xLayout = getLayout(x);
  auto xTy = mlir::dyn_cast<RankedTensorType>(x.getType());
  if (!xTy || xTy.getRank() != 4 || !xTy.hasStaticShape() || !xLayout ||
      !xLayout.isTiled() || !isHeightShardedL1(xLayout) ||
      xTy.getDimSize(3) % kTileWidth == 0) {
    return failure();
  }
  // Several convs reading the same tensor share one untilize.
  if (Value done = untilized.lookup(x)) {
    conv.getInputMutable().assign(done);
    return success();
  }
  // Keep the TILE shard's tile-padded [rows, cols]: the untilize then runs
  // in place, shard by shard (a C-wide shard would be re-derived from the
  // logical, non-padded shape by the layout decomposition).
  auto tile = mlir::cast<ttcore::TileType>(xLayout.getElementType());
  const int64_t shardRows =
      xLayout.getMemref().getShape()[0] * tile.getHeight();
  const int64_t shardCols = xLayout.getMemref().getShape()[1] * tile.getWidth();

  MLIRContext *ctx = conv.getContext();
  Type elemTy = xTy.getElementType();
  TTNNLayoutAttr base = TTNNLayoutAttr::Builder(ctx, xTy.getShape(), elemTy)
                            .setLayout(Layout::RowMajor)
                            .setBufferType(BufferType::L1)
                            .setMemoryLayout(TensorMemoryLayout::Interleaved)
                            .build();
  MemRefType memref = MemRefType::get({shardRows, shardCols}, elemTy,
                                      MemRefLayoutAttrInterface{},
                                      xLayout.getMemref().getMemorySpace());
  auto rmTy = RankedTensorType::get(
      xTy.getShape(), elemTy,
      TTNNLayoutAttr::get(ctx, base.getLinear(), xLayout.getGridShape(), memref,
                          xLayout.getMemLayout(), xLayout.getTensorMesh(),
                          /*ignorePhysicalLayout=*/false,
                          xLayout.getCoreRangeSet()));
  // Right after the producer so it dominates every conv that reads it.
  OpBuilder b(conv.getContext());
  if (Operation *def = x.getDefiningOp()) {
    b.setInsertionPointAfter(def);
  } else {
    b.setInsertionPointToStart(mlir::cast<BlockArgument>(x).getOwner());
  }
  Value rm = b.create<ToLayoutOp>(conv.getLoc(), rmTy, x).getResult();
  untilized[x] = rm;
  conv.getInputMutable().assign(rm);
  return success();
}

class TTNNRowMajorConvInputs
    : public impl::TTNNRowMajorConvInputsBase<TTNNRowMajorConvInputs> {
public:
  using impl::TTNNRowMajorConvInputsBase<
      TTNNRowMajorConvInputs>::TTNNRowMajorConvInputsBase;

  void runOnOperation() final {
    SmallVector<Conv2dOp> convs;
    getOperation()->walk([&](Conv2dOp op) { convs.push_back(op); });
    llvm::DenseMap<Value, Value> untilized;
    for (Conv2dOp conv : convs) {
      (void)rowMajorInput(conv, untilized);
    }
  }
};

// ---- CONV_RULES ------------------------------------------------------------

static void applyRules(Conv2dOp conv) {
  Conv2dConfigAttr cfg = conv.getConv2dConfigAttr();
  std::optional<TensorMemoryLayout> shard = convShardLayout(conv);
  if (!cfg || !shard) {
    return;
  }
  const int64_t px = static_cast<int64_t>(conv.getInputHeight()) *
                     conv.getInputWidth() * conv.getBatchSize();
  ArrayRef<int32_t> k = conv.getKernelSize();
  ArrayRef<int32_t> s = conv.getStride();
  Conv2dConfigAttr next = cfg;
  if (*shard == TensorMemoryLayout::BlockSharded) {
    next = next.withFullInnerDim(true)
               .withEnableActDoubleBuffer(true)
               .withEnableWeightsDoubleBuffer(true);
  } else if (*shard == TensorMemoryLayout::HeightSharded) {
    if (px <= 4096) {
      next = next.withEnableActDoubleBuffer(true).withEnableWeightsDoubleBuffer(
          true);
    } else if (px > 30000 && px <= 100000 && k == ArrayRef<int32_t>{3, 3} &&
               s == ArrayRef<int32_t>{1, 1} && conv.getInChannels() <= 32) {
      next = next.withFullInnerDim(true)
                 .withActBlockHOverride(128)
                 .withEnableActDoubleBuffer(true);
    }
  }
  if (next != cfg) {
    conv.setConv2dConfigAttr(next);
  }
}

class TTNNConvConfigRules
    : public impl::TTNNConvConfigRulesBase<TTNNConvConfigRules> {
public:
  using impl::TTNNConvConfigRulesBase<
      TTNNConvConfigRules>::TTNNConvConfigRulesBase;

  void runOnOperation() final {
    getOperation()->walk([&](Conv2dOp op) { applyRules(op); });
  }
};

} // namespace
} // namespace mlir::tt::ttnn
