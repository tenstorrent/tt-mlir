// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MINSERTEXPLICITSTREAMS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {
class D2MInsertExplicitStreamsRewriter
    : public OpRewritePattern<d2m::GenericOp> {
public:
  using OpRewritePattern<d2m::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(d2m::GenericOp op,
                                PatternRewriter &rewriter) const final {
    if (!op.hasExplicitBlockFactors()) {
      return failure();
    }

    func::FuncOp func = op->getParentOfType<func::FuncOp>();
    std::optional<ArrayAttr> argAttrs = func.getArgAttrs();
    if (!argAttrs) {
      return failure();
    }

    SmallVector<bool> requestedStream =
        llvm::map_to_vector(*argAttrs, [](Attribute attr) -> bool {
          Attribute stream = llvm::cast<DictionaryAttr>(attr).get("d2m.stream");
          return stream && llvm::cast<BoolAttr>(stream).getValue();
        });

    if (!llvm::any_of(requestedStream, [](bool b) { return b; })) {
      return failure();
    }

    bool modified = false;
    for (OpOperand *operand : op.getDpsInputOperands()) {
      if (!requestedStream[operand->getOperandNumber()]) {
        continue;
      }

      const bool hasStream =
          mlir::isa<d2m::StreamLayoutOp>(operand->get().getDefiningOp());
      if (hasStream) {
        continue;
      }

      d2m::StreamLayoutOp stream = createStream(rewriter, op.getLoc(), operand);

      // Replace uses involved with this generic op:
      //  - Op operand
      //  - Used inside the region of this generic op (i.e. dma op)
      rewriter.replaceUsesWithIf(
          operand->get(), stream.getResult(), [&](OpOperand &use) {
            auto useParent = use.getOwner()->getParentOfType<GenericOp>();
            return use.getOwner() == op || useParent == op;
          });

      modified = true;
    }

    return modified ? success() : failure();
  }

  static d2m::StreamLayoutOp createStream(PatternRewriter &rewriter,
                                          Location loc, OpOperand *opOperand) {
    Value operand = opOperand->get();
    MemRefType memref = mlir::cast<MemRefType>(operand.getType());
    auto layout = mlir::cast<ttcore::DeviceLayoutInterface>(memref.getLayout());
    SmallVector<int64_t> viewGrid(layout.getGridShape(memref));
    SmallVector<int64_t> storageGrid(layout.getGridShape(memref));
    SmallVector<int64_t> storageShard(layout.getShardShape(memref));

    ttcore::ViewLayoutAttr streamLayout;
    if (auto view =
            mlir::dyn_cast<ViewLayoutOp>(opOperand->get().getDefiningOp())) {
      operand = view.getInput();
      streamLayout = mlir::cast<ttcore::ViewLayoutAttr>(
          mlir::cast<MemRefType>(view.getResult().getType()).getLayout());
    } else {
      streamLayout = rewriter.getAttr<ttcore::ViewLayoutAttr>(
          rewriter.getMultiDimIdentityMap(memref.getRank()));
    }

    auto streamShape =
        llvm::to_vector(llvm::concat<int64_t>(viewGrid, storageShard));
    auto storageShape =
        llvm::to_vector(llvm::concat<int64_t>(storageGrid, storageShard));

    auto streamMemref = MemRefType::get(storageShape, memref.getElementType(),
                                        streamLayout, memref.getMemorySpace());
    // TODO: support buffers
    auto storageAttr = ttcore::ShardLayoutAttr::get(
        storageShard, memref.getElementType(), /*buffers=*/1);
    auto storageMemref = MemRefType::get(storageShape, memref.getElementType(),
                                         storageAttr, memref.getMemorySpace());
    auto storage = rewriter.create<memref::AllocOp>(loc, storageMemref);
    return rewriter.create<d2m::StreamLayoutOp>(loc, streamMemref, operand,
                                                storage);
  }
};
} // namespace

namespace {
class D2MInsertExplicitStreams
    : public impl::D2MInsertExplicitStreamsBase<D2MInsertExplicitStreams> {
public:
  using impl::D2MInsertExplicitStreamsBase<
      D2MInsertExplicitStreams>::D2MInsertExplicitStreamsBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<D2MInsertExplicitStreamsRewriter>(&getContext());
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};
} // namespace

} // namespace mlir::tt::d2m
