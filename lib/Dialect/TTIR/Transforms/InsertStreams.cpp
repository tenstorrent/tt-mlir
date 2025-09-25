// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_TTIRINSERTSTREAMS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {
class TTIRInsertStreamsRewriter final
    : public OpRewritePattern<ttir::GenericOp> {
public:
  TTIRInsertStreamsRewriter(MLIRContext *context, unsigned numStreamBuffers)
      : OpRewritePattern<ttir::GenericOp>(context),
        numStreamBuffers(numStreamBuffers) {}

  LogicalResult matchAndRewrite(ttir::GenericOp op,
                                PatternRewriter &rewriter) const final {
    bool modified = false;
    for (OpOperand &operand : op->getOpOperands()) {
      // If input is not already a stream, insert one.
      // For DMA-only form, stream insertion will break semantics.
      bool opIsStream = mlir::isa_and_nonnull<ttir::StreamLayoutOp>(
          operand.get().getDefiningOp());
      if (opIsStream || op.isDMAOnlyForm()) {
        continue;
      }
      insertStream(rewriter, operand, op);
      modified = true;
    }

    return success(modified);
  }
  void insertStream(PatternRewriter &rewriter, OpOperand &operand,
                    ttir::GenericOp op) const {
    auto memref = mlir::cast<MemRefType>(operand.get().getType());
    auto streamAttr = rewriter.getAttr<ttcore::ViewLayoutAttr>(
        rewriter.getMultiDimIdentityMap(memref.getRank()));
    auto streamMemref =
        MemRefType::get(memref.getShape(), memref.getElementType(), streamAttr,
                        memref.getMemorySpace());
    auto storageAttr =
        ttcore::ShardLayoutAttr::get(memref, /*buffers=*/numStreamBuffers);
    auto storageMemref =
        MemRefType::get(memref.getShape(), memref.getElementType(), storageAttr,
                        memref.getMemorySpace());
    auto storage = rewriter.create<memref::AllocOp>(op.getLoc(), storageMemref);
    auto streamLayout = rewriter.create<ttir::StreamLayoutOp>(
        op.getLoc(), streamMemref, operand.get(), storage);
    rewriter.modifyOpInPlace(
        op, [&]() { operand.assign(streamLayout.getResult()); });
  }
  unsigned numStreamBuffers;
};
} // namespace

namespace {
class TTIRInsertStreams final
    : public impl::TTIRInsertStreamsBase<TTIRInsertStreams> {
public:
  using impl::TTIRInsertStreamsBase<TTIRInsertStreams>::TTIRInsertStreamsBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRInsertStreamsRewriter>(&getContext(), numStreamBuffers);
    if (failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::ttir
