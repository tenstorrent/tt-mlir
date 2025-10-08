// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::d2m {

#define GEN_PASS_DEF_D2MINSERTSTREAMS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {
class D2MInsertStreamsRewriter final : public OpRewritePattern<d2m::GenericOp> {
public:
  D2MInsertStreamsRewriter(MLIRContext *context, unsigned numStreamBuffers,
                           bool allowOutputSpilling)
      : OpRewritePattern<d2m::GenericOp>(context),
        numStreamBuffers(numStreamBuffers),
        allowOutputSpilling(allowOutputSpilling) {}

  LogicalResult matchAndRewrite(d2m::GenericOp op,
                                PatternRewriter &rewriter) const final {
    TT_assertv(!allowOutputSpilling, "Output spilling is not allowed");

    // For DMA-only form, stream insertion will break semantics.
    if (op.isDMAOnlyForm()) {
      return failure();
    }

    bool modified = false;
    for (OpOperand &operand : op->getOpOperands()) {
      bool isOutput = !op.isDpsInput(&operand);
      bool alreadyStreamed = mlir::isa_and_nonnull<d2m::StreamLayoutOp>(
          operand.get().getDefiningOp());
      bool isL1Memspace =
          ttcore::getMemorySpace(mlir::cast<MemRefType>(
              operand.get().getType())) == ttcore::MemorySpace::DeviceL1;

      if ((isOutput && !allowOutputSpilling) || alreadyStreamed ||
          isL1Memspace) {
        continue;
      }
      insertStream(rewriter, operand, op);
      modified = true;
    }

    return success(modified);
  }

  void insertStream(PatternRewriter &rewriter, OpOperand &operand,
                    d2m::GenericOp op) const {
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
                        rewriter.getAttr<ttcore::MemorySpaceAttr>(
                            ttcore::MemorySpace::DeviceL1));
    auto storage = rewriter.create<memref::AllocOp>(op.getLoc(), storageMemref);
    auto streamLayout = rewriter.create<d2m::StreamLayoutOp>(
        op.getLoc(), streamMemref, operand.get(), storage);
    rewriter.modifyOpInPlace(
        op, [&]() { operand.assign(streamLayout.getResult()); });
  }

private:
  unsigned numStreamBuffers;
  bool allowOutputSpilling;
};
} // namespace

namespace {
class D2MInsertStreams final
    : public impl::D2MInsertStreamsBase<D2MInsertStreams> {
public:
  using impl::D2MInsertStreamsBase<D2MInsertStreams>::D2MInsertStreamsBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<D2MInsertStreamsRewriter>(&getContext(), numStreamBuffers,
                                           allowOutputSpilling);
    if (failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
