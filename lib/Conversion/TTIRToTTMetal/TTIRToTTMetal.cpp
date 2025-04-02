// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTMetal/TTIRToTTMetal.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir::tt::ttmetal {

namespace {
class TTIRGenericRewriter : public OpRewritePattern<ttir::GenericOp> {
public:
  using OpRewritePattern<ttir::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::GenericOp op,
                                PatternRewriter &rewriter) const final {
    auto coreRanges = llvm::SmallVector<Attribute>();
    coreRanges.reserve(op.getThreads().size());
    for (size_t i = 0; i < op.getThreads().size(); i++) {
      coreRanges.push_back(
          rewriter.getAttr<ttmetal::CoreRangeAttr>(op.getGrid()));
    }
    rewriter.replaceOpWithNewOp<ttmetal::EnqueueProgramOp>(
        op, op->getResultTypes(), op.getInputs(), op.getOutputs(),
        op.getThreads(), rewriter.getArrayAttr(coreRanges),
        rewriter.getArrayAttr({}));
    return success();
  };
};
} // namespace
namespace {
class MemrefAllocRewriter : public OpRewritePattern<memref::AllocOp> {
public:
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter &rewriter) const final {

    auto address = op->getAttr("address")
                       ? op->getAttrOfType<IntegerAttr>("address")
                       : rewriter.getI64IntegerAttr(
                             1000); // arbitrary default for now, remove when
                                    // allocate pass is implemented
    assert(op.getMemref().getType().getMemorySpace() &&
           "No memref memroy space found, failing.");
    assert(mlir::isa<TileType>(op.getMemref().getType().getElementType()) &&
           "Expected memref to have tile element type, failing.");
    auto size = mlir::cast<TileType>(op.getMemref().getType().getElementType())
                    .getSizeBytes() *
                op.getMemref().getType().getNumElements();
    auto memorySpace = mlir::cast<tt::MemorySpaceAttr>(
        op.getMemref().getType().getMemorySpace());
    auto createBufferOp = rewriter.create<ttmetal::CreateBufferOp>(
        op->getLoc(), op.getMemref().getType(), address.getInt(), size,
        memorySpace.getValue());
    rewriter.replaceOp(op, createBufferOp);

    return success();
  };
};
} // namespace

} // namespace mlir::tt::ttmetal

namespace mlir::tt {

void populateTTIRToTTMetalPatterns(MLIRContext *ctx,
                                   RewritePatternSet &patterns,
                                   TypeConverter & /*typeConverter*/) {
  patterns.add<ttmetal::TTIRGenericRewriter, ttmetal::MemrefAllocRewriter>(ctx);
}

} // namespace mlir::tt
