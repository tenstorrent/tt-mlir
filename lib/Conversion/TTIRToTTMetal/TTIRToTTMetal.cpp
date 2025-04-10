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
class TTIRGenericRewriter : public OpConversionPattern<ttir::GenericOp> {
public:
  using OpConversionPattern<ttir::GenericOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::GenericOp op, ttir::GenericOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    llvm::SmallVector<Attribute> coreRanges;
    coreRanges.reserve(op.getThreads().size());
    for (size_t i = 0; i < op.getThreads().size(); i++) {
      coreRanges.emplace_back(
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
class MemrefAllocRewriter : public OpConversionPattern<memref::AllocOp> {
public:
  using OpConversionPattern<memref::AllocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, memref::AllocOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    auto address = op->getAttr("address")
                       ? op->getAttrOfType<IntegerAttr>("address")
                       : rewriter.getI64IntegerAttr(
                             1000); // TODO(#1909): arbitrary default for now,
                                    // remove when allocate pass is implemented
    assert(op.getMemref().getType().getMemorySpace() &&
           "No memref memroy space found, failing.");
    assert(mlir::isa<TileType>(op.getMemref().getType().getElementType()) &&
           "Expected memref to have tile element type, failing.");
    auto memrefType = op.getMemref().getType();
    auto size =
        mlir::cast<TileType>(memrefType.getElementType()).getSizeBytes() *
        memrefType.getNumElements();
    auto memorySpace =
        mlir::cast<tt::MemorySpaceAttr>(memrefType.getMemorySpace());
    auto createBufferOp = rewriter.create<ttmetal::CreateBufferOp>(
        op->getLoc(), memrefType, address.getInt(), size,
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
