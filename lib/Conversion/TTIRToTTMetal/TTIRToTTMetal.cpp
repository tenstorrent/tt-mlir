// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTMetal/TTIRToTTMetal.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include <cstdint>
#include <utility>

namespace mlir::tt::ttmetal {

namespace {
class TTIRGenericRewriter : public OpConversionPattern<ttir::GenericOp> {
public:
  using OpConversionPattern<ttir::GenericOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::GenericOp op, ttir::GenericOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    llvm::SmallVector<Attribute> coreRanges(
        op.getThreads().size(),
        rewriter.getAttr<ttmetal::CoreRangeAttr>(op.getGrid()));
    rewriter.replaceOpWithNewOp<ttmetal::EnqueueProgramOp>(
        op, op.getInputs(), op.getOutputs(), op.getThreads(),
        rewriter.getArrayAttr(coreRanges), rewriter.getArrayAttr({}));
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
    auto device = lookupDevice(op);
    auto address = op->getAttr("address")
                       ? op->getAttrOfType<IntegerAttr>("address")
                       : rewriter.getI64IntegerAttr(
                             1000); // TODO(#1909): arbitrary default for now,
                                    // remove when allocate pass is implemented
    assert(op.getMemref().getType().getMemorySpace() &&
           "No memref memory space found, failing.");
    auto memrefType = op.getMemref().getType();
    auto size = device.getMemrefSizeBytes(memrefType, 0);
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

namespace {
class TTIRToLayoutRewriter : public OpConversionPattern<ttir::ToLayoutOp> {
public:
  using OpConversionPattern<ttir::ToLayoutOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ToLayoutOp op, ttir::ToLayoutOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value input = op.getInput();
    Value output = op.getOutput();
    MemRefType inputTy = mlir::cast<MemRefType>(input.getType());
    MemRefType outputTy = mlir::cast<MemRefType>(output.getType());
    tt::MemorySpaceAttr inputMemorySpace =
        mlir::dyn_cast_if_present<tt::MemorySpaceAttr>(
            inputTy.getMemorySpace());
    tt::MemorySpaceAttr outputMemorySpace =
        mlir::dyn_cast_if_present<tt::MemorySpaceAttr>(
            outputTy.getMemorySpace());
    bool inputMemorySpaceSet = inputMemorySpace != nullptr;
    bool outputMemorySpaceSet = outputMemorySpace != nullptr;
    assert((inputMemorySpaceSet != outputMemorySpaceSet) &&
           "expected either input or output to have memory space");

    // No memoryspace implicitly means host
    if (inputMemorySpace) {
      rewriter.replaceOpWithNewOp<ttmetal::EnqueueReadBufferOp>(op, input,
                                                                output);
    } else {
      rewriter.replaceOpWithNewOp<ttmetal::EnqueueWriteBufferOp>(op, input,
                                                                 output);
    }
    return success();
  }
};
} // namespace

} // namespace mlir::tt::ttmetal

namespace mlir::tt {

void populateTTIRToTTMetalPatterns(MLIRContext *ctx,
                                   RewritePatternSet &patterns,
                                   TypeConverter & /*typeConverter*/) {
  patterns.add<ttmetal::TTIRGenericRewriter, ttmetal::MemrefAllocRewriter,
               ttmetal::TTIRToLayoutRewriter>(ctx);
}

} // namespace mlir::tt
