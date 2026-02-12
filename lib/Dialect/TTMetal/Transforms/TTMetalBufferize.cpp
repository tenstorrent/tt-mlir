// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"
#include "ttmlir/Dialect/TTMetal/Transforms/Passes.h"

namespace mlir::tt::ttmetal {

#define GEN_PASS_DEF_TTMETALBUFFERIZEPASS
#include "ttmlir/Dialect/TTMetal/Transforms/Passes.h.inc"

namespace {

struct CreateGlobalSemaphoreBufferize
    : public OpRewritePattern<CreateGlobalSemaphoreOp> {
  using OpRewritePattern<CreateGlobalSemaphoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CreateGlobalSemaphoreOp op,
                                PatternRewriter &rewriter) const override {
    // Get the initial_value attribute.
    auto initialValueAttr = op.getInitialValueAttr();
    int32_t initialValue = initialValueAttr.getValue().getSExtValue();

    // Get the result type, which should be a TTKernel_GlobalSemaphoreType and
    // convert it to MemRefType.
    auto globalSemaphoreType =
        mlir::cast<tt::ttkernel::GlobalSemaphoreType>(op.getResult().getType());
    auto memRefType = MemRefType::get({1}, rewriter.getI32Type());

    // Allocate memory for the global semaphore.
    Location loc = op.getLoc();
    Value alloc = rewriter.create<memref::AllocOp>(loc, memRefType);

    // Store the initial value into the allocated memory.
    Value initialValueConst =
        rewriter.create<arith::ConstantOp>(loc, initialValueAttr);
    rewriter.create<memref::StoreOp>(loc, initialValueConst, alloc,
                                     ValueRange{});

    // Replace the original op with the allocated memref.
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

struct TTMetalBufferizePass
    : public impl::TTMetalBufferizePassBase<TTMetalBufferizePass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<memref::MemRefDialect, arith::ArithDialect>();
    target.addIllegalOp<CreateGlobalSemaphoreOp>();

    RewritePatternSet patterns(context);
    patterns.add<CreateGlobalSemaphoreBufferize>(context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::tt::ttmetal
