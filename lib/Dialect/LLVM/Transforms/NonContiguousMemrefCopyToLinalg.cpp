// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "ttmlir/Dialect/LLVM/Transforms/Passes.h"
#include <llvm/Support/Casting.h>

namespace mlir::tt::llvm_util {
#define GEN_PASS_DEF_NONCONTIGUOUSMEMREFCOPYTOLINALGPASS
#include "ttmlir/Dialect/LLVM/Transforms/Passes.h.inc"

// Pattern to convert `memref.copy` to `linalg.generic` using MLIR's
// builtin `makeMemRefCopyOp`.
struct MemrefCopyToLinalgPattern : public OpConversionPattern<memref::CopyOp> {
  using OpConversionPattern<memref::CopyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::CopyOp copyOp, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = copyOp.getLoc();
    Value src = copyOp.getSource();
    Value dst = copyOp.getTarget();

    auto copyGeneric = mlir::linalg::makeMemRefCopyOp(rewriter, loc, src, dst);
    rewriter.replaceOp(copyOp, copyGeneric->getResults());
    return success();
  }
};

// During FinalizeMemRefToLLVMConversion pass, `memref.copy` gets lowered to
// `llvm.memcpy` only if the operands are non-empty and have an identity layout
// or are contiguous with an arbitrary offset. Otherwise, it gets lowered to
// invocation of `memrefCopy` function, which lives in the
// `libmlir_c_runner_utils.so` dylib.
//
// Since we want to avoid linking the MLIR runtime dylibs, this pass ensures
// that all non-trivial `memref.copy` ops are lowered to `linalg.generic`, which
// can further be lowered to simple loops.
class NonContiguousMemrefCopyToLinalg
    : public impl::NonContiguousMemrefCopyToLinalgPassBase<
          NonContiguousMemrefCopyToLinalg> {
  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<linalg::LinalgDialect, memref::MemRefDialect>();

    auto isContiguousMemrefType = [&](BaseMemRefType type) {
      auto memrefType = dyn_cast<mlir::MemRefType>(type);
      return memrefType &&
             (memrefType.getLayout().isIdentity() ||
              (memrefType.hasStaticShape() && memrefType.getNumElements() > 0 &&
               memref::isStaticShapeAndContiguousRowMajor(memrefType)));
    };

    // Dynamically mark memref.copy as illegal unless it operates on contiguous
    // memrefs.
    target.addDynamicallyLegalOp<memref::CopyOp>(
        [isContiguousMemrefType](memref::CopyOp op) {
          return isContiguousMemrefType(op.getSource().getType()) &&
                 isContiguousMemrefType(op.getTarget().getType());
        });

    RewritePatternSet patterns(&getContext());
    patterns.add<MemrefCopyToLinalgPattern>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace mlir::tt::llvm_util
