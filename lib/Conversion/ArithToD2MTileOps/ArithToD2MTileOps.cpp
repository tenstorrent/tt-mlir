// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/ArithToD2MTileOps/ArithToD2MTileOps.h"

#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#include <array>

namespace mlir::tt::ttir {

namespace {
template <typename ArithOp, typename D2MTileOp>
class ArithOpRewriter : public OpConversionPattern<ArithOp> {
public:
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto operands = adaptor.getOperands();
    if (operands.size() == 1) {
      rewriter.replaceOpWithNewOp<D2MTileOp>(op, op.getResult().getType(),
                                             operands[0]);
      return success();
    } else if (operands.size() == 2) {
      rewriter.replaceOpWithNewOp<D2MTileOp>(op, op.getResult().getType(),
                                             operands[0], operands[1]);
      return success();
    }

    return failure();
  }
};
} // namespace

} // namespace mlir::tt::ttir

namespace mlir::tt {

void populateArithToD2MTileOpsPatterns(MLIRContext *ctx,
                                       RewritePatternSet &patterns,
                                       TypeConverter &typeConverter) {
  patterns.add<ttir::ArithOpRewriter<arith::AddFOp, ttir::TileAddOp>,
               ttir::ArithOpRewriter<arith::DivFOp, ttir::TileDivOp>>(
      typeConverter, ctx);
}

} // namespace mlir::tt
// ----------------------------------------------------------------------------
