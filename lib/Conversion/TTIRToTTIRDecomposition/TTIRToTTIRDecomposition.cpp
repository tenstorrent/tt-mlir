// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTIRDecomposition/TTIRToTTIRDecomposition.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include <llvm/Support/raw_ostream.h>
#include <mlir/Support/LLVM.h>

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt {

// Decompose IndexOp into SliceOp
//
// This transformation adjusts IndexOp attributes so that `begin`, `end`, and
// `step` become arrays, where each array element corresponds to a dimension of
// the input tensor. For dimensions other than the sliced dimension, default
// values are used.
//
struct IndexToSliceConversionPattern
    : public OpConversionPattern<ttir::IndexOp> {
  using OpConversionPattern<ttir::IndexOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::IndexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inputType =
        ::mlir::dyn_cast<mlir::RankedTensorType>(adaptor.getInput().getType());
    if (!inputType || !inputType.hasRank())
      return failure();

    int64_t rank = inputType.getRank();
    llvm::SmallVector<mlir::Attribute, 4> begins, ends, steps;

    for (int64_t i = 0; i < rank; ++i) {
      if (i == op.getDim()) {
        begins.push_back(rewriter.getI32IntegerAttr(adaptor.getBegin()));
        ends.push_back(rewriter.getI32IntegerAttr(adaptor.getEnd()));
        steps.push_back(rewriter.getI32IntegerAttr(adaptor.getStep()));
      } else {
        begins.push_back(rewriter.getI32IntegerAttr(0));
        ends.push_back(rewriter.getI32IntegerAttr(inputType.getDimSize(i)));
        steps.push_back(rewriter.getI32IntegerAttr(1));
      }
    }

    auto newOp = rewriter.create<ttir::SliceOp>(
        op.getLoc(), op.getType(), adaptor.getInput(), adaptor.getOutput(),
        rewriter.getArrayAttr(begins), rewriter.getArrayAttr(ends),
        rewriter.getArrayAttr(steps), adaptor.getOperandConstraints());

    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

void populateTTIRToTTIRDecompositionPatterns(MLIRContext *ctx,
                                             RewritePatternSet &patterns,
                                             TypeConverter &typeConverter) {
  patterns.add<IndexToSliceConversionPattern>(typeConverter, ctx);
}

} // namespace mlir::tt
