// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTIRFusion/TTIRToTTIRFusion.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/BuiltinAttributes.h>

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt {

struct StaticKVCacheUpdateScattterConversionPattern
    : public OpRewritePattern<ttir::ScatterOp> {
  using OpRewritePattern<ttir::ScatterOp>::OpRewritePattern;

  std::vector<mlir::Value> getRoots(mlir::Value value) const {
    if (value.getDefiningOp() == nullptr) {
      return {value};
    }

    std::vector<mlir::Operation *> toVisit;
    toVisit.push_back(value.getDefiningOp());

    std::vector<mlir::Value> roots;
    while (!toVisit.empty()) {
      mlir::Operation *op = toVisit.back();
      toVisit.pop_back();

      if (op->getNumOperands() == 0) {
        for (mlir::Value result : op->getResults()) {
          roots.push_back(result);
        }
      } else {
        for (mlir::Value operand : op->getOperands()) {
          if (operand.getDefiningOp() == nullptr) {
            roots.push_back(operand);
          } else {
            toVisit.push_back(operand.getDefiningOp());
          }
        }
      }
    }

    return roots;
  }

  LogicalResult constantDetermined(mlir::Value value) const {
    // If this value has no defining operation it must have been a graph input
    auto roots = getRoots(value);

    for (mlir::Value root : roots) {
      if (!root.getDefiningOp()) {
        return failure();
      }
    }

    // In the recursive case, this value is the result of an operation,
    // and all inputs of that operation are the result of operations done
    // on constants, so, this is a constant.
    //
    // In the base case, if this is an op, but it has no operands, then
    // it must be a creation op, and what it creates is known at compile time.
    return success();
  }

  LogicalResult matchAndRewrite(ttir::ScatterOp op,
                                PatternRewriter &rewriter) const final {

    if (!mlir::isa<ttir::ConcatOp>(op.getScatterIndices().getDefiningOp())) {
      return failure();
    }

    ttir::ConcatOp concat =
        mlir::cast<ttir::ConcatOp>(op.getScatterIndices().getDefiningOp());

    // There must be 4 inputs to this concat op
    if (concat.getInputs().size() != 4) {
      return failure();
    }

    // Input 0, 1, and 3 must be derived from constant roots only
    if (failed(constantDetermined(concat.getInputs()[0])) ||
        failed(constantDetermined(concat.getInputs()[1])) ||
        failed(constantDetermined(concat.getInputs()[3]))) {
      return failure();
    }

    // If this is a cache update then the cache position should be a runtime arg
    if (succeeded(constantDetermined(concat.getInputs()[2]))) {
      return failure();
    }

    // There should be only one runtime arg for cache position
    std::vector<mlir::Value> indicesRoots = getRoots(concat.getInputs()[2]);

    if (std::count_if(indicesRoots.begin(), indicesRoots.end(),
                      [=](const mlir::Value &value) {
                        return value.getDefiningOp() == nullptr;
                      }) != 1) {
      return failure();
    }

    mlir::Value cachePosition =
        *std::find_if(indicesRoots.begin(), indicesRoots.end(),
                      [=](const mlir::Value &value) {
                        return value.getDefiningOp() == nullptr;
                      });
    if (!mlir::isa<RankedTensorType>(cachePosition.getType())) {
      return failure();
    }

    RankedTensorType cachePositionType =
        mlir::cast<RankedTensorType>(cachePosition.getType());
    if (cachePositionType.getRank() != 1) {
      return failure();
    }

    if (elementTypeToDataType(cachePositionType.getElementType()) !=
        DataType::UInt32) {
      return failure();
    }

    RankedTensorType cacheType =
        mlir::cast<RankedTensorType>(op.getInput().getType());
    if (cacheType.getRank() != 4) {
      return failure();
    }

    RankedTensorType kvUpdateType =
        mlir::cast<RankedTensorType>(op.getUpdate().getType());
    if (kvUpdateType.getRank() != 4) {
      return failure();
    }

    // Whether this is a static cache update or fill, the cache poition tensor
    // must have its single dimensions size be equal to that of the kvUpdate
    // shape at dim -2
    if (cachePositionType.getShape()[0] !=
        kvUpdateType.getShape()[kvUpdateType.getRank() - 2]) {
      return failure();
    }

    // If the shape of the cache position is > 1 then this is a static cache
    // FILL
    if (cachePositionType.getShape()[0] > 1) {
      rewriter.replaceOpWithNewOp<ttir::FillCacheOp>(
          op, op.getType(), op.getInput(), op.getUpdate(),
          rewriter.getI32IntegerAttr(0), op.getOperandConstraintsAttr());
    } else {
      rewriter.replaceOpWithNewOp<ttir::UpdateCacheOp>(
          op, op.getType(), op.getInput(), op.getUpdate(), cachePosition,
          rewriter.getI32IntegerAttr(0), op.getOperandConstraintsAttr());
    }

    return success();
  }
};

void populateTTIRToTTIRFusionPatterns(MLIRContext *ctx,
                                      RewritePatternSet &patterns) {
  patterns.add<StaticKVCacheUpdateScattterConversionPattern>(ctx);
}

} // namespace mlir::tt
