// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRATTACHMETALLAYOUT
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

class TTIRAttachMetalLayoutRewriter : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const final {
    if (mlir::isa<GenericOp>(op.getOperation()->getParentOp())) {
      return failure();
    } // might not be necessary

    for (auto operand : op.getFunctionType().getInputs()) {
      if (isa<RankedTensorType>(operand)) {
        auto operandType = mlir::cast<RankedTensorType>(operand);
        if (operandType.getEncoding()) {
          return failure();
        }
      }
    }

    auto appendLayoutToType = [&](Type type) -> Type {
      auto operandType = mlir::cast<RankedTensorType>(type);
      auto layout = rewriter.getAttr<MetalLayoutAttr>(
          operandType, MemorySpace::DeviceL1, GridAttr());
      auto newTensorType = RankedTensorType::get(
          operandType.getShape(), operandType.getElementType(), layout);
      return newTensorType;
    };

    auto appendLayoutToValue = [&](Value operand) {
      if (isa<RankedTensorType>(operand.getType())) {
        auto operandType = mlir::cast<RankedTensorType>(operand.getType());
        auto layout = rewriter.getAttr<MetalLayoutAttr>(
            operandType, MemorySpace::DeviceL1, GridAttr());
        operand.setType(RankedTensorType::get(
            operandType.getShape(), operandType.getElementType(), layout));
      }
    };

    op.walk([&](Operation *op) {
      if (isa<func::FuncOp>(op)) {
        auto funcOp = mlir::cast<func::FuncOp>(op);
        std::vector inputs = std::vector<Type>();
        std::vector results = std::vector<Type>();
        for (auto operand : funcOp.getFunctionType().getInputs()) {
          if (isa<RankedTensorType>(operand)) {
            inputs.push_back(appendLayoutToType(operand));
          }
        }
        for (auto result : funcOp.getFunctionType().getResults()) {
          if (isa<RankedTensorType>(result)) {
            results.push_back(appendLayoutToType(result));
          }
        }
        mlir::FunctionType newFuncType =
            FunctionType::get(funcOp.getContext(), inputs, results);
        funcOp.setFunctionType(newFuncType);
        return;
      }
      for (auto operand : op->getOperands()) {
        appendLayoutToValue(operand);
      }
      for (auto result : op->getResults()) {
        appendLayoutToValue(result);
      }
    });

    return success();
  }
};

class TTIRAttachMetalLayout
    : public impl::TTIRAttachMetalLayoutBase<TTIRAttachMetalLayout> {

  using impl::TTIRAttachMetalLayoutBase<
      TTIRAttachMetalLayout>::TTIRAttachMetalLayoutBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRAttachMetalLayoutRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
      signalPassFailure();
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
  }
};

} // namespace mlir::tt::ttir
