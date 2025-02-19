// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Iterators.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRTENSORLAYOUT
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

GridAttr getOptimalGrid(MLIRContext *ctx, ArrayRef<int64_t> memref_shape,
                               ArrayRef<int64_t> device_grid_shape) {
  std::vector<int64_t> grid_shape;
  for (size_t i = 0; i < memref_shape.size(); i++) {
    int64_t dim = memref_shape[i];
    for (int grid = device_grid_shape[i]; grid > 1; grid--) {
      for (int pad = 0; pad < 9; pad++) {
        if ((dim + pad) % grid == 0 && pad < (dim + pad) / grid) {
          grid_shape.push_back(grid);
          break;
        }
      }
      if (grid_shape.size() == i + 1) {
        break;
      }
    }
    if (grid_shape.size() == i + 1) {
      continue;
    }
    grid_shape.push_back(1);
  }
  return GridAttr::get(ctx, grid_shape);
}

RankedTensorType getLocalLayout(Value tensor, PatternRewriter &rewriter,
                                       DeviceAttr &device) {
  // llvm::errs() << "Assigning local layout for " << op->getName() << "\n";

  RankedTensorType result_type =
      mlir::dyn_cast<RankedTensorType>(tensor.getType());
  auto result_encoding =
      mlir::dyn_cast_or_null<MetalLayoutAttr>(result_type.getEncoding());
  assert(result_encoding && "Tensor type must have a MetalLayoutAttr encoding");
  auto optimal_output_grid = getOptimalGrid(
      tensor.getContext(), result_encoding.getMemref().getShape(),
      device.getWorkerGrid().getShape());

  auto new_result_encoding =
      MetalLayoutAttr::get(tensor.getContext(), result_type,
                           result_encoding.getMemorySpace(),
                           optimal_output_grid)
          .withElementType(tensor.getContext(),
                           result_encoding.getMemref().getElementType());

  auto new_tensor_type =
      RankedTensorType::get(result_type.getShape(),
                            result_type.getElementType(), new_result_encoding);
  return new_tensor_type;
}

class TTIRGenericTensorLayoutRewriter : public RewritePattern {
public:
  TTIRGenericTensorLayoutRewriter(MLIRContext *ctx)
      : RewritePattern(ttir::GenericOp::getOperationName(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    auto device = getCurrentScopeDevice(op);
    assert(device && "Device not found");
    auto genericOp = mlir::cast<ttir::GenericOp>(op);

    // Update output tensor type
    assert(op->getResults().size() == 1 &&
           "Only one result tensor is supported for now");
    auto optimal_layout = getLocalLayout(op->getResult(0), rewriter, device);
    op->getResult(0).setType(optimal_layout);
    auto dpsOp = mlir::cast<DestinationStyleOpInterface>(op);
    assert(dpsOp.getNumDpsInits() == 1 &&
           "Only one result tensor is supported for now");
    dpsOp.getDpsInits()[0].setType(optimal_layout);

    // Update generic grid (match worker cores to output grid)
    genericOp.setGridAttr(mlir::cast<MetalLayoutAttr>(optimal_layout.getEncoding()).getGrid());    

    return failure(); // need some better way to exit cond. the rewriter than 
                      // always returning false!
  }
};

class TTIRMemrefLayoutRewriter : public RewritePattern {
public:
  TTIRMemrefLayoutRewriter(MLIRContext *ctx)
      : RewritePattern(ttir::GenericOp::getOperationName(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    auto genericOp = mlir::cast<ttir::GenericOp>(op);

    Block *genericBlock = &genericOp.getRegion().front();
    assert(genericBlock->getNumArguments() == genericOp->getNumOperands() &&
           "Number of block arguments should match the number of generic op "
           "operands");
    for (size_t i = 0; i < genericBlock->getNumArguments(); i++) {
      auto arg = genericBlock->getArgument(i);
      auto operand = mlir::cast<RankedTensorType>(genericOp->getOperand(i).getType());
      auto operand_encoding = mlir::cast<MetalLayoutAttr>(operand.getEncoding());
      arg.setType(operand_encoding.getMemref());
    }
    return failure(); // need some better way to exit cond. the rewriter than
                      // always returning false!
  }
};

class TTIRFuncOperandsTensorLayoutRewriter : public RewritePattern {
public:
  TTIRFuncOperandsTensorLayoutRewriter(MLIRContext *ctx)
      : RewritePattern(func::FuncOp::getOperationName(), /*benefit=*/1, ctx) {
  }
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto device = getCurrentScopeDevice(op);
    assert(device && "Device not found");

    func::FuncOp funcOp = mlir::cast<func::FuncOp>(op);

    llvm::errs() << "Operand Rewriter hit\n";
    rewriter.setInsertionPointToStart(&op->getRegion(0).front());;
    bool modified = false;
    for (Value operand : funcOp.getArguments()) {
      auto optimal_layout = getLocalLayout(operand, rewriter, device);
      auto tensor = mlir::cast<RankedTensorType>(operand.getType());
      auto encoding = mlir::cast<MetalLayoutAttr>(tensor.getEncoding());
      if (mlir::cast<MetalLayoutAttr>(optimal_layout.getEncoding()).getGrid().getShape() == encoding.getGrid().getShape()||
          (std::distance(operand.getUses().begin(), operand.getUses().end()) == 1 && mlir::isa<ttir::ToLayoutOp>(operand.getUses().begin().getUser()))) {
        continue;
      }
      modified = true;
      auto emptyOp = rewriter.create<tensor::EmptyOp>(op->getLoc(), optimal_layout.getShape(), optimal_layout.getElementType(), optimal_layout.getEncoding());
      auto toLayoutOp = rewriter.create<ttir::ToLayoutOp>(op->getLoc(), emptyOp.getType(), operand, emptyOp);
      rewriter.replaceAllUsesExcept(operand, toLayoutOp.getResult(), llvm::SmallPtrSet<Operation *, 2>{op, toLayoutOp});
    }
    return modified ? success() : failure();
  }
};

class TTIRFuncReturnTensorLayoutRewriter : public RewritePattern {
public:
  TTIRFuncReturnTensorLayoutRewriter(MLIRContext *ctx)
      : RewritePattern(func::ReturnOp::getOperationName(), /*benefit=*/1,
                       ctx) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (mlir::isa<ttir::ToLayoutOp>(op->getOperand(0).getDefiningOp())) {
      return failure();
    }
    auto returnTensorType = mlir::cast<RankedTensorType>(op->getOperand(0).getType());
    auto emptyOp = rewriter.create<tensor::EmptyOp>(op->getLoc(), returnTensorType.getShape(), returnTensorType.getElementType(), returnTensorType.getEncoding());
    auto toLayoutOp = rewriter.create<ttir::ToLayoutOp>(op->getLoc(), returnTensorType, op->getOperand(0), emptyOp);
    op->setOperand(0, toLayoutOp.getResult());
    return success();
  }
};

class TTIRTensorLayout : public impl::TTIRTensorLayoutBase<TTIRTensorLayout> {

  using impl::TTIRTensorLayoutBase<TTIRTensorLayout>::TTIRTensorLayoutBase;

  void runOnOperation() final {
    {
      RewritePatternSet patterns(&getContext());
      patterns.add<
          TTIRFuncReturnTensorLayoutRewriter>(
          &getContext());
      FrozenRewritePatternSet patternSet(std::move(patterns));
      if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
        signalPassFailure();
        return;
      }
    }

    {
      RewritePatternSet patterns(&getContext());
      patterns.add<TTIRGenericTensorLayoutRewriter>(&getContext());
      GreedyRewriteConfig config = GreedyRewriteConfig();
      FrozenRewritePatternSet patternSet(std::move(patterns));
      if (failed(
              applyPatternsAndFoldGreedily(getOperation(), patternSet, config))) {
        signalPassFailure();
        return;
      }
    }

    {
      RewritePatternSet patterns(&getContext());
      patterns.add<TTIRFuncOperandsTensorLayoutRewriter>(&getContext());
      GreedyRewriteConfig config = GreedyRewriteConfig();
      config.strictMode = GreedyRewriteStrictness::ExistingOps;
      FrozenRewritePatternSet patternSet(std::move(patterns));
      if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet,
                                              config))) {
        signalPassFailure();
        return;
      }
    }

    {
      RewritePatternSet patterns(&getContext());
      patterns.add<TTIRMemrefLayoutRewriter>(&getContext());
      FrozenRewritePatternSet patternSet(std::move(patterns));
      if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
        signalPassFailure();
        return;
      }
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
  }
};

} // namespace mlir::tt::ttir