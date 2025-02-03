// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"    
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include <mlir/IR/Iterators.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRTENSORLAYOUT
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

class TTIRGenericTensorTypeRewriter : public RewritePattern {
public:
  TTIRGenericTensorTypeRewriter(MLIRContext *ctx)
      : RewritePattern(ttir::GenericOp::getOperationName(), /*benefit=*/1, ctx) {}

  static GridAttr getOptimalGrid(MLIRContext *ctx,
                                 ArrayRef<int64_t> memref_shape,
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

  static void assignLocalLayout(Operation *op, PatternRewriter &rewriter, DeviceAttr &device) {
    // Output tensor
    llvm::errs() << "Assigning local layout for " << op->getName() << "\n";
    // assert(op->getResults().size() == 1 && "Only one result tensor is supported for now");
    if (op->getNumResults() > 0) {
      RankedTensorType result_type =
          mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
      auto result_encoding =
          mlir::dyn_cast_or_null<MetalLayoutAttr>(result_type.getEncoding());
      assert(result_encoding &&
             "Tensor type must have a MetalLayoutAttr encoding");
      auto optimal_output_grid = getOptimalGrid(
          op->getContext(), result_encoding.getMemref().getShape(),
          device.getWorkerGrid().getShape());
      optimal_output_grid.dump();

      auto new_result_encoding =
          MetalLayoutAttr::get(op->getContext(), result_type,
                               result_encoding.getMemorySpace(),
                               optimal_output_grid)
              .withElementType(op->getContext(),
                               result_encoding.getMemref().getElementType());

      auto new_tensor_type = RankedTensorType::get(result_type.getShape(),
                                                   result_type.getElementType(),
                                                   new_result_encoding);
      op->getResult(0).setType(new_tensor_type);
    }
  
    // // Input tensors
    // for (auto operand : op->getOperands()) {
    //   RankedTensorType operand_type =
    //       mlir::dyn_cast<RankedTensorType>(operand.getType());
    //   auto operand_encoding =
    //       mlir::dyn_cast_or_null<MetalLayoutAttr>(operand_type.getEncoding());
    //   assert(operand_encoding &&
    //          "Tensor type must have a MetalLayoutAttr encoding");
    //   auto optimal_input_grid = getOptimalGrid(
    //       op->getContext(), operand_encoding.getMemref().getShape(), device.getWorkerGrid().getShape());
    //   auto new_operand_encoding = MetalLayoutAttr::get(op->getContext(), operand_type, 
    //                                                    operand_encoding.getMemorySpace(), optimal_input_grid).withElementType(op->getContext(), operand_encoding.getMemref().getElementType());
    //   auto new_tensor_type = RankedTensorType::get(operand_type.getShape(),
    //                                                operand_type.getElementType(),
    //                                                new_operand_encoding);
    //   operand.setType(new_tensor_type);
    // }
  }

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    // op = mlir::dyn_cast<ttir::GenericOp>(op);
    auto device = getCurrentScopeDevice(op);
    assert(device && "Device not found");
    assignLocalLayout(op, rewriter, device);
    return success();
  }

    // const TypeConverter *converter; 
};


class TTIRTensorLayout
    : public impl::TTIRTensorLayoutBase<TTIRTensorLayout> {

  using impl::TTIRTensorLayoutBase<TTIRTensorLayout>::TTIRTensorLayoutBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRGenericTensorTypeRewriter>(&getContext());
    GreedyRewriteConfig config = GreedyRewriteConfig();
    config.maxNumRewrites = 1;
    config.maxIterations = 1;
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet, config))) {
      signalPassFailure();
      return;
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
  }
};

} // namespace mlir::tt::ttir