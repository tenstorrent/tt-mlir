// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_TTNNMERGELAYOUTSWITHFUNCARGS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

// Update the input layouts of a function
// This rewriter checks for to_layout_ops on function arguments
// and updates the function input layouts accordingly
namespace {
class MergeLayoutsWithFuncArgsPattern
    : public OpRewritePattern<mlir::func::FuncOp> {
public:
  MergeLayoutsWithFuncArgsPattern(MLIRContext *ctx)
      : OpRewritePattern<mlir::func::FuncOp>(ctx) {}

  LogicalResult matchAndRewrite(mlir::func::FuncOp funcOp,
                                PatternRewriter &rewriter) const final {
    bool modified = false;
    rewriter.startOpModification(funcOp);
    modified |= rewriteInput(funcOp, rewriter);
    if (!modified) {
      rewriter.cancelOpModification(funcOp);
      return failure();
    }
    rewriter.finalizeOpModification(funcOp);
    return success();
  }

private:
  bool rewriteInput(mlir::func::FuncOp funcOp,
                    PatternRewriter &rewriter) const {
    // Skip const eval and cpu hoisted functions
    if (ttmlir::utils::isConstEvalFunc(funcOp) || funcOp.isDeclaration()) {
      return false;
    }

    Block &entryBlock = funcOp.getBody().front();
    SmallVector<Type> inputTypes;
    bool modified = false;

    for (BlockArgument &arg : entryBlock.getArguments()) {
      TTNNLayoutAttr desiredLayout = nullptr;
      // Set to false if no uses
      bool allUsesAreSameToLayout = !arg.use_empty();

      // Check if all the uses of the argument are ToLayoutOps with the same
      // layout
      for (OpOperand &argUse : arg.getUses()) {
        if (auto toLayoutOp = mlir::dyn_cast<ToLayoutOp>(argUse.getOwner())) {
          auto layoutAttr =
              mlir::cast<TTNNLayoutAttr>(toLayoutOp.getType().getEncoding());

          if (!desiredLayout) {
            desiredLayout = layoutAttr;
          } else if (desiredLayout != layoutAttr) {
            allUsesAreSameToLayout = false;
            break;
          }
        } else {
          allUsesAreSameToLayout = false;
          break;
        }
      }

      if (!desiredLayout || !allUsesAreSameToLayout) {
        inputTypes.push_back(arg.getType());
        continue;
      }

      // If all uses are the same layout, we can use that layout,
      // and later fold all to_layout ops into function arguments
      // during canonicalization pass.
      RankedTensorType ty = mlir::cast<RankedTensorType>(arg.getType());
      RankedTensorType newType = RankedTensorType::get(
          ty.getShape(), desiredLayout.getScalarElementType(), desiredLayout);

      if (arg.getType() != newType) {
        inputTypes.push_back(newType);
        modified = true;
      } else {
        inputTypes.push_back(arg.getType());
      }
    }

    if (modified) {
      SmallVector<Type> outputTypes(funcOp.getResultTypes());
      FunctionType newFuncType =
          rewriter.getFunctionType(inputTypes, outputTypes);
      funcOp.setFunctionType(newFuncType);

      for (uint32_t i = 0; i < entryBlock.getNumArguments(); i++) {
        entryBlock.getArgument(i).setType(inputTypes[i]);
      }
    }

    return modified;
  }
};
} // namespace

namespace {
// This pass rewrites function arguments with subsequent to_layout ops
// into function arguments with the to_layout op layout.
// Later canonicalization pass will remove the to_layout ops.
class TTNNMergeLayoutsWithFuncArgsPass
    : public impl::TTNNMergeLayoutsWithFuncArgsBase<
          TTNNMergeLayoutsWithFuncArgsPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // Add the pattern to merge to_layout ops with function arguments
    patterns.add<MergeLayoutsWithFuncArgsPattern>(context);

    // Apply the patterns
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::tt::ttnn
