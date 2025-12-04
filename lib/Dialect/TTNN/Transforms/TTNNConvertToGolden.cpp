// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNTraits.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNCONVERTTOGOLDEN
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

// Helper to check if an operation supports golden function
static bool supportsGoldenFunction(Operation *op) {
  return !op->hasTrait<NoGoldenFunctionTrait>();
}

// Helper to check if an operation is a TTNN dialect op (excluding
// from_torch/to_torch)
static bool isTTNNOp(Operation *op) {
  if (!op->getDialect()) {
    return false;
  }
  if (op->getDialect()->getNamespace() != "ttnn") {
    return false;
  }
  // Exclude from_torch and to_torch operations
  if (isa<FromTorchOp, ToTorchOp>(op)) {
    return false;
  }
  return true;
}

// Pattern to convert individual TTNN operations to golden mode
class ConvertOpToGoldenPattern : public RewritePattern {
public:
  ConvertOpToGoldenPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Only convert TTNN dialect ops
    if (!isTTNNOp(op)) {
      return failure();
    }

    // Skip ops that don't support golden function
    if (!supportsGoldenFunction(op)) {
      return failure();
    }

    // Skip if already has use_golden attribute
    if (op->hasAttr("use_golden")) {
      return failure();
    }

    // Check if all operands come from torch tensor sources
    // (to_torch ops or other ops with use_golden)
    for (Value operand : op->getOperands()) {
      if (!isTorchTensorValue(operand)) {
        return failure();
      }
    }

    // Add use_golden attribute to the op
    rewriter.modifyOpInPlace(
        op, [&]() { op->setAttr("use_golden", rewriter.getUnitAttr()); });

    return success();
  }
};

} // namespace

class TTNNConvertToGolden
    : public impl::TTNNConvertToGoldenBase<TTNNConvertToGolden> {
public:
  using impl::TTNNConvertToGoldenBase<
      TTNNConvertToGolden>::TTNNConvertToGoldenBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    IRRewriter rewriter(ctx);

    // Process each function
    module->walk([&](func::FuncOp func) {
      if (func.isDeclaration()) {
        return;
      }

      // Step 1: Insert to_torch ops for function arguments
      insertInputConversions(func, rewriter);

      // Step 2: Apply patterns to convert ops to golden mode
      // (This happens after input conversions so operands come from to_torch)
      RewritePatternSet patterns(ctx);
      patterns.add<ConvertOpToGoldenPattern>(ctx);

      if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
        signalPassFailure();
        return;
      }

      // Step 3: Insert from_torch ops for function returns
      insertOutputConversions(func, rewriter);
    });
  }

private:
  // Insert to_torch ops at the beginning of the function for tensor arguments
  void insertInputConversions(func::FuncOp func, IRRewriter &rewriter) {
    if (func.getBody().empty()) {
      return;
    }

    Block &entryBlock = func.getBody().front();
    rewriter.setInsertionPointToStart(&entryBlock);

    for (BlockArgument arg : entryBlock.getArguments()) {
      auto tensorType = mlir::dyn_cast<RankedTensorType>(arg.getType());
      if (!tensorType) {
        continue;
      }

      // Insert to_torch op - result type is the same as input
      // The to_torch op marks the transition to "torch tensor" domain
      auto toTorchOp =
          rewriter.create<ToTorchOp>(func.getLoc(), tensorType, arg);

      // Replace all uses of the argument (except in the to_torch op itself)
      arg.replaceAllUsesExcept(toTorchOp.getResult(), toTorchOp);
    }
  }

  // Insert from_torch ops before return statements
  void insertOutputConversions(func::FuncOp func, IRRewriter &rewriter) {
    // Find all return operations
    SmallVector<func::ReturnOp> returnOps;
    func.walk([&](func::ReturnOp returnOp) { returnOps.push_back(returnOp); });

    for (func::ReturnOp returnOp : returnOps) {
      rewriter.setInsertionPoint(returnOp);

      SmallVector<Value> newOperands;
      bool changed = false;

      for (auto [operandIdx, operand] :
           llvm::enumerate(returnOp.getOperands())) {
        auto tensorType = mlir::dyn_cast<RankedTensorType>(operand.getType());
        if (!tensorType) {
          newOperands.push_back(operand);
          continue;
        }

        // Check if operand comes from a torch tensor source
        if (!isTorchTensorValue(operand)) {
          newOperands.push_back(operand);
          continue;
        }

        Type expectedResultType = func.getResultTypes()[operandIdx];

        // Insert from_torch op to convert back from torch tensor domain
        // TODO(dmilinkovic): Device argument is required when the output layout
        // is on device.
        auto fromTorchOp = rewriter.create<FromTorchOp>(
            returnOp.getLoc(), expectedResultType, operand,
            /*device=*/Value(), /*layout=*/nullptr, /*dtype=*/nullptr,
            /*memoryConfig=*/nullptr);

        newOperands.push_back(fromTorchOp.getResult());
        changed = true;
      }

      if (changed) {
        rewriter.replaceOpWithNewOp<func::ReturnOp>(returnOp, newOperands);
      }
    }
  }
};

} // namespace mlir::tt::ttnn
