// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitPy/EmitPyConversion.h"
#include "ttmlir/Conversion/TTNNToEmitPy/TTNNToEmitPy.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h"
#include "ttmlir/Dialect/TTCore/Transforms/Passes.h"

#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Pass/PassManager.h"
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_CONVERTTTNNTOEMITPY
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::ttnn

namespace {

class TTNNToEmitPyTypeConverter : public TypeConverter {
public:
  TTNNToEmitPyTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](tt::ttnn::DeviceType type) -> emitpy::OpaqueType {
      return emitpy::OpaqueType::get(ctx, "ttnn.Device");
    });
    addConversion([ctx](mlir::TensorType type) -> emitpy::OpaqueType {
      return emitpy::OpaqueType::get(ctx,
                                     ttnn_to_emitpy::TypeNameV<::ttnn::Tensor>);
    });
    addConversion([ctx](mlir::TupleType type) -> emitpy::OpaqueType {
      return emitpy::OpaqueType::get(
          ctx, ttnn_to_emitpy::TypeNameV<std::vector<::ttnn::Tensor>>);
    });
  }
};

// ttnn.to_torch converts a TTNN tensor to a torch tensor
// ttnn.from_torch converts a torch tensor to a TTNN tensor
// This pass should perform these conversions on function arguments and return
// values.
class TorchConversionRewriter : public OpRewritePattern<func::FuncOp> {
public:
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                PatternRewriter &rewriter) const override {

    auto torchTensorType =
        emitpy::OpaqueType::get(rewriter.getContext(), "torch.Tensor");

    auto ttnnTensorType = emitpy::OpaqueType::get(
        rewriter.getContext(), ttnn_to_emitpy::TypeNameV<::ttnn::Tensor>);

    // to_torch for each operand
    rewriter.setInsertionPointToStart(&funcOp.getBody().front());
    for (auto &arg : funcOp.getArguments()) {
      auto toTorchOp = rewriter.create<emitpy::CallOpaqueOp>(
          funcOp.getLoc(), torchTensorType, "ttnn.to_torch", arg, nullptr,
          nullptr);

      // Replace all uses of the original argument
      // except in the to_torch op itself, and for deallocation
      llvm::SmallVector<OpOperand *> usesToReplace;
      for (auto &use : arg.getUses()) {
        // skip the to_torch op itself
        if (use.getOwner() == toTorchOp) {
          continue;
        }

        usesToReplace.push_back(&use);
      }

      for (auto *use : usesToReplace) {
        use->set(toTorchOp.getResult(0));
      }
    }

    // CallOpaque to each op which contains "golden_function" in its name
    // should have its type modified so that it returns torch tensors
    funcOp.walk([&](emitpy::CallOpaqueOp callOp) {
      if (callOp.getCallee().str().find("golden_function") !=
          std::string::npos) {
        llvm::SmallVector<Type, 4> newResultTypes;
        for (auto resultType : callOp.getResultTypes()) {
          if (resultType == ttnnTensorType) {
            newResultTypes.push_back(torchTensorType);
          } else {
            newResultTypes.push_back(resultType);
          }
        }

        // create new CallOpaqueOp with modified result types
        auto newCallOp = rewriter.create<emitpy::CallOpaqueOp>(
            callOp.getLoc(), newResultTypes, callOp.getCallee(),
            callOp.getOperands(), callOp.getArgsAttr(),
            callOp.getKeywordArgsAttr());
        rewriter.replaceOp(callOp, newCallOp.getResults());
      }
    });

    // delete deallocation ops whenever the argument is torch tensor
    funcOp.walk([&](emitpy::CallOpaqueOp callOp) {
      if (callOp.getCallee().str().find("deallocate") != std::string::npos) {
        auto operandType = callOp.getOperand(0).getType();
        if (operandType == torchTensorType) {
          rewriter.eraseOp(callOp);
        }
      }
    });

    // from_torch for each return value
    llvm::SmallVector<func::ReturnOp> returnOps;
    for (auto retOp : funcOp.getBody().front().getOps<func::ReturnOp>()) {
      returnOps.push_back(retOp);
    }

    for (auto retOp : returnOps) {
      rewriter.setInsertionPoint(retOp);

      llvm::SmallVector<mlir::Value> newResults;
      for (auto retVal : retOp.getOperands()) {
        auto fromTorch = rewriter.create<emitpy::CallOpaqueOp>(
            funcOp.getLoc(), ttnnTensorType, "ttnn.from_torch", retVal, nullptr,
            nullptr);
        newResults.push_back(fromTorch.getResult(0));
      }

      rewriter.replaceOpWithNewOp<func::ReturnOp>(retOp, newResults);
    }

    return success();
  }
};

struct ConvertTTNNToEmitPyPass
    : public tt::ttnn::impl::ConvertTTNNToEmitPyBase<ConvertTTNNToEmitPyPass> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<emitpy::EmitPyDialect>();
    target.addIllegalDialect<tt::ttnn::TTNNDialect>();
    // mlir::ModuleOp is legal only if no attributes are present on it
    //
    target.addDynamicallyLegalOp<mlir::ModuleOp>(
        [&](mlir::ModuleOp op) { return op->getAttrs().empty(); });

    OpBuilder builder(module);

    if (module.getBodyRegion().empty()) {
      // Parent module is empty, nothing to do here
      //
      signalPassFailure();
    }

    // Set insertion point to start of first module child
    //
    builder.setInsertionPointToStart(module.getBody(0));

    // Include headers
    //
    builder.create<emitpy::ImportOp>(module->getLoc(), "ttnn", nullptr, nullptr,
                                     nullptr, nullptr);
    builder.create<emitpy::ImportOp>(module->getLoc(), "utils", nullptr,
                                     nullptr, nullptr, nullptr);

    // TTNN -> EmitPy
    //
    {
      TTNNToEmitPyTypeConverter typeConverter(&getContext());
      RewritePatternSet patterns(&getContext());

      // Func dialect handling
      //
      populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
          patterns, typeConverter);
      target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
        return typeConverter.isSignatureLegal(op.getFunctionType()) &&
               typeConverter.isLegal(&op.getBody());
      });
      populateReturnOpTypeConversionPattern(patterns, typeConverter);
      target.addDynamicallyLegalOp<func::ReturnOp>(
          [&](func::ReturnOp op) { return typeConverter.isLegal(op); });
      populateCallOpTypeConversionPattern(patterns, typeConverter);
      target.addDynamicallyLegalOp<func::CallOp>(
          [&](func::CallOp op) { return typeConverter.isLegal(op); });

      // TTNN -> EmitPy patterns
      //
      populateTTNNToEmitPyPatterns(&getContext(), patterns, typeConverter);

      // Apply full conversion
      //
      if (failed(applyFullConversion(getOperation(), target,
                                     std::move(patterns)))) {
        signalPassFailure();
        return;
      }
      // For hoisted functions, convert inputs to Torch tensors, and results
      // back to TTNN tensors before returning to the caller.
      TorchConversionRewriter torchRewriter(&getContext());

      module.walk([&](mlir::func::FuncOp funcOp) {
        if (funcOp.getSymName().starts_with("hoisted_")) {
          PatternRewriter rewriter(funcOp.getContext());
          if (failed(torchRewriter.matchAndRewrite(funcOp, rewriter))) {
            signalPassFailure();
            return;
          }
        }
      });
    }
  }
};

} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTNNToEmitPyPass() {
  return std::make_unique<ConvertTTNNToEmitPyPass>();
}

} // namespace mlir::tt
