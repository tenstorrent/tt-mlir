// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitPy/EmitPyConversion.h"
#include "ttmlir/Conversion/TTNNToEmitPy/TTNNToEmitPy.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPy.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyTypes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/FunctionTypes.h"

#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/MapVector.h"

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
    addConversion([ctx](tt::ttcore::DictType type) -> emitpy::DictType {
      return emitpy::DictType::get(ctx, Type(), Type());
    });
  }
};

struct ConvertTTNNToEmitPyPass
    : public tt::ttnn::impl::ConvertTTNNToEmitPyBase<ConvertTTNNToEmitPyPass> {

  using tt::ttnn::impl::ConvertTTNNToEmitPyBase<
      ConvertTTNNToEmitPyPass>::ConvertTTNNToEmitPyBase;

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

    // If we are in the module-export path (i.e., `target-module=true`),
    // const-eval functions must also take `device` as an explicit argument so
    // they can avoid materializing `ttnn.get_device` in the function body.
    //
    if (this->targetModule) {
      targetModuleConversion(module);
    }

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
    }

    // Lower ImportedDeclaration func declarations to emitpy.import ops.
    // These are private declarations created by TTNNFileSplit to represent
    // functions defined in other files.
    lowerImportedDeclarations(module);
  }

private:
  // Converts ImportedDeclaration func ops into emitpy.import ops within each
  // file scope. Groups declarations by source file to produce a single import
  // statement per source file.
  void lowerImportedDeclarations(ModuleOp moduleOp) {
    moduleOp.walk([&](emitpy::FileOp fileOp) {
      // Group imported declarations by source file.
      llvm::MapVector<StringRef, SmallVector<func::FuncOp>> importsByFile;
      for (auto funcOp : fileOp.getOps<func::FuncOp>()) {
        if (!ttmlir::utils::isImportedDeclarationFunc(funcOp)) {
          continue;
        }
        auto sourceFile = ttmlir::utils::getImportedFrom(funcOp);
        assert(sourceFile && "ImportedDeclaration missing tt.imported_from");
        importsByFile[*sourceFile].push_back(funcOp);
      }

      // Create an emitpy.import op for each source file, inserted before
      // the first imported declaration in the file.
      OpBuilder builder(&getContext());
      for (auto &[sourceFile, funcOps] : importsByFile) {
        builder.setInsertionPoint(funcOps.front());

        SmallVector<StringRef> memberNames;
        for (auto funcOp : funcOps) {
          memberNames.push_back(funcOp.getSymName());
        }

        auto membersAttr = builder.getStrArrayAttr(memberNames);
        // member_aliases must match size of members_to_import; use empty
        // strings to indicate no alias.
        SmallVector<StringRef> emptyAliases(memberNames.size(), "");
        auto aliasesAttr = builder.getStrArrayAttr(emptyAliases);

        builder.create<emitpy::ImportOp>(funcOps.front()->getLoc(), sourceFile,
                                         /*module_alias=*/nullptr, membersAttr,
                                         aliasesAttr,
                                         /*import_all=*/nullptr);
      }
    });
  }

  // This function is used to convert the const-eval functions to accept a
  // device argument. This is duplicated from
  // `lib/Dialect/TTNN/Transforms/Passes.cpp@TTNNPrepareModuleForExport` because
  // `ttcore.load_cached` op expects const-eval function without device
  // argument. Issue: https://github.com/tenstorrent/tt-mlir/issues/6746
  void targetModuleConversion(ModuleOp moduleOp) {
    IRRewriter rewriter(&getContext());
    mlir::tt::ttnn::DeviceType deviceType =
        mlir::tt::ttnn::DeviceType::get(&getContext());

    moduleOp.walk([&](func::FuncOp funcOp) {
      if (!ttmlir::utils::isConstEvalFunc(funcOp) || funcOp.isExternal()) {
        return;
      }

      // Add device argument to the const-eval function signature.
      //
      auto originalFuncType = funcOp.getFunctionType();
      SmallVector<Type> newInputTypes(originalFuncType.getInputs().begin(),
                                      originalFuncType.getInputs().end());
      newInputTypes.push_back(deviceType);
      auto newFuncType = FunctionType::get(&getContext(), newInputTypes,
                                           originalFuncType.getResults());

      funcOp.setFunctionType(newFuncType);

      Block &entryBlock = funcOp.getBody().front();
      BlockArgument deviceArg =
          entryBlock.addArgument(deviceType, funcOp.getLoc());
      funcOp.setArgAttr(newInputTypes.size() - 1, ttnn_to_emitpy::kNameAttr,
                        rewriter.getStringAttr("device"));

      // Replace all GetDeviceOp operations with the new device argument.
      //
      SmallVector<mlir::tt::ttnn::GetDeviceOp> getDeviceOps;
      funcOp.walk(
          [&](mlir::tt::ttnn::GetDeviceOp op) { getDeviceOps.push_back(op); });
      for (auto op : getDeviceOps) {
        rewriter.replaceOp(op, deviceArg);
      }
    });
  }
};

} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTNNToEmitPyPass() {
  return std::make_unique<ConvertTTNNToEmitPyPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTTNNToEmitPyPass(const ConvertTTNNToEmitPyOptions &options) {
  return std::make_unique<ConvertTTNNToEmitPyPass>(options);
}

} // namespace mlir::tt
