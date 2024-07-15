#include "llvm/ADT/ScopeExit.h"
#include <algorithm>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir-c/IR.h>
#include <mlir/Support/LogicalResult.h>

#include "../PassDetail.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttmlir/Conversion/TTNNToEmitC/TTNNToEmitC.h"
#include "ttmlir/Dialect/TT/IR/TTOpsDialect.h.inc"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include <iostream>

using namespace mlir;
using namespace mlir::emitc;
using namespace mlir::tt;
using namespace mlir::tt::ttnn;

class TTNNToEmitCTypeConverter : public TypeConverter {
public:
  TTNNToEmitCTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](tt::DeviceType type) -> Type {
      return emitc::OpaqueType::get(ctx, "ttnn::Device");
    });
    addConversion([ctx](mlir::TensorType type) -> Type {
      return emitc::OpaqueType::get(ctx, "ttnn::Tensor");
    });
  }
};

template <typename SrcOp, typename Adaptor = typename SrcOp::Adaptor>
class DefaultOpConversionPattern : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

public:
  DefaultOpConversionPattern(MLIRContext *ctx)
      : OpConversionPattern<SrcOp>(ctx) {}

  DefaultOpConversionPattern(const TypeConverter &typeConverter,
                             MLIRContext *context, bool convertTypes = true,
                             PatternBenefit benefit = 1)
      : OpConversionPattern<SrcOp>(typeConverter, context, benefit) {
    this->convertTypes = convertTypes;
  }

  std::string getOpName(SrcOp op) const {
    auto name = op.getOperationName();
    if (name.starts_with("ttnn.")) {
      return "ttnn::" + name.drop_front(5).str();
    }
    return "ttnn::" + name.str();
  }

  LogicalResult
  matchAndRewrite(SrcOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Attribute, 4> templateArguments;

    auto operands = adaptor.getOperands();
    (void)operands;

    auto sz = srcOp->getResultTypes().size();
    if (sz > 0 && convertTypes) {
      auto resultTy = cast<emitc::OpaqueType>(
          this->getTypeConverter()->convertType(srcOp->getResult(0).getType()));
      rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
          srcOp, resultTy, getOpName(srcOp), nullptr, nullptr,
          adaptor.getOperands());
    } else {
      rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
          srcOp, srcOp->getResultTypes(), getOpName(srcOp), nullptr, nullptr,
          adaptor.getOperands());
    }

    return success();
  }

private:
  bool convertTypes = false;
};

void populateTTNNToEmitCPatterns(mlir::MLIRContext *ctx,
                                 mlir::RewritePatternSet &patterns,
                                 TypeConverter &typeConverter) {
  // Device ops
  //
  patterns.add<DefaultOpConversionPattern<mlir::tt::ttnn::OpenDeviceOp>>(
      typeConverter, ctx, true);
  patterns.add<DefaultOpConversionPattern<mlir::tt::ttnn::CloseDeviceOp>>(
      typeConverter, ctx);

  // Memory ops
  //
  patterns.add<DefaultOpConversionPattern<mlir::tt::ttnn::ToMemoryConfigOp>>(
      typeConverter, ctx);

  // Tensor ops
  //
  patterns.add<DefaultOpConversionPattern<mlir::tt::ttnn::FullOp>>(
      typeConverter, ctx);

  // Math ops
  //
  patterns.add<DefaultOpConversionPattern<mlir::tt::ttnn::AddOp>>(typeConverter,
                                                                  ctx);
  patterns.add<DefaultOpConversionPattern<mlir::tt::ttnn::MultiplyOp>>(
      typeConverter, ctx);
}

namespace {

struct ConvertTTNNToEmitCPass
    : public mlir::tt::ttnn::ConvertTTNNToEmitCBase<ConvertTTNNToEmitCPass> {
  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());

    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalDialect<emitc::EmitCDialect>();
    target.addIllegalDialect<mlir::tt::ttnn::TTNNDialect>();

    // Add header imports to front of module
    //
    {
      auto module = getOperation();
      OpBuilder builder(module);

      builder.create<emitc::IncludeOp>(module.getLoc(), "ttnn/device.h",
                                       /*isStandard=*/false);
      builder.create<emitc::IncludeOp>(
          module.getLoc(), "ttnn/operations/eltwise/binary/binary.hpp",
          /*isStandard=*/false);
      builder.create<emitc::IncludeOp>(
          module.getLoc(), "ttnn/operations/core.hpp", /*isStandard=*/false);
      builder.create<emitc::IncludeOp>(module.getLoc(),
                                       "ttnn/operations/creation.hpp",
                                       /*isStandard=*/false);
      builder.create<emitc::IncludeOp>(
          module.getLoc(),
          "ttnn/operations/reduction/generic/generic_reductions.hpp",
          /*isStandard=*/false);
    }

    // TTNN -> EmitC
    //
    {
      TTNNToEmitCTypeConverter typeConverter(&getContext());
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

      // TTNN -> EmitC patterns
      //
      populateTTNNToEmitCPatterns(&getContext(), patterns, typeConverter);

      // Apply conversion
      //
      if (failed(applyFullConversion(getOperation(), target,
                                     std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }
  };
};

} // namespace

namespace mlir::tt::ttnn {

std::unique_ptr<OperationPass<func::FuncOp>> createConvertTTNNToEmitCPass() {
  return std::make_unique<ConvertTTNNToEmitCPass>();
}

LogicalResult emitTTNNAsCpp(ModuleOp origOp, llvm::raw_ostream &os) {
  ModuleOp op = cast<ModuleOp>(origOp->clone());
  auto cleanupDispatchClone = llvm::make_scope_exit([&op] { op->erase(); });

  auto pm = PassManager::on<ModuleOp>(op.getContext());
  pm.addPass(createConvertTTNNToEmitCPass());

  if (pm.run(op).failed()) {
    return failure();
  }

  if (mlir::emitc::translateToCpp(op, os).failed()) {
    return failure();
  }

  return success();
}

} // namespace mlir::tt::ttnn
