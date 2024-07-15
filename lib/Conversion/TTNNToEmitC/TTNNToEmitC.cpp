#include "llvm/ADT/ScopeExit.h"
#include <algorithm>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
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

// class TTNNToEmitCTypeRewriter : public ConversionPattern {
// public:
//   TTNNToEmitCTypeRewriter(const TypeConverter &converter, MLIRContext *ctx)
//       : ConversionPattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx),
//         converter(&converter) {}

//   template <typename ValueRange>
//   bool convertTypes(ValueRange valueRange, SmallVector<Type> &newTypes) const
//   {
//     bool updated = false;
//     auto result = converter->convertTypes(valueRange.getTypes(), newTypes);
//     std::cout << "VESELO" << std::endl;
//     llvm::errs() << "ZA SVAKI SLUCAJ" << "\n";
//     if (result.failed()) {
//       return false;
//     }
//     for (auto [operand, newType] : llvm::zip(valueRange, newTypes)) {
//       if (operand.getType() == newType) {
//         continue;
//       }
//       operand.setType(newType);
//       updated = true;
//     }
//     return updated;
//   }

//   bool convertFuncType(Operation *op, PatternRewriter &rewriter) const {
//     std::cout << "VESELO2" << std::endl;
//     llvm::errs() << "ZA SVAKI SLUCAJ2" << "\n";
//     auto funcOp = dyn_cast<func::FuncOp>(op);
//     if (not funcOp) {
//       return false;
//     }
//     SmallVector<Type> inputTypes(funcOp.getArgumentTypes());
//     SmallVector<Type> outputTypes(funcOp.getResultTypes());
//     for (Type &ty : inputTypes) {
//       ty = converter->convertType(ty);
//     }
//     for (Type &ty : outputTypes) {
//       ty = converter->convertType(ty);
//     }
//     auto newType = rewriter.getType<FunctionType>(inputTypes, outputTypes);
//     if (funcOp.getFunctionType() == newType) {
//       return false;
//     }
//     funcOp.setFunctionType(newType);
//     return true;
//   }

//   LogicalResult
//   matchAndRewrite(Operation *op, ArrayRef<Value> operands,
//                   ConversionPatternRewriter &rewriter) const override {
//     bool updated = false;
//     SmallVector<Type> operandss;
//     SmallVector<Type> results;
//     updated |= convertTypes(op->getOperands(), operandss);
//     updated |= convertTypes(op->getResults(), results);
//     updated |= convertFuncType(op, rewriter);
//     return updated ? success() : failure();
//   }

//   const TypeConverter *converter;
// };

// template <typename OpTy>
// class TTNNToEmitCOpaqueRewriter : public OpRewritePattern<OpTy> {
// public:
//   using OpRewritePattern<OpTy>::OpRewritePattern;

//   std::string getOpName(OpTy op) const {
//     auto name = op.getOperation()->getName().getStringRef();
//     if (name.starts_with("ttnn.")) {
//       name = name.drop_front(5);
//     }
//     return "ttnn::" + name.str();
//   }

//   LogicalResult matchAndRewrite(OpTy op,
//                                 PatternRewriter &rewriter) const final {
//     rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
//         op, op->getResultTypes(), getOpName(op), nullptr, nullptr,
//         op->getOperands());
//     return success();
//   }
// };

template <typename SrcOp, typename Adaptor = typename SrcOp::Adaptor>
class DefaultOpConversionPattern : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

public:
  DefaultOpConversionPattern(MLIRContext *ctx)
      : OpConversionPattern<SrcOp>(ctx) {}

  DefaultOpConversionPattern(const TypeConverter &typeConverter,
                             MLIRContext *context, bool convertTypes = true, PatternBenefit benefit = 1)
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

    // srcOp->dump();
    // auto r = srcOp->getResultTypes();
    // std::for_each(r.begin(), r.end(), [](Type t) { t.dump(); });

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
    // resultTy => srcOp->getResultTypes()

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
  // patterns.add<DefaultOpConversionPattern<mlir::tt::ttnn::OpenDeviceOp>>(typeConverter);
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


  // patterns.add<DefaultOpConversionPattern<mlir::func::ReturnOp>>(typeConverter, ctx);
}

namespace {

struct ConvertTTNNToEmitCPass
    : public mlir::tt::ttnn::ConvertTTNNToEmitCBase<ConvertTTNNToEmitCPass> {
  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());

    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalDialect<emitc::EmitCDialect>();

    target.addLegalOp<mlir::func::FuncOp>();
    target.addLegalOp<mlir::func::ReturnOp>();
    // target.addLegalDialect<mlir::tt::ttnn::TTNNDialect>();
    // target.addIllegalDialect<mlir::tt::TTDialect>();
    target.addIllegalDialect<mlir::tt::ttnn::TTNNDialect>();

    std::cout << "PRE PRINT" << std::endl;

    // auto x = getOperation();
    // auto r = x->getResultTypes();
    getOperation()->dump();
    // std::for_each(r.begin(), r.end(), [](Type t) { t.dump(); });
    std::cout
        << "=================================================================="
        << std::endl;

    TTNNToEmitCTypeConverter typeConverter(&getContext());
    RewritePatternSet patterns(&getContext());

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

    // populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
    // target.markUnknownOpDynamicallyLegal([&](Operation *op) {
    //   return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
    //          isLegalForBranchOpInterfaceTypeConversionPattern(op,
    //                                                           typeConverter) ||
    //          isLegalForReturnOpTypeConversionPattern(op, typeConverter);
    // });


    populateTTNNToEmitCPatterns(&getContext(), patterns, typeConverter);
    // FrozenRewritePatternSet patternSet(std::move(patterns));

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // {
    //   RewritePatternSet patterns(&getContext());
    //   populateTTNNToEmitCPatterns(&getContext(), patterns);
    //   if (failed(applyPatternsAndFoldGreedily(getOperation(),
    //                                           std::move(patterns)))) {
    //     signalPassFailure();
    //     return;
    //   }
    // }

    // {
    //   auto module = getOperation();
    //   OpBuilder builder(module);
    //   module.getBody()->push_front(builder.create<emitc::IncludeOp>(
    //       module.getLoc(), "ttnn/device.h", /*isStandard=*/false));
    //   module.getBody()->push_front(builder.create<emitc::IncludeOp>(
    //       module.getLoc(), "ttnn/operations/eltwise/binary/binary.hpp",
    //       /*isStandard=*/false));
    //   module.getBody()->push_front(builder.create<emitc::IncludeOp>(
    //       module.getLoc(), "ttnn/operations/core.hpp",
    //       /*isStandard=*/false));
    //   module.getBody()->push_front(builder.create<emitc::IncludeOp>(
    //       module.getLoc(), "ttnn/operations/creation.hpp",
    //       /*isStandard=*/false));
    //   module.getBody()->push_front(builder.create<emitc::IncludeOp>(
    //       module.getLoc(),
    //       "ttnn/operations/reduction/generic/generic_reductions.hpp",
    //       /*isStandard=*/false));
    // }
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
