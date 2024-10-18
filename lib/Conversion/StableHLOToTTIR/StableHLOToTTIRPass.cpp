// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/StableHLOToTTIR/StableHLOToTTIR.h"

#include <llvm/ADT/ArrayRef.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>

#include <stablehlo/dialect/StablehloOps.h>

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_CONVERTSTABLEHLOTOTTIR
#define GEN_PASS_DEF_CONVERTARITHTOSTABLEHLO
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::ttir

namespace {

class StablehloTypeConverter : public TypeConverter {
public:
  StablehloTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) {
      assert(isa<RankedTensorType>(type) &&
             "only ranked tensor type supported");
      return type;
    });

    // TTNN doesn't support either scalars or boolean data. This transformation
    // converts boolean to bfloat16 and scalars to 1-D tensors.
    addConversion([&](RankedTensorType type) -> RankedTensorType {
      bool changed = false;
      Type elementType = type.getElementType();
      llvm::ArrayRef<int64_t> shape = type.getShape();
      // Convert the element type to bfloat16 if the input is boolean.
      if (type.getElementTypeBitWidth() == 1) {
        elementType = BFloat16Type::get(elementType.getContext());
        changed = true;
      }
      // Create shape of 1-D tensor in case of scalar input.
      if (shape.size() == 0) {
        shape = RankedTensorType::get({1}, elementType).getShape();
        changed = true;
      }
      return changed ? RankedTensorType::get(shape, elementType) : type;
    });
  }
};

struct ConvertStableHLOToTTIRPass
    : public ttir::impl::ConvertStableHLOToTTIRBase<
          ConvertStableHLOToTTIRPass> {
  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());

    target.addIllegalDialect<mlir::stablehlo::StablehloDialect>();

    target.addLegalDialect<ttir::TTIRDialect>();
    target.addLegalOp<mlir::tensor::EmptyOp>();
    target.addLegalOp<mlir::ModuleOp>();
    target.addLegalOp<mlir::func::FuncOp>();
    target.addLegalOp<mlir::func::ReturnOp>();

    // For now keep the same type assuming StableHLO ops operate on builtin
    // tensor.
    StablehloTypeConverter typeConverter(&getContext());
    RewritePatternSet patterns(&getContext());

    // Func type conversions
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

    populateStableHLOToTTIRPatterns(&getContext(), patterns, typeConverter);

    // Apply conversion.
    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

struct ConvertArithToStableHLOPass
    : public ttir::impl::ConvertArithToStableHLOBase<
          ConvertArithToStableHLOPass> {
  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());

    target.addIllegalDialect<mlir::arith::ArithDialect>();

    // For now keep the same type assuming StableHLO ops operate on builtin
    // tensor.
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) {
      assert(isa<RankedTensorType>(type) &&
             "only ranked tensor type supported");
      return type;
    });
    RewritePatternSet patterns(&getContext());

    populateArithToStableHLOPatterns(&getContext(), patterns, typeConverter);

    // Apply conversion.
    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertStableHLOToTTIRPass() {
  return std::make_unique<ConvertStableHLOToTTIRPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertArithToStableHLOPass() {
  return std::make_unique<ConvertArithToStableHLOPass>();
}

} // namespace mlir::tt
