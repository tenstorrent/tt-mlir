// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/StableHLOToTTIR/StableHLOToTTIR.h"

#include "ttmlir/Conversion/StableHLOToTTIR/EmptyOpTypeConversion.h"
#include "ttmlir/Conversion/StableHLOToTTIR/ShardyToTTIR.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/ArrayRef.h"

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_CONVERTSTABLEHLOTOTTIR
#define GEN_PASS_DEF_CONVERTSHARDYTOTTIR
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
    // This transformation also convert 64 bit float/integer types to 32 bit
    // types. In the case of a quantized type, the type is returned as is.
    addConversion([&](RankedTensorType type) -> RankedTensorType {
      bool changed = false;
      Type elementType = type.getElementType();
      llvm::ArrayRef<int64_t> shape = type.getShape();

      if (mlir::isa<mlir::quant::QuantizedType>(elementType)) {
        return type;
      }

      size_t bitWidth = type.getElementTypeBitWidth();
      mlir::Attribute encoding = type.getEncoding();
      MLIRContext *context = elementType.getContext();
      // Convert the element type to bfloat16 if the input is boolean.
      if (bitWidth == 1) {
        elementType = BFloat16Type::get(context);
        changed = true;
      } else if (bitWidth == 64) {
        // Convert 64 bit integer element type to 32 bit integer.
        // If element is unsigned, we explicitly assign it Unsigned semantics to
        // it (like `ui32`). Otherwise, we don't explicitly use Signed semantics
        // (like `si32`), but rather Signless (like `i32`) which is the default.
        if (isa<IntegerType>(elementType)) {
          elementType = IntegerType::get(
              context, 32,
              elementType.isUnsignedInteger()
                  ? IntegerType::SignednessSemantics::Unsigned
                  : IntegerType::SignednessSemantics::Signless);
          changed = true;
        }
        // Convert 64 bit float element type to 32 bit float.
        else if (isa<FloatType>(elementType)) {
          elementType = Float32Type::get(context);
          changed = true;
        }
      }
      // Create shape of 1-D tensor in case of scalar input.
      if (shape.size() == 0) {
        shape = RankedTensorType::get({1}, elementType).getShape();
        changed = true;
      }
      return changed ? RankedTensorType::get(shape, elementType, encoding)
                     : type;
    });
  }
};

struct ConvertStableHLOToTTIRPass
    : public ttir::impl::ConvertStableHLOToTTIRBase<
          ConvertStableHLOToTTIRPass> {
  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());

    target.addIllegalDialect<mlir::stablehlo::StablehloDialect>();
    target.addIllegalDialect<mlir::sdy::SdyDialect>();
    target.addIllegalOp<mlir::tensor::EmptyOp>();

    target.addLegalDialect<mlir::quant::QuantDialect>();
    target.addLegalDialect<ttir::TTIRDialect>();
    target.addLegalOp<mlir::tt::ttir::EmptyOp>();
    target.addLegalOp<mlir::ModuleOp>();
    target.addLegalOp<mlir::func::FuncOp>();
    target.addLegalOp<mlir::func::ReturnOp>();
    target.addLegalOp<mlir::func::CallOp>();

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

    addEmptyOpTypeConversionPattern(&getContext(), patterns, typeConverter);

    populateStableHLOToTTIRPatterns(&getContext(), patterns, typeConverter);
    populateShardyToTTIRPatterns(&getContext(), patterns, typeConverter);

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

} // namespace mlir::tt
