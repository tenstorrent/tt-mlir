// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTNN/TTIRToTTNN.h"

#include "ttmlir/Conversion/TTIRToTTIRDecomposition/TTIRToTTIRDecomposition.h"
#include "ttmlir/Dialect/Debug/IR/Debug.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_CONVERTTTIRTOTTNN
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::ttir

namespace {

struct ConvertTTIRToTTNNPass
    : public ttir::impl::ConvertTTIRToTTNNBase<ConvertTTIRToTTNNPass> {
  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<BuiltinDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<ttnn::TTNNDialect>();
    target.addLegalDialect<quant::QuantDialect>();
    target.addLegalDialect<debug::DebugDialect>();
    target.addIllegalDialect<ttir::TTIRDialect>();
    target.addLegalOp<ttcore::DeviceOp>();
    target.addLegalOp<ttcore::OptimizationBarrierOp>();
    target.addLegalOp<ttcore::LoadCachedOp>();

    TypeConverter typeConverter;
    // All types map 1:1.
    typeConverter.addConversion([](Type type) { return type; });

    // Add materialization callbacks for si32 <-> ui32 conversions (for argmax)
    // ArgMax operations convert si32 indices to ui32 (as required by tt-metal),
    // but consumers may still expect si32. Since both have the same bit
    // representation, we use ttnn::TypecastOp to convert between them.
    auto materializeCast = [](OpBuilder &builder, Type resultType,
                              ValueRange inputs, Location loc) -> Value {
      if (inputs.size() != 1) {
        return nullptr;
      }

      auto sourceType = inputs[0].getType();
      if (auto sourceTensor = mlir::dyn_cast<RankedTensorType>(sourceType)) {
        if (auto resultTensor = mlir::dyn_cast<RankedTensorType>(resultType)) {
          auto sourceElem = sourceTensor.getElementType();
          auto resultElem = resultTensor.getElementType();

          if (auto sourceInt = mlir::dyn_cast<IntegerType>(sourceElem)) {
            if (auto resultInt = mlir::dyn_cast<IntegerType>(resultElem)) {
              // Allow si32 <-> ui32 conversion (bitcast, same representation)
              if (sourceInt.getWidth() == 32 && resultInt.getWidth() == 32 &&
                  sourceInt.getSignedness() != resultInt.getSignedness()) {
                ttcore::DataType targetDataType;
                if (resultInt.isUnsigned()) {
                  targetDataType = ttcore::DataType::UInt32;
                } else {
                  targetDataType = ttcore::DataType::Int32;
                }
                auto dtypeAttr = ttcore::DataTypeAttr::get(builder.getContext(),
                                                           targetDataType);
                return builder
                    .create<ttnn::TypecastOp>(loc, resultType, inputs[0],
                                              dtypeAttr)
                    .getResult();
              }
            }
          }
        }
      }
      return nullptr;
    };

    typeConverter.addTargetMaterialization(materializeCast);
    typeConverter.addSourceMaterialization(materializeCast);

    RewritePatternSet patterns(&getContext());
    populateTTIRToTTNNPatterns(&getContext(), patterns, typeConverter);

    // Apply full conversion
    //
    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTIRToTTNNPass() {
  return std::make_unique<ConvertTTIRToTTNNPass>();
}

} // namespace mlir::tt
