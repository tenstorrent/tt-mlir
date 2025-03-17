// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/UniformTypeRewriter.h"

#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRPREPARETENSORSFORBUFFERIZATION
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {
class TTIRTensorBufferizeShapeConverter : public TypeConverter {
public:
  TTIRTensorBufferizeShapeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([](RankedTensorType type) -> Type {
      auto layout = mlir::cast<MetalLayoutAttr>(type.getEncoding());
      auto bufferType = layout.getBufferType();
      return RankedTensorType::get(bufferType.getShape(),
                                   bufferType.getElementType(), layout);
    });
  }
};
} // namespace

namespace {
class TTIRPrepareTensorsForBufferization
    : public impl::TTIRPrepareTensorsForBufferizationBase<
          TTIRPrepareTensorsForBufferization> {

  using impl::TTIRPrepareTensorsForBufferizationBase<
      TTIRPrepareTensorsForBufferization>::
      TTIRPrepareTensorsForBufferizationBase;

  void runOnOperation() final {
    TTIRTensorBufferizeShapeConverter typeConverter(&getContext());
    RewritePatternSet patterns(&getContext());
    patterns.add<UniformTypeRewriter>(typeConverter, &getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet))) {
      signalPassFailure();
      return;
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
  }
};
} // namespace

void initializeOneShotBufferizationOptions(
    mlir::bufferization::OneShotBufferizationOptions &options) {
  options.bufferizeFunctionBoundaries = true;
  options.functionArgTypeConverterFn =
      [](mlir::TensorType tensorType, mlir::Attribute memorySpace,
         func::FuncOp funcOp,
         const bufferization::BufferizationOptions &options)
      -> ::mlir::BaseMemRefType {
    auto rankedTensorType = mlir::cast<::mlir::RankedTensorType>(tensorType);
    return mlir::cast<tt::MetalLayoutAttr>(rankedTensorType.getEncoding())
        .getBufferType();
  };
  options.defaultMemorySpaceFn =
      [](mlir::TensorType tensorType) -> std::optional<mlir::Attribute> {
    auto rankedTensorType = mlir::cast<::mlir::RankedTensorType>(tensorType);
    return mlir::cast<tt::MetalLayoutAttr>(rankedTensorType.getEncoding())
        .getMemref()
        .getMemorySpace();
  };
  options.unknownTypeConverterFn =
      [](Value value, Attribute memorySpace,
         const bufferization::BufferizationOptions &) -> BaseMemRefType {
    auto rankedTensorType =
        mlir::cast<::mlir::RankedTensorType>(value.getType());
    return mlir::cast<tt::MetalLayoutAttr>(rankedTensorType.getEncoding())
        .getBufferType();
  };
}

} // namespace mlir::tt::ttir
