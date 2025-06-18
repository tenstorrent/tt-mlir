// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
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
      auto layout =
          mlir::dyn_cast_if_present<MetalLayoutAttr>(type.getEncoding());
      if (!layout) {
        return type;
      }
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
    registry.insert<mlir::tt::TTCoreDialect>();
  }
};
} // namespace

} // namespace mlir::tt::ttir
