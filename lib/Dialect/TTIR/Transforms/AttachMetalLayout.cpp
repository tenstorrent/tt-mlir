// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/UniformTypeRewriter.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRATTACHMETALLAYOUT
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {
class TTIRLayoutTensorTypeConverter : public TypeConverter {
public:
  TTIRLayoutTensorTypeConverter(MLIRContext *ctx, MemorySpace initMemorySpace,
                                bool useStreamLayout, GridAttr deviceGrid) {
    addConversion([](Type type) { return type; });
    addConversion([ctx, useStreamLayout, deviceGrid,
                   initMemorySpace](RankedTensorType type) -> Type {
      if (type.getEncoding()) {
        return type;
      }
      std::int64_t deviceGridRank = deviceGrid.getShape().size();
      // Default to single core grid
      auto tensorGrid = GridAttr::get(ctx, deviceGridRank);

      MetalLayoutAttr newLayout = [&]() {
        // Default to initMemorySpace, the optimizer might decide otherwise:
        auto layout =
            MetalLayoutAttr::get(ctx, type, initMemorySpace, tensorGrid);
        if (!useStreamLayout) {
          return layout;
        }

        auto tileType = TileType::get(ctx, type.getElementType());
        return layout.withElementType(ctx, tileType);
      }();

      return RankedTensorType::get(type.getShape(), type.getElementType(),
                                   newLayout);
    });
  }
};
} // namespace

namespace {
class TTIRAttachMetalLayout
    : public impl::TTIRAttachMetalLayoutBase<TTIRAttachMetalLayout> {

  using impl::TTIRAttachMetalLayoutBase<
      TTIRAttachMetalLayout>::TTIRAttachMetalLayoutBase;

  void runOnOperation() final {
    auto device = getCurrentScopeDevice(getOperation());
    assert(device && "Device not found");
    TTIRLayoutTensorTypeConverter typeConverter(&getContext(), initMemorySpace,
                                                useStreamLayout,
                                                device.getWorkerGrid());
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

} // namespace mlir::tt::ttir
