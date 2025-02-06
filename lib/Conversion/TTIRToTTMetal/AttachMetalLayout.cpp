// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

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
      std::int64_t collapsedTensorRank = deviceGridRank;

      // Default to single core grid
      auto tensorGrid = GridAttr::get(ctx, deviceGridRank);

      MetalLayoutAttr newLayout = [&]() {
        // Default to initMemorySpace, the optimizer might decide otherwise:
        auto layout =
            MetalLayoutAttr::get(ctx, type, initMemorySpace, tensorGrid);
        if (!useStreamLayout) {
          return layout;
        }

        // Default to '1x1x...' for outer scaling:
        llvm::SmallVector<int64_t> outerScale(collapsedTensorRank, 1);

        // Select some stream layout defaults for the given memory space:
        StreamMode streamMode;
        uint32_t numBuffers;
        std::tie(streamMode, numBuffers) =
            StreamLayoutAttr::getDefaults(initMemorySpace);

        return layout.withOuterScale(ctx, outerScale, streamMode, numBuffers);
      }();

      return RankedTensorType::get(type.getShape(), type.getElementType(),
                                   newLayout);
    });
  }
};
} // namespace

namespace {
class TTIRLayoutTensorTypeRewriter : public RewritePattern {
public:
  TTIRLayoutTensorTypeRewriter(const TypeConverter &converter, MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx),
        converter(&converter) {}

  template <typename ValueRange>
  bool convertTypes(ValueRange valueRange, SmallVector<Type> &newTypes) const {
    bool updated = false;
    auto result = converter->convertTypes(valueRange.getTypes(), newTypes);
    if (result.failed()) {
      return false;
    }
    for (auto [operand, newType] : llvm::zip(valueRange, newTypes)) {
      if (operand.getType() == newType) {
        continue;
      }
      operand.setType(newType);
      updated = true;
    }
    return updated;
  }

  bool convertFuncType(Operation *op, PatternRewriter &rewriter) const {
    auto funcOp = dyn_cast<func::FuncOp>(op);
    if (not funcOp) {
      return false;
    }
    SmallVector<Type> inputTypes(funcOp.getArgumentTypes());
    SmallVector<Type> outputTypes(funcOp.getResultTypes());
    for (Type &ty : inputTypes) {
      ty = converter->convertType(ty);
    }
    for (Type &ty : outputTypes) {
      ty = converter->convertType(ty);
    }
    auto newType = rewriter.getType<FunctionType>(inputTypes, outputTypes);
    if (funcOp.getFunctionType() == newType) {
      return false;
    }
    funcOp.setFunctionType(newType);

    Block &entryBlock = funcOp.getBody().front();
    for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
      entryBlock.getArgument(i).setType(inputTypes[i]);
    }

    return true;
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Skip if we're inside a GenericOp
    if (mlir::isa<GenericOp>(op->getParentOp())) {
      return failure();
    }
    bool updated = false;
    SmallVector<Type> operands;
    SmallVector<Type> results;
    updated |= convertTypes(op->getOperands(), operands);
    updated |= convertTypes(op->getResults(), results);
    updated |= convertFuncType(op, rewriter);
    return updated ? success() : failure();
  }

  const TypeConverter *converter;
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
    patterns.add<TTIRLayoutTensorTypeRewriter>(typeConverter, &getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
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
