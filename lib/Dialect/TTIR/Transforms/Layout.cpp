// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/UniformTypeRewriter.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRLAYOUT
#define GEN_PASS_DEF_TTIRSPLITCOMPOUNDLAYOUT
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// To layout pass
//===----------------------------------------------------------------------===//

namespace {
class TTIRLayoutTensorTypeConverter : public TypeConverter {
public:
  TTIRLayoutTensorTypeConverter(MLIRContext *ctx, MemorySpace initMemorySpace,
                                GridAttr deviceGrid) {
    addConversion([](Type type) { return type; });
    addConversion(
        [ctx, initMemorySpace, deviceGrid](RankedTensorType type) -> Type {
          auto layout = type.getEncoding();
          if (layout) {
            return type;
          }
          std::int64_t deviceGridRank = deviceGrid.getShape().size();
          // Default to single core grid
          auto tensorGrid = GridAttr::get(ctx, deviceGridRank);
          // Default to initMemorySpace, the optimizer might decide otherwise
          auto newLayout =
              MetalLayoutAttr::get(ctx, type, initMemorySpace, tensorGrid);
          return RankedTensorType::get(type.getShape(), type.getElementType(),
                                       newLayout);
        });
  }
};
} // namespace

static std::optional<Value> createToLayoutOp(PatternRewriter &rewriter,
                                             Location loc, Value input,
                                             MemorySpace desiredMemorySpace,
                                             bool tiled) {

  auto ty = mlir::cast<RankedTensorType>(input.getType());
  auto currLayout = mlir::cast<MetalLayoutAttr>(ty.getEncoding());
  auto currMemorySpace = currLayout.getMemorySpace();
  auto currElementType = currLayout.getElementType();
  auto desiredElementType =
      tiled ? rewriter.getType<TileType>(ty.getElementType())
            : ty.getElementType();
  if (currMemorySpace == desiredMemorySpace &&
      currElementType == desiredElementType) {
    return std::nullopt;
  }

  auto desiredLayout = rewriter.getAttr<MetalLayoutAttr>(
      ty, desiredMemorySpace, currLayout.getGrid(), desiredElementType);

  ttir::EmptyOp existingEmpty = input.getDefiningOp<ttir::EmptyOp>();
  if (existingEmpty) {
    return rewriter
        .replaceOpWithNewOp<ttir::EmptyOp>(existingEmpty, ty.getShape(),
                                           ty.getElementType(), desiredLayout)
        .getResult();
  }

  ttir::ConstantOp existingConstant = input.getDefiningOp<ttir::ConstantOp>();
  if (existingConstant) {
    return rewriter
        .replaceOpWithNewOp<ttir::ConstantOp>(
            existingConstant,
            mlir::RankedTensorType::get(ty.getShape(), ty.getElementType(),
                                        desiredLayout),
            existingConstant.getValue())
        .getResult();
  }

  return ttmlir::utils::createDPSOp<ttir::ToLayoutOp>(
             rewriter, loc, ty.getShape(), ty.getElementType(), desiredLayout,
             input)
      ->getResult(0);
}

class TTIRLayoutDPSOperandsRewriter
    : public OpInterfaceRewritePattern<DestinationStyleOpInterface> {
public:
  TTIRLayoutDPSOperandsRewriter(MLIRContext *ctx,
                                MemorySpace defaultMemorySpace)
      : OpInterfaceRewritePattern<DestinationStyleOpInterface>(ctx),
        defaultMemorySpace(defaultMemorySpace) {}

  LogicalResult matchAndRewrite(DestinationStyleOpInterface op,
                                PatternRewriter &rewriter) const final {
    if (mlir::isa<GenericOp>(op->getParentOp())) {
      // Skip if we're inside a GenericOp.
      return failure();
    }

    if (mlir::isa<ToLayoutOp>(op.getOperation())) {
      // Skip the ToLayoutOp itself.
      return failure();
    }

    assert(op->template hasTrait<TTIROp::Trait>());
    bool modified = false;
    for (auto &operand : op->getOpOperands()) {
      bool isResult = op.isDpsInit(&operand);

      // TTNN Conv2d moves input, weight, and bias from host to device
      // itself. Inserting the ToLayoutOp on these operands is thus problematic.
      if (mlir::isa<Conv2dOp>(op.getOperation()) && !isResult) {
        continue;
      }

      Location newLoc = ttmlir::utils::appendLocationSuffix(
          op.getLoc(),
          "_in_" + std::to_string(operand.getOperandNumber()) + "_layout");
      auto desiredLayout =
          createToLayoutOp(rewriter, newLoc, operand.get(), defaultMemorySpace,
                           true /* isTiled */);

      if (desiredLayout) {
        rewriter.modifyOpInPlace(op, [&]() {
          modified = true;
          op->setOperand(operand.getOperandNumber(), *desiredLayout);
          if (isResult) {
            // If this is the output operand, update the result type
            op->getResult(0).setType(desiredLayout->getType());
          }
        });
      }
    }

    return modified ? success() : failure();
  }

private:
  MemorySpace defaultMemorySpace;
};

class TTIRLayoutFuncReturnRewriter
    : public OpRewritePattern<mlir::func::ReturnOp> {
public:
  TTIRLayoutFuncReturnRewriter(MLIRContext *ctx, MemorySpace initMemorySpace)
      : OpRewritePattern<mlir::func::ReturnOp>(ctx),
        initMemorySpace(initMemorySpace) {}

  LogicalResult matchAndRewrite(mlir::func::ReturnOp op,
                                PatternRewriter &rewriter) const final {
    bool modified = false;
    for (auto &operand : op->getOpOperands()) {
      // Leave the return values in initMemorySpace, optimizer might decide
      // otherwise
      bool tiled = false;
      Location newLoc = ttmlir::utils::appendLocationSuffix(
          op.getLoc(),
          "_in_" + std::to_string(operand.getOperandNumber()) + "_layout");
      if (auto layout = createToLayoutOp(rewriter, newLoc, operand.get(),
                                         initMemorySpace, tiled);
          layout) {
        rewriter.modifyOpInPlace(
            op, [&]() { op.setOperand(operand.getOperandNumber(), *layout); });
        modified = true;
      }
    }
    return modified ? success() : failure();
  }

private:
  MemorySpace initMemorySpace;
};

class TTIRLayout : public impl::TTIRLayoutBase<TTIRLayout> {
public:
  using impl::TTIRLayoutBase<TTIRLayout>::TTIRLayoutBase;

  void runOnOperation() final {
    {
      auto device = lookupDevice(getOperation());
      assert(device && "Device not found");
      TTIRLayoutTensorTypeConverter typeConverter(
          &getContext(), initMemorySpace, device.getWorkerGrid());
      RewritePatternSet patterns(&getContext());
      patterns.add<UniformTypeRewriter>(typeConverter, &getContext());
      FrozenRewritePatternSet patternSet(std::move(patterns));
      if (failed(applyPatternsGreedily(getOperation(), patternSet))) {
        signalPassFailure();
        return;
      }
    }
    {
      RewritePatternSet patterns(&getContext());
      patterns.add<TTIRLayoutDPSOperandsRewriter>(&getContext(),
                                                  defaultMemorySpace);
      patterns.add<TTIRLayoutFuncReturnRewriter>(&getContext(),
                                                 initMemorySpace);
      FrozenRewritePatternSet patternSet(std::move(patterns));
      GreedyRewriteConfig config = GreedyRewriteConfig();
      config.useTopDownTraversal = true;
      if (failed(applyPatternsGreedily(getOperation(), patternSet, config))) {
        signalPassFailure();
        return;
      }
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
    registry.insert<mlir::func::FuncDialect>();
  }
};

//===----------------------------------------------------------------------===//
// Split compound layout pass
//===----------------------------------------------------------------------===//

class TTIRSplitCompoundLayoutRewriter : public OpRewritePattern<ToLayoutOp> {
public:
  using OpRewritePattern<ToLayoutOp>::OpRewritePattern;

  ttir::ToLayoutOp createToLayoutOp(PatternRewriter &rewriter, Location loc,
                                    Value input,
                                    MetalLayoutAttr desiredLayout) const {
    auto ty = mlir::cast<RankedTensorType>(input.getType());
    return ttmlir::utils::createDPSOp<ttir::ToLayoutOp>(
        rewriter, loc, ty.getShape(), ty.getElementType(), desiredLayout,
        input);
  }

  Value bounce(PatternRewriter &rewriter, ToLayoutOp op,
               MetalLayoutAttr bounceLayout) const {
    auto bounced =
        createToLayoutOp(rewriter, op.getLoc(), op.getInput(), bounceLayout);
    return rewriter
        .replaceOpWithNewOp<ttir::ToLayoutOp>(
            op, op.getOutput().getType(), bounced->getResult(0), op.getOutput())
        ->getResult(0);
  }

  LogicalResult matchAndRewrite(ToLayoutOp op,
                                PatternRewriter &rewriter) const final {
    auto components = op.compoundComponents();
    bool isCompound = (static_cast<int>(components.isLayoutChange) +
                       static_cast<int>(components.isGridChange) +
                       static_cast<int>(components.isFormatChange) +
                       static_cast<int>(components.isMemorySpaceChange)) > 1;

    if (!isCompound) {
      return failure();
    }

    auto inputType = mlir::cast<RankedTensorType>(op.getInput().getType());
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    auto inputLayout = mlir::cast<MetalLayoutAttr>(inputType.getEncoding());
    auto outputLayout = mlir::cast<MetalLayoutAttr>(outputType.getEncoding());

    bool inputL1 = inputLayout.getMemorySpace() == MemorySpace::DeviceL1;
    bool outputL1 = outputLayout.getMemorySpace() == MemorySpace::DeviceL1;

    // First prioritize moving the data into L1 so we can work with it in L1
    if (!inputL1) {
      // read first into L1, then format convert
      bounce(rewriter, op,
             inputLayout.withMemorySpace(rewriter.getContext(),
                                         MemorySpace::DeviceL1));
    } else if (!outputL1) {
      // format convert first in L1 first, then write
      assert(inputL1 && "input should guaranteed be in L1 because of the "
                        "previous case");
      bounce(rewriter, op,
             outputLayout.withMemorySpace(rewriter.getContext(),
                                          MemorySpace::DeviceL1));
    } else if (inputLayout.isTiled() != outputLayout.isTiled()) {
      // Prioritize moving tiled data
      if (inputLayout.isTiled()) {
        bounce(rewriter, op,
               outputLayout.withElementType(rewriter.getContext(),
                                            inputLayout.getElementType()));
      } else {
        assert(outputLayout.isTiled());
        bounce(rewriter, op,
               inputLayout.withElementType(rewriter.getContext(),
                                           outputLayout.getElementType()));
      }
    } else if (components.isLayoutChange && inputLayout.isTiled()) {
      // For now to flexibly support layout changes, we need to bounce to scalar
      // first
      bounce(rewriter, op,
             inputLayout.withElementType(rewriter.getContext(),
                                         inputLayout.getScalarElementType()));
    } else if (components.isGridChange) {
      assert(!components.isLayoutChange &&
             "Changing layout and grid at the same time is currently "
             "not supported");
      bounce(rewriter, op,
             outputLayout.withGrid(rewriter.getContext(), outputType,
                                   inputLayout.getGrid()));
    } else {
      // Note we should eventually support DRAM <-> DRAM, or System <-> System
      // w/ format conversion via streaming supported
      assert(false && "Unsupported compound layout change");
      return failure();
    }

    return success();
  }
};

class TTIRSplitCompoundLayout
    : public impl::TTIRSplitCompoundLayoutBase<TTIRSplitCompoundLayout> {
public:
  using impl::TTIRSplitCompoundLayoutBase<
      TTIRSplitCompoundLayout>::TTIRSplitCompoundLayoutBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRSplitCompoundLayoutRewriter>(&getContext());
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

} // namespace mlir::tt::ttir
