// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Layout.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {

inline Location appendInputSuffix(Location loc, int64_t operandIndex) {
  if (isa<NameLoc>(loc)) {
    NameLoc oldLoc = mlir::cast<NameLoc>(loc);
    StringAttr newName = StringAttr::get(
        loc->getContext(), oldLoc.getName().str() + "_in_" +
                               std::to_string(operandIndex) + "_layout");

    return NameLoc::get(newName, oldLoc.getChildLoc());
  }

  return loc;
}

//===----------------------------------------------------------------------===//
// To layout pass
//===----------------------------------------------------------------------===//

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

static std::optional<Value>
createToLayoutOp(PatternRewriter &rewriter, Location loc, Value input,
                 MemorySpace desiredMemorySpace,
                 TensorMemoryLayout desiredMemLayout, bool tiled) {

  auto ty = mlir::cast<RankedTensorType>(input.getType());
  auto currLayout = mlir::cast<MetalLayoutAttr>(ty.getEncoding());
  auto currMemorySpace = currLayout.getMemorySpace();
  auto currElementType = currLayout.getElementType();
  auto currMemLayout = currLayout.getMemLayout();
  auto desiredElementType =
      tiled ? rewriter.getType<TileType>(ty.getElementType())
            : ty.getElementType();
  if (currMemorySpace == desiredMemorySpace &&
      currElementType == desiredElementType &&
      currMemLayout == desiredMemLayout) {
    return std::nullopt;
  }

  auto desiredLayout = rewriter.getAttr<MetalLayoutAttr>(
      ty, desiredMemorySpace, currLayout.getGrid(), desiredElementType,
      desiredMemLayout);

  tensor::EmptyOp existingEmpty = input.getDefiningOp<tensor::EmptyOp>();
  if (existingEmpty) {
    return rewriter
        .replaceOpWithNewOp<tensor::EmptyOp>(existingEmpty, ty.getShape(),
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

  tensor::EmptyOp output = rewriter.create<tensor::EmptyOp>(
      loc, ty.getShape(), ty.getElementType(), desiredLayout);

  return rewriter
      .create<ttir::ToLayoutOp>(loc, output.getType(), input, output)
      ->getResult(0);
}

class TTIRLayoutDPSOperandsRewriter
    : public OpInterfaceRewritePattern<DestinationStyleOpInterface> {
public:
  TTIRLayoutDPSOperandsRewriter(MLIRContext *ctx,
                                MemorySpace defaultMemorySpace,
                                TensorMemoryLayout defaultDeviceMemoryLayout)
      : OpInterfaceRewritePattern<DestinationStyleOpInterface>(ctx),
        defaultMemorySpace(defaultMemorySpace),
        defaultDeviceMemoryLayout(defaultDeviceMemoryLayout) {}

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

      Location newLoc = TTIRLayout::loc(
          appendInputSuffix(op.getLoc(), operand.getOperandNumber()));
      auto desiredLayout =
          createToLayoutOp(rewriter, newLoc, operand.get(), defaultMemorySpace,
                           defaultDeviceMemoryLayout, true /* isTiled */);

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
  TensorMemoryLayout defaultDeviceMemoryLayout;
};

class TTIRLayoutFuncReturnRewriter
    : public OpRewritePattern<mlir::func::ReturnOp> {
public:
  TTIRLayoutFuncReturnRewriter(MLIRContext *ctx, MemorySpace initMemorySpace,
                               TensorMemoryLayout defaultDeviceMemoryLayout)
      : OpRewritePattern<mlir::func::ReturnOp>(ctx),
        initMemorySpace(initMemorySpace),
        defaultDeviceMemoryLayout(defaultDeviceMemoryLayout) {}

  LogicalResult matchAndRewrite(mlir::func::ReturnOp op,
                                PatternRewriter &rewriter) const final {
    bool modified = false;
    for (auto &operand : op->getOpOperands()) {
      // Leave the return values in initMemorySpace, optimizer might decide
      // otherwise
      bool tiled = false;
      TensorMemoryLayout initMemoryLayout = TensorMemoryLayout::None;
      if (isDeviceMemorySpace(initMemorySpace)) {
        initMemoryLayout = defaultDeviceMemoryLayout;
      }
      Location newLoc = TTIRLayout::loc(
          appendInputSuffix(op.getLoc(), operand.getOperandNumber()));
      if (auto layout =
              createToLayoutOp(rewriter, newLoc, operand.get(), initMemorySpace,
                               initMemoryLayout, tiled);
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
  TensorMemoryLayout defaultDeviceMemoryLayout;
};

void TTIRLayout::runOnOperation() {
  {
    auto device = getCurrentScopeDevice(getOperation());
    assert(device && "Device not found");
    TTIRLayoutTensorTypeConverter typeConverter(&getContext(), initMemorySpace,
                                                device.getWorkerGrid());
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRLayoutTensorTypeRewriter>(typeConverter, &getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
      signalPassFailure();
      return;
    }
  }
  {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRLayoutDPSOperandsRewriter>(
        &getContext(), defaultMemorySpace, defaultDeviceMemoryLayout);
    patterns.add<TTIRLayoutFuncReturnRewriter>(&getContext(), initMemorySpace,
                                               defaultDeviceMemoryLayout);
    FrozenRewritePatternSet patternSet(std::move(patterns));
    GreedyRewriteConfig config = GreedyRewriteConfig();
    config.useTopDownTraversal = true;
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), patternSet, config))) {
      signalPassFailure();
      return;
    }
  }
}

void TTIRLayout::getDependentDialects(mlir::DialectRegistry &registry) const {
  registry.insert<mlir::tt::ttir::TTIRDialect>();
  registry.insert<mlir::tt::TTDialect>();
  registry.insert<mlir::func::FuncDialect>();
}

//===----------------------------------------------------------------------===//
// Split compound layout pass
//===----------------------------------------------------------------------===//

class TTIRSplitCompoundLayoutRewriter : public OpRewritePattern<ToLayoutOp> {
public:
  using OpRewritePattern<ToLayoutOp>::OpRewritePattern;

  Value createToLayoutOp(PatternRewriter &rewriter, Location loc, Value input,
                         MetalLayoutAttr desiredLayout) const {
    auto ty = mlir::cast<RankedTensorType>(input.getType());
    auto output = rewriter.create<tensor::EmptyOp>(
        loc, ty.getShape(), ty.getElementType(), desiredLayout);
    return rewriter
        .create<ttir::ToLayoutOp>(loc, output.getType(), input, output)
        ->getResult(0);
  }

  Value bounce(PatternRewriter &rewriter, ToLayoutOp op,
               MetalLayoutAttr bounceLayout) const {
    auto bounced =
        createToLayoutOp(rewriter, TTIRSplitCompoundLayout::loc(op.getLoc()),
                         op.getInput(), bounceLayout);
    return rewriter.replaceOpWithNewOp<ttir::ToLayoutOp>(
        op, op.getOutput().getType(), bounced, op.getOutput());
  }

  LogicalResult matchAndRewrite(ToLayoutOp op,
                                PatternRewriter &rewriter) const final {
    auto components = op.compoundComponents();
    bool isCompound = (static_cast<int>(components.isLayoutChange) +
                       static_cast<int>(components.isGridChange) +
                       static_cast<int>(components.isFormatChange) +
                       static_cast<int>(components.isMemorySpaceChange) +
                       static_cast<int>(components.isMemoryLayoutChange)) > 1;

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
    } else if (components.isMemoryLayoutChange) {
      bounce(rewriter, op,
             inputLayout.withMemoryLayout(rewriter.getContext(),
                                          outputLayout.getMemLayout()));
    } else {
      // Note we should eventually support DRAM <-> DRAM, or System <-> System
      // w/ format conversion via streaming supported
      assert(false && "Unsupported compound layout change");
      return failure();
    }

    return success();
  }
};

void TTIRSplitCompoundLayout::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<TTIRSplitCompoundLayoutRewriter>(&getContext());
  FrozenRewritePatternSet patternSet(std::move(patterns));
  if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
    signalPassFailure();
    return;
  }
}

void TTIRSplitCompoundLayout::getDependentDialects(
    mlir::DialectRegistry &registry) const {
  registry.insert<mlir::tt::ttir::TTIRDialect>();
  registry.insert<mlir::tt::TTDialect>();
}

} // namespace mlir::tt::ttir
