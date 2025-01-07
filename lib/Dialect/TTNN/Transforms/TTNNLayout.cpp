// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNLAYOUT
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

// Default collapse dims for affine map (d0, d1, d2) -> (d0 <> d1, d2)
static const std::array<std::pair<int64_t, int64_t>, 1> g_defaultCollapseDims =
    {{{0, -1}}};

// Default memory space for tensors on host
static const BufferType g_defaultMemorySpaceHost = BufferType::SystemMemory;

// Default memory space for tesnors on device
static const BufferType g_defaultMemorySpaceDevice = BufferType::DRAM;

// Default memory layout for tensors on device
static const TensorMemoryLayout g_defaultMemoryLayout =
    TensorMemoryLayout::Interleaved;

//===----------------------------------------------------------------------===//
// Helper methods
//===----------------------------------------------------------------------===//

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

// Converts tensor types to have a ttnn layout attribute with deault values
//
// Example: tensor<15x10x32xf32> -> tensor<15x10x32xf32, ttnn_layout<...>>
// where ttnn_layout<...> is constructed with default values
// SystemMemory, MemoryLayout::None, Grid<1x1>
class TTNNLayoutTensorTypeConverter : public TypeConverter {
public:
  TTNNLayoutTensorTypeConverter(MLIRContext *ctx, GridAttr deviceGrid) {
    addConversion([](Type type) { return type; });
    addConversion([ctx, deviceGrid](RankedTensorType type) -> Type {
      Attribute layout = type.getEncoding();
      if (layout) {
        return type;
      }

      std::int64_t deviceGridRank = deviceGrid.getShape().size();

      // Default to single core grid
      GridAttr tensorGrid = GridAttr::get(ctx, deviceGridRank);

      llvm::ArrayRef<std::pair<int64_t, int64_t>> collapseDimsRef(
          g_defaultCollapseDims);

      TTNNLayoutAttr newLayout = TTNNLayoutAttr::get(
          ctx, type.getShape(), type.getElementType(), g_defaultMemorySpaceHost,
          tensorGrid, nullptr /* memLayoutAttr */, collapseDimsRef);
      return RankedTensorType::get(type.getShape(), type.getElementType(),
                                   newLayout);
    });
  }
};

// Rewrites tensor types to have a ttnn layout attribute with default values
class TTNNLayoutTensorTypeRewriter : public RewritePattern {
public:
  TTNNLayoutTensorTypeRewriter(const TypeConverter &converter, MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx),
        converter(&converter) {}

  bool convertTypes(ValueRange valueRange, SmallVector<Type> &newTypes) const {
    if (failed(converter->convertTypes(valueRange.getTypes(), newTypes))) {
      return false;
    }

    bool updated = false;
    for (auto [value, newType] : llvm::zip(valueRange, newTypes)) {
      if (value.getType() != newType) {
        value.setType(newType);
        updated = true;
      }
    }
    return updated;
  }

  // FuncOp requires special handling because it has a FunctionType attribute
  bool convertFuncType(Operation *op, PatternRewriter &rewriter) const {
    func::FuncOp funcOp = dyn_cast<func::FuncOp>(op);
    if (!funcOp) {
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
    FunctionType newType =
        rewriter.getType<FunctionType>(inputTypes, outputTypes);
    if (funcOp.getFunctionType() == newType) {
      return false;
    }
    funcOp.setFunctionType(newType);

    if (funcOp.isDeclaration()) {
      return true;
    }

    Block &entryBlock = funcOp.getBody().front();

    for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
      entryBlock.getArgument(i).setType(inputTypes[i]);
    }

    return true;
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    bool updated = false;
    SmallVector<Type> operands;
    SmallVector<Type> results;
    updated |= convertTypes(op->getOperands(), operands);
    updated |= convertTypes(op->getResults(), results);
    updated |= convertFuncType(op, rewriter);
    if (updated) {
      llvm::outs() << "TTNNLayoutTensorTypeRewriter::mAR succeeded on op:\n";
      op->dump();
    }
    return updated ? success() : failure();
  }

  const TypeConverter *converter;
};

// Given desired buffer type, memory layout and type checks if the input tensor
// needs to be converted to the desired layout. If it does, creates a new
// EmptyOp/ConstantOp/EmptyOp + ToLayoutOp depending on the input.
static std::optional<Value>
createToLayoutOp(PatternRewriter &rewriter, Location loc, Value input,
                 BufferType desiredBufferType,
                 TensorMemoryLayoutAttr desiredMemLayoutAttr, bool tiled) {

  // Get type
  RankedTensorType ty = mlir::cast<RankedTensorType>(input.getType());

  // Get ttnn layout from the type
  TTNNLayoutAttr ttnnLayoutAttr = mlir::cast<TTNNLayoutAttr>(ty.getEncoding());

  // Get buffer type (i.e DRAM/L1 etc)
  BufferType currBufferType = ttnnLayoutAttr.getBufferType();

  // Get the current element type (i.e bf16/TileType etc)
  Type currElementType = ttnnLayoutAttr.getElementType();

  // Get mem layout. If the tensor is on host layout is null
  TensorMemoryLayoutAttr currMemLayout = ttnnLayoutAttr.getMemLayout();

  // Get element type that should be used in the new ttnn layout
  Type desiredElementType =
      tiled ? rewriter.getType<TileType>(ty.getElementType())
            : ty.getElementType();

  // If the current buffer type, element type and memory layout are the same as
  // the desired ones, we don't need to do anything
  if (currBufferType == desiredBufferType &&
      currElementType == desiredElementType &&
      currMemLayout == desiredMemLayoutAttr) {
    return std::nullopt;
  }

  // Create a new ttnn layout with the desired buffer type, element type and
  // memory layout
  TTNNLayoutAttr desiredLayout = rewriter.getAttr<TTNNLayoutAttr>(
      ty.getShape(), desiredElementType, desiredBufferType,
      ttnnLayoutAttr.getGrid(), desiredMemLayoutAttr, g_defaultCollapseDims);

  // If the input tensor is a constant or empty tensor, we can replace it with a
  // new tensor with the desired layout
  tensor::EmptyOp existingEmpty = input.getDefiningOp<tensor::EmptyOp>();
  if (existingEmpty) {
    return rewriter
        .replaceOpWithNewOp<tensor::EmptyOp>(existingEmpty, ty.getShape(),
                                             ty.getElementType(), desiredLayout)
        .getResult();
  }

  // If the input tensor is a constant, we can replace it with a new constant
  // with the desired layout
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

  // If the input tensor is an arange, we want to set the desired layout just
  // like the other creation ops. However, a caveat is that in ttnn, arange is
  // hardcoded to be ROW_MAJOR. So we must ensure that the layout we assign to
  // it is ROW_MAJOR - and to make it tile layout we still must insert
  // ToLayoutOp on its output. We can do this by setting the element type to
  // ty.getElementType() in case desiredElementType is a TileType.
  ttir::ArangeOp existingArange = input.getDefiningOp<ttir::ArangeOp>();
  if (existingArange) {
    TTNNLayoutAttr arangeLayout = rewriter.getAttr<TTNNLayoutAttr>(
        ty.getShape(), ty.getElementType(), desiredBufferType,
        ttnnLayoutAttr.getGrid(), desiredMemLayoutAttr, g_defaultCollapseDims);
    input =
        rewriter
            .replaceOpWithNewOp<ttir::ArangeOp>(
                existingArange,
                mlir::RankedTensorType::get(ty.getShape(), ty.getElementType(),
                                            arangeLayout),
                existingArange.getStart(), existingArange.getEnd(),
                existingArange.getStep(), existingArange.getArangeDimension())
            .getResult();
  }

  // If the input tensor is not a constant or empty tensor, we need to create a
  // new tensor with the desired layout which will be used as the output of the
  // ToLayoutOp
  tensor::EmptyOp output = rewriter.create<tensor::EmptyOp>(
      loc, ty.getShape(), ty.getElementType(), desiredLayout);

  // Create the ToLayoutOp which will convert the input tensor to the desired
  // layout
  return rewriter
      .create<ttir::ToLayoutOp>(loc, output.getType(), input, output)
      ->getResult(0);
}

static bool changeLayoutToHost(DestinationStyleOpInterface &op,
                               OpOperand &operand, PatternRewriter &rewriter) {
  Location newLoc = appendInputSuffix(op.getLoc(), operand.getOperandNumber());
  std::optional<Value> layout =
      createToLayoutOp(rewriter, newLoc, operand.get(),
                       BufferType::SystemMemory, nullptr, false /* tiled */);
  if (layout.has_value()) {
    rewriter.modifyOpInPlace(
        op, [&]() { op->setOperand(operand.getOperandNumber(), *layout); });
    return true;
  }
  return false;
}

// Updates the layout of the operands of a TTIR ops which have DPS operands.
// This function rewrites the operands and result to have the correct layout
// with respect to operand constraints.
class TTNNLayoutDPSOperandsRewriter
    : public OpInterfaceRewritePattern<DestinationStyleOpInterface> {
public:
  TTNNLayoutDPSOperandsRewriter(MLIRContext *ctx)
      : OpInterfaceRewritePattern<DestinationStyleOpInterface>(ctx) {}

  LogicalResult matchAndRewrite(DestinationStyleOpInterface op,
                                PatternRewriter &rewriter) const final {
    llvm::outs() << "s TTNNLayoutDPSOperandsRewriter::mAR\n";
    op.dump();

    // To layout op is a special case, we don't want to rewrite it
    if (mlir::isa<ttir::ToLayoutOp>(op.getOperation()) || op->getParentOfType<tt::CPUModuleOp>()) {
      llvm::outs() << "e -- f TTNNLayoutDPSOperandsRewriter::mAR\n";
      return failure();
    }

    assert(op->template hasTrait<ttir::TTIROp::Trait>());
    bool modified = false;
    for (OpOperand &operand : op->getOpOperands()) {
      // Check if the operand is a dps result
      bool isResult = op.isDpsInit(&operand);

      // TTNN Conv2d moves input, weight, and bias from host to device
      // itself. Inserting the ToLayoutOp on these operands is thus problematic.
      if (mlir::isa<ttir::Conv2dOp>(op.getOperation()) && !isResult) {
        // For the weight input of the conv2d op, it specifically needs to be on
        // host, so we create a host to layout op (issue
        // https://github.com/tenstorrent/tt-mlir/issues/1528).
        if (operand.getOperandNumber() == 1) {
          modified = changeLayoutToHost(op, operand, rewriter);
        }
        continue;
      }

      // If the operand is a BroadcastOp or a ToLayout op do not put a
      // ToLayoutOp on its output
      if (operand.get().getDefiningOp<ttir::BroadcastOp>() ||
          operand.get().getDefiningOp<ttir::ToLayoutOp>()) {
        continue;
      }

      Location newLoc =
          appendInputSuffix(op.getLoc(), operand.getOperandNumber());
      // Given the operand constraint, create the desired layout for the operand
      std::optional<Value> desiredLayout = createToLayoutOp(
          rewriter, newLoc, operand.get(), g_defaultMemorySpaceDevice,
          TensorMemoryLayoutAttr::get(rewriter.getContext(),
                                      g_defaultMemoryLayout),
          true /* isTiled */);

      // If layout changed update the operand
      if (desiredLayout) {
        rewriter.modifyOpInPlace(op, [&]() {
          modified = true;
          op->setOperand(operand.getOperandNumber(), *desiredLayout);
          // If operand is dps result, update the result type on current op
          if (isResult) {
            op->getResult(0).setType(desiredLayout->getType());
          }
        });
      }
    }
    llvm::outs() << "e -- " << (modified ? "s" : "f")
                 << " TTNNLayoutDPSOperandsRewriter::mAR\n";

    return modified ? success() : failure();
  }
};

class TTNNLayoutHoistedFuncCallRewriter
    : public OpRewritePattern<func::CallOp> {
public:
  TTNNLayoutHoistedFuncCallRewriter(MLIRContext *ctx)
      : OpRewritePattern<func::CallOp>(ctx) {}

  // Match and rewrite the CallOp.
  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    llvm::outs() << "s TTNNLayoutHoistedFuncCallRewriter::mAR\n";

    llvm::outs() << "0\n";

    llvm::outs() << "Address of callOp: " << &callOp << "\n";
    llvm::outs() << "Details of callOp: " << callOp << "\n";

    llvm::outs() << callOp->getName() << "\n";
    if (!callOp->hasAttr("hoisted_call") || callOp->hasAttr("replaced")) {
      llvm::outs() << "e -- f TTNNLayoutHoistedFuncCallRewriter::mAR\n";
      return failure();
    }
    auto device = utils::getOrInsertDevice(rewriter, callOp);

    llvm::outs() << "1\n";

    // Create a FromDevice operation for each operand.
    SmallVector<Value, 4> fromDeviceOperands;
    size_t locIdx = 0;
    for (auto operand : callOp.getOperands()) {
      // Insert FromDevice op before the operand and collect the new operands.
      auto fromDeviceOp = rewriter.create<ttnn::FromDeviceOp>(
          callOp.getLoc(), operand.getType(), operand);
      fromDeviceOp.dump();
      Location newLoc = appendInputSuffix(callOp.getLoc(), locIdx++);
      std::optional<Value> maybeLayoutOp = createToLayoutOp(
          rewriter, newLoc, fromDeviceOp.getResult(), BufferType::SystemMemory,
          nullptr /* tensorMemoryLayoutAttr */, false /* tiled */);
      Value hostOpValue = maybeLayoutOp.has_value() ? maybeLayoutOp.value()
                                                    : fromDeviceOp.getResult();
      fromDeviceOperands.push_back(hostOpValue);
    }

    llvm::outs() << "2\n";

    // Create the original CallOp with the new operands (FromDevice'd).
    auto newCallOp = rewriter.create<func::CallOp>(
        callOp.getLoc(), callOp.getCallee(), callOp.getResultTypes(),
        fromDeviceOperands);

    llvm::outs() << "3\n";

    // Now, insert ToDevice ops for the results of the CallOp
    SmallVector<Value, 4> toDeviceResults;
    for (auto result : newCallOp.getResults()) {
      // Insert ToDevice op after the result
      auto toDeviceOp = rewriter.create<ttnn::ToDeviceOp>(
          callOp.getLoc(), result.getType(), result, device,
          ttnn::MemoryConfigAttr{});
      Location newLoc =
          appendInputSuffix(callOp.getLoc(), result.getResultNumber() + locIdx);
      std::optional<Value> maybeLayoutOp = createToLayoutOp(
          rewriter, newLoc, toDeviceOp.getResult(), BufferType::SystemMemory,
          nullptr /* tensorMemoryLayoutAttr */, true /* tiled */);
      Value deviceResultValue = maybeLayoutOp.has_value()
                                    ? maybeLayoutOp.value()
                                    : toDeviceOp.getResult();
      toDeviceResults.push_back(deviceResultValue);
    }
    llvm::outs() << "4\n";

    // Replace the original call with the new ToDevice results.
    rewriter.replaceOp(callOp, toDeviceResults);
    // rewriter.eraseOp(callOp);
    callOp->setAttr("replaced", rewriter.getUnitAttr());

    llvm::outs() << "5\n";

    llvm::outs() << "e -- s TTNNLayoutHoistedFuncCallRewriter::mAR\n";

    return success();
  }
};

// Updates the layout of the operands of a func::ReturnOp.
// The intent is to move the result to host.
class TTNNLayoutFuncReturnRewriter
    : public OpRewritePattern<mlir::func::ReturnOp> {
public:
  TTNNLayoutFuncReturnRewriter(MLIRContext *ctx)
      : OpRewritePattern<mlir::func::ReturnOp>(ctx) {}

  LogicalResult matchAndRewrite(mlir::func::ReturnOp op,
                                PatternRewriter &rewriter) const final {
    llvm::outs() << "s TTNNLayoutFuncReturnRewriter::mAR\n";

    bool modified = false;
    for (OpOperand &operand : op->getOpOperands()) {
      Location newLoc =
          appendInputSuffix(op.getLoc(), operand.getOperandNumber());
      std::optional<Value> layout = createToLayoutOp(
          rewriter, newLoc, operand.get(), BufferType::SystemMemory,
          nullptr /* tensorMemoryLayoutAttr */, false /* tiled */);
      if (layout.has_value()) {
        rewriter.modifyOpInPlace(
            op, [&]() { op.setOperand(operand.getOperandNumber(), *layout); });
        modified = true;
      }
    }
    llvm::outs() << "e -- ";
    if (modified) {
      llvm::outs() << "s";
      op.dump();
    } else {
      llvm::outs() << "f";
    }
    llvm::outs() << " TTNNLayoutFuncReturnRewriter::mAR\n";

    return modified ? success() : failure();
  }

private:
};

class TTNNLayout : public impl::TTNNLayoutBase<TTNNLayout> {
public:
  using impl::TTNNLayoutBase<TTNNLayout>::TTNNLayoutBase;

  void runOnOperation() final {
    // First add default attribute to all tensors. Example:
    // Given tensor type: tensor<15x10x32xf32>
    // we construct a ttnn layout attribute with default values:
    // ttnn_layout<affine_map, grid<1x1>, memref<<15x64>xf32, #system_memory>
    {
      DeviceAttr device = getCurrentScopeDevice(getOperation());
      assert(device && "Device not found");
      TTNNLayoutTensorTypeConverter typeConverter(&getContext(),
                                                  device.getWorkerGrid());
      RewritePatternSet patterns(&getContext());
      patterns.add<TTNNLayoutTensorTypeRewriter>(typeConverter, &getContext());
      FrozenRewritePatternSet patternSet(std::move(patterns));
      if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
        signalPassFailure();
        return;
      }
    }
    {
      RewritePatternSet patterns(&getContext());
      // Move operands + results of hoisted funcs to and from device
      // appropriately
      patterns.add<TTNNLayoutHoistedFuncCallRewriter>(&getContext());
      FrozenRewritePatternSet patternSet(std::move(patterns));
      if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
        signalPassFailure();
        return;
      }
      llvm::outs() << "IR dump:\n";
      getOperation()->dump();
    }
    {
      RewritePatternSet patterns(&getContext());
      // Takes all TTIR ops which have DPS operands
      // and rewrites its operands and result to have the correct layout
      // with respect to operand constraints.
      patterns.add<TTNNLayoutDPSOperandsRewriter>(&getContext());
      // Takes func::Return op and sets layout which will
      // move it's operands to host
      patterns.add<TTNNLayoutFuncReturnRewriter>(&getContext());

      FrozenRewritePatternSet patternSet(std::move(patterns));
      GreedyRewriteConfig config = GreedyRewriteConfig();
      config.useTopDownTraversal = true;
      if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet,
                                              config))) {
        signalPassFailure();
        return;
      }
      llvm::outs() << "IR dump:\n";
      getOperation()->dump();
      llvm::outs() << "e TTNNLayout::rOO\n";
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
    registry.insert<mlir::func::FuncDialect>();
  }
};

} // namespace mlir::tt::ttnn
