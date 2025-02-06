// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

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

// Default memory space for tensors on device
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

static TTNNLayoutAttr createLayoutAttr(
    MLIRContext *ctx, GridAttr deviceGrid, RankedTensorType type,
    std::optional<BufferType> bufferTypeOpt = std::nullopt,
    std::optional<TensorMemoryLayout> memoryLayoutOpt = std::nullopt,
    std::optional<bool> isTiledOpt = std::nullopt) {

  BufferType bufferType = bufferTypeOpt.value_or(g_defaultMemorySpaceDevice);
  TensorMemoryLayout memoryLayout =
      memoryLayoutOpt.value_or(g_defaultMemoryLayout);
  bool isTiled = isTiledOpt.value_or(true);
  std::int64_t deviceGridRank = deviceGrid.getShape().size();
  // Default to single core grid
  GridAttr tensorGrid = GridAttr::get(ctx, deviceGridRank);

  llvm::ArrayRef<std::pair<int64_t, int64_t>> collapseDimsRef(
      g_defaultCollapseDims);

  // Force TileType for tensors
  auto elementType = isTiled ? TileType::get(ctx, type.getElementType())
                             : type.getElementType();
  return TTNNLayoutAttr::get(
      ctx, type.getShape(), elementType, bufferType, tensorGrid,
      TensorMemoryLayoutAttr::get(ctx, memoryLayout), collapseDimsRef);
}

//===----------------------------------------------------------------------===//
// To layout pass
//===----------------------------------------------------------------------===//

// Converts tensor types to have a ttnn layout attribute with default values
//
// Example: tensor<15x10x32xf32> -> tensor<15x10x32xf32, ttnn_layout<...>>
// where ttnn_layout<...> is constructed with default values
// Dram, MemoryLayout::Interleaved, Grid<1x1>
class TTNNLayoutTensorTypeConverter : public TypeConverter {
public:
  TTNNLayoutTensorTypeConverter(MLIRContext *ctx, GridAttr deviceGrid) {
    addConversion([](Type type) { return type; });
    addConversion([ctx, deviceGrid](RankedTensorType type) -> Type {
      Attribute layout = type.getEncoding();
      if (layout) {
        return type;
      }

      TTNNLayoutAttr newLayout = createLayoutAttr(ctx, deviceGrid, type);
      // Convert mlir data types to tt data types
      Type elementType = mlir::tt::ttnn::utils::dataTypeToElementType(
          ctx, elementTypeToDataType(type.getElementType()));
      return RankedTensorType::get(type.getShape(), elementType, newLayout);
    });
  }
};

// Converts tensor types to have a ttnn layout attribute for in SystemMemory
//
// Example: tensor<15x10x32xf32> -> tensor<15x10x32xf32, ttnn_layout<...>>
// where ttnn_layout<...> is constructed with SystemMemory
class TTNNLayoutTensorTypeSystemMemoryConverter : public TypeConverter {
public:
  TTNNLayoutTensorTypeSystemMemoryConverter(MLIRContext *ctx,
                                            GridAttr deviceGrid) {
    addConversion([](Type type) { return type; });
    addConversion([ctx, deviceGrid](RankedTensorType type) -> Type {
      Attribute layout = type.getEncoding();
      if (layout) {
        return type;
      }

      TTNNLayoutAttr newLayout = createLayoutAttr(
          ctx, deviceGrid, type, BufferType::SystemMemory, std::nullopt, false);
      // Convert mlir data types to tt data types
      Type elementType = mlir::tt::ttnn::utils::dataTypeToElementType(
          ctx, elementTypeToDataType(type.getElementType()));
      return RankedTensorType::get(type.getShape(), elementType, newLayout);
    });
  }
};

// Rewrites tensor types to have a ttnn layout attribute with default values
class TTNNLayoutTensorTypeRewriter : public RewritePattern {
public:
  TTNNLayoutTensorTypeRewriter(const TypeConverter &dramConverter,
                               const TypeConverter &sysMemoryConverter,
                               MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx),
        dramConverter(&dramConverter), sysMemoryConverter(&sysMemoryConverter) {
  }

  bool convertTypes(ValueRange valueRange, SmallVector<Type> &newTypes) const {
    if (failed(dramConverter->convertTypes(valueRange.getTypes(), newTypes))) {
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

    const TypeConverter *converter =
        funcOp.isDeclaration() ? sysMemoryConverter : dramConverter;
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
    SmallVector<Type> operands;
    SmallVector<Type> results;
    if (isa<func::FuncOp>(op)) {
      return convertFuncType(op, rewriter) ? success() : failure();
    }
    bool updated = false;
    updated |= convertTypes(op->getOperands(), operands);
    updated |= convertTypes(op->getResults(), results);
    return updated ? success() : failure();
  }

  const TypeConverter *dramConverter;
  const TypeConverter *sysMemoryConverter;
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
  // If the defining op is arange, then we need to assume ROW_MAJOR (scalar)
  // element type.
  Type currElementType = ttnnLayoutAttr.getElementType();
  ttir::ArangeOp existingArange = input.getDefiningOp<ttir::ArangeOp>();
  if (existingArange) {
    currElementType = ttnnLayoutAttr.getScalarElementType();
  }

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

  // Create the ToLayoutOp which will convert the input tensor to the desired
  // layout.
  return ttmlir::utils::createDPSOp<ttir::ToLayoutOp>(
      rewriter, loc, ty.getShape(), ty.getElementType(), desiredLayout, input);
}

static bool changeLayoutToHost(DestinationStyleOpInterface &op,
                               OpOperand &operand, PatternRewriter &rewriter,
                               bool isDPSResult) {
  Location newLoc = appendInputSuffix(op.getLoc(), operand.getOperandNumber());
  std::optional<Value> layout = createToLayoutOp(
      rewriter, newLoc, operand.get(), BufferType::SystemMemory,
      nullptr /* desiredMemLayoutAttr */, false /* tiled */);
  if (layout.has_value()) {
    rewriter.modifyOpInPlace(op, [&]() {
      op->setOperand(operand.getOperandNumber(), *layout);
      if (isDPSResult) {
        op->getResult(0).setType(layout->getType());
      }
    });
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
    // Skip toLayout ops
    if (mlir::isa<ttir::ToLayoutOp>(op.getOperation())) {
      return failure();
    }

    assert(op->template hasTrait<ttir::TTIROp::Trait>());
    bool modified = false;
    for (OpOperand &operand : op->getOpOperands()) {
      // Check if the operand is a dps result
      bool isDPSResult = op.isDpsInit(&operand);

      // TTNN Conv2d moves input, weight, and bias from host to device
      // itself. Inserting the ToLayoutOp on these operands is thus problematic.
      if (!isDPSResult &&
          (mlir::isa<ttir::Conv2dOp>(op.getOperation()) ||
           mlir::isa<ttir::ConvTranspose2dOp>(op.getOperation()))) {
        // For the weight input of the conv2d op, it specifically needs to be on
        // host, so we create a host to layout op (issue
        // https://github.com/tenstorrent/tt-mlir/issues/1528).
        if (operand.getOperandNumber() == 1) {
          modified = changeLayoutToHost(op, operand, rewriter, isDPSResult);
        }
        continue;
      }

      // TTNN mesh shard expects host input and output
      // TODO(#2102): This can be removed once the workaround pass can correctly
      // handle cannonicalization of toLayout ops. Currently the workaround pass
      // cannot detect redundant toLayout ops as a result of forcing the output
      // layout and removing them.
      if (mlir::isa<ttir::MeshShardOp>(op.getOperation())) {
        modified = changeLayoutToHost(op, operand, rewriter, isDPSResult);
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

      bool isTiled = shouldTilize(op, operand.getOperandNumber());

      // Given the operand constraint, create the desired layout for the operand
      std::optional<Value> desiredLayout = createToLayoutOp(
          rewriter, newLoc, operand.get(), g_defaultMemorySpaceDevice,
          TensorMemoryLayoutAttr::get(rewriter.getContext(),
                                      g_defaultMemoryLayout),
          isTiled);

      // If layout changed update the operand
      if (desiredLayout) {
        rewriter.modifyOpInPlace(op, [&]() {
          modified = true;
          op->setOperand(operand.getOperandNumber(), *desiredLayout);
          // If operand is dps result, update the result type on current op
          if (isDPSResult) {
            op->getResult(0).setType(desiredLayout->getType());
          }
        });
      }
    }

    return modified ? success() : failure();
  }

private:
  bool shouldTilize(DestinationStyleOpInterface dpsOp,
                    int64_t operandNumber) const {

    Operation *operation = dpsOp.getOperation();

    // TTNN Reshape does not support implicit tilization/untilization
    // Therefore input output layouts should be the same
    if (mlir::isa<ttir::ReshapeOp>(operation) && operandNumber == 1) {
      Value input = dpsOp->getOperand(0);
      RankedTensorType inputType =
          mlir::cast<RankedTensorType>(input.getType());
      TTNNLayoutAttr inputLayout =
          mlir::cast<TTNNLayoutAttr>(inputType.getEncoding());
      return mlir::isa<TileType>(inputLayout.getElementType());
    }

    // These ops constrain to ROW_MAJOR on their operands
    if (mlir::isa<ttir::Conv2dOp>(operation) ||
        mlir::isa<ttir::SliceOp>(operation)) {
      return false;
    }
    return true;
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
    if (!callOp->hasAttr("hoisted_call")) {
      return failure();
    }

    // Create a FromDevice operation for each operand.
    SmallVector<Value, 4> fromDeviceOperands;
    size_t locIdx = 0;
    for (auto operand : callOp.getOperands()) {
      Location newLoc = appendInputSuffix(callOp.getLoc(), locIdx++);
      std::optional<Value> optionalLayoutOp = createToLayoutOp(
          rewriter, newLoc, operand, BufferType::SystemMemory,
          nullptr /* desiredMemLayoutAttr */, false /* tiled */);
      fromDeviceOperands.push_back(
          optionalLayoutOp.has_value() ? optionalLayoutOp.value() : operand);
    }

    // Original CallOp defaults to device tensor return type now, we need to
    // replace with return types which TTNNLayoutFuncInputOutputTypeRewriter
    // updated.
    func::FuncOp funcOp = dyn_cast<func::FuncOp>(
        SymbolTable::lookupNearestSymbolFrom(callOp, callOp.getCalleeAttr()));
    // Create the original CallOp with the new inputs on host.
    auto newCallOp = rewriter.create<func::CallOp>(
        callOp.getLoc(), callOp.getCallee(), funcOp.getResultTypes(),
        fromDeviceOperands);

    rewriter.replaceOp(callOp, newCallOp);

    return success();
  }
};

// Update the input/output layouts of a function
// This rewriter checks for special ops (e.g. mesh shard ops) and updates the
// function input/output layouts accordingly
class TTNNLayoutFuncInputOutputTypeRewriter
    : public OpRewritePattern<mlir::func::FuncOp> {
public:
  TTNNLayoutFuncInputOutputTypeRewriter(MLIRContext *ctx, GridAttr deviceGrid)
      : OpRewritePattern<mlir::func::FuncOp>(ctx), deviceGrid(deviceGrid) {}

  LogicalResult matchAndRewrite(mlir::func::FuncOp funcOp,
                                PatternRewriter &rewriter) const final {
    bool modified = false;
    modified |= rewriteInput(funcOp, rewriter);
    modified |= rewriteOutput(funcOp, rewriter);
    return modified ? success() : failure();
  }

private:
  GridAttr deviceGrid;

  bool rewriteInput(mlir::func::FuncOp funcOp,
                    PatternRewriter &rewriter) const {
    bool modified = false;
    SmallVector<Type> inputTypes;
    SmallVector<Type> outputTypes(funcOp.getResultTypes());
    // Func declarations are always CPU-hoisted funcs, which means all inputs
    // should be in system_memory.
    if (funcOp.isDeclaration()) {
      return false;
    }

    Block &entryBlock = funcOp.getBody().front();

    for (BlockArgument &arg : entryBlock.getArguments()) {
      if (!mlir::isa<RankedTensorType>(arg.getType()) ||
          !shouldForceInputSystemMemory(arg)) {
        inputTypes.push_back(arg.getType());
        continue;
      }
      RankedTensorType ty = mlir::cast<RankedTensorType>(arg.getType());
      RankedTensorType newType = toSystemMemoryType(funcOp.getContext(), ty);

      inputTypes.push_back(newType);
      modified = arg.getType() != newType;
      // Ensure entryBlock's arg types match func's arg types.
      if (modified) {
        for (uint32_t i = 0; i < entryBlock.getNumArguments(); i++) {
          entryBlock.getArgument(i).setType(inputTypes[i]);
        }
      }
    }
    if (modified) {
      FunctionType newFuncType =
          rewriter.getFunctionType(inputTypes, outputTypes);
      funcOp.setFunctionType(newFuncType);
    }
    return modified;
  }

  bool rewriteOutput(mlir::func::FuncOp funcOp,
                     PatternRewriter &rewriter) const {
    // Func declarations are always CPU-hoisted funcs, which means all outputs
    // should stay in system  memory.
    if (funcOp.isDeclaration()) {
      return false;
    }
    SmallVector<mlir::func::ReturnOp> returnOps;
    funcOp.walk(
        [&](mlir::func::ReturnOp returnOp) { returnOps.push_back(returnOp); });

    bool forceSysMem = false;
    for (auto returnOp : returnOps) {
      forceSysMem |= shouldForceOutputSystemMemory(returnOp);
    }
    if (!forceSysMem) {
      return false;
    }

    bool modified = false;
    SmallVector<Type> inputTypes(funcOp.getArgumentTypes());
    SmallVector<Type> outputTypes;
    for (auto type : funcOp.getResultTypes()) {
      if (!mlir::isa<RankedTensorType>(type)) {
        outputTypes.push_back(type);
        continue;
      }
      RankedTensorType tensorType = mlir::cast<RankedTensorType>(type);
      RankedTensorType newType =
          toSystemMemoryType(funcOp.getContext(), tensorType);
      outputTypes.push_back(newType);
      modified |= (tensorType != newType);
    }
    if (modified) {
      FunctionType newFuncType =
          rewriter.getFunctionType(inputTypes, outputTypes);
      funcOp.setFunctionType(newFuncType);
    }
    return modified;
  }

  RankedTensorType toSystemMemoryType(MLIRContext *ctx,
                                      RankedTensorType ty) const {
    TTNNLayoutAttr newLayout = createLayoutAttr(
        ctx, deviceGrid, ty, BufferType::SystemMemory, std::nullopt, false);
    auto newType =
        RankedTensorType::get(ty.getShape(), ty.getElementType(), newLayout);
    return newType;
  }

  bool shouldForceInputSystemMemory(BlockArgument arg) const {
    for (Operation *user : arg.getUsers()) {
      if (mlir::isa<ttir::MeshShardOp>(user)) {
        return true;
      }
      // For the weight input of the conv2d op, it specifically needs to be on
      // host (issue https://github.com/tenstorrent/tt-mlir/issues/1528).
      if ((mlir::isa<ttir::Conv2dOp>(user) ||
           mlir::isa<ttir::ConvTranspose2dOp>(user)) &&
          user->getOperand(1) == arg) {
        return true;
      }
    }
    return false;
  }

  bool shouldForceOutputSystemMemory(mlir::func::ReturnOp returnOp) const {
    for (Value operand : returnOp.getOperands()) {
      if (!mlir::isa<RankedTensorType>(operand.getType())) {
        continue;
      }
      if (operand.getDefiningOp<ttir::MeshShardOp>()) {
        return true;
      }
    }
    return false;
  }
};

// Updates the layout of the operands of a func::ReturnOp
// forces it to dram interleaved tiled unless we need special handling
class TTNNLayoutFuncReturnRewriter
    : public OpRewritePattern<mlir::func::ReturnOp> {
public:
  TTNNLayoutFuncReturnRewriter(MLIRContext *ctx)
      : OpRewritePattern<mlir::func::ReturnOp>(ctx) {}

  LogicalResult matchAndRewrite(mlir::func::ReturnOp op,
                                PatternRewriter &rewriter) const final {
    bool modified = false;
    for (OpOperand &operand : op->getOpOperands()) {
      if (!mlir::isa<RankedTensorType>(operand.get().getType())) {
        continue;
      }
      bool forceHost = shouldForceSystemMemory(operand.get());
      BufferType desiredBufferType =
          forceHost ? BufferType::SystemMemory : g_defaultMemorySpaceDevice;

      TensorMemoryLayoutAttr desiredMemLayoutAttr =
          forceHost ? nullptr
                    : TensorMemoryLayoutAttr::get(rewriter.getContext(),
                                                  g_defaultMemoryLayout);

      bool isTiled = !forceHost;

      Location newLoc =
          appendInputSuffix(op.getLoc(), operand.getOperandNumber());
      std::optional<Value> updatedLayout =
          createToLayoutOp(rewriter, newLoc, operand.get(), desiredBufferType,
                           desiredMemLayoutAttr, isTiled);
      if (updatedLayout.has_value()) {
        rewriter.modifyOpInPlace(op, [&]() {
          op.setOperand(operand.getOperandNumber(), *updatedLayout);
        });

        modified = true;
      }
    }
    return modified ? success() : failure();
  }

private:
  // Return op output should be on host if it's a result of a mesh shard op
  bool shouldForceSystemMemory(Value operandValue) const {
    if (operandValue.getDefiningOp<ttir::MeshShardOp>()) {
      return true;
    }
    return false;
  }
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
      TTNNLayoutTensorTypeConverter typeDefaultConverter(
          &getContext(), device.getWorkerGrid());
      TTNNLayoutTensorTypeSystemMemoryConverter typeSystemMemoryConverter(
          &getContext(), device.getWorkerGrid());
      RewritePatternSet patterns(&getContext());
      // Set the tensor layouts to have proper values
      patterns.add<TTNNLayoutTensorTypeRewriter>(
          typeDefaultConverter, typeSystemMemoryConverter, &getContext());
      // Set the tensor layouts of the func op inputs and outputs based on their
      // consumers/producers. For example, if a func op input is consumed by a
      // mesh shard op, that input tensor should be on host
      patterns.add<TTNNLayoutFuncInputOutputTypeRewriter>(
          &getContext(), device.getWorkerGrid());
      FrozenRewritePatternSet patternSet(std::move(patterns));
      if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
        signalPassFailure();
        return;
      }
    }
    {
      RewritePatternSet patterns(&getContext());
      // Takes all TTIR ops which have DPS operands
      // and rewrites its operands and result to have the correct layout
      // with respect to operand constraints.
      patterns.add<TTNNLayoutDPSOperandsRewriter>(&getContext());
      // Update the return op output layout based on its consumers
      // Logic here should match that of TTNNLayoutFuncInputOutputTypeRewriter
      patterns.add<TTNNLayoutFuncReturnRewriter>(&getContext());
      patterns.add<TTNNLayoutHoistedFuncCallRewriter>(&getContext());
      FrozenRewritePatternSet patternSet(std::move(patterns));
      GreedyRewriteConfig config = GreedyRewriteConfig();
      config.useTopDownTraversal = true;
      if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet,
                                              config))) {
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

} // namespace mlir::tt::ttnn
