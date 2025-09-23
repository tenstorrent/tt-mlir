// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/UniformTypeRewriter.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNLAYOUT
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

// Default collapse dims for affine map (d0, d1, d2) -> (d0 <> d1, d2)
static const std::array<std::pair<int64_t, int64_t>, 1> g_defaultCollapseDims =
    {{{0, -1}}};

static const llvm::SmallDenseMap<BufferType, std::optional<TensorMemoryLayout>,
                                 4>
    g_bufferLayoutMap = {
        {BufferType::DRAM, TensorMemoryLayout::Interleaved},
        {BufferType::L1, TensorMemoryLayout::Interleaved},
        {BufferType::SystemMemory, std::nullopt},
};

static const BufferType g_defaultMemorySpaceDevice = BufferType::DRAM;

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

static TensorMemoryLayoutAttr getMemoryLayoutAttr(MLIRContext *ctx,
                                                  BufferType bufferType) {
  std::optional<TensorMemoryLayout> layout = g_bufferLayoutMap.at(bufferType);
  if (layout) {
    return TensorMemoryLayoutAttr::get(ctx, layout.value());
  }

  return TensorMemoryLayoutAttr{};
}

static TTNNLayoutAttr createLayoutAttr(
    MLIRContext *ctx, ttcore::GridAttr deviceGrid, RankedTensorType type,
    BufferType bufferType = g_defaultMemorySpaceDevice, bool isTiled = true) {

  // Default to single core grid.
  ttcore::GridAttr tensorGrid = ttcore::GridAttr::get(ctx);

  llvm::ArrayRef<std::pair<int64_t, int64_t>> collapseDimsRef(
      g_defaultCollapseDims);

  // Force TileType for tensors
  Type elementType = type.getElementType();
  // The tile type for a quantized type is the desired type.
  // Ex: for a quant p of fp32->int8, the storage type is int8.
  if (auto quantType =
          mlir::dyn_cast<mlir::quant::QuantizedType>(elementType)) {
    elementType = isTiled ? ttcore::TileType::get(quantType.getStorageType())
                          : quantType.getStorageType();
  } else {
    elementType = isTiled ? ttcore::TileType::get(type.getElementType())
                          : type.getElementType();
  }
  mlir::Attribute encoding = type.getEncoding();
  ttcore::TensorMeshAttr tensorMeshAttr;
  if (auto encodingMeshSharding =
          mlir::dyn_cast_if_present<ttcore::TensorMeshAttr>(encoding)) {
    tensorMeshAttr = encodingMeshSharding;
  } else if (auto layout =
                 mlir::dyn_cast_if_present<TTNNLayoutAttr>(encoding)) {
    tensorMeshAttr = layout.getTensorMesh();
  }
  TensorMemoryLayoutAttr memoryLayoutAttr =
      getMemoryLayoutAttr(ctx, bufferType);
  return TTNNLayoutAttr::get(ctx, type.getShape(), elementType, bufferType,
                             tensorGrid, memoryLayoutAttr, tensorMeshAttr,
                             collapseDimsRef);
}

static bool shouldMeshShardOpForceSystemMemory(mlir::Operation *srcOp) {
  auto meshShardOp = mlir::dyn_cast_if_present<ttir::MeshShardOp>(srcOp);
  return meshShardOp && meshShardOp.getShardType() !=
                            mlir::tt::ttcore::MeshShardType::Identity;
}

//===----------------------------------------------------------------------===//
// To layout pass
//===----------------------------------------------------------------------===//

// Converts tensor types to have a ttnn layout attribute with default values
//
// Example: tensor<15x10x32xf32> -> tensor<15x10x32xf32, ttnn_layout<...>>
// where ttnn_layout<...> is constructed with default values
// Dram, MemoryLayout::Interleaved, Grid<1x1>
namespace {
class TTNNLayoutTensorTypeConverter : public TypeConverter {
public:
  TTNNLayoutTensorTypeConverter(MLIRContext *ctx, ttcore::GridAttr deviceGrid) {
    addConversion([](Type type) { return type; });
    addConversion([ctx, deviceGrid](RankedTensorType type) -> Type {
      if (isa_and_nonnull<TTNNLayoutAttr>(type.getEncoding())) {
        return type;
      }

      TTNNLayoutAttr newLayout = createLayoutAttr(ctx, deviceGrid, type);
      return RankedTensorType::get(type.getShape(), type.getElementType(),
                                   newLayout);
    });
  }
};
} // namespace

// Given desired buffer type, memory layout and type checks if the input tensor
// needs to be converted to the desired layout. If it does, creates a new
// EmptyOp/ConstantOp/EmptyOp + ToLayoutOp depending on the input.
static std::optional<Value> createToLayoutOp(PatternRewriter &rewriter,
                                             Location loc, Value input,
                                             BufferType desiredBufferType,
                                             bool tiled) {
  TensorMemoryLayoutAttr desiredMemLayoutAttr =
      getMemoryLayoutAttr(rewriter.getContext(), desiredBufferType);

  // Get type
  RankedTensorType ty = mlir::cast<RankedTensorType>(input.getType());

  // Get ttnn layout from the type
  TTNNLayoutAttr ttnnLayoutAttr = mlir::cast<TTNNLayoutAttr>(ty.getEncoding());
  // Get buffer type (i.e DRAM/L1 etc)
  BufferType currBufferType = ttnnLayoutAttr.getBufferType();

  // Get mesh sharding
  ttcore::TensorMeshAttr desiredTensorMesh = ttnnLayoutAttr.getTensorMesh();

  // Get the current element type (i.e bf16/TileType etc)
  Type currElementType = ttnnLayoutAttr.getElementType();

  // Get mem layout. If the tensor is on host layout is null
  TensorMemoryLayoutAttr currMemLayout = ttnnLayoutAttr.getMemLayout();

  // Get element type that should be used in the new ttnn layout
  Type desiredElementType =
      tiled ? ttcore::TileType::get(ty.getElementType()) : ty.getElementType();

  // If the element type is quantized, use the desired type.
  // Ex: for a quant op of fp32->int8, the storage type is int8.
  if (auto quantType =
          mlir::dyn_cast<mlir::quant::QuantizedType>(ty.getElementType())) {
    desiredElementType = tiled
                             ? ttcore::TileType::get(quantType.getStorageType())
                             : quantType.getStorageType();
  }

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
      ttnnLayoutAttr.getGrid(), desiredMemLayoutAttr, desiredTensorMesh,
      g_defaultCollapseDims);

  // Create the ToLayoutOp which will convert the input tensor to the desired
  // layout.
  return ttir::utils::createDPSOp<ttir::ToLayoutOp>(
             rewriter, loc, ty.getShape(), ty.getElementType(), desiredLayout,
             input, nullptr)
      ->getResult(0);
}

static bool changeLayoutToHost(DestinationStyleOpInterface &op,
                               OpOperand &operand, PatternRewriter &rewriter,
                               bool isDPSResult) {
  Location newLoc = appendInputSuffix(op.getLoc(), operand.getOperandNumber());
  std::optional<Value> layout =
      createToLayoutOp(rewriter, newLoc, operand.get(),
                       BufferType::SystemMemory, false /* tiled */);
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
namespace {
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

      // TTNN mesh shard expects host input and output
      // TODO(#2291): This can be removed once the workaround pass can correctly
      // handle canonicalization of toLayout ops (#2102). Currently the
      // workaround pass cannot detect redundant toLayout ops as a result of
      // forcing the output layout and removing them.
      if (shouldMeshShardOpForceSystemMemory(op.getOperation())) {
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

      bool isTiled = shouldTilize(op, operand.getOperandNumber(), isDPSResult);

      // Given the operand constraint, create the desired layout for the operand
      std::optional<Value> desiredLayout = createToLayoutOp(
          rewriter, newLoc, operand.get(), g_defaultMemorySpaceDevice, isTiled);

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
  bool shouldTilize(DestinationStyleOpInterface dpsOp, int64_t operandNumber,
                    bool isDPSResult) const {

    Operation *operation = dpsOp.getOperation();

    // TTNN Reshape does not support implicit tilization/untilization
    // Therefore input output layouts should be the same
    if (mlir::isa<ttir::ReshapeOp>(operation) && operandNumber == 1) {
      Value input = dpsOp->getOperand(0);
      RankedTensorType inputType =
          mlir::cast<RankedTensorType>(input.getType());
      TTNNLayoutAttr inputLayout =
          mlir::cast<TTNNLayoutAttr>(inputType.getEncoding());
      return mlir::isa<ttcore::TileType>(inputLayout.getElementType());
    }

    return true;
  }
};
} // namespace

namespace {
class TTNNLayoutHoistedFuncCallRewriter
    : public OpRewritePattern<func::CallOp> {
public:
  TTNNLayoutHoistedFuncCallRewriter(MLIRContext *ctx)
      : OpRewritePattern<func::CallOp>(ctx) {}

  // Match and rewrite the CallOp.
  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    if (!callOp->hasAttr(ttir::HoistedCallAttr::name)) {
      return failure();
    }

    // Create a FromDevice operation for each operand.
    SmallVector<Value, 4> fromDeviceOperands;
    size_t locIdx = 0;
    for (auto operand : callOp.getOperands()) {
      Location newLoc = appendInputSuffix(callOp.getLoc(), locIdx++);
      std::optional<Value> optionalLayoutOp =
          createToLayoutOp(rewriter, newLoc, operand, BufferType::SystemMemory,
                           false /* tiled */);
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

    newCallOp->setAttr(ttmlir::utils::g_cpuHoistFuncCallAttrName,
                       mlir::UnitAttr::get(rewriter.getContext()));

    rewriter.replaceOp(callOp, newCallOp);

    return success();
  }
};
} // namespace

// Update the input/output layouts of a function
// This rewriter checks for special ops (e.g. mesh shard ops) and updates the
// function input/output layouts accordingly
namespace {
class TTNNLayoutFuncInputOutputTypeRewriter
    : public OpRewritePattern<mlir::func::FuncOp> {
public:
  TTNNLayoutFuncInputOutputTypeRewriter(MLIRContext *ctx,
                                        ttcore::GridAttr deviceGrid)
      : OpRewritePattern<mlir::func::FuncOp>(ctx), deviceGrid(deviceGrid) {}

  LogicalResult matchAndRewrite(mlir::func::FuncOp funcOp,
                                PatternRewriter &rewriter) const final {
    bool modified = false;
    rewriter.startOpModification(funcOp);
    modified |= rewriteInput(funcOp, rewriter);
    modified |= rewriteOutput(funcOp, rewriter);
    modified |= rewriteFuncDecl(funcOp, rewriter);
    if (!modified) {
      rewriter.cancelOpModification(funcOp);
      return failure();
    }
    rewriter.finalizeOpModification(funcOp);
    return success();
  }

private:
  ttcore::GridAttr deviceGrid;

  // Rewrite the function declaration to have system memory in/out types
  // Func declarations are used by CPU-hoisted functions.
  bool rewriteFuncDecl(mlir::func::FuncOp funcOp,
                       PatternRewriter &rewriter) const {
    if (!funcOp.isDeclaration()) {
      return false;
    }

    MLIRContext *context = funcOp.getContext();

    auto convertType = [&](Type type) -> Type {
      return toSystemMemoryType(context, mlir::cast<RankedTensorType>(type));
    };

    SmallVector<Type> inputTypes, outputTypes;

    for (Type arg : funcOp.getArgumentTypes()) {
      inputTypes.push_back(convertType(arg));
    }

    for (Type result : funcOp.getResultTypes()) {
      outputTypes.push_back(convertType(result));
    }

    FunctionType newType =
        rewriter.getType<FunctionType>(inputTypes, outputTypes);

    if (funcOp.getFunctionType() == newType) {
      return false;
    }

    funcOp.setFunctionType(newType);
    return true;
  }

  bool rewriteInput(mlir::func::FuncOp funcOp,
                    PatternRewriter &rewriter) const {
    // Func declarations are always CPU-hoisted funcs, which means all inputs
    // should stay in system  memory.
    if (funcOp.isDeclaration()) {
      return false;
    }
    bool modified = false;
    Block &entryBlock = funcOp.getBody().front();

    SmallVector<Type> inputTypes;
    SmallVector<Type> outputTypes(funcOp.getResultTypes());
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
    }

    if (modified) {
      FunctionType newFuncType =
          rewriter.getFunctionType(inputTypes, outputTypes);
      funcOp.setFunctionType(newFuncType);
      for (uint32_t i = 0; i < entryBlock.getNumArguments(); i++) {
        entryBlock.getArgument(i).setType(inputTypes[i]);
      }
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

    bool modified = false;
    SmallVector<Type> inputTypes(funcOp.getArgumentTypes());
    SmallVector<Type> outputTypes;
    funcOp.walk([&](mlir::func::ReturnOp returnOp) {
      for (auto [type, operand] :
           llvm::zip_equal(funcOp.getResultTypes(), returnOp->getOperands())) {
        if (!mlir::isa<RankedTensorType>(type) ||
            !shouldMeshShardOpForceSystemMemory(operand.getDefiningOp())) {
          outputTypes.push_back(type);
          continue;
        }
        RankedTensorType tensorType = mlir::cast<RankedTensorType>(type);
        RankedTensorType newType =
            toSystemMemoryType(funcOp.getContext(), tensorType);
        outputTypes.push_back(newType);
        modified |= (tensorType != newType);
      }
      return mlir::WalkResult::interrupt();
    });

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
        ctx, deviceGrid, ty, BufferType::SystemMemory, false /* isTiledOpt */);
    auto newType =
        RankedTensorType::get(ty.getShape(), ty.getElementType(), newLayout);
    return newType;
  }

  bool shouldForceInputSystemMemory(BlockArgument arg) const {
    for (Operation *user : arg.getUsers()) {
      if (shouldMeshShardOpForceSystemMemory(user)) {
        return true;
      }
    }

    // For block arguments which are maked as conv2d weights leave them on host.
    func::FuncOp owningFunc = cast<func::FuncOp>(arg.getOwner()->getParentOp());
    uint32_t argIdx = arg.getArgNumber();

    if (owningFunc.getArgAttr(argIdx, ttmlir::utils::g_conv2dWeightAttrName)) {
      return true;
    }

    return false;
  }
};
} // namespace

// Updates the layout of the operands of a func::ReturnOp
// forces it to dram interleaved tiled unless we need special handling
namespace {
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

      bool isTiled = !forceHost;

      Location newLoc =
          appendInputSuffix(op.getLoc(), operand.getOperandNumber());
      std::optional<Value> updatedLayout = createToLayoutOp(
          rewriter, newLoc, operand.get(), desiredBufferType, isTiled);
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
    if (shouldMeshShardOpForceSystemMemory(operandValue.getDefiningOp())) {
      return true;
    }
    return false;
  }
};
} // namespace

namespace {
class TTNNLayout : public impl::TTNNLayoutBase<TTNNLayout> {
public:
  using impl::TTNNLayoutBase<TTNNLayout>::TTNNLayoutBase;

  void runOnOperation() final {
    // First add default attribute to all tensors. Example:
    // Given tensor type: tensor<15x10x32xf32>
    // we construct a ttnn layout attribute with default values:
    // ttnn_layout<affine_map, grid<1x1>, memref<<15x64>xf32, #system_memory>
    {
      ttcore::DeviceAttr device = ttcore::lookupDevice(getOperation());
      assert(device && "Device not found");
      TTNNLayoutTensorTypeConverter typeDefaultConverter(
          &getContext(), device.getWorkerGrid());
      RewritePatternSet patterns(&getContext());
      // Set the tensor layouts to have proper values
      patterns.add<ttir::UniformTypeRewriter>(typeDefaultConverter,
                                              &getContext());
      // Set the tensor layouts of the func op inputs and outputs based on their
      // consumers/producers. For example, if a func op input is consumed by a
      // mesh shard op, that input tensor should be on host
      patterns.add<TTNNLayoutFuncInputOutputTypeRewriter>(
          &getContext(), device.getWorkerGrid());
      FrozenRewritePatternSet patternSet(std::move(patterns));
      if (failed(applyPatternsGreedily(getOperation(), patternSet))) {
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
      config.setUseTopDownTraversal(true);
      if (failed(applyPatternsGreedily(getOperation(), patternSet, config))) {
        signalPassFailure();
        return;
      }
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::ttnn::TTNNDialect>();
    registry.insert<mlir::tt::ttcore::TTCoreDialect>();
    registry.insert<mlir::func::FuncDialect>();
  }
};
} // namespace

} // namespace mlir::tt::ttnn
