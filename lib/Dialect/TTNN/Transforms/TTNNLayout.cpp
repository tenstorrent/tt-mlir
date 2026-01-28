// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/UniformTypeRewriter.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/FunctionTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Casting.h"

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

static TTNNLayoutAttr createLayoutAttr(MLIRContext *ctx,
                                       ttcore::GridAttr deviceGrid,
                                       RankedTensorType type,
                                       BufferType bufferType, bool isTiled) {

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

// Converts tensor types to have a ttnn layout attribute with provided encoding
// parameters.
//
namespace {
class TTNNLayoutTensorTypeConverter : public TypeConverter {
public:
  TTNNLayoutTensorTypeConverter(MLIRContext *ctx, ttcore::GridAttr deviceGrid,
                                BufferType bufferType, bool isTiled) {
    addConversion([](Type type) { return type; });
    addConversion([=](RankedTensorType type) -> Type {
      if (isa_and_nonnull<TTNNLayoutAttr>(type.getEncoding())) {
        return type;
      }

      TTNNLayoutAttr newLayout =
          createLayoutAttr(ctx, deviceGrid, type, bufferType, isTiled);
      return RankedTensorType::get(type.getShape(), type.getElementType(),
                                   newLayout);
    });
  }
};
} // namespace

static std::optional<RankedTensorType>
createDesiredType(PatternRewriter &rewriter, RankedTensorType ty,
                  BufferType desiredBufferType, bool tiled) {
  TensorMemoryLayoutAttr desiredMemLayoutAttr =
      getMemoryLayoutAttr(rewriter.getContext(), desiredBufferType);

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
  TTNNLayoutAttr encoding = rewriter.getAttr<TTNNLayoutAttr>(
      ty.getShape(), desiredElementType, desiredBufferType,
      ttnnLayoutAttr.getGrid(), desiredMemLayoutAttr, desiredTensorMesh,
      g_defaultCollapseDims);

  return mlir::RankedTensorType::get(ty.getShape(), ty.getElementType(),
                                     encoding);
}

// Given desired buffer type, memory layout and type checks if the input tensor
// needs to be converted to the desired layout. If it does, creates a new
// EmptyOp/ConstantOp/EmptyOp + ToLayoutOp depending on the input.
static std::optional<Value> createToLayoutOp(PatternRewriter &rewriter,
                                             Location loc, Value input,
                                             BufferType desiredBufferType,
                                             bool tiled) {
  // Get type
  RankedTensorType inputType = mlir::cast<RankedTensorType>(input.getType());
  std::optional<RankedTensorType> desiredType =
      createDesiredType(rewriter, inputType, desiredBufferType, tiled);

  if (!desiredType) {
    return std::nullopt;
  }

  // Create the ToLayoutOp which will convert the input tensor to the desired
  // layout.
  return ttir::utils::createDPSOp<ttir::ToLayoutOp>(rewriter, loc, *desiredType,
                                                    input, nullptr)
      ->getResult(0);
}

// Updates the layout of the operands of a TTIR ops which have DPS operands.
// This function rewrites the operands and result to have the correct layout
// with respect to operand constraints.
namespace {
class TTNNLayoutRewriter : public OpInterfaceRewritePattern<ttir::TTIROp> {
public:
  TTNNLayoutRewriter(MLIRContext *ctx)
      : OpInterfaceRewritePattern<ttir::TTIROp>(ctx) {}

  LogicalResult matchAndRewrite(ttir::TTIROp op,
                                PatternRewriter &rewriter) const final {
    // Skip toLayout ops
    if (mlir::isa<ttir::ToLayoutOp>(op) ||
        op->hasTrait<mlir::tt::ttcore::Trait::TTCoreCreationOpTrait>() ||
        mlir::isa<ttir::MeshShardOp>(op)) {
      return failure();
    }

    bool modified = false;
    for (OpOperand &operand : op->getOpOperands()) {

      // If the operand is a BroadcastOp or a ToLayout op do not put a
      // ToLayoutOp on its output
      if (operand.get().getDefiningOp<ttir::BroadcastOp>() ||
          operand.get().getDefiningOp<ttir::ToLayoutOp>()) {
        continue;
      }

      bool shouldTilize = true;
      if (mlir::isa<ttir::MeshPartitionOp>(op)) {
        shouldTilize = false;
      }

      Location newLoc =
          appendInputSuffix(op->getLoc(), operand.getOperandNumber());

      // Given the operand constraint, create the desired layout for the operand
      std::optional<Value> desiredLayout =
          createToLayoutOp(rewriter, newLoc, operand.get(),
                           g_defaultMemorySpaceDevice, /*tiled=*/shouldTilize);

      // If layout changed update the operand
      if (desiredLayout) {
        rewriter.modifyOpInPlace(op, [&]() {
          modified = true;
          op->setOperand(operand.getOperandNumber(), *desiredLayout);
        });
      }
    }

    for (auto it : llvm::enumerate(op->getResultTypes())) {
      RankedTensorType ty = mlir::cast<RankedTensorType>(it.value());
      std::optional<RankedTensorType> desiredType = createDesiredType(
          rewriter, ty, g_defaultMemorySpaceDevice, /*tiled=*/shouldTilize(op));
      if (desiredType) {
        rewriter.modifyOpInPlace(op, [&]() {
          modified = true;
          op->getResult(it.index()).setType(*desiredType);
        });
      }
    }

    return modified ? success() : failure();
  }

private:
  bool shouldTilize(Operation *op) const {

    // TTNN Reshape does not support implicit tilization/untilization
    // Therefore input output layouts should be the same
    if (auto reshapeOp = mlir::dyn_cast<ttir::ReshapeOp>(op)) {
      RankedTensorType inputType =
          mlir::cast<RankedTensorType>(reshapeOp.getType());
      TTNNLayoutAttr inputLayout =
          mlir::cast<TTNNLayoutAttr>(inputType.getEncoding());
      return mlir::isa<ttcore::TileType>(inputLayout.getElementType());
    }

    // Conv3d produces ROW_MAJOR output at runtime (experimental op)
    if (mlir::isa<ttir::Conv3dOp>(op)) {
      return false;
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
    if (!callOp->hasAttr(ttir::CPUHoistedCallAttr::name)) {
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

// Rewrite LoadCachedOp result types to match the callee function signature.
namespace {
class TTNNLayoutLoadCachedOpTypeRewriter
    : public OpRewritePattern<ttcore::LoadCachedOp> {
public:
  TTNNLayoutLoadCachedOpTypeRewriter(MLIRContext *ctx)
      : OpRewritePattern<ttcore::LoadCachedOp>(ctx) {}

  LogicalResult matchAndRewrite(ttcore::LoadCachedOp loadCachedOp,
                                PatternRewriter &rewriter) const override {
    // Look up the callee function.
    func::FuncOp funcOp =
        dyn_cast<func::FuncOp>(SymbolTable::lookupNearestSymbolFrom(
            loadCachedOp, loadCachedOp.getCalleeAttr()));
    if (!funcOp) {
      return failure();
    }

    bool modified = false;

    // Rewrite result types to match function signature.
    for (auto [idx, callResultType] :
         llvm::enumerate(loadCachedOp->getResultTypes())) {
      if (idx >= funcOp.getResultTypes().size()) {
        break;
      }
      auto funcResultType = funcOp.getResultTypes()[idx];
      if (callResultType != funcResultType) {
        loadCachedOp->getResult(idx).setType(funcResultType);
        modified = true;
      }
    }

    return success(modified);
  }
};
} // namespace

namespace {
class TTNNLayoutMeshShardRewriter : public OpRewritePattern<ttir::MeshShardOp> {
public:
  TTNNLayoutMeshShardRewriter(MLIRContext *ctx)
      : OpRewritePattern<ttir::MeshShardOp>(ctx) {}
  // Match and rewrite the MeshShardOp.
  LogicalResult matchAndRewrite(ttir::MeshShardOp op,
                                PatternRewriter &rewriter) const override {
    // TTNN mesh shard expects host input and output
    // TODO(#2291): This can be removed once the workaround pass can correctly
    // handle canonicalization of toLayout ops (#2102). Currently the
    // workaround pass cannot detect redundant toLayout ops as a result of
    // forcing the output layout and removing them.
    if (!shouldMeshShardOpForceSystemMemory(op.getOperation())) {
      return failure();
    }

    bool modified = false;
    Value input = op.getOperand();
    Location newLoc = appendInputSuffix(op.getLoc(), 0);
    std::optional<Value> inputLayout = createToLayoutOp(
        rewriter, newLoc, input, BufferType::SystemMemory, /* tiled */ false);
    if (inputLayout.has_value()) {
      rewriter.modifyOpInPlace(op, [&]() { op->setOperand(0, *inputLayout); });
      modified = true;
    }

    RankedTensorType resultType =
        mlir::cast<RankedTensorType>(op.getResult().getType());
    TTNNLayoutAttr newLayout =
        createLayoutAttr(rewriter.getContext(), nullptr, resultType,
                         BufferType::SystemMemory, /* isTiled */ false);
    if (newLayout != resultType.getEncoding()) {
      auto resultSystemMemoryType = RankedTensorType::get(
          resultType.getShape(), resultType.getElementType(), newLayout);
      rewriter.modifyOpInPlace(
          op, [&]() { op->getResult(0).setType(resultSystemMemoryType); });
      modified = true;
    }
    return success(modified);
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

  // Rewrite the CPU-hoisted function declarations to have system memory in/out
  // types.
  bool rewriteFuncDecl(mlir::func::FuncOp funcOp,
                       PatternRewriter &rewriter) const {
    if (!ttmlir::utils::isForwardCPUDeclarationFunc(funcOp)) {
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
    // For CPU-hoisted declarations, all inputs should stay in the system
    // memory.
    if (ttmlir::utils::isForwardCPUDeclarationFunc(funcOp)) {
      return false;
    }
    bool modified = false;
    Block &entryBlock = funcOp.getBody().front();

    SmallVector<Type> inputTypes;
    SmallVector<Type> outputTypes(funcOp.getResultTypes());
    for (BlockArgument &arg : entryBlock.getArguments()) {
      if (!mlir::isa<RankedTensorType>(arg.getType())) {
        inputTypes.push_back(arg.getType());
        continue;
      }

      RankedTensorType currentType =
          mlir::cast<RankedTensorType>(arg.getType());

      RankedTensorType newType;
      if (shouldForceInputSystemMemory(arg)) {
        newType = toSystemMemoryType(funcOp.getContext(), currentType);
      } else {
        newType = currentType;
      }

      if (shouldForceInputRowMajor(arg)) {
        newType = toRowMajorType(funcOp.getContext(), newType);
      }

      inputTypes.push_back(newType);
      modified |= arg.getType() != newType;
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
    // For CPU-hoisted declarations, all outputs should stay in the system
    // memory.
    if (ttmlir::utils::isForwardCPUDeclarationFunc(funcOp)) {
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
        ctx, deviceGrid, ty, BufferType::SystemMemory, /*isTiled=*/false);
    auto newType =
        RankedTensorType::get(ty.getShape(), ty.getElementType(), newLayout);
    return newType;
  }

  bool shouldForceInputSystemMemory(BlockArgument arg) const {
    func::FuncOp owningFunc = cast<func::FuncOp>(arg.getOwner()->getParentOp());

    // For block arguments which are maked as conv2d weights leave them on host.
    uint32_t argIdx = arg.getArgNumber();
    if (owningFunc.getArgAttr(argIdx, ttmlir::utils::g_conv2dWeightAttrName)) {
      return true;
    }

    // If function is marked as const-eval leave inputs as is.
    // TTNNConstEvalInputsToSystemMemory pass will handle them.
    if (ttmlir::utils::isConstEvalFunc(owningFunc)) {
      return false;
    }

    for (Operation *user : arg.getUsers()) {
      if (shouldMeshShardOpForceSystemMemory(user)) {
        return true;
      }
    }

    return false;
  }

  bool shouldForceInputRowMajor(BlockArgument arg) const {
    func::FuncOp owningFunc = cast<func::FuncOp>(arg.getOwner()->getParentOp());

    // KV cache arguments should not be forced to row major.
    if (owningFunc.getArgAttr(arg.getArgNumber(), ttcore::g_kvCacheAttrName)) {
      return false;
    }

    for (Operation *user : arg.getUsers()) {
      // MeshShardOp inputs should be tiled.
      if (mlir::isa<ttir::MeshShardOp>(user)) {
        return false;
      }
    }

    if (auto typeAttr = owningFunc.getArgAttrOfType<ttcore::ArgumentTypeAttr>(
            arg.getArgNumber(), ttcore::ArgumentTypeAttr::name)) {
      return typeAttr.getValue() == ttcore::ArgumentType::Input;
    }
    return false;
  }

  RankedTensorType toRowMajorType(MLIRContext *ctx, RankedTensorType ty) const {
    BufferType bufferType = g_defaultMemorySpaceDevice;

    // Preserve existing buffer type if encoding exists
    if (auto currentLayout =
            mlir::dyn_cast_if_present<TTNNLayoutAttr>(ty.getEncoding())) {
      bufferType = currentLayout.getBufferType();
    }

    TTNNLayoutAttr rmLayout =
        createLayoutAttr(ctx, deviceGrid, ty, bufferType, /*isTiled=*/false);
    return RankedTensorType::get(ty.getShape(), ty.getElementType(), rmLayout);
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
    ttcore::DeviceOp deviceOp = ttcore::lookupDeviceOp(getOperation());

    // If there is no device registered in the module, we simply want all
    // tensors to be laid out in the system memory (e.g., CPU module).
    if (!deviceOp) {
      TTNNLayoutTensorTypeConverter typeConverter(
          &getContext(), ttcore::GridAttr::get(&getContext()),
          BufferType::SystemMemory, /* isTiled */ false);

      RewritePatternSet patterns(&getContext());
      patterns.add<ttir::UniformTypeRewriter>(typeConverter, &getContext());

      FrozenRewritePatternSet patternSet(std::move(patterns));
      if (failed(applyPatternsGreedily(getOperation(), patternSet))) {
        signalPassFailure();
      }
      return;
    }

    // First add default attribute to all tensors. Example:
    // Given tensor type: tensor<15x10x32xf32>
    // we construct a ttnn layout attribute with default values:
    // ttnn_layout<affine_map, grid<1x1>,
    // memref<1x1x!ttcore.tile<32x32>, #dram>, <interleaved>>
    {
      ttcore::DeviceAttr device = deviceOp.getDeviceAttr();
      assert(device && "Device not found");
      TTNNLayoutTensorTypeConverter typeDefaultConverter(
          &getContext(), device.getWorkerGrid(), g_defaultMemorySpaceDevice,
          /* isTiled */ true);
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
      patterns.add<TTNNLayoutRewriter>(&getContext());
      // Update the return op output layout based on its consumers
      // Logic here should match that of TTNNLayoutFuncInputOutputTypeRewriter
      patterns.add<TTNNLayoutFuncReturnRewriter>(&getContext());
      patterns.add<TTNNLayoutHoistedFuncCallRewriter>(&getContext());
      patterns.add<TTNNLayoutMeshShardRewriter>(&getContext());

      // Rewrite LoadCachedOp call sites to have correct result types matching
      // callee function signatures in case const-eval function signatures have
      // been updated.
      patterns.add<TTNNLayoutLoadCachedOpTypeRewriter>(&getContext());

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
