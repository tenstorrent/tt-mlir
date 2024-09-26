// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTNN/TTIRToTTNN.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::tt;

namespace {

// Gets or inserts a GetDeviceOp at the top of the current block of the given
// operation.
static Value getOrInsertDevice(ConversionPatternRewriter &rewriter,
                               Operation *op) {
  Block *block = op->getBlock();
  for (auto &op : block->getOperations()) {
    if (auto deviceOp = dyn_cast<ttnn::GetDeviceOp>(op)) {
      return deviceOp.getResult();
    }
  }

  DeviceAttr deviceAttr = getCurrentScopeDevice(op);
  auto currentInsertionPoint = rewriter.saveInsertionPoint();
  rewriter.setInsertionPoint(block, block->begin());
  auto deviceOp = rewriter.create<ttnn::GetDeviceOp>(
      op->getLoc(), rewriter.getType<DeviceType>(deviceAttr),
      ttnn::MeshShapeAttr::get(op->getContext(), 1, 1));
  rewriter.restoreInsertionPoint(currentInsertionPoint);
  return deviceOp.getResult();
}

class TensorEmptyConversionPattern
    : public OpConversionPattern<tensor::EmptyOp> {
public:
  using OpConversionPattern<tensor::EmptyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tensor::EmptyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Get tt::LayoutAttr of the result type
    //
    tt::LayoutAttr ttLayoutAttr =
        mlir::cast<tt::LayoutAttr>(op.getResult().getType().getEncoding());

    // Get the shape of the tensor, tensor layout, and data type
    //
    mlir::MemRefType memref = ttLayoutAttr.getMemref();
    ttnn::ShapeAttr shapeAttr = ttnn::ShapeAttr::get(
        rewriter.getContext(),
        mlir::cast<RankedTensorType>(op->getResult(0).getType()).getShape());
    Type elementType = memref.getElementType();
    DataType dtype = DataType::Float32;
    ttnn::Layout ttnnLayoutEnum = ttnn::Layout::RowMajor;
    if (llvm::isa<TileType>(elementType)) {
      ttnnLayoutEnum = ttnn::Layout::Tile;
      auto tileType = mlir::cast<TileType>(elementType);
      dtype = tileType.getDataType();
    } else {
      ttnnLayoutEnum = ttnn::Layout::RowMajor;
      dtype = elementTypeToDataType(elementType);
    }
    DataTypeAttr dTypeAttr = DataTypeAttr::get(rewriter.getContext(), dtype);
    ttnn::LayoutAttr tensorLayoutAttr =
        ttnn::LayoutAttr::get(op.getContext(), ttnnLayoutEnum);

    // If the tensor is not going to device, we can create the op without
    // device-specific attributes
    //
    tt::TensorMemoryLayout ttTensorMemoryLayout = ttLayoutAttr.getMemLayout();
    if (ttTensorMemoryLayout == TensorMemoryLayout::None) {
      rewriter.replaceOpWithNewOp<ttnn::EmptyOp>(
          op, this->getTypeConverter()->convertType(op.getType()), nullptr,
          shapeAttr, dTypeAttr, tensorLayoutAttr, nullptr);

      return success();
    }

    ttnn::BufferType bufferType =
        ttnn::utils::toTTNNBufferType(ttLayoutAttr.getMemorySpace());
    ttnn::TensorMemoryLayout tensorMemoryLayout =
        ttnn::utils::toTTNNTensorMemoryLayout(ttLayoutAttr.getMemLayout());

    // Create MemoryConfigAttr
    //
    auto device = getOrInsertDevice(rewriter, op);
    ttnn::MemoryConfigAttr memoryConfigAttr = ttnn::MemoryConfigAttr::get(
        op.getContext(),
        ttnn::TensorMemoryLayoutAttr::get(op.getContext(), tensorMemoryLayout),
        ttnn::BufferTypeAttr::get(op.getContext(), bufferType));

    rewriter.replaceOpWithNewOp<ttnn::EmptyOp>(
        op, this->getTypeConverter()->convertType(op.getType()), device,
        shapeAttr, dTypeAttr, tensorLayoutAttr, memoryConfigAttr);

    return success();
  }
};

// TTIR::ToLayoutOp is a rather generic op that dictates how all the layout
// properties of a tensor should be set. However, in TTNN world, multiple APIs
// are required to achieve an arbitrary layout, namely:
// - ToLayoutOp: to set the layout (ROW_MAJOR, TILE) of the tensor
// - TypecastOp: to change the data type of the tensor
// - ToDeviceOp: to move the tensor to a specific device
// - FromDeviceOp: to move the tensor from a specific device to host
// - ToMemoryConfigOp: to set the memory configuration (dram, l1, interleaved,
// sharded) of the tensor
class ToLayoutOpConversionPattern
    : public OpConversionPattern<ttir::ToLayoutOp> {
public:
  using OpConversionPattern<ttir::ToLayoutOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ToLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Get the DPS operand and delete it's creator op, if it's tensor::emptyOp
    //
    Value dpsOperand = adaptor.getOperands().back();
    ttnn::EmptyOp emptyOp = dpsOperand.getDefiningOp<ttnn::EmptyOp>();
    if (emptyOp) {
      rewriter.eraseOp(emptyOp);
    }

    if (failed(createLayoutConversionOps(op, rewriter))) {
      return failure();
    };

    rewriter.eraseOp(op);
    return success();
  }

private:
  struct LayoutInfo {
    ttnn::BufferType bufferType;
    ttnn::Layout layoutEnum;
    DataType dataType;
    ttnn::TensorMemoryLayout tensorMemoryLayout;

    bool isOnHost() const {
      return bufferType == ttnn::BufferType::SystemMemory;
    }
    bool isOnDevice() const { return not isOnHost(); }
    bool isTilized() const { return layoutEnum == ttnn::Layout::Tile; }
  };

  struct OpsToCreate {
    bool createToDeviceOp = false;
    bool createFromDeviceOp = false;
    bool createToLayoutOp = false;
    bool createTypecastOp = false;
    bool createToMemoryConfigOp = false;

    bool createSomeOp() const {
      return createToLayoutOp or createTypecastOp or createToDeviceOp or
             createFromDeviceOp or createToMemoryConfigOp;
    }
  };

  struct OpCreationInfo {
    mlir::Value device;
    LayoutInfo input;
    LayoutInfo output;
    OpsToCreate opsToCreate;

    OpCreationInfo(mlir::Value device, const LayoutInfo &input,
                   const LayoutInfo &output, const OpsToCreate &opsToCreate)
        : device(device), input(input), output(output),
          opsToCreate(opsToCreate) {}

    bool shouldUntilize() const {
      return opsToCreate.createToLayoutOp and not output.isTilized();
    }

    bool shouldTilize() const {
      return opsToCreate.createToLayoutOp and output.isTilized();
    }
  };

  bool shouldForceRowMajor(ttir::ToLayoutOp op) const {
    for (mlir::Operation *user : op.getResult().getUsers()) {
      if (isa<ttir::Conv2dOp>(user) || isa<ttir::MaxPool2dOp>(user)) {
        return true;
      }
    }

    return false;
  }

  ttnn::Layout getLayoutFromMemRef(mlir::MemRefType memref) const {
    ttnn::Layout ttnnLayoutEnum = ttnn::Layout::RowMajor;
    Type elementType = memref.getElementType();
    if (llvm::isa<TileType>(elementType)) {
      ttnnLayoutEnum = ttnn::Layout::Tile;
    } else {
      ttnnLayoutEnum = ttnn::Layout::RowMajor;
    }
    return ttnnLayoutEnum;
  }

  ttnn::MemoryConfigAttr
  createMemoryConfigAttr(MLIRContext *context,
                         ttnn::TensorMemoryLayout tensorMemoryLayout,
                         ttnn::BufferType bufferType) const {
    return ttnn::MemoryConfigAttr::get(
        context, ttnn::TensorMemoryLayoutAttr::get(context, tensorMemoryLayout),
        ttnn::BufferTypeAttr::get(context, bufferType));
  }

  std::pair<LayoutInfo, LayoutInfo>
  getInputOutputLayouts(ttir::ToLayoutOp op) const {
    LayoutInfo input, output;

    auto inputLayoutAttr =
        mlir::cast<tt::LayoutAttr>(op.getInput().getType().getEncoding());
    auto outputLayoutAttr =
        mlir::cast<tt::LayoutAttr>(op.getResult().getType().getEncoding());

    auto inputMemref = inputLayoutAttr.getMemref();
    auto outputMemref = outputLayoutAttr.getMemref();

    input.bufferType =
        ttnn::utils::toTTNNBufferType(inputLayoutAttr.getMemorySpace());
    output.bufferType =
        ttnn::utils::toTTNNBufferType(outputLayoutAttr.getMemorySpace());

    input.layoutEnum = getLayoutFromMemRef(inputMemref);
    output.layoutEnum = getLayoutFromMemRef(outputMemref);

    if (output.bufferType != ttnn::BufferType::SystemMemory) {
      // TODO(bug #665):
      // Binary ops fail with row major layout in ttnn, defaulting to and
      // assuming tile layout for all device tensors...
      // Note: mlir doesn't know about this, so tensors may still appear as row
      // major in the generated mlir
      // TODO(bug #875):
      // Remove the following code block once constraints modelling is
      // implemented on dialect level
      //
      // Default to Tile layout unless op supports only RowMajor layout
      //
      output.layoutEnum =
          shouldForceRowMajor(op) ? ttnn::Layout::RowMajor : ttnn::Layout::Tile;
    }

    input.dataType = ttnn::utils::getDataTypeFromMemRef(inputMemref);
    output.dataType = ttnn::utils::getDataTypeFromMemRef(outputMemref);

    input.tensorMemoryLayout =
        ttnn::utils::toTTNNTensorMemoryLayout(inputLayoutAttr.getMemLayout());
    output.tensorMemoryLayout =
        ttnn::utils::toTTNNTensorMemoryLayout(outputLayoutAttr.getMemLayout());

    return {input, output};
  }

  OpsToCreate determineRequiredOps(const LayoutInfo &input,
                                   const LayoutInfo &output) const {
    OpsToCreate opsToCreate;

    opsToCreate.createToDeviceOp =
        (input.bufferType != output.bufferType) and input.isOnHost();
    opsToCreate.createFromDeviceOp =
        (input.bufferType != output.bufferType) and output.isOnHost();

    opsToCreate.createTypecastOp = input.dataType != output.dataType;
    opsToCreate.createToLayoutOp = input.layoutEnum != output.layoutEnum;
    // TODO(bug #665):
    // Insert a ToLayoutOp manually if we're moving from device to host to
    // untilize. Since we're hardcoding tile layout, the tensor may be row
    // major in mlir, and therefore it would appear as if we don't need to
    // untilize
    opsToCreate.createToLayoutOp |=
        (opsToCreate.createFromDeviceOp and
         output.layoutEnum == ttnn::Layout::RowMajor);

    // TODO(bug #620):
    // Add support for ShardSpec
    // ToDeviceOp can handle the creation of the memory config of the initial
    // device tensor
    if (not opsToCreate.createToDeviceOp) {
      opsToCreate.createToMemoryConfigOp =
          (input.tensorMemoryLayout != output.tensorMemoryLayout) and
          (output.tensorMemoryLayout != ttnn::TensorMemoryLayout::None);
      opsToCreate.createToMemoryConfigOp |=
          (input.bufferType == ttnn::BufferType::DRAM and
           output.bufferType == ttnn::BufferType::L1) or
          (input.bufferType == ttnn::BufferType::L1 and
           output.bufferType == ttnn::BufferType::DRAM);
    }
    return opsToCreate;
  }

  LogicalResult isCreationValid(ttir::ToLayoutOp op, const LayoutInfo &input,
                                const LayoutInfo &output,
                                const OpsToCreate &opsToCreate) const {
    if (not opsToCreate.createSomeOp()) {
      op->emitError("Redundant ttir::ToLayoutOp - no ttnn layout ops "
                    "needed");
      return failure();
    }

    if (opsToCreate.createToDeviceOp and opsToCreate.createFromDeviceOp) {
      op->emitError("Cannot create both ToDeviceOp and FromDeviceOp");
      return failure();
    }

    if (opsToCreate.createToMemoryConfigOp and
        output.bufferType == ttnn::BufferType::SystemMemory) {
      op->emitError(
          "ToMemoryConfigOp only supported for device output tensors");
      return failure();
    }

    if (input.isOnHost() and opsToCreate.createFromDeviceOp) {
      op->emitError("Unexpected FromDeviceOp on host tensor");
      return failure();
    }

    if (input.isOnDevice() and opsToCreate.createToDeviceOp) {
      op->emitError("Unexpected ToDeviceOp on device tensor");
      return failure();
    }
    return success();
  }

  /* Helper functions to create ttnn layout ops */

  template <typename OpType, typename... Args>
  mlir::Value createOp(ttir::ToLayoutOp op, ConversionPatternRewriter &rewriter,
                       mlir::Value currentInput, Args... args) const {
    return rewriter.create<OpType>(
        op.getLoc(), this->getTypeConverter()->convertType(op.getType()),
        currentInput, args...);
  }

  mlir::Value createToDeviceOpIfNeeded(ttir::ToLayoutOp op,
                                       ConversionPatternRewriter &rewriter,
                                       mlir::Value currentInput,
                                       const OpCreationInfo &info) const {
    if (not info.opsToCreate.createToDeviceOp) {
      return currentInput;
    }
    ttnn::MemoryConfigAttr memoryConfigAttr =
        createMemoryConfigAttr(op.getContext(), info.output.tensorMemoryLayout,
                               info.output.bufferType);
    return this->createOp<ttnn::ToDeviceOp>(op, rewriter, currentInput,
                                            info.device, memoryConfigAttr);
  }

  // FromDeviceOp
  mlir::Value createFromDeviceOpIfNeeded(ttir::ToLayoutOp op,
                                         ConversionPatternRewriter &rewriter,
                                         mlir::Value currentInput,
                                         const OpCreationInfo &info,
                                         bool forceCreate = false) const {
    if (not info.opsToCreate.createFromDeviceOp) {
      return currentInput;
    }
    return this->createOp<ttnn::FromDeviceOp>(op, rewriter, currentInput);
  }

  mlir::Value createToLayoutOpIfNeeded(ttir::ToLayoutOp op,
                                       ConversionPatternRewriter &rewriter,
                                       mlir::Value currentInput,
                                       const OpCreationInfo &info) const {
    if (not info.opsToCreate.createToLayoutOp) {
      return currentInput;
    }
    ttnn::LayoutAttr layoutAttr =
        ttnn::LayoutAttr::get(op.getContext(), info.output.layoutEnum);
    return this->createOp<ttnn::ToLayoutOp>(
        op, rewriter, currentInput, layoutAttr, /*dtype*/ nullptr,
        /*memory_config*/ nullptr, /*device*/ nullptr);
  }

  mlir::Value createTypecastOpIfNeeded(ttir::ToLayoutOp op,
                                       ConversionPatternRewriter &rewriter,
                                       mlir::Value currentInput,
                                       const OpCreationInfo &info) const {
    if (not info.opsToCreate.createTypecastOp) {
      return currentInput;
    }
    DataTypeAttr dtypeAttr =
        DataTypeAttr::get(op.getContext(), info.output.dataType);
    return this->createOp<ttnn::TypecastOp>(op, rewriter, currentInput,
                                            dtypeAttr);
  }

  mlir::Value createToMemoryConfigOpIfNeeded(
      ttir::ToLayoutOp op, ConversionPatternRewriter &rewriter,
      mlir::Value currentInput, const OpCreationInfo &info) const {
    if (not info.opsToCreate.createToMemoryConfigOp) {
      return currentInput;
    }
    ttnn::MemoryConfigAttr memoryConfigAttr =
        createMemoryConfigAttr(op.getContext(), info.output.tensorMemoryLayout,
                               info.output.bufferType);
    return this->createOp<ttnn::ToMemoryConfigOp>(op, rewriter, currentInput,
                                                  memoryConfigAttr);
  }

  /* Functions that create ops based on the layouts of the input output tensors
   */

  LogicalResult handleHostInputNoLayoutNoTypecast(
      ttir::ToLayoutOp op, ConversionPatternRewriter &rewriter,
      mlir::Value currentInput, const OpCreationInfo &info) const {
    // If the input tensor is on host and we don't need to create a ToLayoutOp
    // nor a TypecastOp we can create a ToDeviceOp and a ToMemoryConfigOp if
    // needed and return
    currentInput =
        this->createToDeviceOpIfNeeded(op, rewriter, currentInput, info);
    currentInput =
        this->createToMemoryConfigOpIfNeeded(op, rewriter, currentInput, info);
    op.getResult().replaceAllUsesWith(currentInput);
    return success();
  }

  LogicalResult handleHostInputLayoutNoTypecast(
      ttir::ToLayoutOp op, ConversionPatternRewriter &rewriter,
      mlir::Value currentInput, const OpCreationInfo &info) const {
    const LayoutInfo &input = info.input;
    const LayoutInfo &output = info.output;
    assert(input.dataType == output.dataType &&
           "Data type should be the same if we're not creating typecast op");
    /* if we should untilize, untilize on host */
    if (info.shouldUntilize()) {
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return success();
    }

    /* If we should tilize, and the data type is bfloat16, we can tilize on
     * device */
    if (info.shouldTilize() and output.dataType == DataType::BFloat16) {
      currentInput =
          this->createToDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return success();
    }

    /* If we should tilize, and the data type is not bfloat16, we tilize on host
     */
    if (info.shouldTilize() and output.dataType != DataType::BFloat16) {
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return success();
    }

    llvm_unreachable("Unreachable code path");
  }

  LogicalResult handleHostInputNoLayoutTypecast(
      ttir::ToLayoutOp op, ConversionPatternRewriter &rewriter,
      mlir::Value currentInput, const OpCreationInfo &info) const {
    const LayoutInfo &input = info.input;
    const LayoutInfo &output = info.output;
    assert(input.layoutEnum == output.layoutEnum &&
           "Layout should be the same if we're not creating a ToLayoutOp");

    /* If the output is already tilized, we can typecast on device */
    if (output.isTilized()) {
      currentInput =
          this->createToDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createTypecastOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return success();
    }

    /* If the output is not tilized, typecast on host */
    else if (not output.isTilized()) {
      currentInput =
          this->createTypecastOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return success();
    }

    llvm_unreachable("Unreachable code path");
  }

  LogicalResult handleHostInputLayoutTypecast(
      ttir::ToLayoutOp op, ConversionPatternRewriter &rewriter,
      mlir::Value currentInput, const OpCreationInfo &info) const {
    const LayoutInfo &input = info.input;
    const LayoutInfo &output = info.output;

    /* If we need to untilize and typecast, then untilize and typecast on host
     */
    if (info.shouldUntilize()) {
      currentInput =
          this->createTypecastOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return success();
    }

    /* If we need to tilize and the input datatype is bfloat16
       we can tilize on device and then typecast afterwards */
    if (info.shouldTilize() and input.dataType == DataType::BFloat16) {
      currentInput =
          this->createToDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createTypecastOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return success();
    }

    /* if we need to tilize and the output data type is bfloat16
       we can typecast on host and tilize on device */
    if (info.shouldTilize() and output.dataType == DataType::BFloat16) {
      currentInput =
          this->createTypecastOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return success();
    }

    /* if we need to tilize and the input/ output data types are not bfloat16 do
     * everything on host */
    if (info.shouldTilize() and input.dataType != DataType::BFloat16 and
        output.dataType != DataType::BFloat16) {
      currentInput =
          this->createTypecastOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return success();
    }

    llvm_unreachable("Unreachable code path");
  }

  LogicalResult handleHostInputLayoutConversion(
      ttir::ToLayoutOp op, ConversionPatternRewriter &rewriter,
      mlir::Value currentInput, const OpCreationInfo &info) const {
    const OpsToCreate &opsToCreate = info.opsToCreate;
    if (not opsToCreate.createToLayoutOp and not opsToCreate.createTypecastOp) {
      return handleHostInputNoLayoutNoTypecast(op, rewriter, currentInput,
                                               info);
    }
    if (opsToCreate.createToLayoutOp and not opsToCreate.createTypecastOp) {
      return handleHostInputLayoutNoTypecast(op, rewriter, currentInput, info);
    }
    if (not opsToCreate.createToLayoutOp and opsToCreate.createTypecastOp) {
      return handleHostInputNoLayoutTypecast(op, rewriter, currentInput, info);
    }
    if (opsToCreate.createToLayoutOp and opsToCreate.createTypecastOp) {
      return handleHostInputLayoutTypecast(op, rewriter, currentInput, info);
    }
    llvm_unreachable("Unreachable code path");
  }

  LogicalResult handleDeviceInputNoLayoutNoTypecast(
      ttir::ToLayoutOp op, ConversionPatternRewriter &rewriter,
      mlir::Value currentInput, const OpCreationInfo &info) const {
    // If the input tensor is on device and we don't need to create a ToLayoutOp
    // nor a TypecastOp we can create a FromDeviceOp and a ToMemoryConfigOp if
    // needed and return
    currentInput =
        this->createToMemoryConfigOpIfNeeded(op, rewriter, currentInput, info);
    currentInput =
        this->createFromDeviceOpIfNeeded(op, rewriter, currentInput, info);
    op.getResult().replaceAllUsesWith(currentInput);
    return success();
  }

  LogicalResult handleDeviceInputLayoutNoTypecast(
      ttir::ToLayoutOp op, ConversionPatternRewriter &rewriter,
      mlir::Value currentInput, const OpCreationInfo &info) const {
    const LayoutInfo &input = info.input;
    const LayoutInfo &output = info.output;
    const OpsToCreate &opsToCreate = info.opsToCreate;
    assert(input.dataType == output.dataType &&
           "Data type should be the same if we're not creating typecast op");

    /* if we should untilize, untilize on host */
    /* this is the main untilize case hit when we read data from device back to
     * host at the end of the program */
    if (info.shouldUntilize() and opsToCreate.createFromDeviceOp) {
      currentInput =
          this->createFromDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return success();
    }

    /* This is a rare untilize case, where we want to untilize a device tensor
       but keep it on device to handle this we need to move the tensor to host,
       untilize it, and then move it back to device */
    if (info.shouldUntilize() and not opsToCreate.createFromDeviceOp) {
      // Force-create a FromDeviceOp
      currentInput =
          this->createOp<ttnn::FromDeviceOp>(op, rewriter, currentInput);
      // untilize on host
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      // move back to device and convert memory config if needed
      currentInput = this->createOp<ttnn::ToDeviceOp>(
          op, rewriter, currentInput, info.device,
          /* optional MemConfigAttr */ nullptr);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return success();
    }

    /* If we should tilize and the input data type is bfloat16, tilize on device
     */
    if (info.shouldTilize() and input.dataType == DataType::BFloat16) {
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      currentInput =
          this->createFromDeviceOpIfNeeded(op, rewriter, currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return success();
    }

    /* If we should tilize and the input data type is not bfloat16, tilize on
     * host */
    if (info.shouldTilize() and input.dataType != DataType::BFloat16 and
        opsToCreate.createFromDeviceOp) {
      currentInput =
          this->createFromDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return success();
    }

    /* If we want to tilize a device tensor that is not bfloat16, we need to
     * tilize on host and move it back */
    if (info.shouldTilize() and input.dataType != DataType::BFloat16 and
        not opsToCreate.createFromDeviceOp) {
      // Force-create a FromDeviceOp
      currentInput =
          this->createOp<ttnn::FromDeviceOp>(op, rewriter, currentInput);
      // tilize on host
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      // move back to device and convert memory config if needed
      currentInput = this->createOp<ttnn::ToDeviceOp>(
          op, rewriter, currentInput, info.device,
          /* optional MemConfigAttr */ nullptr);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return success();
    }

    llvm_unreachable("Unreachable code path");
  }

  LogicalResult handleDeviceInputNoLayoutTypecast(
      ttir::ToLayoutOp op, ConversionPatternRewriter &rewriter,
      mlir::Value currentInput, const OpCreationInfo &info) const {
    const LayoutInfo &input = info.input;
    const LayoutInfo &output = info.output;
    const OpsToCreate &opsToCreate = info.opsToCreate;
    assert(input.layoutEnum == output.layoutEnum &&
           "Layout should be the same if we're not creating toLayout op");

    /* If the output is tilized, typecast directly on device*/
    if (output.isTilized()) {
      currentInput =
          this->createTypecastOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      currentInput =
          this->createFromDeviceOpIfNeeded(op, rewriter, currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return success();
    }

    /* If the output is not tilized, typecast on host */
    if (not output.isTilized() and opsToCreate.createFromDeviceOp) {
      currentInput =
          this->createFromDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createTypecastOpIfNeeded(op, rewriter, currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return success();
    }

    /* Device to device untilized typecast, need to move to host first */
    if (not output.isTilized() and not opsToCreate.createFromDeviceOp) {
      // Force-create a FromDeviceOp
      currentInput =
          this->createOp<ttnn::FromDeviceOp>(op, rewriter, currentInput);
      // typecast on host
      currentInput =
          this->createTypecastOpIfNeeded(op, rewriter, currentInput, info);
      // move back to device and convert memory config if needed
      currentInput = this->createOp<ttnn::ToDeviceOp>(
          op, rewriter, currentInput, info.device,
          /* optional MemConfigAttr */ nullptr);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return success();
    }

    llvm_unreachable("Unreachable code path");
  }

  LogicalResult handleDeviceInputLayoutTypecast(
      ttir::ToLayoutOp op, ConversionPatternRewriter &rewriter,
      mlir::Value currentInput, const OpCreationInfo &info) const {
    const LayoutInfo &input = info.input;
    const OpsToCreate &opsToCreate = info.opsToCreate;

    /* If we need to untilize, typecast on device and untilize on host */
    if (info.shouldUntilize() and opsToCreate.createFromDeviceOp) {
      currentInput =
          this->createTypecastOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createFromDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return success();
    }

    /* Rare case of device to device untilize, typecast on device, untilize on
     * host, move back to device */
    if (info.shouldUntilize() and not opsToCreate.createFromDeviceOp) {
      // typecast on device
      currentInput =
          this->createTypecastOpIfNeeded(op, rewriter, currentInput, info);
      // Force-create a FromDeviceOp
      currentInput =
          this->createOp<ttnn::FromDeviceOp>(op, rewriter, currentInput);
      // untilize on host
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      // move back to device and convert memory config if needed
      currentInput = this->createOp<ttnn::ToDeviceOp>(
          op, rewriter, currentInput, info.device,
          /* optional MemConfigAttr */ nullptr);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return success();
    }

    /* If we should tilize and the input data type is bfloat16, tilize and
     * typecast on device */
    if (info.shouldTilize() and input.dataType == DataType::BFloat16) {
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createTypecastOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      currentInput =
          this->createFromDeviceOpIfNeeded(op, rewriter, currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return success();
    }

    /* If we should tilize and the input data type is not bfloat16 and we want
       to read back from device do everything on host */
    if (info.shouldTilize() and input.dataType != DataType::BFloat16 and
        opsToCreate.createFromDeviceOp) {
      currentInput =
          this->createFromDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createTypecastOpIfNeeded(op, rewriter, currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return success();
    }

    /* If we should tilize and the input data type is not bfloat 16 and we don't
       want to read back from device: tilize on host, move back to device, and
       typecast on device */
    if (info.shouldTilize() and input.dataType != DataType::BFloat16 and
        not opsToCreate.createFromDeviceOp) {
      // Force-create a FromDeviceOp
      currentInput =
          this->createOp<ttnn::FromDeviceOp>(op, rewriter, currentInput);
      // tilize on host
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      // move back to device and convert data type/memory config if needed
      currentInput = this->createOp<ttnn::ToDeviceOp>(
          op, rewriter, currentInput, info.device,
          /* optional MemConfigAttr */ nullptr);
      currentInput =
          this->createTypecastOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return success();
    }

    llvm_unreachable("Unreachable code path");
  }

  LogicalResult handleDeviceInputLayoutConversion(
      ttir::ToLayoutOp op, ConversionPatternRewriter &rewriter,
      mlir::Value currentInput, const OpCreationInfo &info) const {
    const OpsToCreate &opsToCreate = info.opsToCreate;
    if (not opsToCreate.createToLayoutOp and not opsToCreate.createTypecastOp) {
      return handleDeviceInputNoLayoutNoTypecast(op, rewriter, currentInput,
                                                 info);
    }
    if (opsToCreate.createToLayoutOp and not opsToCreate.createTypecastOp) {
      return handleDeviceInputLayoutNoTypecast(op, rewriter, currentInput,
                                               info);
    }
    if (not opsToCreate.createToLayoutOp and opsToCreate.createTypecastOp) {
      return handleDeviceInputNoLayoutTypecast(op, rewriter, currentInput,
                                               info);
    }
    if (opsToCreate.createToLayoutOp and opsToCreate.createTypecastOp) {
      return handleDeviceInputLayoutTypecast(op, rewriter, currentInput, info);
    }
    llvm_unreachable("Unreachable code path");
  }

  /*
   * Logic for creating ops. Conditions/constraints include:
   * - When possible, we want to execute operations on device.
   * - Tilize on device requires dataformat of BFLOAT16.
   * - Typecast on device requires TILIZED tensor.
   * - Untilize on device requires even width, and page size >
   *   sizeof(uint32_t). For now, we will always untilize on host. We rarely
   * need device to device untilize, so the perf hit should be acceptable.
   */
  LogicalResult
  createLayoutConversionOps(ttir::ToLayoutOp op,
                            ConversionPatternRewriter &rewriter) const {
    auto [input, output] = getInputOutputLayouts(op);
    OpsToCreate opsToCreate = determineRequiredOps(input, output);

    if (failed(isCreationValid(op, input, output, opsToCreate))) {
      return failure();
    }

    auto device = getOrInsertDevice(rewriter, op);

    OpCreationInfo info(device, input, output, opsToCreate);

    Value currentInput = op.getInput();

    if (input.isOnHost()) {
      return handleHostInputLayoutConversion(op, rewriter, currentInput, info);
    }
    return handleDeviceInputLayoutConversion(op, rewriter, currentInput, info);
  }
};

template <typename TTIROpTy, typename TTNNOpTy,
          typename OpAdaptor = typename TTIROpTy::Adaptor>
class ElementwiseOpConversionPattern : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultTypes))) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<TTNNOpTy>(op, resultTypes, adaptor.getInputs(),
                                          adaptor.getOutputs());
    return success();
  }
};

template <typename TTIROpTy, typename TTNNOpTy,
          typename OpAdaptor = typename TTIROpTy::Adaptor>
class ReductionOpConversionPattern : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TTNNOpTy>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), adaptor.getKeepDim(),
        adaptor.getDimArg().value_or(nullptr));
    return success();
  }
};

class EmbeddingOpConversionPattern
    : public OpConversionPattern<ttir::EmbeddingOp> {
public:
  using OpConversionPattern<ttir::EmbeddingOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::EmbeddingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ttnn::EmbeddingOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), adaptor.getWeight());

    return success();
  }
};

class SoftmaxOpConversionPattern : public OpConversionPattern<ttir::SoftmaxOp> {
public:
  using OpConversionPattern<ttir::SoftmaxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::SoftmaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ttnn::SoftmaxOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), adaptor.getDimension());
    return success();
  }
};

class TransposeOpConversionPattern
    : public OpConversionPattern<ttir::TransposeOp> {
public:
  using OpConversionPattern<ttir::TransposeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ttnn::TransposeOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), adaptor.getDim0(), adaptor.getDim1());
    return success();
  }
};

class ConcatOpConversionPattern : public OpConversionPattern<ttir::ConcatOp> {
public:
  using OpConversionPattern<ttir::ConcatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    int dim = adaptor.getDim();
    if (dim < 0) {
      dim += cast<RankedTensorType>(adaptor.getInputs().front().getType())
                 .getRank();
    }
    rewriter.replaceOpWithNewOp<ttnn::ConcatOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInputs(), adaptor.getOutput(), dim);
    return success();
  }
};

class ReshapeOpConversionPattern : public OpConversionPattern<ttir::ReshapeOp> {
public:
  using OpConversionPattern<ttir::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ttnn::ReshapeOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), adaptor.getShape());
    return success();
  }
};

class SliceOpConversionPattern : public OpConversionPattern<ttir::SliceOp> {
public:
  using OpConversionPattern<ttir::SliceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::SliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ttnn::SliceOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), adaptor.getOutput(), adaptor.getBegins(),
        adaptor.getEnds(), adaptor.getStep());
    return success();
  }
};

class SqueezeOpConversionPattern : public OpConversionPattern<ttir::SqueezeOp> {
public:
  using OpConversionPattern<ttir::SqueezeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::SqueezeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Extract input tensor type
    ::mlir::RankedTensorType inputType =
        mlir::cast<::mlir::RankedTensorType>(adaptor.getInput().getType());

    // Get the squeeze dimension
    int32_t dim = adaptor.getDim();

    if (dim < 0) {
      dim += inputType.getRank();
    }

    // Get the shape of the input tensor
    auto inputShape = inputType.getShape();
    llvm::SmallVector<int32_t, 4> newShape;

    // Build the new shape by removing the specified dimension
    for (int64_t i = 0; i < inputType.getRank(); ++i) {
      if (i == dim) {
        continue;
      }
      newShape.push_back(inputShape[i]);
    }

    // Create the new shape attribute
    auto shapeAttr = rewriter.getI32ArrayAttr(newShape);

    // Replace the SqueezeOp with a ReshapeOp
    rewriter.replaceOpWithNewOp<ttnn::ReshapeOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), shapeAttr);

    return success();
  }
};

class UnsqueezeOpConversionPattern
    : public OpConversionPattern<ttir::UnsqueezeOp> {
public:
  using OpConversionPattern<ttir::UnsqueezeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::UnsqueezeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Extract input tensor type
    ::mlir::RankedTensorType inputType =
        mlir::cast<::mlir::RankedTensorType>(adaptor.getInput().getType());

    // Get the unsqueeze dimension
    int32_t dim = adaptor.getDim();

    // Convert negative dim to its positive equivalent
    if (dim < 0) {
      dim += inputType.getRank() + 1;
    }

    // Get the shape of the input tensor
    auto inputShape = inputType.getShape();
    llvm::SmallVector<int32_t, 5> newShape;

    // Insert the new dimension
    for (int i = 0; i < inputType.getRank(); ++i) {
      if (i == dim) {
        newShape.push_back(1);
      }
      newShape.push_back(inputShape[i]);
    }

    // Create the new shape attribute
    auto shapeAttr = rewriter.getI32ArrayAttr(newShape);

    // Replace the UnsqueezeOp with a ReshapeOp
    rewriter.replaceOpWithNewOp<ttnn::ReshapeOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), shapeAttr);

    return success();
  }
};

class ConstantOpConversionPattern
    : public OpConversionPattern<ttir::ConstantOp> {
public:
  using OpConversionPattern<ttir::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ::mlir::ElementsAttr valueAttr = op.getValue();

    LogicalResult legalityResult = checkBasicLegality(op, valueAttr, rewriter);
    if (!legalityResult.succeeded()) {
      return legalityResult;
    }

    if (valueAttr.isSplat()) {
      Value device = getOrInsertDevice(rewriter, op);
      float fillValue = valueAttr.getElementType().isInteger()
                            ? static_cast<float>(valueAttr.getSplatValue<int>())
                            : valueAttr.getSplatValue<float>();
      if (fillValue == 0) {
        rewriter.replaceOpWithNewOp<tensor::EmptyOp>(
            op, this->getTypeConverter()->convertType(op.getType()), device);
      } else {
        ::mlir::FloatAttr fillValueAttr = rewriter.getF32FloatAttr(fillValue);
        rewriter.replaceOpWithNewOp<ttnn::FullOp>(
            op, this->getTypeConverter()->convertType(op.getType()), device,
            fillValueAttr);
      }
    } else {
      return rewriter.notifyMatchFailure(
          op, "TTNN doesn't currently support tensor creation from multiple "
              "given values (issue #685)");
    }

    return success();
  }

private:
  LogicalResult checkBasicLegality(ttir::ConstantOp &op,
                                   ::mlir::ElementsAttr &valueAttr,
                                   ConversionPatternRewriter &rewriter) const {
    if (!valueAttr.getElementType().isIntOrFloat()) {
      return rewriter.notifyMatchFailure(
          op, "TTNN doesn't currently support tensor creation from values "
              "which are not integer or floating point numbers");
    }

    return success();
  }
};

} // namespace

// ANCHOR: adding_an_op_matmul_op_rewriter
class MatmulOpConversionPattern : public OpConversionPattern<ttir::MatmulOp> {
public:
  using OpConversionPattern<ttir::MatmulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ttnn::MatmulOp>(
        op, this->getTypeConverter()->convertType(op.getType()), adaptor.getA(),
        adaptor.getB(), adaptor.getOutput());
    return success();
  }
};
// ANCHOR_END: adding_an_op_matmul_op_rewriter

class Conv2dOpConversionPattern : public OpConversionPattern<ttir::Conv2dOp> {
public:
  using OpConversionPattern<ttir::Conv2dOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::Conv2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto device = getOrInsertDevice(rewriter, op);
    auto kernel_ty =
        mlir::cast<RankedTensorType>(adaptor.getWeight().getType());
    llvm::ArrayRef<std::int64_t> kernel_shape = kernel_ty.getShape();

    auto input_ty = mlir::cast<RankedTensorType>(adaptor.getInput().getType());
    llvm::ArrayRef<std::int64_t> input_shape = input_ty.getShape();

    auto output_ty =
        mlir::cast<RankedTensorType>(adaptor.getOutput().getType());
    llvm::ArrayRef<std::int64_t> output_shape = output_ty.getShape();

    auto in_channels =
        rewriter.getI32IntegerAttr(input_shape[input_shape.size() - 1]);
    auto out_channels =
        rewriter.getI32IntegerAttr(output_shape[output_shape.size() - 1]);
    auto batch_size =
        rewriter.getI32IntegerAttr(input_shape[input_shape.size() - 4]);
    auto input_height =
        rewriter.getI32IntegerAttr(input_shape[input_shape.size() - 3]);
    auto input_width =
        rewriter.getI32IntegerAttr(input_shape[input_shape.size() - 2]);

    auto kernel_height =
        rewriter.getI32IntegerAttr(kernel_shape[kernel_shape.size() - 2]);
    auto kernel_width =
        rewriter.getI32IntegerAttr(kernel_shape[kernel_shape.size() - 1]);

    auto stride_height = rewriter.getI32IntegerAttr(adaptor.getStrideHeight());
    auto stride_width = rewriter.getI32IntegerAttr(adaptor.getStrideWidth());

    assert(
        adaptor.getPaddingBottom() == adaptor.getPaddingTop() &&
        "TTNN only supports padding height/width attributes. Thus, padding_top "
        "must equal padding_bottom for the op to execute as expected.");
    assert(adaptor.getPaddingLeft() == adaptor.getPaddingRight() &&
           "TTNN only supports padding height/width attributes. Thus, "
           "padding_left must equal padding_right for the op to execute as "
           "expected.");
    auto padding_height = rewriter.getI32IntegerAttr(adaptor.getPaddingTop());
    auto padding_width = rewriter.getI32IntegerAttr(adaptor.getPaddingRight());

    auto dilation_height =
        rewriter.getI32IntegerAttr(adaptor.getDilationHeight());
    auto dilation_width =
        rewriter.getI32IntegerAttr(adaptor.getDilationWidth());
    auto groups = rewriter.getI32IntegerAttr(adaptor.getGroups());
    rewriter.replaceOpWithNewOp<ttnn::Conv2dOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), adaptor.getWeight(), adaptor.getBias(),
        adaptor.getOutput(), device, in_channels, out_channels, batch_size,
        input_height, input_width, kernel_height, kernel_width, stride_height,
        stride_width, padding_height, padding_width, dilation_height,
        dilation_width, groups);
    return success();
  }
};

class MaxPool2dOpConversionPattern
    : public OpConversionPattern<ttir::MaxPool2dOp> {
public:
  using OpConversionPattern<ttir::MaxPool2dOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::MaxPool2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    assert(adaptor.getPaddingBottom() == adaptor.getPaddingTop() &&
           "TTNN max_pool2d does not support padding top/bottom/left/right "
           "separately");
    assert(adaptor.getPaddingLeft() == adaptor.getPaddingRight() &&
           "TTNN max_pool2d does not support padding top/bottom/left/right "
           "separately");

    auto device = getOrInsertDevice(rewriter, op);
    auto input_ty = mlir::cast<RankedTensorType>(adaptor.getInput().getType());
    llvm::ArrayRef<std::int64_t> input_shape = input_ty.getShape();

    auto batch_size =
        rewriter.getSI32IntegerAttr(input_shape[input_shape.size() - 4]);
    auto channels =
        rewriter.getSI32IntegerAttr(input_shape[input_shape.size() - 1]);

    assert(adaptor.getOriginalHeight().has_value() &&
           "ttir::MaxPool2dOp must have original_height set before translating "
           "to TTNN dialect.");
    assert(adaptor.getOriginalWidth().has_value() &&
           "ttir::MaxPool2dOp must have original_width set before translating "
           "to TTNN dialect.");

    rewriter.replaceOpWithNewOp<ttnn::MaxPool2dOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), adaptor.getOutput(), device, batch_size,
        adaptor.getOriginalHeightAttr(), adaptor.getOriginalWidthAttr(),
        channels, adaptor.getKernelHeightAttr(), adaptor.getKernelWidthAttr(),
        adaptor.getStrideHeightAttr(), adaptor.getStrideWidthAttr(),
        adaptor.getDilationHeightAttr(), adaptor.getDilationWidthAttr(),
        adaptor.getCeilModeAttr(), adaptor.getPaddingTopAttr(),
        adaptor.getPaddingRightAttr());
    return success();
  }
};

class TypecastOpConversionPattern
    : public OpConversionPattern<ttir::TypecastOp> {
  using OpConversionPattern<ttir::TypecastOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(ttir::TypecastOp op, ttir::TypecastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto input = ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(
        *op.getInputs().begin());
    auto result = ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(
        *op.getResults().begin());

    tt::LayoutAttr outputLayoutAttr =
        mlir::cast<tt::LayoutAttr>(result.getType().getEncoding());

    mlir::MemRefType outputMemref = outputLayoutAttr.getMemref();

    DataType outputDataType = ttnn::utils::getDataTypeFromMemRef(outputMemref);

    if (op->getUsers().empty()) {
      return rewriter.notifyMatchFailure(
          op, "ttir.typecast op should have at least one use.");
    }
    rewriter.replaceOpWithNewOp<ttnn::TypecastOp>(
        op, this->getTypeConverter()->convertType(op.getType(0)), input,
        outputDataType);
    return success();
  }
};

class BroadcastOpConversionPattern
    : public OpConversionPattern<ttir::BroadcastOp> {
  using OpConversionPattern<ttir::BroadcastOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(ttir::BroadcastOp srcOp, ttir::BroadcastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Fold this operation into all consumer ops. It will only work with TTNN
    // ops that support implicit broadcasting. We expect each Op's verify
    // function to assert their arguments to verify that they can broadcast.

    if (srcOp->getUsers().empty()) {
      // This broadcast chain has already been replaced.
      rewriter.eraseOp(srcOp);
      return success();
    }

    mlir::Value input = srcOp.getOperand(0);

    mlir::Operation *nextOp = srcOp;
    while (isa<ttir::BroadcastOp>(*nextOp->getUsers().begin())) {
      assert(nextOp->hasOneUse() &&
             "Broadcast with multiple uses are not supported");
      nextOp = *nextOp->getUsers().begin();
      if (nextOp->getUsers().empty()) {
        // This broadcast chain has already been replaced.
        rewriter.eraseOp(srcOp);
        return success();
      }
    }

    rewriter.replaceAllOpUsesWith(nextOp, input);
    rewriter.eraseOp(srcOp);

    return success();
  }
};

class SubtractOpConversionPattern
    : public OpConversionPattern<ttir::SubtractOp> {
  using OpConversionPattern<ttir::SubtractOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(ttir::SubtractOp srcOp, ttir::SubtractOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType lhsType =
        mlir::cast<RankedTensorType>(adaptor.getInputs().front().getType());
    RankedTensorType rhsType =
        mlir::cast<RankedTensorType>(adaptor.getInputs().back().getType());

    if (lhsType.getShape() == rhsType.getShape()) {
      rewriter.replaceOpWithNewOp<ttnn::SubtractOp>(
          srcOp, adaptor.getInputs().front(), adaptor.getInputs().back(),
          adaptor.getOutputs().front());

      // Broadcast for rhs operand require the operation to be commutative to
      // allow switching the order of operands. To allow this conversion, the
      // following conversion is applied to SubtractOp: subtractOp(lhs,rhs) ->
      // addOp(lhs, negOp(rhs))

    } else {
      Value device = getOrInsertDevice(rewriter, srcOp);
      tensor::EmptyOp negEmptyOp = rewriter.create<tensor::EmptyOp>(
          srcOp.getLoc(), this->getTypeConverter()->convertType(rhsType),
          device);
      ttnn::NegOp negOp = rewriter.create<ttnn::NegOp>(
          srcOp.getLoc(), adaptor.getInputs().back(), negEmptyOp);

      rewriter.replaceOpWithNewOp<ttnn::AddOp>(
          srcOp, adaptor.getInputs().front(), negOp.getResults().front(),
          adaptor.getOutputs().front());
    }

    return success();
  }
};

namespace mlir::tt {

void populateTTIRToTTNNPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                TypeConverter &typeConverter) {
  // clang-format off
  // ANCHOR: op_rewriter_pattern_set
  patterns
      .add<TensorEmptyConversionPattern,
           ToLayoutOpConversionPattern,
           ElementwiseOpConversionPattern<ttir::AbsOp, ttnn::AbsOp>,
           ElementwiseOpConversionPattern<ttir::AddOp, ttnn::AddOp>,
           ElementwiseOpConversionPattern<ttir::LogicalAndOp, ttnn::LogicalAndOp>,
           ElementwiseOpConversionPattern<ttir::LogicalOrOp, ttnn::LogicalOrOp>,
           ElementwiseOpConversionPattern<ttir::LogicalNotOp, ttnn::LogicalNotOp>,
           ElementwiseOpConversionPattern<ttir::MultiplyOp, ttnn::MultiplyOp>,
           ElementwiseOpConversionPattern<ttir::EqualOp, ttnn::EqualOp>,
           ElementwiseOpConversionPattern<ttir::NotEqualOp, ttnn::NotEqualOp>,
           ElementwiseOpConversionPattern<ttir::GreaterEqualOp, ttnn::GreaterEqualOp>,
           ElementwiseOpConversionPattern<ttir::GreaterThanOp, ttnn::GreaterThanOp>,
           ElementwiseOpConversionPattern<ttir::LessEqualOp, ttnn::LessEqualOp>,
           ElementwiseOpConversionPattern<ttir::LessThanOp, ttnn::LessThanOp>,
           ElementwiseOpConversionPattern<ttir::MaximumOp, ttnn::MaximumOp>,
           ElementwiseOpConversionPattern<ttir::NegOp, ttnn::NegOp>,
           ElementwiseOpConversionPattern<ttir::ReluOp, ttnn::ReluOp>,
           ElementwiseOpConversionPattern<ttir::SqrtOp, ttnn::SqrtOp>,
           ElementwiseOpConversionPattern<ttir::RsqrtOp, ttnn::RsqrtOp>,
           ElementwiseOpConversionPattern<ttir::SigmoidOp, ttnn::SigmoidOp>,
           ElementwiseOpConversionPattern<ttir::ReciprocalOp, ttnn::ReciprocalOp>,
           ElementwiseOpConversionPattern<ttir::ExpOp, ttnn::ExpOp>,
           ElementwiseOpConversionPattern<ttir::DivOp, ttnn::DivOp>,
           ReductionOpConversionPattern<ttir::SumOp, ttnn::SumOp>,
           ReductionOpConversionPattern<ttir::MeanOp, ttnn::MeanOp>,
           ReductionOpConversionPattern<ttir::MaxOp, ttnn::MaxOp>,
           BroadcastOpConversionPattern,
           EmbeddingOpConversionPattern,
           SoftmaxOpConversionPattern,
           TransposeOpConversionPattern,
           TypecastOpConversionPattern,
           ConcatOpConversionPattern,
           ReshapeOpConversionPattern,
           SliceOpConversionPattern,
           SqueezeOpConversionPattern,
           UnsqueezeOpConversionPattern,
           ConstantOpConversionPattern,
           MatmulOpConversionPattern,
           Conv2dOpConversionPattern,
           MaxPool2dOpConversionPattern,
           SubtractOpConversionPattern
           >(typeConverter, ctx);
  // ANCHOR_END: op_rewriter_pattern_set
  // clang-format on
}

} // namespace mlir::tt
