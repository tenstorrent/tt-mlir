// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Support/Logger.h"

#include "llvm/Support/raw_ostream.h"

#include <cassert>

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNDECOMPOSELAYOUTS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

class TTNNDecomposeLayouts
    : public impl::TTNNDecomposeLayoutsBase<TTNNDecomposeLayouts> {

public:
  using impl::TTNNDecomposeLayoutsBase<
      TTNNDecomposeLayouts>::TTNNDecomposeLayoutsBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());
    llvm::SmallVector<Operation *> opsToReplace;
    module->walk([&](func::FuncOp func) {
      if (func.isDeclaration()) {
        return;
      }
      assert(func.getBody().hasOneBlock() &&
             "Found func that didn't have one block!");
      func->walk([&](Operation *op) {
        if (!isa<ttnn::ToLayoutOp>(op)) {
          return;
        }
        opsToReplace.push_back(op);
      });
    });
    for (Operation *op : opsToReplace) {
      if (failed(createLayoutConversionOps(mlir::cast<ttnn::ToLayoutOp>(op),
                                           rewriter))) {
        signalPassFailure();
        return;
      }
      rewriter.eraseOp(op);
    }
  }

private:
  struct LayoutInfo {
    ttnn::BufferType bufferType;
    ttnn::Layout layoutEnum;
    ttcore::DataType dataType;
    ttnn::TensorMemoryLayoutAttr tensorMemoryLayout;
    ttcore::GridAttr deviceGrid;
    ttcore::GridAttr shardGrid;
    llvm::SmallVector<int64_t> shardShape;

    ttnn::MemoryConfigAttr createMemoryConfigAttr(MLIRContext *context) const {
      return ttnn::MemoryConfigAttr::get(
          context, tensorMemoryLayout,
          ttnn::BufferTypeAttr::get(context, bufferType),
          utils::createShardSpecIfNeeded(
              tensorMemoryLayout, ttnn::ShapeAttr::get(context, shardShape),
              shardGrid, deviceGrid));
    }
    bool isL1Sharded() const {
      return isShardedMemoryLayout(tensorMemoryLayout.getValue());
    }
    bool isOnHost() const {
      return bufferType == ttnn::BufferType::SystemMemory;
    }
    bool isOnDevice() const { return !isOnHost(); }
    bool isTilized() const { return layoutEnum == ttnn::Layout::Tile; }
  };

  struct OpsToCreate {
    bool createToDeviceOp = false;
    bool createFromDeviceOp = false;
    bool createToLayoutOp = false;
    bool createDataTypeCastOp = false;
    bool createToMemoryConfigOp = false;

    bool createSomeOp() const {
      return createToLayoutOp || createDataTypeCastOp || createToDeviceOp ||
             createFromDeviceOp || createToMemoryConfigOp;
    }

    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                         const OpsToCreate opsToCreate) {
      os << "OpsToCreate{ \n"
         << "\t"
         << "CreateToDeviceOp: " << opsToCreate.createToDeviceOp << "\n"
         << "\t"
         << "CreateFromDeviceOp: " << opsToCreate.createFromDeviceOp << "\n"
         << "\t"
         << "CreateToLayoutOp: " << opsToCreate.createToLayoutOp << "\n"
         << "\t"
         << "CreateTypecastOp: " << opsToCreate.createDataTypeCastOp << "\n"
         << "\t"
         << "CreateToMemoryConfigOp: " << opsToCreate.createToMemoryConfigOp
         << "\n}\n";
      return os;
    }
  };

  struct OpCreationInfo {
    LayoutInfo input;
    LayoutInfo output;
    OpsToCreate opsToCreate;

    OpCreationInfo(const LayoutInfo &input, const LayoutInfo &output,
                   const OpsToCreate &opsToCreate)
        : input(input), output(output), opsToCreate(opsToCreate) {}

    bool shouldUntilize() const {
      return opsToCreate.createToLayoutOp && !output.isTilized();
    }

    bool shouldTilize() const {
      return opsToCreate.createToLayoutOp && output.isTilized();
    }
  };

  bool canTilizeDataTypeOnDevice(const ttcore::DataType &dataType) const {
    // tt-metal tilize supports: bfloat16, float32, uint32, int32, uint16
    // See: ttnn/operations/data_movement/tilize/device/tilize_op.cpp
    return dataType == ttcore::DataType::BFloat16 ||
           dataType == ttcore::DataType::Float32 ||
           dataType == ttcore::DataType::UInt32 ||
           dataType == ttcore::DataType::UInt16 ||
           dataType == ttcore::DataType::Int32;
  }

  bool canUntilizeDataTypeOnDevice(const ttcore::DataType &dataType) const {
    // tt-metal untilize supports: bfloat16, float32, uint32, int32
    // (requires use_pack_untilize for uint32/int32)
    // See: ttnn/operations/data_movement/untilize/device/untilize_op.cpp
    return dataType == ttcore::DataType::BFloat16 ||
           dataType == ttcore::DataType::Float32 ||
           dataType == ttcore::DataType::UInt32 ||
           dataType == ttcore::DataType::Int32;
  }

  std::pair<LayoutInfo, LayoutInfo>
  getInputOutputLayouts(ttnn::ToLayoutOp op) const {
    LayoutInfo input, output;

    auto inputLayoutAttr =
        mlir::cast<TTNNLayoutAttr>(op.getInput().getType().getEncoding());

    auto outputLayoutAttr =
        mlir::cast<TTNNLayoutAttr>(op.getResult().getType().getEncoding());

    assert(op.getMemoryConfig().has_value());
    MemoryConfigAttr outputMemoryConfig = op.getMemoryConfig().value();

    input.bufferType = inputLayoutAttr.getBufferType();
    output.bufferType = outputMemoryConfig.getBufferType().getValue();

    input.layoutEnum = inputLayoutAttr.getLayout();
    output.layoutEnum = outputLayoutAttr.getLayout();

    input.dataType = inputLayoutAttr.getDataType();
    assert(op.getDtype().has_value());
    output.dataType = op.getDtype().value();

    input.tensorMemoryLayout = inputLayoutAttr.getMemLayout();
    output.tensorMemoryLayout = outputMemoryConfig.getTensorMemoryLayout();

    input.shardGrid = inputLayoutAttr.getGrid();
    output.shardGrid = outputLayoutAttr.getGrid();

    input.shardShape = inputLayoutAttr.getScalarShardShape();
    output.shardShape = outputLayoutAttr.getScalarShardShape();

    ttcore::DeviceAttr deviceAttr =
        ttcore::lookupDevice(op.getResult().getParentBlock()->getParentOp());
    input.deviceGrid = deviceAttr.getWorkerGrid();
    output.deviceGrid = deviceAttr.getWorkerGrid();

    TTMLIR_DEBUG(ttmlir::LogComponent::General,
                 "Decompose layouts pass for op {} \nInput layout: {} \nOutput "
                 "layout: {} \n",
                 op, inputLayoutAttr, outputLayoutAttr);

    return {input, output};
  }

  OpsToCreate determineRequiredOps(const LayoutInfo &input,
                                   const LayoutInfo &output) const {
    OpsToCreate opsToCreate;

    opsToCreate.createToDeviceOp =
        (input.bufferType != output.bufferType) && input.isOnHost();
    opsToCreate.createFromDeviceOp =
        (input.bufferType != output.bufferType) && output.isOnHost();

    opsToCreate.createDataTypeCastOp = input.dataType != output.dataType;
    opsToCreate.createToLayoutOp = input.layoutEnum != output.layoutEnum;

    // ToDeviceOp can handle the creation of the memory config of the initial
    // device tensor
    if (!opsToCreate.createToDeviceOp && output.isOnDevice()) {
      opsToCreate.createToMemoryConfigOp =
          output.tensorMemoryLayout &&
          (input.tensorMemoryLayout != output.tensorMemoryLayout);
      opsToCreate.createToMemoryConfigOp |=
          (input.bufferType == ttnn::BufferType::DRAM &&
           output.bufferType == ttnn::BufferType::L1) ||
          (input.bufferType == ttnn::BufferType::L1 &&
           output.bufferType == ttnn::BufferType::DRAM);
      // If shard grids don't match we need to reshard
      opsToCreate.createToMemoryConfigOp |=
          (input.isL1Sharded() && output.isL1Sharded() &&
           input.shardGrid != output.shardGrid);
    }

    TTMLIR_DEBUG(ttmlir::LogComponent::General,
                 "Decompose layouts pass ops to create: {}", opsToCreate);
    return opsToCreate;
  }

  bool isCreationValid(ttnn::ToLayoutOp op, const LayoutInfo &input,
                       const LayoutInfo &output,
                       const OpsToCreate &opsToCreate) const {

    if (!opsToCreate.createSomeOp()) {
      op->emitError(
          "Redundant ttnn::ToLayoutOp - no ttnn layout ops "
          "needed, this may be due to the forcing of tile/row major layouts.");
      return false;
    }

    if (opsToCreate.createToDeviceOp && opsToCreate.createFromDeviceOp) {
      op->emitError("Cannot create both ToDeviceOp and FromDeviceOp");
      return false;
    }

    if (opsToCreate.createToMemoryConfigOp &&
        output.bufferType == ttnn::BufferType::SystemMemory) {
      op->emitError(
          "ToMemoryConfigOp only supported for device output tensors");
      return false;
    }

    if (input.isOnHost() && opsToCreate.createFromDeviceOp) {
      op->emitError("Unexpected FromDeviceOp on host tensor");
      return false;
    }

    if (input.isOnDevice() && opsToCreate.createToDeviceOp) {
      op->emitError("Unexpected ToDeviceOp on device tensor");
      return false;
    }
    return true;
  }

  /* Helper functions to create ttnn layout ops */

  template <typename OpType, typename... Args>
  mlir::Value createOp(ttnn::ToLayoutOp op, IRRewriter &rewriter,
                       mlir::Value currentInput, Args &&...args) const {

    rewriter.setInsertionPoint(op);
    return rewriter.create<OpType>(op.getLoc(), op.getType(), currentInput,
                                   std::forward<Args>(args)...);
  }

  template <typename OpType, typename... Args>
  mlir::Value createOp(IRRewriter &rewriter, ttnn::ToLayoutOp op,
                       RankedTensorType newResultType, mlir::Value currentInput,
                       Args &&...args) const {
    rewriter.setInsertionPoint(op);
    return rewriter.create<OpType>(op.getLoc(), newResultType, currentInput,
                                   std::forward<Args>(args)...);
  }

  mlir::Value createToDeviceOpIfNeeded(ttnn::ToLayoutOp op,
                                       IRRewriter &rewriter,
                                       mlir::Value currentInput,
                                       const OpCreationInfo &info,
                                       bool forceCreate = false) const {
    if (!info.opsToCreate.createToDeviceOp && !forceCreate) {
      return currentInput;
    }
    ttnn::MemoryConfigAttr memoryConfigAttr =
        info.output.createMemoryConfigAttr(op.getContext());
    RankedTensorType currentInputType =
        mlir::cast<RankedTensorType>(currentInput.getType());
    RankedTensorType newResultType = utils::RankedTensorTypeFactory::create(
        currentInputType, info.output.bufferType);
    newResultType = utils::RankedTensorTypeFactory::create(
        newResultType, info.output.tensorMemoryLayout.getValue());

    // Respect grid attribute of the output layout
    newResultType = utils::RankedTensorTypeFactory::create(
        newResultType, info.output.shardGrid);

    mlir::Value device = utils::getOrInsertDevice(rewriter, op);

    // Create new ranked tensor type with host memory buffer type
    return this->createOp<ttnn::ToDeviceOp>(
        rewriter, op, newResultType, currentInput, device, memoryConfigAttr);
  }

  // FromDeviceOp
  mlir::Value createFromDeviceOpIfNeeded(ttnn::ToLayoutOp op,
                                         IRRewriter &rewriter,
                                         mlir::Value currentInput,
                                         const OpCreationInfo &info,
                                         bool forceCreate = false) const {
    if (!info.opsToCreate.createFromDeviceOp && !forceCreate) {
      return currentInput;
    }
    RankedTensorType currentInputType =
        mlir::cast<RankedTensorType>(currentInput.getType());
    RankedTensorType newResultType = utils::RankedTensorTypeFactory::create(
        currentInputType, ttnn::BufferType::SystemMemory);
    return this->createOp<ttnn::FromDeviceOp>(rewriter, op, newResultType,
                                              currentInput);
  }

  mlir::Value createToLayoutOpIfNeeded(ttnn::ToLayoutOp op,
                                       IRRewriter &rewriter,
                                       mlir::Value currentInput,
                                       const OpCreationInfo &info) const {
    if (!info.opsToCreate.createToLayoutOp) {
      return currentInput;
    }
    ttnn::LayoutAttr layoutAttr =
        ttnn::LayoutAttr::get(op.getContext(), info.output.layoutEnum);
    RankedTensorType newResultType = utils::RankedTensorTypeFactory::create(
        mlir::cast<RankedTensorType>(currentInput.getType()),
        info.output.layoutEnum);

    return this->createOp<ttnn::ToLayoutOp>(rewriter, op, newResultType,
                                            currentInput, layoutAttr,
                                            /*dtype*/ nullptr,
                                            /*memory_config*/ nullptr);
  }

  mlir::Value createDataTypeCastingOp(ttnn::ToLayoutOp op, IRRewriter &rewriter,
                                      mlir::Value currentInput,
                                      const OpCreationInfo &info) const {
    ttcore::DataTypeAttr dtypeAttr =
        ttcore::DataTypeAttr::get(op.getContext(), info.output.dataType);
    RankedTensorType newResultType = utils::RankedTensorTypeFactory::create(
        mlir::cast<RankedTensorType>(currentInput.getType()),
        info.output.dataType);
    return this->createOp<ttnn::TypecastOp>(rewriter, op, newResultType,
                                            currentInput, dtypeAttr);
  }

  mlir::Value
  createDataTypeCastingOpIfNeeded(ttnn::ToLayoutOp op, IRRewriter &rewriter,
                                  mlir::Value currentInput,
                                  const OpCreationInfo &info) const {
    if (!info.opsToCreate.createDataTypeCastOp) {
      return currentInput;
    }

    RankedTensorType currentInputType =
        mlir::cast<RankedTensorType>(currentInput.getType());

    TTNNLayoutAttr inputLayout =
        mlir::cast<TTNNLayoutAttr>(currentInputType.getEncoding());
    if (!inputLayout.isSystemBufferType()) {
      assert(inputLayout.getLayout() == Layout::Tile &&
             "Only tilized tensors are supported for device typecast");
    }
    return this->createDataTypeCastingOp(op, rewriter, currentInput, info);
  }

  mlir::Value createToMemoryConfigOpIfNeeded(ttnn::ToLayoutOp op,
                                             IRRewriter &rewriter,
                                             mlir::Value currentInput,
                                             const OpCreationInfo &info) const {
    if (!info.opsToCreate.createToMemoryConfigOp) {
      return currentInput;
    }
    ttnn::MemoryConfigAttr memoryConfigAttr =
        info.output.createMemoryConfigAttr(op.getContext());
    RankedTensorType currentInputType =
        mlir::cast<RankedTensorType>(currentInput.getType());
    TTNNLayoutAttr newLayout =
        utils::getLayoutAttrFromTensor(currentInputType)
            .withBufferType(info.output.bufferType)
            .withMemoryLayout(info.output.tensorMemoryLayout)
            .withGrid(currentInputType.getShape(), info.output.shardGrid)
            .withShardShape(info.output.shardShape);
    RankedTensorType newResultType =
        utils::RankedTensorTypeFactory::create(currentInputType, newLayout);
    return this->createOp<ttnn::ToMemoryConfigOp>(
        rewriter, op, newResultType, currentInput, memoryConfigAttr);
  }

  /* Functions that create ops based on the layouts of the input output tensors
   */

  void handleHostInputNoLayoutNoTypecast(ttnn::ToLayoutOp op,
                                         IRRewriter &rewriter,
                                         mlir::Value currentInput,
                                         const OpCreationInfo &info) const {
    // If the input tensor is on host and we don't need to create a ToLayoutOp
    // nor a TypecastOp we can create a ToDeviceOp and a ToMemoryConfigOp if
    // needed and return
    currentInput =
        this->createToDeviceOpIfNeeded(op, rewriter, currentInput, info);
    currentInput =
        this->createToMemoryConfigOpIfNeeded(op, rewriter, currentInput, info);
    op.getResult().replaceAllUsesWith(currentInput);
  }

  void handleHostInputLayoutNoTypecast(ttnn::ToLayoutOp op,
                                       IRRewriter &rewriter,
                                       mlir::Value currentInput,
                                       const OpCreationInfo &info) const {
    const LayoutInfo &input = info.input;
    const LayoutInfo &output = info.output;
    assert(input.dataType == output.dataType &&
           "Data type should be the same if we're not creating typecast op");

    // If the output is on the host, we can perform layout conversion on host.
    if (output.isOnHost()) {
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    // If the output is on device and we can untilize on device, move to device
    // and untilize.
    if (info.shouldUntilize() && canUntilizeDataTypeOnDevice(output.dataType)) {
      currentInput =
          this->createToDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    // If the output is on device and we should untilize, we untilize on
    // host and then move the tensor to device.
    if (info.shouldUntilize() &&
        !canUntilizeDataTypeOnDevice(output.dataType)) {
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    // If the tensor tilizable on device, we can move the tensor to device and
    // perform the tilization on device.
    if (info.shouldTilize() && canTilizeDataTypeOnDevice(output.dataType)) {
      currentInput =
          this->createToDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    // Otherwise, if tensor is not in a tilizable data format, we perform
    // tilizing on host and than move the tensor to device.
    if (info.shouldTilize() && !canTilizeDataTypeOnDevice(output.dataType)) {
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    llvm_unreachable("Unreachable code path");
  }

  void handleHostInputNoLayoutTypecast(ttnn::ToLayoutOp op,
                                       IRRewriter &rewriter,
                                       mlir::Value currentInput,
                                       const OpCreationInfo &info) const {
    const LayoutInfo &input = info.input;
    const LayoutInfo &output = info.output;
    assert(input.layoutEnum == output.layoutEnum &&
           "Layout should be the same if we're not creating a ToLayoutOp");

    // If the output is on the host, we can perform the data type cast directly
    // on the host.
    if (output.isOnHost()) {
      currentInput = this->createDataTypeCastingOpIfNeeded(op, rewriter,
                                                           currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    // Device typecast only supports tilized tensors. Therefore, if the output
    // tensor is in row-major (input as well is in row-major) and resides on the
    // device, we should perform the data type casting on the host before moving
    // the tensor back to the device.
    if (!output.isTilized()) {
      currentInput = this->createDataTypeCastingOpIfNeeded(op, rewriter,
                                                           currentInput, info);
      currentInput =
          this->createToDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    // If the output tensor is tilized and resides on the device, we can move
    // the tensor to the device and perform the data type cast directly on the
    // device.
    if (output.isTilized()) {
      currentInput =
          this->createToDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createDataTypeCastingOpIfNeeded(op, rewriter,
                                                           currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    llvm_unreachable("Unreachable code path");
  }

  void handleHostInputLayoutTypecast(ttnn::ToLayoutOp op, IRRewriter &rewriter,
                                     mlir::Value currentInput,
                                     const OpCreationInfo &info) const {
    const LayoutInfo &input = info.input;
    const LayoutInfo &output = info.output;

    // If the output tensor is on host, we can perform the data type cast and
    // layout conversion on host.
    if (output.isOnHost()) {
      currentInput = this->createDataTypeCastingOpIfNeeded(op, rewriter,
                                                           currentInput, info);
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    // If we can untilize the output data type on device, move to device if
    // needed, perform the typecast first and then untilize
    if (info.shouldUntilize() && canUntilizeDataTypeOnDevice(output.dataType)) {
      currentInput =
          this->createToDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createDataTypeCastingOpIfNeeded(op, rewriter,
                                                           currentInput, info);
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    // If we cannot untilize the output data type on device, untilize and
    // typecast on host
    if (info.shouldUntilize() &&
        !canUntilizeDataTypeOnDevice(output.dataType)) {
      currentInput = this->createDataTypeCastingOpIfNeeded(op, rewriter,
                                                           currentInput, info);
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    // If we need to tilize and change the data type from a tilizable data
    // format to another format, we can move the tensor to the device, perform
    // the tilization, and then cast the data type on the device
    if (info.shouldTilize() && canTilizeDataTypeOnDevice(input.dataType)) {
      currentInput =
          this->createToDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createDataTypeCastingOpIfNeeded(op, rewriter,
                                                           currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    // If we need to tilize and change the data format from another format
    // to a tilizable data format, we can cast the data type on host, move
    // the tensor to device, and then tilize on device.
    if (info.shouldTilize() && canTilizeDataTypeOnDevice(output.dataType)) {
      currentInput = this->createDataTypeCastingOpIfNeeded(op, rewriter,
                                                           currentInput, info);
      currentInput =
          this->createToDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    // If we need to tilize and the input/output data types are not device
    // tilizable do everything on host
    if (info.shouldTilize() && !canTilizeDataTypeOnDevice(input.dataType) &&
        !canTilizeDataTypeOnDevice(output.dataType)) {
      currentInput = this->createDataTypeCastingOpIfNeeded(op, rewriter,
                                                           currentInput, info);
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    llvm_unreachable("Unreachable code path");
  }

  void handleHostInputLayoutConversion(ttnn::ToLayoutOp op,
                                       IRRewriter &rewriter,
                                       mlir::Value currentInput,
                                       const OpCreationInfo &info) const {
    const OpsToCreate &opsToCreate = info.opsToCreate;
    if (!opsToCreate.createToLayoutOp && !opsToCreate.createDataTypeCastOp) {
      return handleHostInputNoLayoutNoTypecast(op, rewriter, currentInput,
                                               info);
    }
    if (opsToCreate.createToLayoutOp && !opsToCreate.createDataTypeCastOp) {
      return handleHostInputLayoutNoTypecast(op, rewriter, currentInput, info);
    }
    if (!opsToCreate.createToLayoutOp && opsToCreate.createDataTypeCastOp) {
      return handleHostInputNoLayoutTypecast(op, rewriter, currentInput, info);
    }
    if (opsToCreate.createToLayoutOp && opsToCreate.createDataTypeCastOp) {
      return handleHostInputLayoutTypecast(op, rewriter, currentInput, info);
    }
    llvm_unreachable("Unreachable code path");
  }

  void handleDeviceInputNoLayoutNoTypecast(ttnn::ToLayoutOp op,
                                           IRRewriter &rewriter,
                                           mlir::Value currentInput,
                                           const OpCreationInfo &info) const {
    // If the input tensor is on device and we don't need to create a ToLayoutOp
    // nor a TypecastOp we can create a FromDeviceOp and a ToMemoryConfigOp if
    // needed and return
    currentInput =
        this->createToMemoryConfigOpIfNeeded(op, rewriter, currentInput, info);
    currentInput =
        this->createFromDeviceOpIfNeeded(op, rewriter, currentInput, info);
    op.getResult().replaceAllUsesWith(currentInput);
  }

  void handleDeviceInputLayoutNoTypecast(ttnn::ToLayoutOp op,
                                         IRRewriter &rewriter,
                                         mlir::Value currentInput,
                                         const OpCreationInfo &info) const {
    const LayoutInfo &input = info.input;
    const LayoutInfo &output = info.output;
    const OpsToCreate &opsToCreate = info.opsToCreate;
    assert(input.dataType == output.dataType &&
           "Data type should be the same if we're not creating typecast op");

    // If the output data type is untilizable on device, untilize on device then
    // move to host
    if (info.shouldUntilize() && canUntilizeDataTypeOnDevice(input.dataType)) {
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      currentInput =
          this->createFromDeviceOpIfNeeded(op, rewriter, currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    // If we want to untilize, but the output data type is not untilizable on
    // device, move to host and then untilize
    if (info.shouldUntilize() && !canUntilizeDataTypeOnDevice(input.dataType) &&
        opsToCreate.createFromDeviceOp) {
      currentInput =
          this->createFromDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    // This is a rare untilize case, where we want to untilize a device tensor
    // but keep it on device. To handle this we need to move the tensor to host,
    // untilize it, and then move it back to device
    if (info.shouldUntilize() && !canUntilizeDataTypeOnDevice(input.dataType) &&
        !opsToCreate.createFromDeviceOp) {
      // Force-create a FromDeviceOp
      currentInput = this->createFromDeviceOpIfNeeded(
          op, rewriter, currentInput, info, /*forceCreate=*/true);
      // untilize on host
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      // move back to device and convert memory config if needed
      currentInput = createToDeviceOpIfNeeded(op, rewriter, currentInput, info,
                                              /*forceCreate=*/true);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    // If we should tilize and the input data type is device-tilizable, tilize
    // on device
    if (info.shouldTilize() && canTilizeDataTypeOnDevice(input.dataType)) {
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      currentInput =
          this->createFromDeviceOpIfNeeded(op, rewriter, currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    // If we should tilize and the input data type is not device tilizable,
    // tilize on host
    if (info.shouldTilize() && !canTilizeDataTypeOnDevice(input.dataType) &&
        opsToCreate.createFromDeviceOp) {
      currentInput =
          this->createFromDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    // If we want to tilize a device tensor that is not device tilizable, we
    // need to tilize on host and move it back
    if (info.shouldTilize() && !canTilizeDataTypeOnDevice(input.dataType) &&
        !opsToCreate.createFromDeviceOp) {
      // Force-create a FromDeviceOp
      currentInput = this->createFromDeviceOpIfNeeded(
          op, rewriter, currentInput, info, /*forceCreate=*/true);
      // tilize on host
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      // move back to device and convert memory config if needed
      currentInput = this->createToDeviceOpIfNeeded(op, rewriter, currentInput,
                                                    info, /*forceCreate=*/true);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    llvm_unreachable("Unreachable code path");
  }

  void handleDeviceInputNoLayoutTypecast(ttnn::ToLayoutOp op,
                                         IRRewriter &rewriter,
                                         mlir::Value currentInput,
                                         const OpCreationInfo &info) const {
    const LayoutInfo &input = info.input;
    const LayoutInfo &output = info.output;
    const OpsToCreate &opsToCreate = info.opsToCreate;
    assert(input.layoutEnum == output.layoutEnum &&
           "Layout should be the same if we're not creating toLayout op");

    // If the output is tilized, typecast directly on device
    if (output.isTilized()) {
      // If the input is sharded, typecast should happen after converting to
      // memory.
      if (input.isL1Sharded()) {
        currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                            currentInput, info);
        currentInput = this->createDataTypeCastingOpIfNeeded(
            op, rewriter, currentInput, info);
      } else {
        currentInput = this->createDataTypeCastingOpIfNeeded(
            op, rewriter, currentInput, info);
        currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                            currentInput, info);
      }
      currentInput =
          this->createFromDeviceOpIfNeeded(op, rewriter, currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    // If the output is not tilized, typecast on host
    if (!output.isTilized() && opsToCreate.createFromDeviceOp) {
      currentInput =
          this->createFromDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createDataTypeCastingOpIfNeeded(op, rewriter,
                                                           currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    // Device to device untilized typecast, need to move to host first
    if (!output.isTilized() && !opsToCreate.createFromDeviceOp) {
      // Force-create a FromDeviceOp
      currentInput = this->createFromDeviceOpIfNeeded(
          op, rewriter, currentInput, info, /*forceCreate=*/true);
      // typecast on host
      currentInput = this->createDataTypeCastingOpIfNeeded(op, rewriter,
                                                           currentInput, info);
      // move back to device and convert memory config if needed
      currentInput = this->createToDeviceOpIfNeeded(op, rewriter, currentInput,
                                                    info, /*forceCreate=*/true);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    llvm_unreachable("Unreachable code path");
  }

  void handleDeviceInputLayoutTypecast(ttnn::ToLayoutOp op,
                                       IRRewriter &rewriter,
                                       mlir::Value currentInput,
                                       const OpCreationInfo &info) const {
    const LayoutInfo &input = info.input;
    const LayoutInfo &output = info.output;
    const OpsToCreate &opsToCreate = info.opsToCreate;

    // If we need to untilize and the output data type can be untilized on
    // device typecast and untilize on device
    if (info.shouldUntilize() && canUntilizeDataTypeOnDevice(output.dataType)) {
      currentInput = this->createDataTypeCastingOpIfNeeded(op, rewriter,
                                                           currentInput, info);
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createFromDeviceOpIfNeeded(op, rewriter, currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    // If we need to untilize and the output data type cannot be untilized on
    // device typecast on device then untilize on host
    if (info.shouldUntilize() &&
        !canUntilizeDataTypeOnDevice(output.dataType) &&
        opsToCreate.createFromDeviceOp) {
      currentInput = this->createDataTypeCastingOpIfNeeded(op, rewriter,
                                                           currentInput, info);
      currentInput =
          this->createFromDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    // In case of device to device untilize, where the output data type cannot
    // be untilized on device, typecast on device, untilize on host, then move
    // back to device
    if (info.shouldUntilize() &&
        !canUntilizeDataTypeOnDevice(output.dataType) &&
        !opsToCreate.createFromDeviceOp) {
      currentInput = this->createDataTypeCastingOpIfNeeded(op, rewriter,
                                                           currentInput, info);
      currentInput = this->createFromDeviceOpIfNeeded(
          op, rewriter, currentInput, info, /*forceCreate=*/true);
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToDeviceOpIfNeeded(op, rewriter, currentInput,
                                                    info, /*forceCreate=*/true);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    // If we should tilize and the input data type is device tilizable, tilize
    // and typecast on device
    if (info.shouldTilize() && canTilizeDataTypeOnDevice(input.dataType)) {
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createDataTypeCastingOpIfNeeded(op, rewriter,
                                                           currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      currentInput =
          this->createFromDeviceOpIfNeeded(op, rewriter, currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    // If we should tilize and the input data type is not device tilizable and
    // we want to read back from device do everything on host
    if (info.shouldTilize() && !canTilizeDataTypeOnDevice(input.dataType) &&
        opsToCreate.createFromDeviceOp) {
      currentInput =
          this->createFromDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createDataTypeCastingOpIfNeeded(op, rewriter,
                                                           currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    // If we should tilize and the input data type is not device tilizable and
    // we don't want to read back from device: tilize on host, move back to
    // device, and typecast on device
    if (info.shouldTilize() && !canTilizeDataTypeOnDevice(input.dataType) &&
        !opsToCreate.createFromDeviceOp) {
      // Force-create a FromDeviceOp
      currentInput = this->createFromDeviceOpIfNeeded(
          op, rewriter, currentInput, info, /*forceCreate=*/true);
      // tilize on host
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      // move back to device and convert data type/memory config if needed
      currentInput = this->createToDeviceOpIfNeeded(op, rewriter, currentInput,
                                                    info, /*forceCreate=*/true);
      currentInput = this->createDataTypeCastingOpIfNeeded(op, rewriter,
                                                           currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    llvm_unreachable("Unreachable code path");
  }

  void handleDeviceInputLayoutConversion(ttnn::ToLayoutOp op,
                                         IRRewriter &rewriter,
                                         mlir::Value currentInput,
                                         const OpCreationInfo &info) const {
    const OpsToCreate &opsToCreate = info.opsToCreate;
    if (!opsToCreate.createToLayoutOp && !opsToCreate.createDataTypeCastOp) {
      handleDeviceInputNoLayoutNoTypecast(op, rewriter, currentInput, info);
      return;
    }
    if (opsToCreate.createToLayoutOp && !opsToCreate.createDataTypeCastOp) {
      handleDeviceInputLayoutNoTypecast(op, rewriter, currentInput, info);
      return;
    }
    if (not opsToCreate.createToLayoutOp && opsToCreate.createDataTypeCastOp) {
      handleDeviceInputNoLayoutTypecast(op, rewriter, currentInput, info);
      return;
    }
    if (opsToCreate.createToLayoutOp && opsToCreate.createDataTypeCastOp) {
      handleDeviceInputLayoutTypecast(op, rewriter, currentInput, info);
      return;
    }
    llvm_unreachable("Unreachable code path");
  }

  /*
   * Logic for creating ops. Conditions/constraints include:
   * - When possible, we want to execute operations on device.
   * - Tilize on device requires dataformat of bfloat16 or float32.
   * - Typecast on device requires TILIZED tensor.
   * - Untilize on device requires even width, and page size >
   *   sizeof(uint32_t). For now, we will always untilize on host. We rarely
   * need device to device untilize, so the perf hit should be acceptable.
   */
  mlir::LogicalResult createLayoutConversionOps(ttnn::ToLayoutOp op,
                                                IRRewriter &rewriter) const {
    auto [input, output] = getInputOutputLayouts(op);
    OpsToCreate opsToCreate = determineRequiredOps(input, output);
    if (not isCreationValid(op, input, output, opsToCreate)) {
      return failure();
    }

    OpCreationInfo info(input, output, opsToCreate);

    Value currentInput = op.getInput();

    if (input.isOnHost()) {
      handleHostInputLayoutConversion(op, rewriter, currentInput, info);
      return success();
    }
    handleDeviceInputLayoutConversion(op, rewriter, currentInput, info);
    return success();
  }
};
} // namespace mlir::tt::ttnn
