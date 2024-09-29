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

static DataType getDataTypeFromMemRef(mlir::MemRefType memref) {
  Type elementType = memref.getElementType();
  DataType dtype = DataType::Float32;
  if (llvm::isa<TileType>(elementType)) {
    auto tileType = mlir::cast<TileType>(elementType);
    dtype = tileType.getDataType();
  } else {
    dtype = elementTypeToDataType(elementType);
  }
  return dtype;
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

  struct CreationFlags {
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
      output.layoutEnum = ttnn::Layout::Tile;
    }

    input.dataType = getDataTypeFromMemRef(inputMemref);
    output.dataType = getDataTypeFromMemRef(outputMemref);

    input.tensorMemoryLayout =
        ttnn::utils::toTTNNTensorMemoryLayout(inputLayoutAttr.getMemLayout());
    output.tensorMemoryLayout =
        ttnn::utils::toTTNNTensorMemoryLayout(outputLayoutAttr.getMemLayout());

    return {input, output};
  }

  CreationFlags determineRequiredOps(const LayoutInfo &input,
                                     const LayoutInfo &output) const {
    CreationFlags flags;

    flags.createTypecastOp = input.dataType != output.dataType;
    flags.createToDeviceOp =
        (input.bufferType != output.bufferType) and input.isOnHost();
    flags.createFromDeviceOp =
        (input.bufferType != output.bufferType) and output.isOnHost();

    flags.createToLayoutOp = input.layoutEnum != output.layoutEnum;
    // Insert a ToLayoutOp manually if we're moving from device to host to
    // untilize. Since we're hardcoding tile layout, the tensor may still be row
    // major in mlir, and therefore it would appear as if we don't need to
    // untilize
    flags.createToLayoutOp |= (flags.createFromDeviceOp and
                               output.layoutEnum == ttnn::Layout::RowMajor);

    // TODO(bug #620):
    // Add support for ShardSpec
    //
    flags.createToMemoryConfigOp =
        (input.tensorMemoryLayout != output.tensorMemoryLayout) and
        (output.tensorMemoryLayout != ttnn::TensorMemoryLayout::None);
    flags.createToMemoryConfigOp |=
        (input.bufferType == ttnn::BufferType::DRAM and
         output.bufferType == ttnn::BufferType::L1) or
        (input.bufferType == ttnn::BufferType::L1 and
         output.bufferType == ttnn::BufferType::DRAM);

    return flags;
  }

  template <typename OpType, typename... Args>
  void maybeCreateOp(ttir::ToLayoutOp op, ConversionPatternRewriter &rewriter,
                     mlir::Value &currentInput, bool shouldCreate,
                     bool forceCreate, Args... args) const {
    if (not shouldCreate and not forceCreate) {
      return;
    }
    currentInput = rewriter.create<OpType>(
        op.getLoc(), this->getTypeConverter()->convertType(op.getType()),
        currentInput, args...);
  }

  LogicalResult isCreationValid(ttir::ToLayoutOp op, const LayoutInfo &input,
                                const LayoutInfo &output,
                                const CreationFlags &creationFlags) const {
    if (not creationFlags.createSomeOp()) {
      op->emitError("Redundant ttir::ToLayoutOp - no ttnn layout ops "
                    "needed");
      return failure();
    }

    if (creationFlags.createToDeviceOp and creationFlags.createFromDeviceOp) {
      op->emitError("Cannot create both ToDeviceOp and FromDeviceOp");
      return failure();
    }

    if (creationFlags.createToMemoryConfigOp and
        output.bufferType == ttnn::BufferType::SystemMemory) {
      op->emitError(
          "ToMemoryConfigOp only supported for device output tensors");
      return failure();
    }

    if (input.isOnHost() and creationFlags.createFromDeviceOp) {
      op->emitError("Unexpected FromDeviceOp on host tensor");
      return failure();
    }

    if (input.isOnDevice() and creationFlags.createToDeviceOp) {
      op->emitError("Unexpected ToDeviceOp on device tensor");
      return failure();
    }
    return success();
  }

  LogicalResult
  createLayoutConversionOps(ttir::ToLayoutOp op,
                            ConversionPatternRewriter &rewriter) const {
    auto [input, output] = getInputOutputLayouts(op);
    CreationFlags creationFlags = determineRequiredOps(input, output);

    if (failed(isCreationValid(op, input, output, creationFlags))) {
      return failure();
    }

    auto device = getOrInsertDevice(rewriter, op);

    // These values will get updated by the lambdas
    Value currentInput = op.getInput();

    // Lambdas for creating layout conversion ops
    auto maybeCreateToDeviceOp = [this, &op, &rewriter, &currentInput,
                                  creationFlags,
                                  device](bool forceCreate = false) {
      this->maybeCreateOp<ttnn::ToDeviceOp>(op, rewriter, currentInput,
                                            creationFlags.createToDeviceOp,
                                            forceCreate, device);
    };

    auto maybeCreateToLayoutOp = [this, &op, &rewriter, &currentInput,
                                  creationFlags](ttnn::Layout outputLayoutEnum,
                                                 bool forceCreate = false) {
      auto outputLayoutAttr =
          ttnn::LayoutAttr::get(op.getContext(), outputLayoutEnum);
      this->maybeCreateOp<ttnn::ToLayoutOp>(op, rewriter, currentInput,
                                            creationFlags.createToLayoutOp,
                                            forceCreate, outputLayoutAttr);
    };

    auto maybeCreateTypecastOp = [this, &op, &rewriter, &currentInput,
                                  creationFlags](DataType outputDataType,
                                                 bool forceCreate = false) {
      auto outputDataTypeAttr =
          DataTypeAttr::get(op.getContext(), outputDataType);
      this->maybeCreateOp<ttnn::TypecastOp>(op, rewriter, currentInput,
                                            creationFlags.createTypecastOp,
                                            forceCreate, outputDataTypeAttr);
    };

    auto maybeCreateFromDeviceOp = [this, &op, &rewriter, &currentInput,
                                    creationFlags](bool forceCreate = false) {
      this->maybeCreateOp<ttnn::FromDeviceOp>(op, rewriter, currentInput,
                                              creationFlags.createFromDeviceOp,
                                              forceCreate);
    };

    auto maybeCreateToMemoryConfigOp =
        [this, &op, &rewriter, &currentInput, creationFlags](
            ttnn::TensorMemoryLayout outputTensorMemoryLayout,
            ttnn::BufferType outputBufferType, bool forceCreate = false) {
          ttnn::MemoryConfigAttr memoryConfigAttr = ttnn::MemoryConfigAttr::get(
              op.getContext(),
              ttnn::TensorMemoryLayoutAttr::get(op.getContext(),
                                                outputTensorMemoryLayout),
              ttnn::BufferTypeAttr::get(op.getContext(), outputBufferType));

          this->maybeCreateOp<ttnn::ToMemoryConfigOp>(
              op, rewriter, currentInput, creationFlags.createToMemoryConfigOp,
              forceCreate, memoryConfigAttr);
        };

    /*
     * Logic for creating ops. Conditions/constraints include:
     * - When possible, we want to execute operations on device.
     * - Tilize on device requires dataformat of bfloat16.
     * - Typecast on device requires TILIZED tensor.
     * - Untilize on device requires even width, and page size >
     * sizeof(uint32_t)
     *    - Currently not sure how page size is calculated. Typecasting and
     * padding don't seem like good solutions here either. Since pad -> untilize
     * -> unpad is tricky, and typecasting requires tilized tensors.
     *    - Thus for now, we will always untilize on host. We rarely need device
     * to device untilize, so the perf hit should be acceptable.
     */
    bool shouldTilize = (creationFlags.createToLayoutOp and
                         output.layoutEnum == ttnn::Layout::Tile);
    bool shouldUntilize = (creationFlags.createToLayoutOp and
                           output.layoutEnum == ttnn::Layout::RowMajor);

    // Handle host input tensor
    if (input.isOnHost()) {
      // Case 1.1
      // If we don't need to create a ToLayoutOp nor TypecastOp
      // Create to device op and to memory config op if needed and return
      if (not creationFlags.createToLayoutOp and
          not creationFlags.createTypecastOp) {
        maybeCreateToDeviceOp();
        maybeCreateToMemoryConfigOp(output.tensorMemoryLayout,
                                    output.bufferType);
        op.getResult().replaceAllUsesWith(currentInput);
        return success();
      }

      // Case 1.2
      // If we need to create a ToLayoutOp not a TypecastOp, check the data
      // format on tilization
      if (creationFlags.createToLayoutOp and
          not creationFlags.createTypecastOp) {
        if (shouldUntilize) {
          maybeCreateToLayoutOp(output.layoutEnum);
          maybeCreateToDeviceOp();
          maybeCreateToMemoryConfigOp(output.tensorMemoryLayout,
                                      output.bufferType);
          op.getResult().replaceAllUsesWith(currentInput);
          return success();
        }
        // We can tilize on device if data type is bfloat16
        if (shouldTilize and input.dataType == DataType::BFloat16) {
          maybeCreateToDeviceOp();
          maybeCreateToLayoutOp(output.layoutEnum);
          maybeCreateToMemoryConfigOp(output.tensorMemoryLayout,
                                      output.bufferType);
          op.getResult().replaceAllUsesWith(currentInput);
          return success();
        }
        // Currently, tilizing on host
        // TODO (jnie): Investigate if it's better to
        // typecast on host -> move to device -> tilize on device -> typecast
        // back on device
        if (shouldTilize and input.dataType != DataType::BFloat16) {
          maybeCreateToLayoutOp(output.layoutEnum);
          maybeCreateToDeviceOp();
          maybeCreateToMemoryConfigOp(output.tensorMemoryLayout,
                                      output.bufferType);
          op.getResult().replaceAllUsesWith(currentInput);
          return success();
        }
      }

      // Case 1.3
      // If we need need to create a TypecastOp but not a ToLayoutOp
      if (not creationFlags.createToLayoutOp and
          creationFlags.createTypecastOp) {
        if (input.isTilized()) {
          maybeCreateToDeviceOp();
          maybeCreateTypecastOp(output.dataType);
          maybeCreateToMemoryConfigOp(output.tensorMemoryLayout,
                                      output.bufferType);
          op.getResult().replaceAllUsesWith(currentInput);
          return success();
        }
        // Typecast on host
        if (not input.isTilized()) {
          maybeCreateTypecastOp(output.dataType);
          maybeCreateToDeviceOp();
          maybeCreateToMemoryConfigOp(output.tensorMemoryLayout,
                                      output.bufferType);
          op.getResult().replaceAllUsesWith(currentInput);
          return success();
        }
      }

      // Case 1.4
      // If we need to create both TypecastOp and ToLayoutOp
      if (creationFlags.createToLayoutOp and creationFlags.createTypecastOp) {
        // Untilize and typecast on host
        if (shouldUntilize) {
          maybeCreateTypecastOp(output.dataType);
          maybeCreateToLayoutOp(output.layoutEnum);
          maybeCreateToDeviceOp();
          maybeCreateToMemoryConfigOp(output.tensorMemoryLayout,
                                      output.bufferType);
          op.getResult().replaceAllUsesWith(currentInput);
          return success();
        }
        // If we're tilizing and the input datatype is bfloat16
        // try move to device -> tilize -> typecast -> to memory config
        if (shouldTilize and input.dataType == DataType::BFloat16) {
          maybeCreateToDeviceOp();
          maybeCreateToLayoutOp(output.layoutEnum);
          maybeCreateTypecastOp(output.dataType);
          maybeCreateToMemoryConfigOp(output.tensorMemoryLayout,
                                      output.bufferType);
          op.getResult().replaceAllUsesWith(currentInput);
          return success();
        }
        // If we're tilizing and the input data type is not bfloat16
        if (shouldTilize and input.dataType != DataType::BFloat16) {
          // If we want to typecast to bfloat16:
          // typecast on host -> try move to device -> tilize on device
          // potentially
          if (output.dataType == DataType::BFloat16) {
            maybeCreateTypecastOp(output.dataType);
            maybeCreateToDeviceOp();
            maybeCreateToLayoutOp(output.layoutEnum);
            maybeCreateToMemoryConfigOp(output.tensorMemoryLayout,
                                        output.bufferType);
            op.getResult().replaceAllUsesWith(currentInput);
            return success();
          }
          // tilize and typcast on host
          if (not creationFlags.createToDeviceOp) {
            maybeCreateToLayoutOp(output.layoutEnum);
            maybeCreateTypecastOp(output.dataType);
            maybeCreateToMemoryConfigOp(output.tensorMemoryLayout,
                                        output.bufferType);
            op.getResult().replaceAllUsesWith(currentInput);
            return success();
          }
          // move to device -> typecast bfloat16 -> tilize -> typecast to output
          // df
          if (creationFlags.createToDeviceOp) {
            maybeCreateToDeviceOp();
            maybeCreateTypecastOp(DataType::BFloat16, true);
            maybeCreateToLayoutOp(output.layoutEnum);
            maybeCreateTypecastOp(output.dataType);
            maybeCreateToMemoryConfigOp(output.tensorMemoryLayout,
                                        output.bufferType);
            op.getResult().replaceAllUsesWith(currentInput);
            return success();
          }
        }
      }
    }

    else if (input.isOnDevice()) {
      // Case 2.1
      // If we don't need to create a ToLayoutOp nor TypecastOp
      // Create to device op and to memory config op if needed and return
      if (not creationFlags.createToLayoutOp and
          not creationFlags.createTypecastOp) {
        maybeCreateToMemoryConfigOp(output.tensorMemoryLayout,
                                    output.bufferType);
        maybeCreateFromDeviceOp();
        op.getResult().replaceAllUsesWith(currentInput);
        return success();
      }
      // Case 2.2
      // If we need to create a ToLayoutOp not a TypecastOp, check the data
      // format on tilization
      if (creationFlags.createToLayoutOp and
          not creationFlags.createTypecastOp) {
        // This is the main untilize case
        // Where we move data from device to host at the end of the program
        if (shouldUntilize) {
          maybeCreateFromDeviceOp(true);
          maybeCreateToLayoutOp(output.layoutEnum);
          // Move back to device. This is a device to device untilize
          // Try to avoid
          if (not creationFlags.createFromDeviceOp) {
            maybeCreateToDeviceOp(true);
            maybeCreateToMemoryConfigOp(output.tensorMemoryLayout,
                                        output.bufferType);
          }
          op.getResult().replaceAllUsesWith(currentInput);
          return success();
        }
        // We can tilize on device if data type is bfloat16
        if (shouldTilize and input.dataType == DataType::BFloat16) {
          maybeCreateToLayoutOp(output.layoutEnum);
          maybeCreateToMemoryConfigOp(output.tensorMemoryLayout,
                                      output.bufferType);
          maybeCreateFromDeviceOp();
          op.getResult().replaceAllUsesWith(currentInput);
          return success();
        }
        // typecast bfloat16 -> tilize -> typecast output data type
        if (shouldTilize and input.dataType != DataType::BFloat16) {
          maybeCreateTypecastOp(DataType::BFloat16, true);
          maybeCreateToLayoutOp(output.layoutEnum);
          maybeCreateTypecastOp(output.dataType, true);
          maybeCreateToMemoryConfigOp(output.tensorMemoryLayout,
                                      output.bufferType);
          maybeCreateFromDeviceOp();
          op.getResult().replaceAllUsesWith(currentInput);
          return success();
        }
      }

      // Case 2.3
      // If we need to create a TypeCastOp but not a toLayoutOp
      if (not creationFlags.createToLayoutOp and
          creationFlags.createTypecastOp) {
        if (input.isTilized()) {
          maybeCreateTypecastOp(output.dataType);
          maybeCreateToMemoryConfigOp(output.tensorMemoryLayout,
                                      output.bufferType);
          maybeCreateFromDeviceOp();
          op.getResult().replaceAllUsesWith(currentInput);
          return success();
        }
        // if ROW_MAJOR and data format is bfloat16
        // tilize -> typecast on device -> untilize
        if (not input.isTilized() and input.dataType == DataType::BFloat16) {
          maybeCreateToLayoutOp(ttnn::Layout::Tile, true);
          maybeCreateTypecastOp(output.dataType);
          maybeCreateToLayoutOp(output.layoutEnum, true);
          maybeCreateToMemoryConfigOp(output.tensorMemoryLayout,
                                      output.bufferType);
          maybeCreateFromDeviceOp();
          op.getResult().replaceAllUsesWith(currentInput);
          return success();
        }
        // if ROW_MAJOR and data format is not bfloat16
        // typecast on host, because typecast requires TILE layout, tilize on
        // device requires bfloat16
        if (not input.isTilized() and input.dataType != DataType::BFloat16) {
          maybeCreateFromDeviceOp(true);
          maybeCreateTypecastOp(output.dataType);
          // move back to device if necessary
          if (not creationFlags.createFromDeviceOp) {
            maybeCreateToDeviceOp(true);
            maybeCreateToMemoryConfigOp(output.tensorMemoryLayout,
                                        output.bufferType);
          }
          op.getResult().replaceAllUsesWith(currentInput);
          return success();
        }
      }

      // Case 2.4
      // If we need to create both toLayoutOp and TypeCastOp
      if (creationFlags.createToLayoutOp and creationFlags.createTypecastOp) {
        // typecast on device and untilize on host
        if (shouldUntilize) {
          maybeCreateTypecastOp(output.dataType);
          maybeCreateFromDeviceOp(true);
          maybeCreateToLayoutOp(output.layoutEnum);
          // Device to device untilize. Try to avoid
          if (not creationFlags.createFromDeviceOp) {
            maybeCreateToDeviceOp(true);
            maybeCreateToMemoryConfigOp(output.tensorMemoryLayout,
                                        output.bufferType);
          }
          op.getResult().replaceAllUsesWith(currentInput);
          return success();
        }
        if (shouldTilize and input.dataType == DataType::BFloat16) {
          maybeCreateToLayoutOp(output.layoutEnum);
          maybeCreateTypecastOp(output.dataType);
          maybeCreateToMemoryConfigOp(output.tensorMemoryLayout,
                                      output.bufferType);
          maybeCreateFromDeviceOp();
          op.getResult().replaceAllUsesWith(currentInput);
          return success();
        }
        if (shouldTilize and input.dataType != DataType::BFloat16) {
          // move to host -> typecast on host -> move to device -> tilize
          if (output.dataType == DataType::BFloat16 and
              not creationFlags.createFromDeviceOp) {
            maybeCreateFromDeviceOp(true);
            maybeCreateTypecastOp(output.dataType);
            maybeCreateToDeviceOp(true);
            maybeCreateToLayoutOp(output.layoutEnum);
            maybeCreateToMemoryConfigOp(output.tensorMemoryLayout,
                                        output.bufferType);
            op.getResult().replaceAllUsesWith(currentInput);
            return success();
          }
          if (output.dataType != DataType::BFloat16 or
              creationFlags.createFromDeviceOp) {
            maybeCreateFromDeviceOp(true);
            maybeCreateTypecastOp(output.dataType);
            maybeCreateToLayoutOp(output.layoutEnum);
            if (not creationFlags.createFromDeviceOp) {
              maybeCreateToDeviceOp(true);
              maybeCreateToMemoryConfigOp(output.tensorMemoryLayout,
                                          output.bufferType);
            }
            op.getResult().replaceAllUsesWith(currentInput);
            return success();
          }
        }
      }
    }

    llvm_unreachable(
        "Invalid combination of ops created, reaching unreachable code path");
  }

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
        adaptor.getInput(), adaptor.getOutput(), adaptor.getKeepDim(),
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
        adaptor.getInput(), adaptor.getWeight(), adaptor.getOutput());

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
        adaptor.getInput(), adaptor.getOutput(), adaptor.getDimension());
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
        adaptor.getInput(), adaptor.getOutput(), adaptor.getDim0(),
        adaptor.getDim1());
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
        adaptor.getInput(), adaptor.getOutput(), adaptor.getShape());
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
        adaptor.getInput(), adaptor.getOutput(), shapeAttr);

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
        adaptor.getInput(), adaptor.getOutput(), shapeAttr);

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

    DataType outputDataType = getDataTypeFromMemRef(outputMemref);

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

    mlir::Value input = srcOp.getOperand(0);
    mlir::Value result = srcOp.getResult();

    if (srcOp->getUsers().empty()) {
      return rewriter.notifyMatchFailure(
          srcOp, "ttir.broadcast op should have at least one use.");
    }

    rewriter.replaceAllUsesWith(result, input);
    rewriter.eraseOp(srcOp);

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
           ElementwiseOpConversionPattern<ttir::SubtractOp, ttnn::SubtractOp>,
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
           SqueezeOpConversionPattern,
           UnsqueezeOpConversionPattern,
           ConstantOpConversionPattern,
           MatmulOpConversionPattern,
           Conv2dOpConversionPattern,
           MaxPool2dOpConversionPattern
           >(typeConverter, ctx);
  // ANCHOR_END: op_rewriter_pattern_set
  // clang-format on
}

} // namespace mlir::tt
