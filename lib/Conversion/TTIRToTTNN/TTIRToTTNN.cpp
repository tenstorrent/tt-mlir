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

struct CompoundLayoutOpCreationFlags {
  bool createToDeviceOp = false;
  bool createFromDeviceOp = false;
  bool createToLayoutOp = false;
  bool createTypecastOp = false;
  bool createToMemoryConfigOp = false;

  bool createSomeOp() {
    return createToLayoutOp or createTypecastOp or createToDeviceOp or
           createFromDeviceOp or createToMemoryConfigOp;
  }

  bool validate() {
    bool valid = true;
    // shouldn't create a to device op and a from device op
    valid &= not(createToDeviceOp and createFromDeviceOp);
    // shouldn't set memory config for a host tensor
    valid &= not(createFromDeviceOp and createToMemoryConfigOp);
    return valid;
  }
};

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
// are required to achieve an arbitrary layout. There are two main distinct
// paths in this conversion pattern:
//
// 1. If the layout calls for device memory, we will call TTNN::ToLayoutOp and
//    TTNN::ToDeviceOp to achieve the desired layout.
//
// 2. If the layout calls for system memory, we will call TTNN::ToLayoutOp to
//    change the tensor to RowMajor layout, and then the TTNN::FromDeviceOp to
//    move to host memory
//
class ToLayoutOpConversionPattern
    : public OpConversionPattern<ttir::ToLayoutOp> {
public:
  using OpConversionPattern<ttir::ToLayoutOp>::OpConversionPattern;

  ttnn::Layout getLayoutFromMemRef(mlir::MemRefType memref) {
    ttnn::Layout ttnnLayoutEnum = ttnn::Layout::RowMajor;
    Type elementType = memref.getElementType();
    if (llvm::isa<TileType>(elementType)) {
      ttnnLayoutEnum = ttnn::Layout::Tile;
    } else {
      ttnnLayoutEnum = ttnn::Layout::RowMajor;
    }
    return ttnnLayoutEnum;
  }

  DataType getDataTypeFromMemRef(mlir::MemRefType memref) {
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

  LogicalResult createLayoutConversionOps(ttir::ToLayoutOp op,
                                          ConversionPatternRewriter &rewriter) {

    CompoundLayoutOpCreationFlags creationFlags;
    tt::LayoutAttr inputLayoutAttr =
        mlir::cast<tt::LayoutAttr>(op.getInput().getType().getEncoding());

    tt::LayoutAttr outputLayoutAttr =
        mlir::cast<tt::LayoutAttr>(op.getResult().getType().getEncoding());

    mlir::MemRefType inputMemref = inputLayoutAttr.getMemref();
    mlir::MemRefType outputMemref = outputLayoutAttr.getMemref();

    // Check if we should create ToLayoutOp
    ttnn::Layout inputLayoutEnum = getLayoutFromMemRef(inputMemref);
    ttnn::Layout outputLayoutEnum = getLayoutFromMemRef(outputMemref);
    // TODO(bug #665):
    // Binary ops fail with row major layout in ttnn, defaulting to tile
    // layout for all ops...
    //
    outputLayoutEnum = ttnn::Layout::Tile;
    creationFlags.createToLayoutOp = inputLayoutEnum != outputLayoutEnum;

    // Check if we should create TypecastOp
    DataType inputDataType = getDataTypeFromMemRef(inputMemref);
    DataType outputDataType = getDataTypeFromMemRef(outputMemref);
    creationFlags.createTypecastOp = inputDataType != outputDataType;

    // check if we should create ToDeviceOp
    ttnn::BufferType inputBufferType =
        ttnn::utils::toTTNNBufferType(inputLayoutAttr.getMemorySpace());
    ttnn::BufferType outputBufferType =
        ttnn::utils::toTTNNBufferType(outputLayoutAttr.getMemorySpace());
    creationFlags.createToDeviceOp =
        (inputBufferType != outputBufferType) and
        (inputBufferType == ttnn::BufferType::SystemMemory);

    // check if we should create FromDeviceOp
    creationFlags.createFromDeviceOp =
        (inputBufferType != outputBufferType) and
        (outputBufferType == ttnn::BufferType::SystemMemory);

    // check if we should create ToMemoryConfigOp
    // TODO (jnie): we should also check core range set and shard shape for
    // potential resharding with the same sharding method
    ttnn::TensorMemoryLayout inputTensorMemoryLayout =
        ttnn::utils::toTTNNTensorMemoryLayout(inputLayoutAttr.getMemLayout());
    ttnn::TensorMemoryLayout outputTensorMemoryLayout =
        ttnn::utils::toTTNNTensorMemoryLayout(outputLayoutAttr.getMemLayout());
    creationFlags.createToMemoryConfigOp |=
        (inputTensorMemoryLayout != outputTensorMemoryLayout);
    creationFlags.createToMemoryConfigOp |=
        (inputBufferType == ttnn::BufferType::DRAM and
         outputBufferType == ttnn::BufferType::L1) or
        (inputBufferType == ttnn::BufferType::L1 and
         outputBufferType == ttnn::BufferType::DRAM);

    if (not creationFlags.validate()) {
      return failure() << "Invalid combination of ops to create";
    }

    if (not creationFlags.createSomeOp()) {
      return failure() << "Potential redundant ttir::ToLayoutOp - no ttnn "
                          "layout ops needed";
    }

    if (creationFlags.createToMemoryConfigOp and
        outputBufferType == ttnn::BufferType::SystemMemory) {
      return failure()
             << "ToMemoryConfigOp only supported for device output tensors";
    }

    auto device = getOrInsertDevice(rewriter, op);

    // These values will get updated by the lambdas
    Value currentInput = op.getInput();
    bool currentIsOnDevice =
        (inputBufferType != ttnn::BufferType::SystemMemory);
    bool currentIsTilized = (inputLayoutEnum == ttnn::Layout::Tile);
    bool currentDataType = inputDataType;

    // Lambdas for creating layout conversion ops
    auto maybeCreateToDeviceOp = [&currentInput, &currentIsOnDevice,
                                  creationFlags,
                                  device](bool forceCreate = false) {
      if (not creationFlags.createToDeviceOp and not forceCreate) {
        return;
      }
      currentInput = rewriter.create<ttnn::ToLayoutOp>(
          op.getLoc(), this->getTypeConverter()->convertType(op.getType()),
          currentInput, device);
      currentIsOnDevice = true;
    };

    auto maybeCreateToLayoutOp : [&currentInput, &currentIsTilized,
                                  creationFlags](ttnn::Layout outputLayoutEnum,
                                                 bool forceCreate = false) {
      if (not creationFlags.createToLayoutOp and not forceCreate) {
        return;
      }
      currentInput = rewriter.create<ttnn::ToLayoutOp>(
          op.getLoc(), this->getTypeConverter()->convertType(op.getType()),
          currentInput,
          ttnn::LayoutAttr::get(op.getContext(), outputLayoutEnum));
      currentIsTilized = outputLayoutEnum == ttnn::Layout::Tile;
    };

    auto maybeCreateTypecastOp = [&currentInput, &currentDataType,
                                  creationFlags](DataType outputDataType,
                                                 bool forceCreate = false) {
      if (not creationFlags.createTypecastOp and not forceCreate) {
        return;
      }
      currentInput = rewriter.create<ttnn::TypecastOp>(
          op.getLoc(), this->getTypeConverter()->convertType(op.getType()),
          currentInput, DataTypeAttr::get(op.getContext(), outputDataType));
    };

    auto maybeCreateFromDeviceOp = [&currentInput, &currentIsOnDevice,
                                    creationFlags](bool forceCreate = false) {
      if (not creationFlags.createFromDeviceOp and not forceCreate) {
        return;
      }
      currentInput = rewriter.create<ttnn::FromDeviceOp>(
          op.getLoc(), this->getTypeConverter()->convertType(op.getType()),
          currentInput);
      currentIsOnDevice = false;
    };

    auto maybeCreateToMemoryConfigOp = [&currentInput, creationFlags,
                                        outputTensorMemoryLayout,
                                        outputBufferType]() {
      if (not creationFlags.createToMemoryConfigOp) {
        return;
      }
      ttnn::MemoryConfigAttr memoryConfigAttr = ttnn::MemoryConfigAttr::get(
          op.getContext(),
          ttnn::TensorMemoryLayoutAttr::get(op.getContext(),
                                            outputTensorMemoryLayout),
          ttnn::BufferTypeAttr::get(op.getContext(), outputBufferType));
      currentInput = rewriter.create<ttnn::ToMemoryConfigOp>(
          op.getLoc(), this->getTypeConverter()->convertType(op.getType()),
          currentInput, memoryConfigAttr);
    };

    /*
     * Logic for creating ops. Currently there are a few conditions/constraints:
     * - When possible, we want to execute operations on device.
     * - Tilize on device requires dataformat of bfloat16.
     * - Typecast on device requires TILIZED tensor.
     */
    bool shouldTilize = (creationFlags.createToLayoutOp and
                         outputLayoutAttr == ttnn::Layout::Tile);

    // Handle host input tensor
    if (not currentIsOnDevice) {
      if (creationFlags.createFromDeviceOp) {
        return failure() << "Unexpected from device op on host tensor";
      }
      // Case 1.1
      // If we don't need to create a ToLayoutOp nor TypecastOp
      // Create to device op and to memory config op if needed and return
      if (not creationFlags.createToLayoutOp and
          not creationFlags.createTypecastOp) {
        maybeCreateToDeviceOp();
        maybeCreateToMemoryConfigOp();
        return success();
      }

      // Case 1.2
      // If we need to create a ToLayoutOp not a TypecastOp, check the data
      // format on tilization
      else if (creationFlags.createToLayoutOp and
               not creationFlags.createTypecastOp) {
        if (not shouldTilize) {
          maybeCreateToDeviceOp();
          maybeCreateToLayoutOp(outputLayoutEnum);
          maybeCreateToMemoryConfigOp();
          return success();
        }
        // We can tilize on device if data type is bfloat16
        else if (shouldTilize and currentDataType == DataType::BFloat16) {
          maybeCreateToDeviceOp();
          maybeCreateToLayoutOp(outputLayoutEnum);
          maybeCreateToMemoryConfigOp();
          return success();
        }
        // Currently, tilizing on host
        // TODO (jnie): Investigate if it's better to
        // typecast on host -> move to device -> tilize on device -> typecast
        // back on device
        else if (shouldTilize and currentDataType != DataType::BFloat16) {
          maybeCreateToLayoutOp(outputLayoutEnum);
          maybeCreateToDeviceOp();
          maybeCreateToMemoryConfigOp();
          return success();
        }
      }

      // Case 1.3
      // If we need need to create a TypecastOp but not a ToLayoutOp
      else if (not creationFlags.createToLayoutOp and
               creationFlags.createTypecastOp) {
        if (currentIsTilized) {
          maybeCreateToDeviceOp();
          maybeCreateTypecastOp(outputDataType);
          maybeCreateToMemoryConfigOp();
          return success();
        }
        // Currently typecasting on host
        // TODO (jnie): Investigate if it's better to
        // tilize on device -> typecast -> untilize on device
        else if (not currentIsTilized) {
          maybeCreateTypecastOp(outputDataType);
          maybeCreateToDeviceOp();
          maybeCreateToMemoryConfigOp();
          return success();
        }
      }

      // Case 1.4
      // If we need to create both TypecastOp and ToLayoutOp
      else if (creationFlags.createToLayoutOp and
               creationFlags.createTypecastOp) {
        // If we're untilizing
        // try move to device -> typecast -> untilize -> to memory config
        if (not shouldTilize) {
          maybeCreateToDeviceOp();
          maybeCreateTypecastOp(outputDataType);
          maybeCreateToLayoutOp(outputLayoutEnum);
          maybeCreateToMemoryConfigOp();
          return success();
        }
        // If we're tilizing and the input datatype is bfloat16
        // try move to device -> tilize -> typecast -> to memory config
        else if (shouldTilize and currentDataType == DataType::BFloat16) {
          maybeCreateToDeviceOp();
          maybeCreateToLayoutOp(outputLayoutEnum);
          maybeCreateTypecastOp(outputDataType);
          maybeCreateToMemoryConfigOp();
          return success();
        }
        // If we're tilizing and the input data type is not bfloat16
        else if (shouldTilize and currentDataType != DataType::BFloat16) {
          // If we want to typecast to bfloat16:
          // typecast on host -> try move to device -> tilize on device
          // potentially
          if (outputDataType == DataType::BFloat16) {
            maybeCreateTypecastOp(outputDataType);
            maybeCreateToDeviceOp();
            maybeCreateToLayoutOp(outputLayoutEnum);
            maybeCreateToMemoryConfigOp();
            return success();
          }
          // tilize and typcast on host
          else if (not creationFlags.createToDeviceOp) {
            maybeCreateToLayoutOp(outputLayoutEnum);
            maybeCreateTypecastOp(outputDataType);
            maybeCreateToMemoryConfigOp();
            return success();
          }
          // move to device -> typecast bfloat16 -> tilize -> typecast to output
          // df
          else if (creationFlags.createToDeviceOp) {
            maybeCreateToDeviceOp();
            maybeCreateTypecastOp(DataType::BFloat16, true);
            maybeCreateToLayoutOp(outputLayoutEnum);
            maybeCreateTypecastOp(outputDataType);
            maybeCreateToMemoryConfigOp();
            return success();
          }
        }
      }
    }

    else if (currentIsOnDevice) {
      if (creationFlags.createToDeviceOp) {
        return failure() << "Unexpected to device op on device tensor";
      }

      // Case 2.1
      // If we don't need to create a ToLayoutOp nor TypecastOp
      // Create to device op and to memory config op if needed and return
      if (not creationFlags.createToLayoutOp and
          not creationFlags.createTypecastOp) {
        maybeCreateToMemoryConfigOp();
        maybeCreateFromDeviceOp();
        return success();
      }
      // Case 2.2
      // If we need to create a ToLayoutOp not a TypecastOp, check the data
      // format on tilization
      else if (creationFlags.createToLayoutOp and
               not creationFlags.createTypecastOp) {
        if (not shouldTilize) {
          maybeCreateToLayoutOp(outputLayoutEnum);
          maybeCreateToMemoryConfigOp();
          maybeCreateFromDeviceOp();
          return success();
        }
        // We can tilize on device if data type is bfloat16
        else if (shouldTilize and currentDataType == DataType::BFloat16) {
          maybeCreateToLayoutOp(outputLayoutEnum);
          maybeCreateToMemoryConfigOp();
          maybeCreateFromDeviceOp();
          return success();
        }
        // typecast bfloat16 -> tilize -> typecast output data type
        else if (shouldTilize and currentDataType != DataType::BFloat16) {
          maybeCreateTypecastOp(DataType::BFloat16, true);
          maybeCreateToLayoutOp(outputLayoutEnum);
          maybeCreateTypecastOp(outputDataType, true);
          maybeCreateToMemoryConfigOp();
          maybeCreateFromDeviceOp();
          return success();
        }
      }

      // Case 2.3
      // If we need to create a TypeCastOp but not a toLayoutOp
      else if (not creationFlags.createToLayoutOp and
               creationFlags.createTypecastOp) {
        if (currentIsTilized) {
          maybeCreateTypecastOp(outputDataType);
          maybeCreateToMemoryConfigOp();
          maybeCreateFromDeviceOp();
          return success();
        }
        // if ROW_MAJOR and data format is bfloat16
        // tilize -> typecast on device -> untilize
        else if (not currentIsTilized and
                 currentDataType == DataType::BFloat16) {
          maybeCreateToLayoutOp(ttnn::Layout::Tile, true);
          maybeCreateTypecastOp(outputDataType);
          maybeCreateToLayoutOp(outputLayoutEnum, true);
          maybeCreateToMemoryConfigOp();
          maybeCreateFromDeviceOp()
        }
        // if ROW_MAJOR and data format is not bfloat16
        // typecast on host, because typecast requires TILE layout, tilize on
        // device requires bfloat16
        else if (not currentIsTilized and
                 currentDataType != DataType::BFloat16) {
          maybeCreateFromDeviceOp(true);
          maybeCreateTypecastOp(outputDataType);
          // move back to device if necessary
          if (not creationFlags.createFromDeviceOp) {
            maybeCreateToDeviceOp(true);
            maybeCreateToMemoryConfigOp();
          }
          return success();
        }
      }

      // Case 2.4
      // If we need to create both toLayoutOp and TypeCastOp
      else if (creationFlags.createToLayoutOp and
               creationFlags.createTypecastOp) {
        if (not shouldTilize) {
          maybeCreateTypecastOp(outputDataType);
          maybeCreateToLayoutOp(outputLayoutEnum);
          maybeCreateToMemoryConfigOp();
          maybeCreateFromDeviceOp();
          return success();
        } else if (shouldTilize and currentDataType == DataType::BFloat16) {
          maybeCreateToLayoutOp(outputLayoutEnum);
          maybeCreateTypecastOp(outputDataType);
          maybeCreateToMemoryConfigOp();
          maybeCreateFromDeviceOp();
          return success();
        } else if (shouldTilize and currentDataType != DataType::BFloat16) {
          // move to host -> typecast on host -> move to device -> tilize
          if (outputDataType == DataType::BFloat16 and
              not creationFlags.createFromDeviceOp) {
            maybeCreateFromDeviceOp(true);
            maybeCreateTypecastOp(outputDataType);
            maybeCreateToDeviceOp(true);
            maybeCreateToLayoutOp(outputLayoutEnum);
            maybeCreateToMemoryConfigOp();
            return success();
          }
          maybeCreateFromDeviceOp(true);
          maybeCreateTypecastOp(outputDataType);
          maybeCreateToLayoutOp(outputLayoutEnum);
          if (not creationFlags.createFromDeviceOp) {
            maybeCreateToDeviceOp(true);
            maybeCreateToMemoryConfigOp();
          }
          return success();
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

    return createLayoutConversionOps(op, rewriter);
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
           ElementwiseOpConversionPattern<ttir::TypecastOp, ttnn::TypecastOp>,
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
