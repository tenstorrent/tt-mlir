// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTNN/TTIRToTTNN.h"
#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#include <cstdint>
#include <optional>

using namespace mlir;
using namespace mlir::tt;

namespace {
class TensorEmptyConversionPattern
    : public OpConversionPattern<tensor::EmptyOp> {
public:
  using OpConversionPattern<tensor::EmptyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tensor::EmptyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Get ttnn::TTNNLayoutAttr of the result type
    //
    ttnn::TTNNLayoutAttr layoutAttr = mlir::cast<ttnn::TTNNLayoutAttr>(
        op.getResult().getType().getEncoding());

    // Get the shape of the tensor, tensor layout, and data type
    //
    ttnn::ShapeAttr shapeAttr = ttnn::ShapeAttr::get(
        rewriter.getContext(),
        mlir::cast<RankedTensorType>(op->getResult(0).getType()).getShape());
    DataType dtype = layoutAttr.getDataType();
    ttnn::Layout ttnnLayoutEnum = ttnn::Layout::RowMajor;
    if (layoutAttr.isTiled()) {
      ttnnLayoutEnum = ttnn::Layout::Tile;
    } else {
      ttnnLayoutEnum = ttnn::Layout::RowMajor;
    }
    DataTypeAttr dTypeAttr = DataTypeAttr::get(rewriter.getContext(), dtype);
    ttnn::LayoutAttr tensorLayoutAttr =
        ttnn::LayoutAttr::get(op.getContext(), ttnnLayoutEnum);

    // Device
    //
    auto device = ::ttnn::utils::getOrInsertDevice(rewriter, op);

    // Create MemoryConfigAttr
    //
    ttnn::BufferTypeAttr bufferTypeAttr =
        ttnn::BufferTypeAttr::get(op.getContext(), layoutAttr.getBufferType());
    ttnn::ShardSpecAttr shardSpecAttr = ttnn::ShardSpecAttr::get(
        op.getContext(),
        ttnn::ShapeAttr::get(op.getContext(), layoutAttr.getShardShape()));
    ttnn::MemoryConfigAttr memoryConfigAttr =
        ttnn::MemoryConfigAttr::get(op.getContext(), bufferTypeAttr,
                                    shardSpecAttr, layoutAttr.getMemLayout());

    // Replace op
    //
    rewriter.replaceOpWithNewOp<ttnn::EmptyOp>(
        op, this->getTypeConverter()->convertType(op.getType()), shapeAttr,
        dTypeAttr, tensorLayoutAttr, device, memoryConfigAttr);

    return success();
  }
};
} // namespace

namespace {
class ZerosOpConversionPattern : public OpConversionPattern<ttir::ZerosOp> {
public:
  using OpConversionPattern<ttir::ZerosOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ZerosOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get ttnn::TTNNLayoutAttr of the result type
    //
    ttnn::TTNNLayoutAttr layoutAttr = mlir::cast<ttnn::TTNNLayoutAttr>(
        op.getResult().getType().getEncoding());

    // Get the shape of tensor
    //
    // TODO(svuckovic): (#1435) ShapeAttr accepts int64_t, when it should be
    // uint32_t
    //
    ttnn::ShapeAttr shapeAttr = ttnn::ShapeAttr::get(
        rewriter.getContext(), llvm::SmallVector<int64_t, 4>(
                                   op.getShape().begin(), op.getShape().end()));

    // Get memref
    //
    mlir::MemRefType memref = layoutAttr.getMemref();

    // Get data type, tensor layout, device and memory config
    //
    DataTypeAttr dTypeAttr =
        DataTypeAttr::get(rewriter.getContext(), layoutAttr.getDataType());
    ttnn::BufferType bufferType = layoutAttr.getBufferType();
    ttnn::Layout ttnnLayoutEnum = llvm::isa<TileType>(memref.getElementType())
                                      ? ttnn::Layout::Tile
                                      : ttnn::Layout::RowMajor;
    ttnn::LayoutAttr tensorLayoutAttr =
        ttnn::LayoutAttr::get(op.getContext(), ttnnLayoutEnum);
    ttnn::TensorMemoryLayoutAttr memLayout = layoutAttr.getMemLayout();

    // Device only exists if memLayout is *not* null
    //
    auto device =
        memLayout ? mlir::Value(::ttnn::utils::getOrInsertDevice(rewriter, op))
                  : nullptr;

    // MemoryConfigAttr only exists if memLayout is *not* null
    //
    ttnn::MemoryConfigAttr memoryConfigAttr =
        memLayout
            ? ttnn::MemoryConfigAttr::get(
                  op.getContext(),
                  ttnn::BufferTypeAttr::get(op.getContext(), bufferType),
                  ttnn::ShardSpecAttr::get(
                      op.getContext(),
                      ttnn::ShapeAttr::get(op.getContext(), memref.getShape())),
                  memLayout)
            : nullptr;

    rewriter.replaceOpWithNewOp<ttnn::ZerosOp>(
        op, this->getTypeConverter()->convertType(op.getType()), shapeAttr,
        dTypeAttr, tensorLayoutAttr, device, memoryConfigAttr);

    return success();
  }
};
} // namespace

namespace {
class OnesOpConversionPattern : public OpConversionPattern<ttir::OnesOp> {
public:
  using OpConversionPattern<ttir::OnesOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::OnesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get ttnn::TTNNLayoutAttr of the result type
    //
    ttnn::TTNNLayoutAttr layoutAttr = mlir::cast<ttnn::TTNNLayoutAttr>(
        op.getResult().getType().getEncoding());

    // Get the shape of tensor
    //
    // TODO(svuckovic): (#1435) ShapeAttr accepts int64_t, when it should be
    // uint32_t
    //
    ttnn::ShapeAttr shapeAttr = ttnn::ShapeAttr::get(
        rewriter.getContext(), llvm::SmallVector<int64_t, 4>(
                                   op.getShape().begin(), op.getShape().end()));

    // Get memref
    //
    mlir::MemRefType memref = layoutAttr.getMemref();

    // Get data type, tensor layout, device and memory config
    //
    DataTypeAttr dTypeAttr =
        DataTypeAttr::get(rewriter.getContext(), layoutAttr.getDataType());
    ttnn::BufferType bufferType = layoutAttr.getBufferType();
    ttnn::Layout ttnnLayoutEnum = llvm::isa<TileType>(memref.getElementType())
                                      ? ttnn::Layout::Tile
                                      : ttnn::Layout::RowMajor;
    ttnn::LayoutAttr tensorLayoutAttr =
        ttnn::LayoutAttr::get(op.getContext(), ttnnLayoutEnum);
    ttnn::TensorMemoryLayoutAttr memLayout = layoutAttr.getMemLayout();

    // Device only exists if memLayout is *not* null
    //
    auto device =
        memLayout ? mlir::Value(::ttnn::utils::getOrInsertDevice(rewriter, op))
                  : nullptr;

    // MemoryConfigAttr only exists if memLayout is *not* null
    //
    ttnn::MemoryConfigAttr memoryConfigAttr =
        memLayout
            ? ttnn::MemoryConfigAttr::get(
                  op.getContext(),
                  ttnn::BufferTypeAttr::get(op.getContext(), bufferType),
                  ttnn::ShardSpecAttr::get(
                      op.getContext(),
                      ttnn::ShapeAttr::get(op.getContext(), memref.getShape())),
                  memLayout)
            : nullptr;

    rewriter.replaceOpWithNewOp<ttnn::OnesOp>(
        op, this->getTypeConverter()->convertType(op.getType()), shapeAttr,
        dTypeAttr, tensorLayoutAttr, device, memoryConfigAttr);

    return success();
  }
};
} // namespace

namespace {
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

    assert(mlir::isa<mlir::RankedTensorType>(adaptor.getInput().getType()) &&
           "Expected RankedTensorType for ToLayoutOp input");

    auto outputLayoutAttr = mlir::cast<ttnn::TTNNLayoutAttr>(
        op.getResult().getType().getEncoding());

    // Determine the output data type
    DataType dtype = outputLayoutAttr.getDataType();
    DataTypeAttr outputDataType =
        DataTypeAttr::get(rewriter.getContext(), dtype);

    // Determine the output layout (tile or row major)
    ttnn::BufferType outputBufferType = outputLayoutAttr.getBufferType();

    ttnn::Layout outputLayoutEnum = outputLayoutAttr.getLayout();

    bool isOutputOnHost = (outputBufferType == ttnn::BufferType::SystemMemory);

    RankedTensorType result = mlir::cast<RankedTensorType>(op.getType());

    ttnn::LayoutAttr outputLayout =
        ttnn::LayoutAttr::get(rewriter.getContext(), outputLayoutEnum);
    llvm::SmallVector<int64_t> outputShardShape =
        outputLayoutAttr.getShardShape();

    ttnn::MemoryConfigAttr outputMemConfigAttr = ttnn::MemoryConfigAttr::get(
        rewriter.getContext(),
        ttnn::BufferTypeAttr::get(rewriter.getContext(), outputBufferType),
        ttnn::ShardSpecAttr::get(
            op.getContext(),
            ttnn::ShapeAttr::get(rewriter.getContext(), outputShardShape)),
        outputLayoutAttr.getMemLayout());

    rewriter.replaceOpWithNewOp<ttnn::ToLayoutOp>(
        op, this->getTypeConverter()->convertType(result), adaptor.getInput(),
        outputLayout, outputDataType, outputMemConfigAttr,
        isOutputOnHost
            ? nullptr
            : mlir::Value(::ttnn::utils::getOrInsertDevice(rewriter, op)));

    return success();
  }
};
} // namespace

namespace {
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
} // namespace

namespace {
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
} // namespace

namespace {
class ReductionProdOpConversionPattern
    : public OpConversionPattern<ttir::ProdOp> {
public:
  using OpConversionPattern<ttir::ProdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ProdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    int64_t inputRank = op.getInput().getType().getRank();
    auto dimArg = op.getDimArg();
    int64_t size = dimArg ? dimArg->size() : inputRank;

    // [TODO](mmanzoor) Decompose ttnn.prod op into multiple ttnn.prod to handle
    // reduction along multiple dimensions.
    // https://github.com/tenstorrent/tt-mlir/issues/1861
    if ((size > 1) && (size < inputRank)) {
      return rewriter.notifyMatchFailure(
          op, "tt-metal only supports reduce(prod) along one dimension or all "
              "dimensions.");
    }

    bool allDimensions = (size == inputRank) ? true : false;
    int64_t dimension =
        dimArg ? (mlir::cast<mlir::IntegerAttr>(dimArg->getValue()[0])).getInt()
               : 0;

    rewriter.replaceOpWithNewOp<ttnn::ProdOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), allDimensions, adaptor.getKeepDim(), dimension,
        /*memoryConfig*/ nullptr);
    return success();
  }
};
} // namespace

namespace {
class ReductionArgMaxOpConversionPattern
    : public OpConversionPattern<ttir::ArgMaxOp> {
public:
  using OpConversionPattern<ttir::ArgMaxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ArgMaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dimArg = op.getDimArg();

    rewriter.replaceOpWithNewOp<ttnn::ArgMaxOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(),
        (dimArg && dimArg->size())
            ? mlir::cast<mlir::IntegerAttr>(dimArg->getValue().front())
            : nullptr,
        /*use_multicore*/ false, // Default tt-metal value.
        /*memoryConfig*/ nullptr, adaptor.getOutput());
    return success();
  }
};
} // namespace

namespace {
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
} // namespace

namespace {
class EmbeddingBackwardOpConversionPattern
    : public OpConversionPattern<ttir::EmbeddingBackwardOp> {
public:
  using OpConversionPattern<ttir::EmbeddingBackwardOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::EmbeddingBackwardOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto gradType = adaptor.getInGradient().getType();
    auto gradTensor = mlir::cast<RankedTensorType>(gradType);
    auto gradShape = gradTensor.getShape();

    // Reshape grad tensor to [1, 1, R, C] where R is all the first N-1
    // dimensions of grad tensor squeezed and C is the last dimension of grad
    // tensor. This must be done to obey the constraints of the
    // ttnn::EmbeddingBackwardOp.
    int32_t R = 1;
    for (size_t i = 0; i < gradShape.size() - 1; ++i) {
      R *= gradShape[i];
    }
    llvm::SmallVector<int64_t, 4> reshapedGradShape{1, 1, R, gradShape.back()};

    auto reshapedGrad = mlir::tt::ttir_to_ttnn::utils::generateReshape(
        mlir::cast<TypedValue<mlir::RankedTensorType>>(adaptor.getInGradient()),
        reshapedGradShape, rewriter);

    // Get TTNNLayoutAttr of the result type.
    ttnn::TTNNLayoutAttr layoutAttr = mlir::cast<ttnn::TTNNLayoutAttr>(
        op.getResult().getType().getEncoding());
    mlir::MemRefType memref = layoutAttr.getMemref();

    // Get data type, tensor layout, buffer type and memory config.
    DataTypeAttr dTypeAttr =
        DataTypeAttr::get(rewriter.getContext(), layoutAttr.getDataType());
    ttnn::TensorMemoryLayoutAttr memLayout = layoutAttr.getMemLayout();
    ttnn::BufferType bufferType = layoutAttr.getBufferType();

    ttnn::MemoryConfigAttr memoryConfigAttr = ttnn::MemoryConfigAttr::get(
        op.getContext(), ttnn::BufferTypeAttr::get(op.getContext(), bufferType),
        ttnn::ShardSpecAttr::get(
            op.getContext(),
            ttnn::ShapeAttr::get(rewriter.getContext(), memref.getShape())),
        memLayout);

    rewriter.replaceOpWithNewOp<ttnn::EmbeddingBackwardOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), adaptor.getWeight(), reshapedGrad, dTypeAttr,
        memoryConfigAttr, adaptor.getOutput());
    return success();
  }
};
} // namespace

namespace {
class CumSumOpConversionPattern : public OpConversionPattern<ttir::CumSumOp> {
public:
  using OpConversionPattern<ttir::CumSumOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::CumSumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ttnn::MorehCumSumOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), adaptor.getDim(), adaptor.getOutput(), nullptr);
    return success();
  }
};
} // namespace

namespace {
class RepeatInterleaveOpConversionPattern
    : public OpConversionPattern<ttir::RepeatInterleaveOp> {
public:
  using OpConversionPattern<ttir::RepeatInterleaveOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::RepeatInterleaveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ttnn::RepeatInterleaveOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), adaptor.getRepeats(), adaptor.getDim(),
        ttnn::MemoryConfigAttr());
    return success();
  }
};
} // namespace

namespace {
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
} // namespace

namespace {
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
} // namespace

namespace {
class ClampOpConversionPattern : public OpConversionPattern<ttir::ClampOp> {
public:
  using OpConversionPattern<ttir::ClampOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ClampOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ttnn::ClampOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), adaptor.getMin(), adaptor.getMax());
    return success();
  }
};
} // namespace

namespace {
class UpdateCacheOpConversionPattern
    : public OpConversionPattern<ttir::UpdateCacheOp> {
public:
  using OpConversionPattern<ttir::UpdateCacheOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::UpdateCacheOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // The TTIR version of this op is pure. In TTNN this op is in-place.
    // We need to replace uses of the result ot the TTIR op with uses
    // of the cache argument.
    //
    // The presence of the MemWrite trait of this op should preserve
    // the order of this op relative to the cache arguments uses, preserving
    // program correctness.

    // This op can only work if it is the final use of the cache tensor in the
    // order of execution. For now, checking that there is only one user (this
    // op) of the cache tensor will suffice.
    std::vector<mlir::Operation *> users(op.getCache().getUsers().begin(),
                                         op.getCache().getUsers().end());
    if (users.size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "UpdateCacheOp must have exactly one user");
    }

    rewriter.create<ttnn::UpdateCacheOp>(
        op.getLoc(), adaptor.getCache(), adaptor.getInput(),
        adaptor.getUpdateIndex(), adaptor.getBatchOffset());

    rewriter.replaceOp(op, adaptor.getCache());
    return success();
  }
};
} // namespace

namespace {
class FillCacheOpConversionPattern
    : public OpConversionPattern<ttir::FillCacheOp> {
public:
  using OpConversionPattern<ttir::FillCacheOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::FillCacheOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // The TTIR version of this op is pure. In TTNN this op is in-place.
    // We need to replace uses of the result ot the TTIR op with uses
    // of the cache argument.
    //
    // The presence of the MemWrite trait of this op should preserve
    // the order of this op relative to the cache arguments uses, preserving
    // program correctness.

    // This op can only work if it is the final use of the cache tensor in the
    // order of execution. For now, checking that there is only one user (this
    // op) of the cache tensor will suffice.
    std::vector<mlir::Operation *> users(op.getCache().getUsers().begin(),
                                         op.getCache().getUsers().end());
    if (users.size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "FillCacheOp must have exactly one user");
    }

    rewriter.create<ttnn::FillCacheOp>(op.getLoc(), adaptor.getCache(),
                                       adaptor.getInput(),
                                       adaptor.getBatchOffset());

    rewriter.replaceOp(op, adaptor.getCache());
    return success();
  }
};
} // namespace

namespace {
template <typename TTIROpTy, typename TTNNOpTy,
          typename OpAdaptor = typename TTIROpTy::Adaptor>
class ElementwiseUnaryWithFloatParameterOpConversionPattern
    : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TTNNOpTy>(
        op, this->getTypeConverter()->convertType(op.getType(0)),
        adaptor.getInputs(), adaptor.getOutputs(), adaptor.getParameter());
    return success();
  }
};
} // namespace

namespace {
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
        adaptor.getInputs(), adaptor.getOutput(), dim,
        /* memory_config */ nullptr);
    return success();
  }
};
} // namespace

namespace {
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
} // namespace

namespace {
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
} // namespace

namespace {
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
} // namespace

namespace {
class BroadcastOpConversionPattern
    : public OpConversionPattern<ttir::BroadcastOp> {
  using OpConversionPattern<ttir::BroadcastOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(ttir::BroadcastOp op, ttir::BroadcastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn::ShapeAttr shapeAttr = ttnn::ShapeAttr::get(
        rewriter.getContext(), op.getBroadcastDimensions());

    rewriter.replaceOpWithNewOp<ttnn::RepeatOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), shapeAttr);

    return success();
  }
};

class RepeatOpConversionPattern : public OpConversionPattern<ttir::RepeatOp> {
  using OpConversionPattern<ttir::RepeatOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(ttir::RepeatOp op, ttir::RepeatOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ttnn::ShapeAttr repeatDimensionsAttr =
        ttnn::ShapeAttr::get(rewriter.getContext(), op.getRepeatDimensions());

    rewriter.replaceOpWithNewOp<ttnn::RepeatOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), repeatDimensionsAttr);

    return success();
  }
};
} // namespace
namespace {
class PadOpConversionPattern : public OpConversionPattern<ttir::PadOp> {
  using OpConversionPattern<ttir::PadOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(ttir::PadOp op, ttir::PadOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn::MemoryConfigAttr memcfg = nullptr;
    if (ttnn::TTNNLayoutAttr layoutAttr =
            mlir::dyn_cast_or_null<ttnn::TTNNLayoutAttr>(
                op.getResult().getType().getEncoding());
        layoutAttr.getBufferType() != ttnn::BufferType::SystemMemory) {
      memcfg = ttnn::MemoryConfigAttr::get(
          op.getContext(),
          ttnn::BufferTypeAttr::get(op.getContext(),
                                    layoutAttr.getBufferType()),
          ttnn::ShardSpecAttr::get(
              op.getContext(),
              ttnn::ShapeAttr::get(op.getContext(),
                                   layoutAttr.getShardShape())),
          layoutAttr.getMemLayout());
    }
    rewriter.replaceOpWithNewOp<ttnn::PadOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), adaptor.getPaddingAttr(), adaptor.getValue(),
        /* use_multicore */ true, memcfg);

    return success();
  }
};
} // namespace

namespace {
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

    if (inputType.getRank() == dim) {
      newShape.push_back(1);
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
} // namespace

namespace {
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

    // If the value is a splat (i.e. single value), we can use the ttnn::FullOp
    // to create the tensor.
    if (valueAttr.isSplat()) {
      auto device = ::ttnn::utils::getOrInsertDevice(rewriter, op);

      mlir::APFloat fillValue(mlir::APFloat::IEEEsingle());
      if (valueAttr.getElementType().isInteger()) {
        fillValue.convertFromAPInt(valueAttr.getSplatValue<llvm::APInt>(),
                                   valueAttr.getElementType().isSignedInteger(),
                                   llvm::RoundingMode::TowardZero);
      } else {
        fillValue = valueAttr.getSplatValue<mlir::APFloat>();
      }

      rewriter.replaceOpWithNewOp<ttnn::FullOp>(
          op, this->getTypeConverter()->convertType(op.getType()), device,
          rewriter.getF32FloatAttr(fillValue.convertToFloat()));

      // Otherwise, we use the ttnn::ConstantOp to create the tensor.
    } else {
      rewriter.replaceOpWithNewOp<ttnn::ConstantOp>(
          op, this->getTypeConverter()->convertType(op.getType()),
          adaptor.getValue());
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

namespace {
class LinearOpConversionPattern : public OpConversionPattern<ttir::LinearOp> {
public:
  using OpConversionPattern<ttir::LinearOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::LinearOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ttnn::LinearOp>(
        op, this->getTypeConverter()->convertType(op.getType()), adaptor.getA(),
        adaptor.getB(), adaptor.getBias(), adaptor.getOutput());
    return success();
  }
};
} // namespace

// ANCHOR: adding_an_op_matmul_op_rewriter
namespace {
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
} // namespace
// ANCHOR_END: adding_an_op_matmul_op_rewriter

namespace {
class Conv2dOpConversionPattern : public OpConversionPattern<ttir::Conv2dOp> {
public:
  using OpConversionPattern<ttir::Conv2dOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::Conv2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto device = ::ttnn::utils::getOrInsertDevice(rewriter, op);
    auto kernelType =
        mlir::cast<RankedTensorType>(adaptor.getWeight().getType());
    llvm::ArrayRef<std::int64_t> kernelShape = kernelType.getShape();

    auto inputType = mlir::cast<RankedTensorType>(adaptor.getInput().getType());
    llvm::ArrayRef<std::int64_t> inputShape = inputType.getShape();

    auto outputType =
        mlir::cast<RankedTensorType>(adaptor.getOutput().getType());
    llvm::ArrayRef<std::int64_t> outputShape = outputType.getShape();

    auto inChannels = static_cast<int32_t>(inputShape[inputShape.size() - 1]);
    auto outChannels =
        static_cast<int32_t>(outputShape[outputShape.size() - 1]);
    auto batchSize = static_cast<int32_t>(inputShape[inputShape.size() - 4]);
    auto inputHeight = static_cast<int32_t>(inputShape[inputShape.size() - 3]);
    auto inputWidth = static_cast<int32_t>(inputShape[inputShape.size() - 2]);

    auto kernelHeight =
        static_cast<int32_t>(kernelShape[kernelShape.size() - 2]);
    auto kernelWidth =
        static_cast<int32_t>(kernelShape[kernelShape.size() - 1]);

    auto strideHeight = adaptor.getStrideHeight();
    auto strideWidth = adaptor.getStrideWidth();

    assert(
        adaptor.getPaddingBottom() == adaptor.getPaddingTop() &&
        "TTNN only supports padding height/width attributes. Thus, padding_top "
        "must equal padding_bottom for the op to execute as expected.");
    assert(adaptor.getPaddingLeft() == adaptor.getPaddingRight() &&
           "TTNN only supports padding height/width attributes. Thus, "
           "padding_left must equal padding_right for the op to execute as "
           "expected.");
    auto paddingHeight = adaptor.getPaddingTop();
    auto paddingWidth = adaptor.getPaddingRight();

    auto dilationHeight = adaptor.getDilationHeight();
    auto dilationWidth = adaptor.getDilationWidth();
    auto groups = adaptor.getGroups();

    std::vector<int64_t> flattenedInputShape = {
        1, 1, inputShape[0] * inputShape[1] * inputShape[2], inputShape[3]};
    Value flattenedInput = ttir_to_ttnn::utils::generateNHWFlatten(
        mlir::cast<mlir::TypedValue<RankedTensorType>>(adaptor.getInput()),
        rewriter);

    llvm::SmallVector<int64_t> flattenedOutputShape{
        1, 1, outputShape[0] * outputShape[1] * outputShape[2], outputShape[3]};

    outputType = mlir::RankedTensorType::get(flattenedOutputShape,
                                             outputType.getElementType(),
                                             outputType.getEncoding());

    ttnn::Conv2dOp newConv = ttmlir::utils::createDPSOp<ttnn::Conv2dOp>(
        rewriter, op.getLoc(), outputType, flattenedInput, adaptor.getWeight(),
        adaptor.getBias(), device, inChannels, outChannels, batchSize,
        inputHeight, inputWidth, kernelHeight, kernelWidth, strideHeight,
        strideWidth, paddingHeight, paddingWidth, dilationHeight, dilationWidth,
        groups);

    Value output =
        ttir_to_ttnn::utils::generateReshape(newConv, outputShape, rewriter);

    rewriter.replaceOp(op, output);
    return success();
  }
};
} // namespace

namespace {
class ConvTranspose2dOpConversionPattern
    : public OpConversionPattern<ttir::ConvTranspose2dOp> {
public:
  using OpConversionPattern<ttir::ConvTranspose2dOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ConvTranspose2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto device = ::ttnn::utils::getOrInsertDevice(rewriter, op);

    auto inputTy = mlir::cast<RankedTensorType>(adaptor.getInput().getType());
    auto kernelTy = mlir::cast<RankedTensorType>(adaptor.getWeight().getType());
    auto outputTy = mlir::cast<RankedTensorType>(adaptor.getOutput().getType());

    auto batchSizeAttr = rewriter.getI32IntegerAttr(inputTy.getDimSize(0));
    auto inputHeightAttr = rewriter.getI32IntegerAttr(inputTy.getDimSize(1));
    auto inputWidthAttr = rewriter.getI32IntegerAttr(inputTy.getDimSize(2));
    auto inChannelsAttr = rewriter.getI32IntegerAttr(inputTy.getDimSize(3));
    auto outChannelsAttr = rewriter.getI32IntegerAttr(outputTy.getDimSize(3));

    auto kernelSizeAttr = rewriter.getDenseI32ArrayAttr(
        {static_cast<int32_t>(kernelTy.getDimSize(2)),
         static_cast<int32_t>(kernelTy.getDimSize(3))});

    auto strideAttr = attrToDenseI32ArrayAttr(adaptor.getStride(), rewriter);
    if (auto error = strideAttr.takeError()) {
      return LogicalResult::failure();
    }

    auto paddingAttr =
        attrToDenseI32ArrayAttr(adaptor.getPadding(), rewriter, 4);
    if (auto error = paddingAttr.takeError()) {
      return LogicalResult::failure();
    }

    auto paddingArrayRef = paddingAttr->asArrayRef();
    if (paddingArrayRef[0] != paddingArrayRef[1] ||
        paddingArrayRef[2] != paddingArrayRef[3]) {
      return rewriter.notifyMatchFailure(
          op,
          "TTNN only supports padding height/width attributes. Thus, "
          "padding_top/padding_left must equal padding_bottom/padding_right "
          "for the op to execute as expected.");
    }

    // Padding only supports 2 values in ttnn
    auto reducedPaddingAttr =
        rewriter.getDenseI32ArrayAttr({paddingArrayRef[0], paddingArrayRef[1]});

    auto outputPaddingAttr =
        attrToDenseI32ArrayAttr(adaptor.getOutputPadding(), rewriter);
    if (auto error = outputPaddingAttr.takeError()) {
      return LogicalResult::failure();
    }

    auto dilationAttr =
        attrToDenseI32ArrayAttr(adaptor.getDilation(), rewriter);
    if (auto error = dilationAttr.takeError()) {
      return LogicalResult::failure();
    }

    auto groupsAttr = rewriter.getI32IntegerAttr(adaptor.getGroups());

    // Transposed convolution in ttnn returns a tensor in a flattened shape
    // (1 x 1 x N * H * W x C)
    llvm::ArrayRef<std::int64_t> output_shape = outputTy.getShape();
    llvm::SmallVector<std::int64_t, 4> flattenedOutputShape = {
        1, 1, output_shape[0] * output_shape[1] * output_shape[2],
        output_shape[3]};
    outputTy = mlir::cast<RankedTensorType>(getTypeConverter()->convertType(
        outputTy.cloneWith(flattenedOutputShape, outputTy.getElementType())));

    // Using a tensor::EmptyOp so that the rewriter for EmptyOp can handle the
    // attribute determination
    auto convDPSOutput = rewriter.replaceOpWithNewOp<tensor::EmptyOp>(
        adaptor.getOutput().getDefiningOp(), flattenedOutputShape,
        outputTy.getElementType());

    // Must set the type to the output type to maintain the layout attributes
    convDPSOutput.getResult().setType(outputTy);

    ttnn::ConvTranspose2dOp new_conv = rewriter.create<ttnn::ConvTranspose2dOp>(
        op.getLoc(), outputTy, adaptor.getInput(), adaptor.getWeight(),
        adaptor.getBias(), convDPSOutput, device, inChannelsAttr,
        outChannelsAttr, batchSizeAttr, inputHeightAttr, inputWidthAttr,
        kernelSizeAttr, *strideAttr, reducedPaddingAttr, *outputPaddingAttr,
        *dilationAttr, groupsAttr);

    // Restore the normal shape (N x H x W x C)
    Value output =
        ttir_to_ttnn::utils::generateReshape(new_conv, output_shape, rewriter);

    rewriter.replaceOp(op, output);
    return success();
  }

private:
  llvm::Expected<DenseI32ArrayAttr>
  attrToDenseI32ArrayAttr(mlir::Attribute attr,
                          ConversionPatternRewriter &rewriter,
                          uint32_t elementCount = 2) const {
    switch (elementCount) {
    case 2: {
      // Handles attributes requiring 2 spatial dimensions (e.g., stride,
      // dilation). Converts the attribute into a pair of integers.
      auto pair = ttmlir::utils::getPairOfInteger<int32_t>(attr);
      if (auto error = pair.takeError()) {
        return std::move(error);
      }
      return rewriter.getDenseI32ArrayAttr({pair->first, pair->second});
    }
    case 4: {
      // Handles attributes requiring 4 spatial dimensions (e.g., padding in
      // this case). Converts the attribute into a quadruple of integers.
      auto quadruple = ttmlir::utils::getQuadrupleOfInteger<int32_t>(attr);
      if (auto error = quadruple.takeError()) {
        return std::move(error);
      }
      return rewriter.getDenseI32ArrayAttr(
          {std::get<0>(*quadruple), std::get<1>(*quadruple),
           std::get<2>(*quadruple), std::get<3>(*quadruple)});
    }
    default: {
      return llvm::createStringError(std::errc::invalid_argument,
                                     "Unsupported element count: %d",
                                     elementCount);
    }
    }
  }
};
} // namespace

namespace {
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

    auto device = mlir::tt::ttnn::utils::getOrInsertDevice(rewriter, op);
    auto inputType = mlir::cast<RankedTensorType>(adaptor.getInput().getType());
    llvm::ArrayRef<std::int64_t> inputShape = inputType.getShape();

    auto batchSize = static_cast<int32_t>(inputShape[inputShape.size() - 4]);
    auto channels = static_cast<int32_t>(inputShape[inputShape.size() - 1]);

    Value flattenedInput = ttir_to_ttnn::utils::generateNHWFlatten(
        mlir::cast<mlir::TypedValue<RankedTensorType>>(adaptor.getInput()),
        rewriter);

    auto outputType =
        mlir::cast<RankedTensorType>(adaptor.getOutput().getType());
    llvm::ArrayRef<std::int64_t> outputShape = outputType.getShape();

    llvm::SmallVector<int64_t> flattenedOutputShape{
        1, 1, outputShape[0] * outputShape[1] * outputShape[2], outputShape[3]};

    outputType = mlir::RankedTensorType::get(flattenedOutputShape,
                                             outputType.getElementType(),
                                             outputType.getEncoding());

    auto newPool = ttmlir::utils::createDPSOp<ttnn::MaxPool2dOp>(
        rewriter, op.getLoc(), outputType, flattenedInput, device, batchSize,
        static_cast<int32_t>(inputShape[inputShape.size() - 3]),
        static_cast<int32_t>(inputShape[inputShape.size() - 2]), channels,
        adaptor.getKernelHeight(), adaptor.getKernelWidth(),
        adaptor.getStrideHeight(), adaptor.getStrideWidth(),
        adaptor.getDilationHeight(), adaptor.getDilationWidth(),
        adaptor.getCeilMode(), adaptor.getPaddingTop(),
        adaptor.getPaddingRight());

    Value output =
        ttir_to_ttnn::utils::generateReshape(newPool, outputShape, rewriter);

    rewriter.replaceOp(op, output);

    return success();
  }
};
} // namespace

namespace {
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

    ttnn::TTNNLayoutAttr outputLayoutAttr =
        mlir::cast<ttnn::TTNNLayoutAttr>(result.getType().getEncoding());

    DataType outputDataType = outputLayoutAttr.getDataType();

    rewriter.replaceOpWithNewOp<ttnn::TypecastOp>(
        op, this->getTypeConverter()->convertType(op.getType(0)), input,
        outputDataType);
    return success();
  }
};
} // namespace

namespace {
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
      ttnn::NegOp negOp = ttmlir::utils::createDPSOp<ttnn::NegOp>(
          rewriter, srcOp.getLoc(), rhsType, adaptor.getInputs().back());

      rewriter.replaceOpWithNewOp<ttnn::AddOp>(
          srcOp, adaptor.getInputs().front(), negOp.getResults().front(),
          adaptor.getOutputs().front());
    }

    return success();
  }
};
} // namespace

namespace {
class AllReduceOpConversionPattern
    : public OpConversionPattern<ttir::AllReduceOp> {
public:
  using OpConversionPattern<ttir::AllReduceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::AllReduceOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto device = ::ttnn::utils::getOrInsertDevice(rewriter, srcOp);

    rewriter.replaceOpWithNewOp<ttnn::AllReduceOp>(
        srcOp, this->getTypeConverter()->convertType(srcOp.getType()),
        adaptor.getInput(), device, adaptor.getReduceType(),
        adaptor.getClusterAxis());

    return success();
  }
};
} // namespace

namespace {
class MeshShardOpConversionPattern
    : public OpConversionPattern<ttir::MeshShardOp> {
public:
  using OpConversionPattern<ttir::MeshShardOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::MeshShardOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto device = ::ttnn::utils::getOrInsertDevice(rewriter, op);
    rewriter.replaceOpWithNewOp<ttnn::MeshShardOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), device, adaptor.getShardDirection(),
        adaptor.getShardType(), adaptor.getShardShape(),
        adaptor.getShardDims());

    return success();
  }
};
} // namespace

namespace {
class AllGatherOpConversionPattern
    : public OpConversionPattern<ttir::AllGatherOp> {
public:
  using OpConversionPattern<ttir::AllGatherOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::AllGatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto device = ::ttnn::utils::getOrInsertDevice(rewriter, op);

    rewriter.replaceOpWithNewOp<ttnn::AllGatherOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), device, adaptor.getAllGatherDim(),
        static_cast<uint32_t>(adaptor.getClusterAxis()));
    return success();
  }
};
} // namespace

namespace {
class ArangeOpConversionPattern : public OpConversionPattern<ttir::ArangeOp> {
public:
  using OpConversionPattern<ttir::ArangeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ArangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    RankedTensorType outputType =
        mlir::cast<RankedTensorType>(op.getResult().getType());
    assert(static_cast<int64_t>(adaptor.getArangeDimension()) ==
               outputType.getRank() - 1 &&
           "Arange dimension must be the final dimension of the output tensor "
           "to convert to ttnn.arange");

    // Get ttnn::TTNNLayoutAttr of the result type
    //
    ttnn::TTNNLayoutAttr layoutAttr =
        mlir::cast<ttnn::TTNNLayoutAttr>(outputType.getEncoding());

    DataTypeAttr dtypeAttr = rewriter.getAttr<DataTypeAttr>(
        elementTypeToDataType(outputType.getElementType()));
    Value device = mlir::tt::ttnn::utils::getOrInsertDevice(rewriter, op);

    ttnn::MemoryConfigAttr memConfigAttr =
        rewriter.getAttr<ttnn::MemoryConfigAttr>(
            rewriter.getAttr<ttnn::BufferTypeAttr>(layoutAttr.getBufferType()),
            rewriter.getAttr<ttnn::ShardSpecAttr>(
                rewriter.getAttr<ttnn::ShapeAttr>(layoutAttr.getShardShape())),
            layoutAttr.getMemLayout());

    rewriter.replaceOpWithNewOp<ttnn::ArangeOp>(
        op, outputType, adaptor.getStart(), adaptor.getEnd(), adaptor.getStep(),
        dtypeAttr, device, memConfigAttr);

    return success();
  }
};
} // namespace

namespace {
class ScatterOpConversionPattern : public OpConversionPattern<ttir::ScatterOp> {
public:
  using OpConversionPattern<ttir::ScatterOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ScatterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // The ttnn interface has the inverse inputs of the TTIR dialect op (which
    // matches torch ops).
    rewriter.replaceOpWithNewOp<ttnn::ScatterOp>(
        op, adaptor.getUpdate(), adaptor.getInput(), adaptor.getOutput());

    return success();
  }
};
} // namespace

namespace {
class PermuteOpConversionPattern : public OpConversionPattern<ttir::PermuteOp> {
public:
  using OpConversionPattern<ttir::PermuteOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::PermuteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ttnn::PermuteOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), adaptor.getPermutationAttr(),
        ttnn::MemoryConfigAttr(), mlir::FloatAttr());

    return success();
  }
};
} // namespace

namespace {
class UpsampleOpConversionPattern
    : public OpConversionPattern<ttir::Upsample2dOp> {
public:
  using OpConversionPattern<ttir::Upsample2dOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::Upsample2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ttnn::UpsampleOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), adaptor.getScaleFactor(), adaptor.getMode(),
        ttnn::MemoryConfigAttr());

    return success();
  }
};
} // namespace

namespace mlir::tt {

void populateTTIRToTTNNPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                TypeConverter &typeConverter) {
  // clang-format off
  // ANCHOR: op_rewriter_pattern_set
  patterns
      .add<TensorEmptyConversionPattern,
           ZerosOpConversionPattern,
           OnesOpConversionPattern,
           ToLayoutOpConversionPattern,
           ElementwiseOpConversionPattern<ttir::AbsOp, ttnn::AbsOp>,
           ElementwiseOpConversionPattern<ttir::AddOp, ttnn::AddOp>,
           ElementwiseOpConversionPattern<ttir::CbrtOp, ttnn::CbrtOp>,
           ElementwiseOpConversionPattern<ttir::FloorOp, ttnn::FloorOp>,
           ElementwiseOpConversionPattern<ttir::IsFiniteOp, ttnn::IsFiniteOp>,
           ElementwiseOpConversionPattern<ttir::LogicalAndOp, ttnn::LogicalAndOp>,
           ElementwiseOpConversionPattern<ttir::LogicalOrOp, ttnn::LogicalOrOp>,
           ElementwiseOpConversionPattern<ttir::LogicalNotOp, ttnn::LogicalNotOp>,
           ElementwiseOpConversionPattern<ttir::LogicalXorOp, ttnn::LogicalXorOp>,
           ElementwiseOpConversionPattern<ttir::BitwiseAndOp, ttnn::BitwiseAndOp>,
           ElementwiseOpConversionPattern<ttir::BitwiseOrOp, ttnn::BitwiseOrOp>,
           ElementwiseOpConversionPattern<ttir::BitwiseXorOp, ttnn::BitwiseXorOp>,
           ElementwiseOpConversionPattern<ttir::BitwiseNotOp, ttnn::BitwiseNotOp>,
           ElementwiseOpConversionPattern<ttir::MultiplyOp, ttnn::MultiplyOp>,
           ElementwiseOpConversionPattern<ttir::EqualOp, ttnn::EqualOp>,
           ElementwiseOpConversionPattern<ttir::NotEqualOp, ttnn::NotEqualOp>,
           ElementwiseOpConversionPattern<ttir::GreaterEqualOp, ttnn::GreaterEqualOp>,
           ElementwiseOpConversionPattern<ttir::GreaterThanOp, ttnn::GreaterThanOp>,
           ElementwiseOpConversionPattern<ttir::LessEqualOp, ttnn::LessEqualOp>,
           ElementwiseOpConversionPattern<ttir::LessThanOp, ttnn::LessThanOp>,
           ElementwiseOpConversionPattern<ttir::MaximumOp, ttnn::MaximumOp>,
           ElementwiseOpConversionPattern<ttir::MinimumOp, ttnn::MinimumOp>,
           ElementwiseOpConversionPattern<ttir::NegOp, ttnn::NegOp>,
           ElementwiseOpConversionPattern<ttir::ReluOp, ttnn::ReluOp>,
           ElementwiseOpConversionPattern<ttir::GeluOp, ttnn::GeluOp>,
           ElementwiseOpConversionPattern<ttir::SqrtOp, ttnn::SqrtOp>,
           ElementwiseOpConversionPattern<ttir::RsqrtOp, ttnn::RsqrtOp>,
           ElementwiseOpConversionPattern<ttir::SignOp, ttnn::SignOp>,
           ElementwiseOpConversionPattern<ttir::SigmoidOp, ttnn::SigmoidOp>,
           ElementwiseOpConversionPattern<ttir::Log1pOp, ttnn::Log1pOp>,
           ElementwiseOpConversionPattern<ttir::ReciprocalOp, ttnn::ReciprocalOp>,
           ElementwiseOpConversionPattern<ttir::ExpOp, ttnn::ExpOp>,
           ElementwiseOpConversionPattern<ttir::LogOp, ttnn::LogOp>,
           ElementwiseOpConversionPattern<ttir::DivOp, ttnn::DivOp>,
           ElementwiseOpConversionPattern<ttir::CeilOp, ttnn::CeilOp>,
           ElementwiseOpConversionPattern<ttir::SinOp, ttnn::SinOp>,
           ElementwiseOpConversionPattern<ttir::CosOp, ttnn::CosOp>,
           ElementwiseOpConversionPattern<ttir::Expm1Op, ttnn::Expm1Op>,
           ElementwiseOpConversionPattern<ttir::RemainderOp, ttnn::RemainderOp>,
           ElementwiseOpConversionPattern<ttir::WhereOp, ttnn::WhereOp>,
           ElementwiseOpConversionPattern<ttir::TanOp, ttnn::TanOp>,
           ElementwiseOpConversionPattern<ttir::TanhOp, ttnn::TanhOp>,
           ElementwiseOpConversionPattern<ttir::PowerOp, ttnn::PowerOp>,
           ReductionOpConversionPattern<ttir::SumOp, ttnn::SumOp>,
           ReductionOpConversionPattern<ttir::MeanOp, ttnn::MeanOp>,
           ReductionOpConversionPattern<ttir::MaxOp, ttnn::MaxOp>,
           ReductionOpConversionPattern<ttir::MinOp, ttnn::MinOp>,
           ReductionProdOpConversionPattern,
           ReductionArgMaxOpConversionPattern,
           ElementwiseUnaryWithFloatParameterOpConversionPattern<ttir::LeakyReluOp, ttnn::LeakyReluOp>,
           BroadcastOpConversionPattern,
           PadOpConversionPattern,
           EmbeddingOpConversionPattern,
           EmbeddingBackwardOpConversionPattern,
           RepeatOpConversionPattern,
           CumSumOpConversionPattern,
           RepeatInterleaveOpConversionPattern,
           SoftmaxOpConversionPattern,
           TransposeOpConversionPattern,
           TypecastOpConversionPattern,
           ClampOpConversionPattern,
           ConcatOpConversionPattern,
           ReshapeOpConversionPattern,
           SliceOpConversionPattern,
           SqueezeOpConversionPattern,
           UnsqueezeOpConversionPattern,
           ConstantOpConversionPattern,
           LinearOpConversionPattern,
           MatmulOpConversionPattern,
           Conv2dOpConversionPattern,
           ConvTranspose2dOpConversionPattern,
           MaxPool2dOpConversionPattern,
           SubtractOpConversionPattern,
           MeshShardOpConversionPattern,
           AllReduceOpConversionPattern,
           AllGatherOpConversionPattern,
           ArangeOpConversionPattern,
           UpdateCacheOpConversionPattern,
           FillCacheOpConversionPattern,
           ScatterOpConversionPattern,
           PermuteOpConversionPattern,
           UpsampleOpConversionPattern
           >(typeConverter, ctx);
  // ANCHOR_END: op_rewriter_pattern_set
  // clang-format on
}

} // namespace mlir::tt
