// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTNN/TTIRToTTNN.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
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
class TensorEmptyConversionPattern : public OpConversionPattern<ttir::EmptyOp> {
public:
  using OpConversionPattern<ttir::EmptyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::EmptyOp op, OpAdaptor adaptor,
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
    DataTypeAttr dTypeAttr = DataTypeAttr::get(rewriter.getContext(), dtype);

    ttnn::Layout ttnnLayoutEnum = ttnn::Layout::RowMajor;

    if (layoutAttr.isTiled()) {
      ttnnLayoutEnum = ttnn::Layout::Tile;
    }
    ttnn::LayoutAttr tensorLayoutAttr =
        ttnn::LayoutAttr::get(op.getContext(), ttnnLayoutEnum);

    // Due to API constraints, we need to use a host_empty op if tensor is in
    // system_memory.
    if (mlir::tt::ttnn::isSystemBufferType(layoutAttr.getBufferType())) {
      // Replace op
      //
      rewriter.replaceOpWithNewOp<ttnn::ZerosOp>(
          op, this->getTypeConverter()->convertType(op.getType()), shapeAttr,
          dTypeAttr, tensorLayoutAttr, /*device=*/nullptr,
          /*memoryConfig=*/nullptr);
      // Otherwise, we use regular empty op, with device-specific fields.
    } else {
      // Device
      //
      auto device = ::ttnn::utils::getOrInsertDevice(rewriter, op);

      // Create MemoryConfigAttr
      //
      ttnn::BufferTypeAttr bufferTypeAttr = ttnn::BufferTypeAttr::get(
          op.getContext(), layoutAttr.getBufferType());
      ttnn::MemoryConfigAttr memoryConfigAttr = ttnn::MemoryConfigAttr::get(
          op.getContext(), layoutAttr.getMemLayout(), bufferTypeAttr,
          std::nullopt);

      // Replace op
      //
      rewriter.replaceOpWithNewOp<ttnn::EmptyOp>(
          op, this->getTypeConverter()->convertType(op.getType()), shapeAttr,
          dTypeAttr, tensorLayoutAttr, device, memoryConfigAttr);
    }
    return success();
  }
};
} // namespace

namespace {
template <typename TTIRType, typename TTNNType>
class NamedFullConversionPattern : public OpConversionPattern<TTIRType> {
public:
  using OpConversionPattern<TTIRType>::OpConversionPattern;
  using OpAdaptor = typename TTIRType::Adaptor;

  LogicalResult
  matchAndRewrite(TTIRType op, OpAdaptor adaptor,
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

    // Get data type, tensor layout, device and memory config
    //
    DataTypeAttr dTypeAttr =
        DataTypeAttr::get(rewriter.getContext(), layoutAttr.getDataType());
    ttnn::BufferType bufferType = layoutAttr.getBufferType();
    ttnn::LayoutAttr tensorLayoutAttr =
        ttnn::LayoutAttr::get(op.getContext(), layoutAttr.getLayout());
    ttnn::TensorMemoryLayoutAttr memLayout = layoutAttr.getMemLayout();

    // Device only exists if memLayout is *not* null
    //
    auto device =
        memLayout ? mlir::Value(::ttnn::utils::getOrInsertDevice(rewriter, op))
                  : nullptr;

    // MemoryConfigAttr only exists if memLayout is *not* null
    //
    ttnn::MemoryConfigAttr memoryConfigAttr =
        memLayout ? ttnn::MemoryConfigAttr::get(
                        op.getContext(), memLayout,
                        ttnn::BufferTypeAttr::get(op.getContext(), bufferType),
                        std::nullopt)
                  : nullptr;

    rewriter.replaceOpWithNewOp<TTNNType>(
        op, this->getTypeConverter()->convertType(op.getType()), shapeAttr,
        dTypeAttr, tensorLayoutAttr, device, memoryConfigAttr);

    return success();
  }
};
} // namespace

namespace {
class FullOpConversionPattern : public OpConversionPattern<ttir::FullOp> {
public:
  using OpConversionPattern<ttir::FullOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::FullOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ttnnLayoutAttr =
        mlir::cast<ttnn::TTNNLayoutAttr>(op.getType().getEncoding());
    bool isOnDevice =
        ttnnLayoutAttr.getBufferType() != ttnn::BufferType::SystemMemory;
    ttnn::GetDeviceOp deviceOp;
    if (isOnDevice) {
      deviceOp = ::ttnn::utils::getOrInsertDevice(rewriter, op);
    }
    rewriter.replaceOpWithNewOp<ttnn::FullOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getFillValue(), deviceOp);

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

    // Get the DPS operand and delete it's creator op, if it's ttir::EmptyOp
    //
    Value dpsOperand = adaptor.getOperands().back();
    ttnn::EmptyOp emptyOp = dpsOperand.getDefiningOp<ttnn::EmptyOp>();
    if (emptyOp) {
      rewriter.eraseOp(emptyOp);
    }

    assert(mlir::isa<mlir::RankedTensorType>(adaptor.getInput().getType()) &&
           "Expected RankedTensorType for ToLayoutOp input");

    auto outputLayoutAttr = mlir::cast<ttnn::TTNNLayoutAttr>(
        mlir::cast<mlir::RankedTensorType>(op.getResult(0).getType())
            .getEncoding());

    // Determine the output data type
    DataType dtype = outputLayoutAttr.getDataType();
    DataTypeAttr outputDataType =
        DataTypeAttr::get(rewriter.getContext(), dtype);

    // Determine the output layout (tile or row major)
    ttnn::BufferType outputBufferType = outputLayoutAttr.getBufferType();

    ttnn::Layout outputLayoutEnum = outputLayoutAttr.getLayout();

    bool isOutputOnHost = (outputBufferType == ttnn::BufferType::SystemMemory);

    RankedTensorType result = mlir::cast<RankedTensorType>(op.getType(0));

    ttnn::LayoutAttr outputLayout =
        ttnn::LayoutAttr::get(rewriter.getContext(), outputLayoutEnum);

    ttnn::MemoryConfigAttr outputMemConfigAttr = ttnn::MemoryConfigAttr::get(
        rewriter.getContext(), outputLayoutAttr.getMemLayout(),
        ttnn::BufferTypeAttr::get(rewriter.getContext(), outputBufferType),
        std::nullopt);

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

    static_assert(ttir::utils::has_dps_trait_v<TTIROpTy>);
    auto inputs =
        ttir::utils::getDpsInputsFromAdaptor(adaptor, op.getNumDpsInits());
    rewriter.replaceOpWithNewOp<TTNNOpTy>(op, resultTypes, inputs);
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
    std::optional<::mlir::ArrayAttr> dimArg = op.getDimArg();
    int64_t size = dimArg ? dimArg->size() : inputRank;

    // [TODO](mmanzoor) Decompose ttnn.prod op into multiple ttnn.prod to handle
    // reduction along multiple dimensions.
    // https://github.com/tenstorrent/tt-mlir/issues/1861
    if ((size > 1) && (size < inputRank)) {
      return rewriter.notifyMatchFailure(
          op, "tt-metal only supports reduce(prod) along one dimension or all "
              "dimensions.");
    }

    // TTNN only supports reduce(prod) along one dimension or all dimensions.
    // That is controlled by dimArg. If dimArg is not present, then all
    // dimensions are reduced. Otherwise, only the specified dimension is
    // reduced.
    mlir::IntegerAttr newDimArg = nullptr;
    if (dimArg && dimArg->size() == 1) {
      auto int32Attr = mlir::cast<mlir::IntegerAttr>(dimArg->getValue()[0]);
      newDimArg =
          mlir::IntegerAttr::get(rewriter.getI64Type(), int32Attr.getInt());
    }

    rewriter.replaceOpWithNewOp<ttnn::ProdOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), newDimArg, adaptor.getKeepDim(),
        /*memoryConfig=*/nullptr);
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
    // Most of the frontends uses signed or sign less integer as return type for
    // argmax op (tt-mlir uses signed integer in this case); whereas, tt-metal
    // uses UINT32 as return type. This difference is ignored as the output
    // indices will always be positive.

    std::optional<mlir::ArrayAttr> dimArg = op.getDimArg();
    IntegerAttr reductionAxis;
    if (dimArg) {
      assert(dimArg->size() == 1 &&
             "ttir::ArgMaxOp dim argument must be a single integer");
      reductionAxis = *dimArg->getAsRange<mlir::IntegerAttr>().begin();
    }
    rewriter.replaceOpWithNewOp<ttnn::ArgMaxOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), reductionAxis, adaptor.getKeepDim(),
        /*use_multicore=*/false, /*memoryConfig=*/nullptr);
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
        adaptor.getInput(), adaptor.getWeight());

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
        reshapedGradShape, rewriter, "_reshaped_grad");

    // Get TTNNLayoutAttr of the result type.
    ttnn::TTNNLayoutAttr layoutAttr = mlir::cast<ttnn::TTNNLayoutAttr>(
        op.getResult().getType().getEncoding());

    // Get data type, tensor layout, buffer type and memory config.
    DataTypeAttr dTypeAttr =
        DataTypeAttr::get(rewriter.getContext(), layoutAttr.getDataType());
    ttnn::TensorMemoryLayoutAttr memLayout = layoutAttr.getMemLayout();
    ttnn::BufferType bufferType = layoutAttr.getBufferType();

    ttnn::MemoryConfigAttr memoryConfigAttr = ttnn::MemoryConfigAttr::get(
        op.getContext(), memLayout,
        ttnn::BufferTypeAttr::get(op.getContext(), bufferType), std::nullopt);

    rewriter.replaceOpWithNewOp<ttnn::EmbeddingBackwardOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), adaptor.getWeight(), reshapedGrad, dTypeAttr,
        memoryConfigAttr);
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
        adaptor.getInput(), adaptor.getDim(), nullptr);
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
template <typename TTIROpTy, typename TTNNOpTy>
class ClampOpConversionPattern : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;
  using OpAdaptor = typename TTIROpTy::Adaptor;

  LogicalResult
  matchAndRewrite(TTIROpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TTNNOpTy>(
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
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), adaptor.getParameter());
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
        adaptor.getInputs(), dim,
        /*memory_config=*/nullptr);
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
        adaptor.getInput(), adaptor.getShape(), /*memory_config=*/nullptr);
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
        adaptor.getInput(), adaptor.getBegins(), adaptor.getEnds(),
        adaptor.getStep());
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
        adaptor.getInput(), shapeAttr, /*memory_config=*/nullptr);

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
            mlir::dyn_cast_if_present<ttnn::TTNNLayoutAttr>(
                op.getResult().getType().getEncoding());
        layoutAttr.getBufferType() != ttnn::BufferType::SystemMemory) {
      memcfg = ttnn::MemoryConfigAttr::get(
          op.getContext(), layoutAttr.getMemLayout(),
          ttnn::BufferTypeAttr::get(op.getContext(),
                                    layoutAttr.getBufferType()),
          std::nullopt);
    }
    rewriter.replaceOpWithNewOp<ttnn::PadOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), adaptor.getPaddingAttr(), adaptor.getValue(),
        /*use_multicore=*/true, memcfg);

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
        adaptor.getInput(), shapeAttr, /*memory_config=*/nullptr);

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

    rewriter.replaceOpWithNewOp<ttnn::ConstantOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getValue());

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
        adaptor.getB(), adaptor.getBias(), adaptor.getTransposeA(),
        adaptor.getTransposeB());
    return success();
  }
};
} // namespace

namespace {
class BatchNormOpConversionPattern
    : public OpConversionPattern<ttir::BatchNormOp> {
public:
  using OpConversionPattern<ttir::BatchNormOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::BatchNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Check if the operand is a 4-dimensional tensor.
    if (mlir::cast<RankedTensorType>(adaptor.getOperand().getType())
            .getRank() != 4) {
      return rewriter.notifyMatchFailure(
          op, "Operand must be a 4-dimensional tensor");
    }

    // We only support excluded_dimension=1 for ttnn::batch_norm
    if (adaptor.getDimension() != 1) {
      return rewriter.notifyMatchFailure(op, "We can only exclude dimension 1");
    }

    mlir::APFloat defaultMomentum(0.1f);

    rewriter.replaceOpWithNewOp<ttnn::BatchNormOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getOperand(), adaptor.getMean(), adaptor.getVariance(),
        adaptor.getTraining(), adaptor.getEpsilon(), defaultMomentum,
        adaptor.getScale(), adaptor.getOffset(),
        /*memoryConfig*/ nullptr);
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
        adaptor.getB(), adaptor.getTransposeA(), adaptor.getTransposeB(),
        nullptr);
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

    if (!adaptor.getFlattenedCompatInfo()) {
      return rewriter.notifyMatchFailure(
          op, "TTNN only supports flattened input tensors for Conv2dOp. Please "
              "run the FlattenSlidingWindow pass before lowering to TTNN.");
    }
    auto flattenedCompatInfo = adaptor.getFlattenedCompatInfo();
    auto device = ::ttnn::utils::getOrInsertDevice(rewriter, op);

    auto kernelTy = mlir::cast<RankedTensorType>(adaptor.getWeight().getType());
    constexpr unsigned int CHANNEL_DIM = 3;
    auto batchSizeAttr =
        rewriter.getI32IntegerAttr(flattenedCompatInfo.getBatchSize());
    auto inputHeightAttr =
        rewriter.getI32IntegerAttr(flattenedCompatInfo.getInputHeight());
    auto inputWidthAttr =
        rewriter.getI32IntegerAttr(flattenedCompatInfo.getInputWidth());
    auto inChannelsAttr = rewriter.getI32IntegerAttr(
        op.getInput().getType().getDimSize(CHANNEL_DIM));
    auto outChannelsAttr = rewriter.getI32IntegerAttr(
        op.getResult().getType().getDimSize(CHANNEL_DIM));

    auto kernelSizeAttr = rewriter.getDenseI32ArrayAttr(
        {static_cast<int32_t>(kernelTy.getDimSize(2)),
         static_cast<int32_t>(kernelTy.getDimSize(3))});

    auto strideAttr =
        attrToDenseI32ArrayAttr(adaptor.getStride(), rewriter, "stride");
    if (auto error = strideAttr.takeError()) {
      return rewriter.notifyMatchFailure(op, llvm::toString(std::move(error)));
    }

    auto expectedPaddingAttr =
        attrToDenseI32ArrayAttr(adaptor.getPadding(), rewriter, "padding");
    if (auto error = expectedPaddingAttr.takeError()) {
      return rewriter.notifyMatchFailure(op, llvm::toString(std::move(error)));
    }

    DenseI32ArrayAttr paddingAttr = *expectedPaddingAttr;
    if (paddingAttr.size() == 4) {
      // Reorders padding from (pT, pL, pB, pR) to (pT, pB, pL, pR).
      paddingAttr = rewriter.getDenseI32ArrayAttr(
          {paddingAttr[0], paddingAttr[2], paddingAttr[1], paddingAttr[3]});
    }

    auto dilationAttr =
        attrToDenseI32ArrayAttr(adaptor.getDilation(), rewriter, "dilation");
    if (auto error = dilationAttr.takeError()) {
      return rewriter.notifyMatchFailure(op, llvm::toString(std::move(error)));
    }

    auto groupsAttr = rewriter.getI32IntegerAttr(adaptor.getGroups());

    rewriter.replaceOpWithNewOp<ttnn::Conv2dOp>(
        op, getTypeConverter()->convertType(op.getResult().getType()),
        adaptor.getInput(), adaptor.getWeight(), adaptor.getBias(), device,
        inChannelsAttr, outChannelsAttr, batchSizeAttr, inputHeightAttr,
        inputWidthAttr, kernelSizeAttr, *strideAttr, paddingAttr, *dilationAttr,
        groupsAttr, /*conv2d_config=*/nullptr, /*compute_config=*/nullptr);

    return success();
  }

private:
  llvm::Expected<DenseI32ArrayAttr>
  attrToDenseI32ArrayAttr(mlir::Attribute attr,
                          ConversionPatternRewriter &rewriter,
                          StringRef attrName) const {
    if (auto tuple = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
      // TTNN does not support passing a single integer so we have to convert it
      // to a pair.
      auto pair = ttmlir::utils::getPairOfInteger<int32_t>(attr);
      if (auto error = pair.takeError()) {
        return std::move(error);
      }
      return rewriter.getDenseI32ArrayAttr({pair->first, pair->second});
    }

    if (auto denseAttr = dyn_cast<DenseI32ArrayAttr>(attr)) {
      return denseAttr;
    }

    return llvm::createStringError("Unexpected attribute type for '%s'",
                                   attrName.data());
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
    auto outputTy = op.getResult().getType();

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
      return op.emitOpError() << llvm::toString(std::move(error));
    }

    auto maybePaddingAttr =
        attrToDenseI32ArrayAttr<4>(adaptor.getPadding(), rewriter);
    if (auto error = maybePaddingAttr.takeError()) {
      return op.emitOpError() << llvm::toString(std::move(error));
    }
    auto paddingAttr = *maybePaddingAttr;

    if (paddingAttr[0] != paddingAttr[2] || paddingAttr[1] != paddingAttr[3]) {
      return op.emitOpError()
             << "TTNN only supports padding height/width attributes. Thus, "
                "padding_top/padding_left must equal "
                "padding_bottom/padding_right "
                "for the op to execute as expected.";
    }

    // Padding only supports 2 values in TTNN.
    auto reducedPaddingAttr =
        rewriter.getDenseI32ArrayAttr({paddingAttr[0], paddingAttr[1]});

    auto outputPaddingAttr =
        attrToDenseI32ArrayAttr(adaptor.getOutputPadding(), rewriter);
    if (auto error = outputPaddingAttr.takeError()) {
      return op.emitOpError() << llvm::toString(std::move(error));
    }

    auto dilationAttr =
        attrToDenseI32ArrayAttr(adaptor.getDilation(), rewriter);
    if (auto error = dilationAttr.takeError()) {
      return op.emitOpError() << llvm::toString(std::move(error));
    }

    auto groupsAttr = rewriter.getI32IntegerAttr(adaptor.getGroups());

    // Transposed convolution in ttnn returns a tensor in a flattened shape
    // (1 x 1 x N * H * W x C).
    llvm::ArrayRef<std::int64_t> output_shape = outputTy.getShape();
    llvm::SmallVector<std::int64_t, 4> flattenedOutputShape = {
        1, 1, output_shape[0] * output_shape[1] * output_shape[2],
        output_shape[3]};
    outputTy = mlir::cast<RankedTensorType>(getTypeConverter()->convertType(
        outputTy.cloneWith(flattenedOutputShape, outputTy.getElementType())));

    ttnn::ConvTranspose2dOp new_conv = rewriter.create<ttnn::ConvTranspose2dOp>(
        op.getLoc(), outputTy, adaptor.getInput(), adaptor.getWeight(),
        adaptor.getBias(), device, inChannelsAttr, outChannelsAttr,
        batchSizeAttr, inputHeightAttr, inputWidthAttr, kernelSizeAttr,
        *strideAttr, reducedPaddingAttr, *outputPaddingAttr, *dilationAttr,
        groupsAttr, /*conv2d_config=*/nullptr, /*memoryConfig=*/nullptr);

    // Restore the normal shape (N x H x W x C).
    Value output =
        ttir_to_ttnn::utils::generateReshape(new_conv, output_shape, rewriter);

    rewriter.replaceOp(op, output);
    return success();
  }

private:
  template <size_t ElementCount = 2>
  llvm::Expected<DenseI32ArrayAttr>
  attrToDenseI32ArrayAttr(mlir::Attribute attr,
                          ConversionPatternRewriter &rewriter) const {
    if constexpr (ElementCount == 2) {
      // Handles attributes requiring 2 spatial dimensions (e.g., stride,
      // dilation). Converts the attribute into a pair of integers.
      auto pair = ttmlir::utils::getPairOfInteger<int32_t>(attr);
      if (auto error = pair.takeError()) {
        return std::move(error);
      }
      return rewriter.getDenseI32ArrayAttr({pair->first, pair->second});
    } else if constexpr (ElementCount == 4) {
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
    return llvm::createStringError(std::errc::invalid_argument,
                                   "Unsupported element count: %d",
                                   ElementCount);
  }
};
} // namespace

namespace {
template <typename TTIROpTy, typename TTNNOpTy>
class Pooling2dOpConversionPattern : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;
  using OpAdaptor = typename TTIROpTy::Adaptor;

  LogicalResult
  matchAndRewrite(TTIROpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!adaptor.getFlattenedCompatInfo()) {
      return rewriter.notifyMatchFailure(
          op, "TTNN only supports flattened input tensors for " +
                  op.getOperationName() +
                  ". Please "
                  "run the FlattenSlidingWindow pass before lowering to TTNN.");
    }
    if (adaptor.getPaddingBottom() != adaptor.getPaddingTop()) {
      return rewriter.notifyMatchFailure(
          op, op.getOperationName() +
                  "does not support asymmetric padding for top/bottom.");
    }
    if (adaptor.getPaddingLeft() != adaptor.getPaddingRight()) {
      return rewriter.notifyMatchFailure(
          op, op.getOperationName() +
                  "does not support asymmetric padding for left/right.");
    }

    auto batchSize = adaptor.getFlattenedCompatInfo().getBatchSize();
    constexpr unsigned int CHANNEL_DIM = 3;
    auto channels = op.getInput().getType().getDimSize(CHANNEL_DIM);

    DenseI32ArrayAttr kernelSizeAttr = rewriter.getDenseI32ArrayAttr(
        {adaptor.getKernelHeight(), adaptor.getKernelWidth()});

    DenseI32ArrayAttr strideAttr = rewriter.getDenseI32ArrayAttr(
        {adaptor.getStrideHeight(), adaptor.getStrideWidth()});

    assert(adaptor.getPaddingTop() == adaptor.getPaddingBottom());
    assert(adaptor.getPaddingLeft() == adaptor.getPaddingRight());
    DenseI32ArrayAttr paddingAttr = rewriter.getDenseI32ArrayAttr(
        {adaptor.getPaddingTop(), adaptor.getPaddingLeft()});

    DenseI32ArrayAttr dilationAttr = rewriter.getDenseI32ArrayAttr(
        {adaptor.getDilationHeight(), adaptor.getDilationWidth()});

    rewriter.replaceOpWithNewOp<TTNNOpTy>(
        op, this->getTypeConverter()->convertType(op.getResult().getType()),
        adaptor.getInput(), batchSize,
        adaptor.getFlattenedCompatInfo().getInputHeight(),
        adaptor.getFlattenedCompatInfo().getInputWidth(), channels,
        kernelSizeAttr, strideAttr, paddingAttr, dilationAttr,
        /*memory_config=*/nullptr,
        /* applied_shard_scheme=*/nullptr, adaptor.getCeilMode(),
        /* in_place_halo=*/false);

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
    auto resultType = op.getType();
    ttnn::TTNNLayoutAttr outputLayoutAttr =
        mlir::cast<ttnn::TTNNLayoutAttr>(resultType.getEncoding());
    DataType outputDataType = outputLayoutAttr.getDataType();

    rewriter.replaceOpWithNewOp<ttnn::TypecastOp>(
        op, this->getTypeConverter()->convertType(resultType),
        adaptor.getInput(), outputDataType);
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
        mlir::cast<mlir::RankedTensorType>(adaptor.getLhs().getType());
    RankedTensorType rhsType =
        mlir::cast<mlir::RankedTensorType>(adaptor.getRhs().getType());
    Type outputType = this->getTypeConverter()->convertType(srcOp.getType());
    if (lhsType.getShape() == rhsType.getShape()) {
      rewriter.replaceOpWithNewOp<ttnn::SubtractOp>(
          srcOp, outputType, adaptor.getLhs(), adaptor.getRhs());

      // Broadcast for rhs operand require the operation to be commutative to
      // allow switching the order of operands. To allow this conversion, the
      // following conversion is applied to SubtractOp: subtractOp(lhs,rhs) ->
      // addOp(lhs, negOp(rhs))
    } else {
      ttnn::NegOp negOp = rewriter.create<ttnn::NegOp>(
          ttmlir::utils::appendLocationSuffix(srcOp.getLoc(), "_neg"),
          adaptor.getRhs().getType(), adaptor.getRhs());

      rewriter.replaceOpWithNewOp<ttnn::AddOp>(srcOp, outputType,
                                               adaptor.getLhs(), negOp);
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
class ReduceScatterOpConversionPattern
    : public OpConversionPattern<ttir::ReduceScatterOp> {
public:
  using OpConversionPattern<ttir::ReduceScatterOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ReduceScatterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto device = ::ttnn::utils::getOrInsertDevice(rewriter, op);

    rewriter.replaceOpWithNewOp<ttnn::ReduceScatterOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), device, adaptor.getReduceType(),
        adaptor.getScatterDim(), adaptor.getClusterAxis());

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
class CollectivePermuteOpConversionPattern
    : public OpConversionPattern<ttir::CollectivePermuteOp> {
public:
  using OpConversionPattern<ttir::CollectivePermuteOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::CollectivePermuteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto device = ::ttnn::utils::getOrInsertDevice(rewriter, op);

    rewriter.replaceOpWithNewOp<ttnn::CollectivePermuteOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), device, adaptor.getSourceTargetPairs());

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
            layoutAttr.getMemLayout(),
            rewriter.getAttr<ttnn::BufferTypeAttr>(layoutAttr.getBufferType()),
            std::nullopt);

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
        op, adaptor.getOutput().getType(), adaptor.getUpdate(),
        adaptor.getInput());

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

// Utility function to get data type for quantized types.
static DataTypeAttr getDataType(mlir::Value val,
                                ConversionPatternRewriter &rewriter,
                                const TypeConverter *typeConverter) {
  ttnn::TTNNLayoutAttr outputLayoutAttr = mlir::cast<ttnn::TTNNLayoutAttr>(
      mlir::cast<RankedTensorType>(typeConverter->convertType(val.getType()))
          .getEncoding());
  DataType dtype = outputLayoutAttr.getDataType();
  return DataTypeAttr::get(rewriter.getContext(), dtype);
}

namespace {
template <typename OpTy, typename TTNNOpTy>
class QuantizationOpConversionPattern : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    DataTypeAttr outputDataType =
        getDataType(op.getResult(), rewriter, this->getTypeConverter());

    rewriter.replaceOpWithNewOp<TTNNOpTy>(
        op, this->getTypeConverter()->convertType(op.getResult().getType()),
        adaptor.getInput(), op.getScale(), op.getZeroPoint(),
        op.getAxisAttr() ? op.getAxisAttr() : mlir::IntegerAttr(),
        outputDataType, ttnn::MemoryConfigAttr());
    return success();
  }
};

class RequantizeOpConversionPattern
    : public OpConversionPattern<ttir::RequantizeUnrolledOp> {
public:
  using OpConversionPattern<ttir::RequantizeUnrolledOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::RequantizeUnrolledOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    DataTypeAttr outputDataType =
        getDataType(op.getResult(), rewriter, this->getTypeConverter());

    rewriter.replaceOpWithNewOp<ttnn::RequantizeOp>(
        op, this->getTypeConverter()->convertType(op.getResult().getType()),
        adaptor.getInput(), op.getInScale(), op.getInZeroPoint(),
        op.getOutScale(), op.getOutZeroPoint(),
        op.getAxisAttr() ? op.getAxisAttr() : mlir::IntegerAttr(),
        outputDataType, ttnn::MemoryConfigAttr());
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
           NamedFullConversionPattern<ttir::ZerosOp, ttnn::ZerosOp>,
           NamedFullConversionPattern<ttir::OnesOp, ttnn::OnesOp>,
           FullOpConversionPattern,
           ToLayoutOpConversionPattern,
           QuantizationOpConversionPattern<ttir::QuantizeUnrolledOp, ttnn::QuantizeOp>,
           QuantizationOpConversionPattern<ttir::DequantizeUnrolledOp, ttnn::DequantizeOp>,
           RequantizeOpConversionPattern,
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
           ElementwiseOpConversionPattern<ttir::ErfOp, ttnn::ErfOp>,
           ElementwiseOpConversionPattern<ttir::ErfcOp, ttnn::ErfcOp>,
           ElementwiseOpConversionPattern<ttir::LogOp, ttnn::LogOp>,
           ElementwiseOpConversionPattern<ttir::DivOp, ttnn::DivideOp>,
           ElementwiseOpConversionPattern<ttir::CeilOp, ttnn::CeilOp>,
           ElementwiseOpConversionPattern<ttir::SinOp, ttnn::SinOp>,
           ElementwiseOpConversionPattern<ttir::CosOp, ttnn::CosOp>,
           ElementwiseOpConversionPattern<ttir::Expm1Op, ttnn::Expm1Op>,
           ElementwiseOpConversionPattern<ttir::RemainderOp, ttnn::RemainderOp>,
           ElementwiseOpConversionPattern<ttir::WhereOp, ttnn::WhereOp>,
           ElementwiseOpConversionPattern<ttir::TanOp, ttnn::TanOp>,
           ElementwiseOpConversionPattern<ttir::TanhOp, ttnn::TanhOp>,
           ElementwiseOpConversionPattern<ttir::AtanOp, ttnn::AtanOp>,
           ElementwiseOpConversionPattern<ttir::Atan2Op, ttnn::Atan2Op>,
           ElementwiseOpConversionPattern<ttir::PowOp, ttnn::PowOp>,
           Pooling2dOpConversionPattern<ttir::MaxPool2dOp, ttnn::MaxPool2dOp>,
           Pooling2dOpConversionPattern<ttir::AvgPool2dOp, ttnn::AvgPool2dOp>,
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
           ClampOpConversionPattern<ttir::ClampScalarOp, ttnn::ClampScalarOp>,
           ClampOpConversionPattern<ttir::ClampTensorOp, ttnn::ClampTensorOp>,
           ConcatOpConversionPattern,
           ReshapeOpConversionPattern,
           SliceOpConversionPattern,
           SqueezeOpConversionPattern,
           UnsqueezeOpConversionPattern,
           ConstantOpConversionPattern,
           LinearOpConversionPattern,
           BatchNormOpConversionPattern,
           MatmulOpConversionPattern,
           Conv2dOpConversionPattern,
           ConvTranspose2dOpConversionPattern,
           SubtractOpConversionPattern,
           MeshShardOpConversionPattern,
           AllReduceOpConversionPattern,
           AllGatherOpConversionPattern,
           ReduceScatterOpConversionPattern,
           CollectivePermuteOpConversionPattern,
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
