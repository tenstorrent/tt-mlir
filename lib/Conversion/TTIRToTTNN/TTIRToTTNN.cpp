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
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
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

#include "llvm/Support/LogicalResult.h"
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
    ttcore::DataType dtype = layoutAttr.getDataType();
    ttcore::DataTypeAttr dTypeAttr =
        ttcore::DataTypeAttr::get(rewriter.getContext(), dtype);

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
          op, this->getTypeConverter()->convertType(op.getType()),
          /*device=*/nullptr, shapeAttr, dTypeAttr, tensorLayoutAttr,
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
          op, this->getTypeConverter()->convertType(op.getType()), device,
          shapeAttr, dTypeAttr, tensorLayoutAttr, memoryConfigAttr);
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
    ttcore::DataTypeAttr dTypeAttr = ttcore::DataTypeAttr::get(
        rewriter.getContext(), layoutAttr.getDataType());
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
        op, this->getTypeConverter()->convertType(op.getType()), device,
        shapeAttr, dTypeAttr, tensorLayoutAttr, memoryConfigAttr);

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
    ttcore::DataType dtype = outputLayoutAttr.getDataType();
    ttcore::DataTypeAttr outputDataType =
        ttcore::DataTypeAttr::get(rewriter.getContext(), dtype);

    // Determine the output layout (tile or row major)
    ttnn::BufferType outputBufferType = outputLayoutAttr.getBufferType();

    ttnn::Layout outputLayoutEnum = outputLayoutAttr.getLayout();

    RankedTensorType result = mlir::cast<RankedTensorType>(op.getType(0));

    ttnn::LayoutAttr outputLayout =
        ttnn::LayoutAttr::get(rewriter.getContext(), outputLayoutEnum);

    ttnn::MemoryConfigAttr outputMemConfigAttr = ttnn::MemoryConfigAttr::get(
        rewriter.getContext(), outputLayoutAttr.getMemLayout(),
        ttnn::BufferTypeAttr::get(rewriter.getContext(), outputBufferType),
        std::nullopt);

    rewriter.replaceOpWithNewOp<ttnn::ToLayoutOp>(
        op, this->getTypeConverter()->convertType(result), adaptor.getInput(),
        outputLayout, outputDataType, outputMemConfigAttr);

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
// Conversion pattern for binary operations
template <typename TTIROpTy, typename TTNNOpTy,
          typename OpAdaptor = typename TTIROpTy::Adaptor>
class ElementwiseBinaryOpConversionPattern
    : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    static_assert(ttir::utils::has_dps_trait_v<TTIROpTy>);

    rewriter.replaceOpWithNewOp<TTNNOpTy>(
        op, this->getTypeConverter()->convertType(op.getResult().getType()),
        adaptor.getLhs(), adaptor.getRhs());
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
        reshapedGradShape, rewriter,
        ttmlir::utils::appendLocationSuffix(op.getLoc(), "_reshaped_grad"));

    // Get TTNNLayoutAttr of the result type.
    ttnn::TTNNLayoutAttr layoutAttr = mlir::cast<ttnn::TTNNLayoutAttr>(
        op.getResult().getType().getEncoding());

    // Get data type, tensor layout, buffer type and memory config.
    ttcore::DataTypeAttr dTypeAttr = ttcore::DataTypeAttr::get(
        rewriter.getContext(), layoutAttr.getDataType());
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
        adaptor.getInput(), adaptor.getDimension(), adaptor.getNumericStable());
    return success();
  }
};
} // namespace

namespace {
class SortOpConversionPattern : public OpConversionPattern<ttir::SortOp> {
public:
  using OpConversionPattern<ttir::SortOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::SortOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultTypes))) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<ttnn::SortOp>(
        op, resultTypes, adaptor.getInput(), adaptor.getDim(),
        adaptor.getDescending(), adaptor.getStable(), ttnn::MemoryConfigAttr());
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
class PowOpConversionPattern : public OpConversionPattern<ttir::PowOp> {
private:
  // Helper function to extract constant value.
  static mlir::Attribute getConstantAttr(mlir::Value value) {
    mlir::Operation *op = value.getDefiningOp();
    while (mlir::isa_and_present<mlir::tt::ttnn::ReshapeOp,
                                 mlir::tt::ttnn::TypecastOp>(op)) {
      op = op->getOperand(0).getDefiningOp();
    }

    auto fullOp = mlir::dyn_cast_if_present<mlir::tt::ttnn::FullOp>(op);
    if (!fullOp) {
      return {};
    }

    mlir::Attribute fillValueAttr = fullOp.getFillValueAttr();

    if (!isa<FloatAttr, IntegerAttr>(fillValueAttr)) {
      return {};
    }
    return fillValueAttr;
  }

public:
  using OpConversionPattern<ttir::PowOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::PowOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    mlir::Attribute exponent = getConstantAttr(adaptor.getRhs());
    if (exponent) {
      rewriter.replaceOpWithNewOp<ttnn::PowScalarOp>(
          op, this->getTypeConverter()->convertType(op.getType()),
          adaptor.getLhs(), exponent);
      return success();
    }

    rewriter.replaceOpWithNewOp<ttnn::PowTensorOp>(
        op, this->getTypeConverter()->convertType(op.getResult().getType()),
        adaptor.getLhs(), adaptor.getRhs());
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
template <typename TTIROpTy, typename TTNNOpTy>
class SliceOpConversionPattern : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;
  using OpAdaptor = typename TTIROpTy::Adaptor;

  LogicalResult
  matchAndRewrite(TTIROpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TTNNOpTy>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), adaptor.getBegins(), adaptor.getEnds(),
        adaptor.getStepAttr());
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

    // Get ttnn::TTNNLayoutAttr of the result type
    //
    ttnn::TTNNLayoutAttr layoutAttr = mlir::cast<ttnn::TTNNLayoutAttr>(
        op.getResult().getType().getEncoding());

    // Get the data type and tensor layout
    //
    ttcore::DataType dtype = layoutAttr.getDataType();
    ttcore::DataTypeAttr dTypeAttr =
        ttcore::DataTypeAttr::get(rewriter.getContext(), dtype);

    ttnn::Layout ttnnLayoutEnum = ttnn::Layout::RowMajor;

    if (layoutAttr.isTiled()) {
      ttnnLayoutEnum = ttnn::Layout::Tile;
    }
    ttnn::LayoutAttr tensorLayoutAttr =
        ttnn::LayoutAttr::get(op.getContext(), ttnnLayoutEnum);

    mlir::Value device = nullptr;
    ttnn::MemoryConfigAttr memoryConfigAttr = nullptr;

    if (!mlir::tt::ttnn::isSystemBufferType(layoutAttr.getBufferType())) {
      device = ::ttnn::utils::getOrInsertDevice(rewriter, op);

      ttnn::BufferTypeAttr bufferTypeAttr = ttnn::BufferTypeAttr::get(
          op.getContext(), layoutAttr.getBufferType());
      memoryConfigAttr = ttnn::MemoryConfigAttr::get(
          op.getContext(), layoutAttr.getMemLayout(), bufferTypeAttr,
          std::nullopt);
    }

    rewriter.replaceOpWithNewOp<ttnn::ConstantOp>(
        op, this->getTypeConverter()->convertType(op.getType()), device,
        adaptor.getValue(), dTypeAttr, tensorLayoutAttr, memoryConfigAttr);

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
// Used by both BatchNormInferenceOp and BatchNormTrainingOp.
template <typename OpType, typename OpAdaptor>
static LogicalResult
checkBatchNormToTTNNLegality(OpType &op, OpAdaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  // Check if the operand is a 4-dimensional tensor.
  if (mlir::cast<RankedTensorType>(adaptor.getOperand().getType()).getRank() !=
      4) {
    return rewriter.notifyMatchFailure(
        op, "Operand must be a 4-dimensional tensor");
  }

  // We only support excluded_dimension=1 for ttnn::batch_norm
  if (adaptor.getDimension() != 1) {
    return rewriter.notifyMatchFailure(op, "We can only exclude dimension 1");
  }

  return success();
}

class BatchNormInferenceOpConversionPattern
    : public OpConversionPattern<ttir::BatchNormInferenceOp> {
public:
  using OpConversionPattern<ttir::BatchNormInferenceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::BatchNormInferenceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Check legality of the conversion.
    LogicalResult legalityResult =
        checkBatchNormToTTNNLegality(op, adaptor, rewriter);
    if (failed(legalityResult)) {
      return legalityResult;
    }

    rewriter.replaceOpWithNewOp<ttnn::BatchNormInferenceOp>(
        op, this->getTypeConverter()->convertType(op.getResult().getType()),
        adaptor.getOperand(), adaptor.getMean(), adaptor.getVariance(),
        adaptor.getEpsilon(), adaptor.getScale(), adaptor.getOffset(),
        /*memoryConfig*/ nullptr);
    return success();
  }
};
} // namespace

namespace {
class BatchNormTrainingOpConversionPattern
    : public OpConversionPattern<ttir::BatchNormTrainingOp> {
public:
  using OpConversionPattern<ttir::BatchNormTrainingOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::BatchNormTrainingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Check legality of the conversion.
    LogicalResult legalityResult =
        checkBatchNormToTTNNLegality(op, adaptor, rewriter);
    if (failed(legalityResult)) {
      return legalityResult;
    }

    // Convert result types
    auto resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());

    auto batchNormTrainingOp = rewriter.create<ttnn::BatchNormTrainingOp>(
        op.getLoc(), resultType, adaptor.getOperand(), adaptor.getRunningMean(),
        adaptor.getRunningVariance(), adaptor.getEpsilon(),
        adaptor.getMomentum(), adaptor.getScale(), adaptor.getOffset(),
        /*memoryConfig*/ nullptr);

    // TTIR expects the running mean and variance to be returned as separate
    // results.
    rewriter.replaceOp(op, {batchNormTrainingOp.getResult(),
                            batchNormTrainingOp.getRunningMean(),
                            batchNormTrainingOp.getRunningVar()});
    return success();
  }
};
} // namespace

namespace {
class RMSNormOpConversionPattern : public OpConversionPattern<ttir::RMSNormOp> {
public:
  using OpConversionPattern<ttir::RMSNormOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::RMSNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType inputType =
        mlir::cast<RankedTensorType>(adaptor.getInput().getType());

    // TTNN RMS norm only supports normalization over the last dimension.
    // We need to validate that the normalized_shape matches this constraint.
    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> normalizedShape = adaptor.getNormalizedShape();

    // For now, TTNN only support normalization over the last dimension (most
    // common case).
    if (normalizedShape.size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "TTNN RMS norm currently only supports normalization over the "
              "last dimension");
    }

    if (normalizedShape[0] != inputShape.back()) {
      return rewriter.notifyMatchFailure(
          op, "TTNN RMS norm requires normalized_shape to match the last "
              "dimension of input shape");
    }

    rewriter.replaceOpWithNewOp<ttnn::RMSNormOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), adaptor.getWeight(), adaptor.getBias(),
        adaptor.getEpsilon(), /*memoryConfig*/ nullptr);
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

    auto outputLayoutAttr =
        mlir::cast<ttnn::TTNNLayoutAttr>(op.getType().getEncoding());
    auto outputDtypeAttr =
        rewriter.getAttr<ttcore::DataTypeAttr>(outputLayoutAttr.getDataType());

    rewriter.replaceOpWithNewOp<ttnn::Conv2dOp>(
        op, getTypeConverter()->convertType(op.getResult().getType()),
        adaptor.getInput(), adaptor.getWeight(), adaptor.getBias(), device,
        inChannelsAttr, outChannelsAttr, batchSizeAttr, inputHeightAttr,
        inputWidthAttr, kernelSizeAttr, *strideAttr, paddingAttr, *dilationAttr,
        groupsAttr, outputDtypeAttr, /*conv2d_config=*/nullptr,
        /*compute_config=*/nullptr, /*conv2d_slice_config=*/nullptr);

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

    if (!adaptor.getFlattenedCompatInfo()) {
      return rewriter.notifyMatchFailure(
          op,
          "Please run the FlattenSlidingWindow pass before lowering to TTNN.");
    }
    auto flattenedCompatInfo = adaptor.getFlattenedCompatInfo();
    auto device = ::ttnn::utils::getOrInsertDevice(rewriter, op);

    auto inputTy = mlir::cast<RankedTensorType>(adaptor.getInput().getType());
    auto kernelTy = mlir::cast<RankedTensorType>(adaptor.getWeight().getType());
    auto outputTy = op.getResult().getType();

    auto batchSizeAttr =
        rewriter.getI32IntegerAttr(flattenedCompatInfo.getBatchSize());
    auto inputHeightAttr =
        rewriter.getI32IntegerAttr(flattenedCompatInfo.getInputHeight());
    auto inputWidthAttr =
        rewriter.getI32IntegerAttr(flattenedCompatInfo.getInputWidth());
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

    auto outputLayoutAttr =
        mlir::cast<ttnn::TTNNLayoutAttr>(op.getType().getEncoding());
    auto outputDtypeAttr =
        rewriter.getAttr<ttcore::DataTypeAttr>(outputLayoutAttr.getDataType());

    rewriter.replaceOpWithNewOp<ttnn::ConvTranspose2dOp>(
        op, getTypeConverter()->convertType(outputTy), adaptor.getInput(),
        adaptor.getWeight(), adaptor.getBias(), device, inChannelsAttr,
        outChannelsAttr, batchSizeAttr, inputHeightAttr, inputWidthAttr,
        kernelSizeAttr, *strideAttr, reducedPaddingAttr, *outputPaddingAttr,
        *dilationAttr, groupsAttr, outputDtypeAttr, /*conv2d_config=*/nullptr,
        /*compute_config=*/nullptr, /*memoryConfig=*/nullptr);

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
      return op.emitOpError()
             << "only supports lowering to TTNN for flattened input tensors."
             << " Please run the FlattenSlidingWindow pass before lowering to "
                "TTNN";
    }

    // Extract kernel dimensions.
    auto kernelPairOrError =
        ttmlir::utils::getPairOfInteger<int32_t>(adaptor.getKernel());
    assert(kernelPairOrError && "Expected valid kernel attribute");
    DenseI32ArrayAttr kernelSizeAttr = rewriter.getDenseI32ArrayAttr(
        {kernelPairOrError->first, kernelPairOrError->second});

    // Extract stride dimensions.
    auto stridePairOrError =
        ttmlir::utils::getPairOfInteger<int32_t>(adaptor.getStride());
    assert(stridePairOrError && "Expected valid stride attribute");
    DenseI32ArrayAttr strideAttr = rewriter.getDenseI32ArrayAttr(
        {stridePairOrError->first, stridePairOrError->second});

    // Extract dilation dimensions.
    auto dilationPairOrError =
        ttmlir::utils::getPairOfInteger<int32_t>(adaptor.getDilation());
    assert(dilationPairOrError && "Expected valid dilation attribute");
    DenseI32ArrayAttr dilationAttr = rewriter.getDenseI32ArrayAttr(
        {dilationPairOrError->first, dilationPairOrError->second});

    // TTNN only supports lowering of AvgPool2dOp with dilation of (1, 1).
    if constexpr (std::is_same_v<TTIROpTy, ttir::AvgPool2dOp>) {
      if (dilationPairOrError->first != 1 || dilationPairOrError->second != 1) {
        return op.emitOpError()
               << "only supports lowering to TTNN for dilation of (1, 1)";
      }
    }

    // Extract padding values.
    auto paddingQuad =
        ttmlir::utils::getQuadrupleOfInteger<int32_t>(adaptor.getPadding());
    assert(paddingQuad && "Expected valid padding attribute");
    int32_t paddingTop = std::get<0>(*paddingQuad);
    int32_t paddingLeft = std::get<1>(*paddingQuad);
    int32_t paddingBottom = std::get<2>(*paddingQuad);
    int32_t paddingRight = std::get<3>(*paddingQuad);

    DenseI32ArrayAttr paddingAttr;
    // If padding is symmetric along both spatial dimensions, we can use the
    // {height, width} definition of padding.
    if (paddingBottom == paddingTop && paddingLeft == paddingRight) {
      paddingAttr = rewriter.getDenseI32ArrayAttr({paddingTop, paddingLeft});
    } else {
      // Otherwise pass {top, left, bottom, right}
      paddingAttr = rewriter.getDenseI32ArrayAttr(
          {paddingTop, paddingLeft, paddingBottom, paddingRight});
    }

    auto batchSize = adaptor.getFlattenedCompatInfo().getBatchSize();
    constexpr unsigned int CHANNEL_DIM = 3;
    auto channels = op.getInput().getType().getDimSize(CHANNEL_DIM);
    if constexpr (std::is_same_v<TTIROpTy, ttir::AvgPool2dOp>) {
      rewriter.replaceOpWithNewOp<TTNNOpTy>(
          op, this->getTypeConverter()->convertType(op.getResult().getType()),
          adaptor.getInput(), batchSize,
          adaptor.getFlattenedCompatInfo().getInputHeight(),
          adaptor.getFlattenedCompatInfo().getInputWidth(), channels,
          kernelSizeAttr, strideAttr, paddingAttr, dilationAttr,
          /*memory_config=*/nullptr,
          /* applied_shard_scheme=*/nullptr, adaptor.getCeilMode(),
          /* in_place_halo=*/false, adaptor.getCountIncludePad());
    } else if constexpr (std::is_same_v<TTIROpTy, ttir::MaxPool2dOp>) {
      rewriter.replaceOpWithNewOp<TTNNOpTy>(
          op, this->getTypeConverter()->convertType(op.getResult().getType()),
          adaptor.getInput(), batchSize,
          adaptor.getFlattenedCompatInfo().getInputHeight(),
          adaptor.getFlattenedCompatInfo().getInputWidth(), channels,
          kernelSizeAttr, strideAttr, paddingAttr, dilationAttr,
          /*memory_config=*/nullptr,
          /* applied_shard_scheme=*/nullptr, adaptor.getCeilMode(),
          /* in_place_halo=*/false);
    } else {
      llvm_unreachable("Pool2dOp must be AvgPool2dOp or MaxPool2dOp");
    }

    return success();
  }
};
} // namespace

namespace {
class GlobalAvgPool2dOpConversionPattern
    : public OpConversionPattern<ttir::GlobalAvgPool2dOp> {
public:
  using OpConversionPattern<ttir::GlobalAvgPool2dOp>::OpConversionPattern;
  using OpAdaptor = typename ttir::GlobalAvgPool2dOp::Adaptor;

  LogicalResult
  matchAndRewrite(ttir::GlobalAvgPool2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // GlobalAvgPool2d is essentially a much simpler operation than AvgPool2d
    // andd MaxPool2d. under the hood, we just perform a sum reduction across
    // the spatial dimensions on metal and we dont take stride/padding/dilation
    // as params. That is why we don't inherit from the
    // Pooling2dOpConversionPattern.

    // Extract output layout and dtype from the result type
    auto outputLayoutAttr =
        mlir::cast<ttnn::TTNNLayoutAttr>(op.getType().getEncoding());
    auto outputDtypeAttr =
        rewriter.getAttr<ttcore::DataTypeAttr>(outputLayoutAttr.getDataType());

    rewriter.replaceOpWithNewOp<ttnn::GlobalAvgPool2dOp>(
        op, this->getTypeConverter()->convertType(op.getResult().getType()),
        adaptor.getInput(),
        /*memory_config=*/nullptr, outputDtypeAttr);

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
    ttcore::DataType outputDataType = outputLayoutAttr.getDataType();

    rewriter.replaceOpWithNewOp<ttnn::TypecastOp>(
        op, this->getTypeConverter()->convertType(resultType),
        adaptor.getInput(), outputDataType);
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
    ttnn::TTNNLayoutAttr ttnnLayoutAttr =
        mlir::cast<ttnn::TTNNLayoutAttr>(outputType.getEncoding());

    ttcore::DataTypeAttr dtypeAttr = rewriter.getAttr<ttcore::DataTypeAttr>(
        ttcore::elementTypeToDataType(outputType.getElementType()));
    Value device = mlir::tt::ttnn::utils::getOrInsertDevice(rewriter, op);

    ttnn::Layout ttnnLayoutEnum = ttnn::Layout::RowMajor;

    if (ttnnLayoutAttr.isTiled()) {
      ttnnLayoutEnum = ttnn::Layout::Tile;
    }
    ttnn::LayoutAttr tensorLayoutAttr =
        ttnn::LayoutAttr::get(op.getContext(), ttnnLayoutEnum);

    ttnn::MemoryConfigAttr memConfigAttr =
        rewriter.getAttr<ttnn::MemoryConfigAttr>(
            ttnnLayoutAttr.getMemLayout(),
            rewriter.getAttr<ttnn::BufferTypeAttr>(
                ttnnLayoutAttr.getBufferType()),
            std::nullopt);

    rewriter.replaceOpWithNewOp<ttnn::ArangeOp>(
        op, outputType, device, adaptor.getStart(), adaptor.getEnd(),
        adaptor.getStep(), dtypeAttr, tensorLayoutAttr, memConfigAttr);

    return success();
  }
};
} // namespace

namespace {
class RandOpConversionPattern : public OpConversionPattern<ttir::RandOp> {
public:
  using OpConversionPattern<ttir::RandOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::RandOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Get ttnn::TTNNLayoutAttr of the result type.
    //
    ttnn::TTNNLayoutAttr layoutAttr = mlir::cast<ttnn::TTNNLayoutAttr>(
        op.getResult().getType().getEncoding());

    mlir::tt::ttcore::DataType dtype =
        mlir::tt::ttcore::elementTypeToDataType(adaptor.getDtype());
    ttcore::DataTypeAttr dTypeAttr =
        ttcore::DataTypeAttr::get(rewriter.getContext(), dtype);

    ttnn::Layout ttnnLayoutEnum =
        layoutAttr.isTiled() ? ttnn::Layout::Tile : ttnn::Layout::RowMajor;
    ttnn::LayoutAttr tensorLayoutAttr =
        ttnn::LayoutAttr::get(op.getContext(), ttnnLayoutEnum);

    auto device = ::ttnn::utils::getOrInsertDevice(rewriter, op);

    ttnn::BufferTypeAttr bufferTypeAttr =
        ttnn::BufferTypeAttr::get(op.getContext(), layoutAttr.getBufferType());
    ttnn::MemoryConfigAttr memoryConfigAttr =
        ttnn::MemoryConfigAttr::get(op.getContext(), layoutAttr.getMemLayout(),
                                    bufferTypeAttr, std::nullopt);

    ttnn::ShapeAttr sizeAttr = ttnn::ShapeAttr::get(
        rewriter.getContext(), op.getResult().getType().getShape());

    rewriter.replaceOpWithNewOp<ttnn::RandOp>(
        op, this->getTypeConverter()->convertType(op.getType()), device,
        sizeAttr, adaptor.getLowAttr(), adaptor.getHighAttr(),
        adaptor.getSeedAttr(), dTypeAttr, tensorLayoutAttr, memoryConfigAttr);
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
    if (!hasValidInsertedWindowDims(op)) {
      return rewriter.notifyMatchFailure(
          op, "ttnn and tt-metal have limited scatter support. Inserted window "
              "dimenstion must be 1 in the input tensor shape.");
    }
    // The ttnn interface has the inverse inputs of the TTIR dialect op (which
    // matches torch ops).
    rewriter.replaceOpWithNewOp<ttnn::ScatterOp>(
        op, adaptor.getOutput().getType(), adaptor.getUpdate(),
        adaptor.getInput());

    return success();
  }

private:
  bool hasValidInsertedWindowDims(ttir::ScatterOp op) const {
    ArrayRef<int64_t> inputShape = op.getInput().getType().getShape();

    for (uint64_t insertedWindowDims : op.getInsertedWindowDims()) {
      if (insertedWindowDims < inputShape.size() &&
          inputShape[insertedWindowDims] != 1) {
        return false;
      }
    }

    return true;
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

// Lowering of TTIR `collective_broadcast` op to a sequence of TTNN
// `point_to_point` ops.
//
// Currently, TTNN does not have a native CollectiveBroadcast op. Instead,
// we lower the collective broadcast operation into multiple point-to-point
// transfers based on the replica group configuration.
//
// For each replica group, the first device ID is treated as the source,
// and a PointToPointOp is created for each remaining target in that group.
// The output of each PointToPointOp overwrites the previous one until the
// last one is used to replace the original op's result.
namespace {
class CollectiveBroadcastOpConversionPattern
    : public OpConversionPattern<ttir::CollectiveBroadcastOp> {
public:
  using OpConversionPattern<ttir::CollectiveBroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::CollectiveBroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ::mlir::RankedTensorType inputType =
        mlir::cast<::mlir::RankedTensorType>(op.getInput().getType());
    auto meshDevice = ttcore::lookupDevice(op);
    llvm::SmallVector<int64_t> meshShape{meshDevice.getMeshShape()};

    Value finalValue;
    auto replicaGroups = ttmlir::utils::denseElementsAttrTo2D<int64_t>(
        adaptor.getReplicaGroups());

    // For each replica group, broadcast the first device's tensor to all
    // others.
    for (const auto &group : replicaGroups) {
      auto sourceCoord = rewriter.getDenseI64ArrayAttr(
          ttmlir::utils::linearIdToCoord(group[0], meshShape));
      for (const auto &targetId : group) {
        finalValue = rewriter.create<ttnn::PointToPointOp>(
            op.getLoc(), inputType, adaptor.getInput(), sourceCoord,
            rewriter.getDenseI64ArrayAttr(
                ttmlir::utils::linearIdToCoord(targetId, meshShape)),
            finalValue);
      }
    }

    // Replace the original collective_broadcast op with the final output value.
    rewriter.replaceOp(op, finalValue);

    return success();
  }
};
} // namespace

// Utility function to get data type for quantized types.
static ttcore::DataTypeAttr getDataType(mlir::Value val,
                                        ConversionPatternRewriter &rewriter,
                                        const TypeConverter *typeConverter) {
  ttnn::TTNNLayoutAttr outputLayoutAttr = mlir::cast<ttnn::TTNNLayoutAttr>(
      mlir::cast<RankedTensorType>(typeConverter->convertType(val.getType()))
          .getEncoding());
  ttcore::DataType dtype = outputLayoutAttr.getDataType();
  return ttcore::DataTypeAttr::get(rewriter.getContext(), dtype);
}

namespace {
template <typename OpTy, typename TTNNOpTy>
class QuantizationOpConversionPattern : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    ttcore::DataTypeAttr outputDataType =
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
    ttcore::DataTypeAttr outputDataType =
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

class ConcatenateHeadsOpConversionPattern
    : public OpConversionPattern<ttir::ConcatenateHeadsOp> {
public:
  using OpConversionPattern<ttir::ConcatenateHeadsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ConcatenateHeadsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ttnn::ConcatenateHeadsOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), /*memory_config=*/nullptr);
    return success();
  }
};

class SplitQueryKeyValueAndSplitHeadsOpConversionPattern
    : public OpConversionPattern<ttir::SplitQueryKeyValueAndSplitHeadsOp> {
public:
  using OpConversionPattern<
      ttir::SplitQueryKeyValueAndSplitHeadsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::SplitQueryKeyValueAndSplitHeadsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert result types
    auto queryType =
        this->getTypeConverter()->convertType(op.getQuery().getType());
    auto keyType = this->getTypeConverter()->convertType(op.getKey().getType());
    auto valueType =
        this->getTypeConverter()->convertType(op.getValue().getType());

    // Create the TTNN op with 3 results
    auto ttnnOp = rewriter.create<ttnn::SplitQueryKeyValueAndSplitHeadsOp>(
        op.getLoc(), TypeRange{queryType, keyType, valueType},
        adaptor.getInputTensor(), adaptor.getKvInputTensor(),
        adaptor.getNumHeadsAttr(), adaptor.getNumKvHeadsAttr(),
        adaptor.getTransposeKeyAttr(),
        /*memory_config=*/nullptr);

    // Replace the original op with the three results
    rewriter.replaceOp(op, ttnnOp.getResults());
    return success();
  }
};
} // namespace

namespace {
class ScaledDotProductAttentionDecodeOpConversionPattern
    : public OpConversionPattern<ttir::ScaledDotProductAttentionDecodeOp> {
public:
  using OpConversionPattern<
      ttir::ScaledDotProductAttentionDecodeOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ttir::ScaledDotProductAttentionDecodeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FloatAttr scaleAttr = op.getScaleAttr() ? op.getScaleAttr() : nullptr;
    rewriter.replaceOpWithNewOp<ttnn::ScaledDotProductAttentionDecodeOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getQuery(), adaptor.getKey(), adaptor.getValue(),
        adaptor.getIsCausal(), adaptor.getAttentionMask(),
        adaptor.getCurPosTensor(), adaptor.getAttentionSink(), scaleAttr,
        /*memory_config=*/nullptr);
    return success();
  }
};
} // namespace

namespace {
class ScaledDotProductAttentionOpConversionPattern
    : public OpConversionPattern<ttir::ScaledDotProductAttentionOp> {
public:
  using OpConversionPattern<
      ttir::ScaledDotProductAttentionOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ttir::ScaledDotProductAttentionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FloatAttr scaleAttr = op.getScaleAttr() ? op.getScaleAttr() : nullptr;
    rewriter.replaceOpWithNewOp<ttnn::ScaledDotProductAttentionOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getQuery(), adaptor.getKey(), adaptor.getValue(),
        adaptor.getAttentionMask(), op.getIsCausal(), scaleAttr,
        /*memory_config=*/nullptr);
    return success();
  }
};
} // namespace

// This rewrite pattern lowers a ttir.all_to_all op into a sequence of
// ttnn.slice_static, ttnn.point_to_point, and ttnn.concat ops.
//
// The goal is to reproduce the behavior expected from StableHLO's all_to_all,
// which involves redistributing data slices across devices according to the
// replica group configuration.
//
// This lowering performs the following steps:
// 1. Slice the input tensor along the split_dimension.
// 2. Use point_to_point ops to exchange the slices between devices according to
//    the replica group configuration.
// 3. Concatenate the received slices along the concat_dimension to reconstruct
// the final output tensor. Please refer to the StableHLO documentation for more
// details: https://openxla.org/stablehlo/spec#all_to_all
namespace {
class AllToAllOpConversionPattern
    : public OpConversionPattern<ttir::AllToAllOp> {
public:
  using OpConversionPattern<ttir::AllToAllOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::AllToAllOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ::mlir::RankedTensorType inputType =
        mlir::cast<::mlir::RankedTensorType>(op.getInput().getType());
    auto inputShape = inputType.getShape();
    int32_t splitDim = op.getSplitDim();
    auto replicaGroups =
        ttmlir::utils::denseElementsAttrTo2D<int64_t>(op.getReplicaGroups());
    int32_t splitCount = static_cast<int32_t>(replicaGroups[0].size());

    // Step 1: Slice the input tensor along the split dimension.
    // Each slice corresponds to a portion that will be sent to another device.
    int32_t splitSize = inputShape[splitDim] / splitCount;
    llvm::SmallVector<int64_t> slicedShape(inputShape.begin(),
                                           inputShape.end());
    slicedShape[splitDim] = splitSize;
    RankedTensorType sliceOutputType =
        ttnn::utils::RankedTensorTypeFactory::create(inputType, slicedShape);
    llvm::SmallVector<Value> sliceOpResults;
    llvm::SmallVector<int32_t> begins(inputShape.size(), 0);
    llvm::SmallVector<int32_t> ends(inputShape.begin(), inputShape.end());
    llvm::SmallVector<int32_t> steps(inputShape.size(), 1);
    for (int32_t sliceIdx = 0; sliceIdx < splitCount; sliceIdx++) {
      begins[splitDim] = sliceIdx * splitSize;
      ends[splitDim] = (sliceIdx + 1) * splitSize;

      // Create a slice for this range
      ttnn::SliceStaticOp sliceOp = rewriter.create<ttnn::SliceStaticOp>(
          loc, sliceOutputType, op.getInput(), rewriter.getI32ArrayAttr(begins),
          rewriter.getI32ArrayAttr(ends), rewriter.getI32ArrayAttr(steps));
      sliceOpResults.push_back(sliceOp.getResult());
    }
    // Step 2: Reorganize sliced data using PointToPoint communication.
    // For each group of devices, perform pairwise sends via PointToPoint ops.
    // Each sender sends its slices to all devices in the group (including
    // itself).

    // Buffers to hold the output for each device (initialized as empty).
    llvm::SmallVector<Value> reorgBuffers(splitCount);

    auto meshShape = ttcore::lookupDevice(op).getMeshShape();
    // for each group of devices,
    for (const auto &group : replicaGroups) {
      // for each device in the group, send its slices to all other devices in
      // the group
      for (size_t senderIdx = 0; senderIdx < group.size(); senderIdx++) {
        auto senderCoord = rewriter.getDenseI64ArrayAttr(
            ttmlir::utils::linearIdToCoord(group[senderIdx], meshShape));
        for (size_t receiverIdx = 0; receiverIdx < group.size();
             receiverIdx++) {
          auto receiverCoord = rewriter.getDenseI64ArrayAttr(
              ttmlir::utils::linearIdToCoord(group[receiverIdx], meshShape));
          reorgBuffers[senderIdx] = rewriter.create<ttnn::PointToPointOp>(
              loc, sliceOpResults[senderIdx].getType(),
              sliceOpResults[receiverIdx], senderCoord, receiverCoord,
              reorgBuffers[senderIdx]);
        }
      }
    }

    // Step 3: Concatenate all received slices along the concat dimension.
    // This forms the final output tensor after the all-to-all reorganization.
    rewriter.replaceOpWithNewOp<ttnn::ConcatOp>(op, op.getType(), reorgBuffers,
                                                op.getConcatDim(),
                                                /*memory_config=*/nullptr);

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
           ElementwiseBinaryOpConversionPattern<ttir::AddOp, ttnn::AddOp>,
           ElementwiseBinaryOpConversionPattern<ttir::LogicalRightShiftOp, ttnn::LogicalRightShiftOp>,
           ElementwiseBinaryOpConversionPattern<ttir::SubtractOp, ttnn::SubtractOp>,
           ElementwiseBinaryOpConversionPattern<ttir::MultiplyOp, ttnn::MultiplyOp>,
           ElementwiseBinaryOpConversionPattern<ttir::DivOp, ttnn::DivideOp>,
           ElementwiseBinaryOpConversionPattern<ttir::EqualOp, ttnn::EqualOp>,
           ElementwiseBinaryOpConversionPattern<ttir::NotEqualOp, ttnn::NotEqualOp>,
           ElementwiseBinaryOpConversionPattern<ttir::GreaterEqualOp, ttnn::GreaterEqualOp>,
           ElementwiseBinaryOpConversionPattern<ttir::GreaterThanOp, ttnn::GreaterThanOp>,
           ElementwiseBinaryOpConversionPattern<ttir::LessEqualOp, ttnn::LessEqualOp>,
           ElementwiseBinaryOpConversionPattern<ttir::LessThanOp, ttnn::LessThanOp>,
           ElementwiseBinaryOpConversionPattern<ttir::LogicalAndOp, ttnn::LogicalAndOp>,
           ElementwiseBinaryOpConversionPattern<ttir::LogicalOrOp, ttnn::LogicalOrOp>,
           ElementwiseBinaryOpConversionPattern<ttir::LogicalXorOp, ttnn::LogicalXorOp>,
           ElementwiseOpConversionPattern<ttir::BitwiseAndOp, ttnn::BitwiseAndOp>,
           ElementwiseOpConversionPattern<ttir::LogicalLeftShiftOp, ttnn::LogicalLeftShiftOp>,
           ElementwiseOpConversionPattern<ttir::BitwiseOrOp, ttnn::BitwiseOrOp>,
           ElementwiseOpConversionPattern<ttir::BitwiseXorOp, ttnn::BitwiseXorOp>,
           ElementwiseOpConversionPattern<ttir::MaximumOp, ttnn::MaximumOp>,
           ElementwiseOpConversionPattern<ttir::MinimumOp, ttnn::MinimumOp>,
           ElementwiseOpConversionPattern<ttir::RemainderOp, ttnn::RemainderOp>,
           ElementwiseOpConversionPattern<ttir::Atan2Op, ttnn::Atan2Op>,
           ElementwiseOpConversionPattern<ttir::AbsOp, ttnn::AbsOp>,
           ElementwiseOpConversionPattern<ttir::CbrtOp, ttnn::CbrtOp>,
           ElementwiseOpConversionPattern<ttir::FloorOp, ttnn::FloorOp>,
           ElementwiseOpConversionPattern<ttir::IsFiniteOp, ttnn::IsFiniteOp>,
           ElementwiseOpConversionPattern<ttir::LogicalNotOp, ttnn::LogicalNotOp>,
           ElementwiseOpConversionPattern<ttir::BitwiseNotOp, ttnn::BitwiseNotOp>,
           ElementwiseOpConversionPattern<ttir::NegOp, ttnn::NegOp>,
           ElementwiseOpConversionPattern<ttir::ReluOp, ttnn::ReluOp>,
           ElementwiseOpConversionPattern<ttir::Relu6Op, ttnn::Relu6Op>,
           ElementwiseOpConversionPattern<ttir::GeluOp, ttnn::GeluOp>,
           ElementwiseOpConversionPattern<ttir::SqrtOp, ttnn::SqrtOp>,
           ElementwiseOpConversionPattern<ttir::RsqrtOp, ttnn::RsqrtOp>,
           ElementwiseOpConversionPattern<ttir::SignOp, ttnn::SignOp>,
           ElementwiseOpConversionPattern<ttir::SigmoidOp, ttnn::SigmoidOp>,
           ElementwiseOpConversionPattern<ttir::SiluOp, ttnn::SiluOp>,
           ElementwiseOpConversionPattern<ttir::Log1pOp, ttnn::Log1pOp>,
           ElementwiseOpConversionPattern<ttir::ReciprocalOp, ttnn::ReciprocalOp>,
           ElementwiseOpConversionPattern<ttir::ExpOp, ttnn::ExpOp>,
           ElementwiseOpConversionPattern<ttir::ErfOp, ttnn::ErfOp>,
           ElementwiseOpConversionPattern<ttir::ErfcOp, ttnn::ErfcOp>,
           ElementwiseOpConversionPattern<ttir::LogOp, ttnn::LogOp>,
           ElementwiseOpConversionPattern<ttir::CeilOp, ttnn::CeilOp>,
           ElementwiseOpConversionPattern<ttir::SinOp, ttnn::SinOp>,
           ElementwiseOpConversionPattern<ttir::CosOp, ttnn::CosOp>,
           ElementwiseOpConversionPattern<ttir::Expm1Op, ttnn::Expm1Op>,
           ElementwiseOpConversionPattern<ttir::WhereOp, ttnn::WhereOp>,
           ElementwiseOpConversionPattern<ttir::TanOp, ttnn::TanOp>,
           ElementwiseOpConversionPattern<ttir::TanhOp, ttnn::TanhOp>,
           ElementwiseOpConversionPattern<ttir::AtanOp, ttnn::AtanOp>,
           Pooling2dOpConversionPattern<ttir::MaxPool2dOp, ttnn::MaxPool2dOp>,
           Pooling2dOpConversionPattern<ttir::AvgPool2dOp, ttnn::AvgPool2dOp>,
           GlobalAvgPool2dOpConversionPattern,
           ReductionOpConversionPattern<ttir::SumOp, ttnn::SumOp>,
           ReductionOpConversionPattern<ttir::MeanOp, ttnn::MeanOp>,
           ReductionOpConversionPattern<ttir::MaxOp, ttnn::MaxOp>,
           ReductionOpConversionPattern<ttir::MinOp, ttnn::MinOp>,
           ReductionProdOpConversionPattern,
           ReductionArgMaxOpConversionPattern,
           ElementwiseUnaryWithFloatParameterOpConversionPattern<ttir::LeakyReluOp, ttnn::LeakyReluOp>,
           BroadcastOpConversionPattern,
           PadOpConversionPattern,
           PowOpConversionPattern,
           EmbeddingOpConversionPattern,
           EmbeddingBackwardOpConversionPattern,
           RepeatOpConversionPattern,
           CumSumOpConversionPattern,
           RepeatInterleaveOpConversionPattern,
           SoftmaxOpConversionPattern,
           SortOpConversionPattern,
           TypecastOpConversionPattern,
           ClampOpConversionPattern<ttir::ClampScalarOp, ttnn::ClampScalarOp>,
           ClampOpConversionPattern<ttir::ClampTensorOp, ttnn::ClampTensorOp>,
           ConcatOpConversionPattern,
           ReshapeOpConversionPattern,
           SliceOpConversionPattern<ttir::SliceStaticOp, ttnn::SliceStaticOp>,
           SliceOpConversionPattern<ttir::SliceDynamicOp, ttnn::SliceDynamicOp>,
           SqueezeOpConversionPattern,
           UnsqueezeOpConversionPattern,
           ConstantOpConversionPattern,
           LinearOpConversionPattern,
           BatchNormInferenceOpConversionPattern,
           BatchNormTrainingOpConversionPattern,
           RMSNormOpConversionPattern,
           MatmulOpConversionPattern,
           Conv2dOpConversionPattern,
           ConvTranspose2dOpConversionPattern,
           MeshShardOpConversionPattern,
           AllReduceOpConversionPattern,
           AllGatherOpConversionPattern,
           ReduceScatterOpConversionPattern,
           CollectivePermuteOpConversionPattern,
           ArangeOpConversionPattern,
           RandOpConversionPattern,
           UpdateCacheOpConversionPattern,
           FillCacheOpConversionPattern,
           ScatterOpConversionPattern,
           PermuteOpConversionPattern,
           UpsampleOpConversionPattern,
           AllToAllOpConversionPattern,
           CollectiveBroadcastOpConversionPattern,
           ConcatenateHeadsOpConversionPattern,
           ScaledDotProductAttentionOpConversionPattern,
           ScaledDotProductAttentionDecodeOpConversionPattern,
           SplitQueryKeyValueAndSplitHeadsOpConversionPattern
           >(typeConverter, ctx);
  // ANCHOR_END: op_rewriter_pattern_set
  // clang-format on
}

} // namespace mlir::tt
