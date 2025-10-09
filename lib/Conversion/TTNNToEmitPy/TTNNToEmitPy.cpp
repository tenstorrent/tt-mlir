// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitPy/TTNNToEmitPy.h"

#include "ttmlir/Conversion/TTNNToEmitPy/EmitPyConversion.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyTypes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <optional>

using namespace mlir;
using namespace mlir::tt;
using ttnn_to_emitpy::operator|;

// Base class for TTNN to EmitPy OpConversionPattern.
//
namespace {
template <typename SourceOp>
class TTNNToEmitPyBaseOpConversionPattern
    : public OpConversionPattern<SourceOp> {
public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using Adaptor = typename SourceOp::Adaptor;

private:
  virtual std::string getPrefixSearchPattern() const { return "ttnn."; }
  virtual std::string getPrefixSwapPattern() const { return "ttnn."; }

public:
  // Converts op name by removing the prefixSearchPattern with
  // prefixSwapPattern, e.g. "ttnn." with "tt_metal."
  //
  std::string convertOpName(SourceOp op) const {
    auto name = op.getOperationName();
    assert(name.starts_with(getPrefixSearchPattern()) &&
           "TTNNToEmitPyBaseOpConversionPattern only supports ops from the "
           "TTNN dialect");

    if (getPrefixSearchPattern() == getPrefixSwapPattern()) {
      return name.str();
    }

    // Exchange search with swap pattern.
    //
    return name.str().replace(0, getPrefixSearchPattern().size(),
                              getPrefixSwapPattern());
  }
};
} // namespace

// ClampOpConversionPattern conversion pattern
//
namespace {
template <typename SourceOp>
class ClampOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<SourceOp> {
private:
  std::string getPrefixSearchPattern() const override {
    if constexpr (std::is_same_v<SourceOp, ::mlir::tt::ttnn::ClampScalarOp>) {
      return "ttnn.clamp_scalar";
    } else if constexpr (std::is_same_v<SourceOp,
                                        mlir::tt::ttnn::ClampTensorOp>) {
      return "ttnn.clamp_tensor";
    }

    llvm_unreachable("Operation not supported.");
  }

  std::string getPrefixSwapPattern() const override { return "ttnn.clamp"; }

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      SourceOp>::TTNNToEmitPyBaseOpConversionPattern;
  using Adaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<SourceOp> emitter(srcOp, adaptor,
                                                        rewriter);
    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getMin()),
        emitter.emit(srcOp.getMax()),
        emitter.emit(srcOp.getMemoryConfig() |
                         emitter.getMemoryConfig(srcOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// EltwiseUnaryOp conversion pattern
//
namespace {
template <typename TTNNOpTy, typename OpAdaptor = typename TTNNOpTy::Adaptor>
class EltwiseUnaryOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<TTNNOpTy> {
public:
  using TTNNToEmitPyBaseOpConversionPattern<
      TTNNOpTy>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(TTNNOpTy eltwiseUnaryOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<TTNNOpTy> emitter(eltwiseUnaryOp, adaptor,
                                                        rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(eltwiseUnaryOp.getInput()),
        emitter.emit(eltwiseUnaryOp.getMemoryConfig() |
                         emitter.getMemoryConfig(eltwiseUnaryOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// EltwiseUnaryWithFastAndApproximateModeOp conversion pattern
//
namespace {
template <typename TTNNOpTy, typename OpAdaptor = typename TTNNOpTy::Adaptor>
class EltwiseUnaryWithFastAndApproximateModeOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<TTNNOpTy> {
public:
  using TTNNToEmitPyBaseOpConversionPattern<
      TTNNOpTy>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(TTNNOpTy eltwiseUnaryOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<TTNNOpTy> emitter(eltwiseUnaryOp, adaptor,
                                                        rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(eltwiseUnaryOp.getInput()),
        /*parameter=*/emitter.emit(false, "fast_and_approximate_mode"),
        emitter.emit(eltwiseUnaryOp.getMemoryConfig() |
                         emitter.getMemoryConfig(eltwiseUnaryOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// EltwiseUnaryWithVectorAndFastAndApproximateModeOp conversion pattern
//
namespace {
template <typename TTNNOpTy, typename OpAdaptor = typename TTNNOpTy::Adaptor>
class EltwiseUnaryWithVectorAndFastAndApproximateModeOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<TTNNOpTy> {
public:
  using TTNNToEmitPyBaseOpConversionPattern<
      TTNNOpTy>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(TTNNOpTy eltwiseUnaryOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<TTNNOpTy> emitter(eltwiseUnaryOp, adaptor,
                                                        rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(eltwiseUnaryOp.getInput()),
        emitter.emit(static_cast<int>(::ttnn::operations::unary::VecMode::RC),
                     "vector_mode"),
        /*parameter=*/emitter.emit(false, "fast_and_approximate_mode"),
        emitter.emit(eltwiseUnaryOp.getMemoryConfig() |
                         emitter.getMemoryConfig(eltwiseUnaryOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// ElementwiseUnaryWithFloatParameterOp conversion pattern
//
namespace {
template <typename SourceOp>
class EltwiseUnaryWithFloatParameterOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<SourceOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      SourceOp>::TTNNToEmitPyBaseOpConversionPattern;
  using Adaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<SourceOp> emitter(srcOp, adaptor,
                                                        rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getParameter()),
        emitter.emit(std::nullopt | emitter.getMemoryConfig(srcOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// EltwiseBinaryOp conversion pattern
//
namespace {
template <typename TTNNOpTy, typename OpAdaptor = typename TTNNOpTy::Adaptor>
class EltwiseBinaryOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<TTNNOpTy> {
public:
  using TTNNToEmitPyBaseOpConversionPattern<
      TTNNOpTy>::TTNNToEmitPyBaseOpConversionPattern;
  LogicalResult

  matchAndRewrite(TTNNOpTy eltwiseBinaryOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<TTNNOpTy> emitter(eltwiseBinaryOp,
                                                        adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(eltwiseBinaryOp.getLhs()),
        emitter.emit(eltwiseBinaryOp.getRhs()),
        emitter.emit(eltwiseBinaryOp.getDtype(), "dtype"),
        emitter.emit(eltwiseBinaryOp.getMemoryConfig() |
                         emitter.getMemoryConfig(eltwiseBinaryOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// EltwiseBinaryCompositeOp conversion pattern
//
namespace {
template <typename TTNNOpTy, typename OpAdaptor = typename TTNNOpTy::Adaptor>
class EltwiseBinaryCompositeOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<TTNNOpTy> {
public:
  using TTNNToEmitPyBaseOpConversionPattern<
      TTNNOpTy>::TTNNToEmitPyBaseOpConversionPattern;
  LogicalResult

  matchAndRewrite(TTNNOpTy eltwiseBinaryOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<TTNNOpTy> emitter(eltwiseBinaryOp,
                                                        adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(eltwiseBinaryOp.getLhs()),
        emitter.emit(eltwiseBinaryOp.getRhs()),
        emitter.emit(eltwiseBinaryOp.getMemoryConfig() |
                         emitter.getMemoryConfig(eltwiseBinaryOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// EltwiseBinaryCompositeWithDTypeOp conversion pattern
//
namespace {
template <typename TTNNOpTy, typename OpAdaptor = typename TTNNOpTy::Adaptor>
class EltwiseBinaryCompositeWithDTypeOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<TTNNOpTy> {
public:
  using TTNNToEmitPyBaseOpConversionPattern<
      TTNNOpTy>::TTNNToEmitPyBaseOpConversionPattern;
  LogicalResult

  matchAndRewrite(TTNNOpTy eltwiseBinaryOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<TTNNOpTy> emitter(eltwiseBinaryOp,
                                                        adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(eltwiseBinaryOp.getLhs()),
        emitter.emit(eltwiseBinaryOp.getRhs()),
        emitter.emit(std::nullopt, "dtype"),
        emitter.emit(eltwiseBinaryOp.getMemoryConfig() |
                         emitter.getMemoryConfig(eltwiseBinaryOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// PowScalar op conversion pattern
//
// Currently, it has to insert nullopts for some parameters that are not
// modelled in the dialect (memcfg).
//
namespace {
class PowScalarOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::PowScalarOp> {

private:
  std::string getPrefixSearchPattern() const override {
    return "ttnn.pow_scalar";
  }

  std::string getPrefixSwapPattern() const override { return "ttnn::pow"; }

  template <typename ExponentT>
  LogicalResult matchAndRewriteImpl(mlir::tt::ttnn::PowScalarOp srcOp,
                                    ExponentT exponent, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
    static_assert(llvm::is_one_of<ExponentT, float, uint32_t>::value,
                  "ExponentT must be float or uint32_t");

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::PowScalarOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getLhs()),
        emitter.template emit<ExponentT>(exponent),
        emitter.emit(std::nullopt | emitter.getMemoryConfig(srcOp.getResult())),
    };

    emitter.replaceOp(*this, args);
    return mlir::success();
  }

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::PowScalarOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::PowScalarOp srcOp,
                  mlir::tt::ttnn::PowScalarOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (auto exponentAttr = mlir::dyn_cast<FloatAttr>(srcOp.getRhs())) {
      auto exponent = exponentAttr.getValue().convertToFloat();
      return matchAndRewriteImpl(srcOp, exponent, adaptor, rewriter);
    }
    if (auto exponentAttr = mlir::dyn_cast<IntegerAttr>(srcOp.getRhs())) {
      auto exponent =
          static_cast<uint32_t>(exponentAttr.getValue().getSExtValue());
      return matchAndRewriteImpl(srcOp, exponent, adaptor, rewriter);
    }
    return failure();
  }
};
} // namespace

// Eltwise Ternary op conversion pattern
//
namespace {
template <typename SourceOp>
class EltwiseTernaryOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<SourceOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      SourceOp>::TTNNToEmitPyBaseOpConversionPattern;
  using Adaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp ternaryOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<SourceOp> emitter(ternaryOp, adaptor,
                                                        rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(ternaryOp.getFirst()),
        emitter.emit(ternaryOp.getSecond()),
        emitter.emit(ternaryOp.getThird()),
        emitter.emit(ternaryOp.getMemoryConfig() |
                         emitter.getMemoryConfig(ternaryOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// ConstantOp conversion pattern
//
namespace {
class ConstantOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::ConstantOp> {
private:
  std::string getPrefixSearchPattern() const override {
    return "ttnn.constant";
  }
  std::string getPrefixSwapPattern() const override { return "ttnn.Tensor"; }

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::ConstantOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::ConstantOp constantOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::ConstantOp> emitter(
        constantOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(constantOp.getValue()),
        emitter.emit(constantOp.getResult().getType().getShape()),
        emitter.emit(constantOp.getDtype()),
        emitter.emit(constantOp.getLayout()),
    };

    if (constantOp.getDevice()) {
      args.push_back(emitter.emit(constantOp.getDevice()));
      args.push_back(emitter.emit(constantOp.getMemoryConfig()));
    }

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// MatmulOp conversion pattern
//
namespace {
class MatmulOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::MatmulOp> {
public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::MatmulOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::MatmulOp matmulOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::MatmulOp> emitter(
        matmulOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(matmulOp.getA()),
        emitter.emit(matmulOp.getB()),
        emitter.emit(matmulOp.getTransposeA(), "transpose_a"),
        emitter.emit(matmulOp.getTransposeB(), "transpose_b"),
        emitter.emit(std::nullopt |
                         emitter.getMemoryConfig(matmulOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// Linear op conversion pattern
//
namespace {
class LinearOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::LinearOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::LinearOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::LinearOp srcOp,
                  mlir::tt::ttnn::LinearOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::LinearOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getA()),
        emitter.emit(srcOp.getB()),
        emitter.emit(srcOp.getBias(), "bias"),
        emitter.emit(srcOp.getTransposeA(), "transpose_a"),
        emitter.emit(srcOp.getTransposeB(), "transpose_b"),
        emitter.emit(std::nullopt | emitter.getMemoryConfig(srcOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// AvgPool2d op conversion pattern
//
namespace {
class AvgPool2dOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::AvgPool2dOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::AvgPool2dOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::AvgPool2dOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::AvgPool2dOp> emitter(
        srcOp, adaptor, rewriter);

    SmallVector<int32_t> padding;
    if (srcOp.getPadding().size() == 2) {
      padding.push_back(static_cast<uint32_t>(srcOp.getPadding()[0]));
      padding.push_back(static_cast<uint32_t>(srcOp.getPadding()[1]));
    } else {
      padding.push_back(static_cast<uint32_t>(srcOp.getPadding()[0]));
      padding.push_back(static_cast<uint32_t>(srcOp.getPadding()[2]));
      padding.push_back(static_cast<uint32_t>(srcOp.getPadding()[1]));
      padding.push_back(static_cast<uint32_t>(srcOp.getPadding()[3]));
    }

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getBatchSize()),
        emitter.emit(srcOp.getInputHeight()),
        emitter.emit(srcOp.getInputWidth()),
        emitter.emit(srcOp.getChannels()),
        emitter.template emit<std::array<uint32_t, 2>>(
            srcOp.getKernelSizeAttr()),
        emitter.template emit<std::array<uint32_t, 2>>(srcOp.getStrideAttr()),
        emitter.template emit<
            std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>>>(
            rewriter.getI32ArrayAttr(padding)),
        emitter.emit(srcOp.getCeilMode()),
        emitter.emit(srcOp.getCountIncludePad()),
        emitter.emit(/*divisor_override=*/std::nullopt),
        emitter.emit(srcOp.getMemoryConfig() |
                         emitter.getMemoryConfig(srcOp.getResult()),
                     "memory_config"),
        emitter.emit(srcOp.getAppliedShardScheme(), "applied_shard_scheme"),
        emitter.emit(srcOp.getInPlaceHalo(), "in_place_halo"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// MaxPool2dOp conversion pattern
//
namespace {
class MaxPool2dOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::MaxPool2dOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::MaxPool2dOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::MaxPool2dOp maxPool2dOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::MaxPool2dOp> emitter(
        maxPool2dOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(maxPool2dOp.getInput()),
        emitter.emit(maxPool2dOp.getBatchSize()),
        emitter.emit(maxPool2dOp.getInputHeight()),
        emitter.emit(maxPool2dOp.getInputWidth()),
        emitter.emit(maxPool2dOp.getChannels()),
        emitter.template emit<std::vector<uint32_t>>(
            maxPool2dOp.getKernelSizeAttr()),
        emitter.template emit<std::vector<uint32_t>>(
            maxPool2dOp.getStrideAttr()),
        emitter.template emit<std::vector<uint32_t>>(
            maxPool2dOp.getPaddingAttr()),
        emitter.template emit<std::vector<uint32_t>>(
            maxPool2dOp.getDilationAttr()),
        emitter.emit(emitter.getMemoryConfig(maxPool2dOp.getResult()),
                     "memory_config"),
        emitter.emit(maxPool2dOp.getAppliedShardScheme(),
                     "applied_shard_scheme"),
        emitter.emit(maxPool2dOp.getCeilMode(), "ceil_mode"),
        emitter.emit(maxPool2dOp.getInPlaceHalo(), "in_place_halo"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// Upsample op conversion pattern
//
namespace {
class UpsampleOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::UpsampleOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::UpsampleOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::UpsampleOp srcOp,
                  mlir::tt::ttnn::UpsampleOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::UpsampleOp> emitter(
        srcOp, adaptor, rewriter);

    Attribute scaleFactorAttr;
    if (mlir::isa<mlir::IntegerAttr>(srcOp.getScaleFactor())) {
      scaleFactorAttr = emitter.emit<int32_t>(srcOp.getScaleFactor());
    } else {
      scaleFactorAttr =
          emitter.emit<std::array<uint32_t, 2>>(srcOp.getScaleFactor());
    }

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        scaleFactorAttr,
        emitter.emit(srcOp.getMode(), "mode"),
        emitter.emit(srcOp.getMemoryConfig() |
                         emitter.getMemoryConfig(srcOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);
    return success();
  }
};
} // namespace

// Moreh CumSum op conversion pattern
//
namespace {
class MorehCumSumOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<
          mlir::tt::ttnn::MorehCumSumOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::MorehCumSumOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::MorehCumSumOp srcOp,
                  mlir::tt::ttnn::MorehCumSumOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::MorehCumSumOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getDim()),
        emitter.emit(srcOp.getMemoryConfig() |
                         emitter.getMemoryConfig(srcOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);
    return success();
  }
};
} // namespace

namespace {
class SoftmaxOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::SoftmaxOp> {
public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::SoftmaxOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::SoftmaxOp softmaxOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::SoftmaxOp> emitter(
        softmaxOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(softmaxOp.getInput()),
        emitter.emit(softmaxOp.getDimension()),
        emitter.emit(std::nullopt |
                         emitter.getMemoryConfig(softmaxOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// arith::ConstantOp conversion pattern
//
namespace {
class ArithConstantOpConversionPattern
    : public OpConversionPattern<arith::ConstantOp> {

public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp constOp, arith::ConstantOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newTy = this->getTypeConverter()->convertType(constOp.getType());
    if (!newTy) {
      return rewriter.notifyMatchFailure(constOp, "type conversion failed");
    }

    rewriter.replaceOpWithNewOp<emitpy::ConstantOp>(constOp, newTy,
                                                    adaptor.getValue());
    return success();
  }
};
} // namespace

// DeallocateOp conversion pattern
//
namespace {
class DeallocateOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::DeallocateOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::DeallocateOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::DeallocateOp deallocateOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::DeallocateOp> emitter(
        deallocateOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(deallocateOp.getInput()),
        emitter.emit(deallocateOp.getForce()),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// mlir::tt::ttcore::DeviceOp conversion pattern
//
namespace {
struct TTDeviceOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttcore::DeviceOp> {
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttcore::DeviceOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttcore::DeviceOp deviceOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(deviceOp);
    return success();
  }
};
} // namespace

// GetDeviceOp conversion pattern
//
namespace {
class GetDeviceOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::GetDeviceOp> {
private:
  std::string getPrefixSearchPattern() const override {
    return "ttnn.get_device";
  }
  std::string getPrefixSwapPattern() const override {
    return "my_get_device.DeviceGetter.get_device";
  }

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::GetDeviceOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::GetDeviceOp getDeviceOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::GetDeviceOp> emitter(
        getDeviceOp, adaptor, rewriter);

    emitter.replaceOp(*this, {});

    return success();
  }
};
} // namespace

// ToDeviceOp conversion pattern
//
namespace {
class ToDeviceOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::ToDeviceOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::ToDeviceOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::ToDeviceOp toDeviceOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::ToDeviceOp> emitter(
        toDeviceOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(toDeviceOp.getInput()),
        emitter.emit(toDeviceOp.getDevice(), "device"),
        emitter.emit(toDeviceOp.getMemoryConfig() |
                         emitter.getMemoryConfig(toDeviceOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// FromDeviceOp conversion pattern
//
namespace {
class FromDeviceOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::FromDeviceOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::FromDeviceOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::FromDeviceOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::FromDeviceOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// TypecastOp conversion pattern
//
namespace {
class TypecastOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::TypecastOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::TypecastOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::TypecastOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::TypecastOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getDtype()),
        emitter.emit(std::nullopt | emitter.getMemoryConfig(srcOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// ToDTypeOp conversion pattern
//
namespace {
class ToDTypeOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::ToDTypeOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::ToDTypeOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::ToDTypeOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::ToDTypeOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getDtype()),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// ToLayoutOp conversion pattern
//
namespace {
class ToLayoutOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::ToLayoutOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::ToLayoutOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::ToLayoutOp toLayoutOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::ToLayoutOp> emitter(
        toLayoutOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(toLayoutOp.getInput()),
        emitter.emit(toLayoutOp.getLayout()),
        emitter.emit(toLayoutOp.getDtype()),
        emitter.emit(toLayoutOp.getMemoryConfig() |
                         emitter.getMemoryConfig(toLayoutOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// ToMemoryConfig conversion pattern
//
namespace {
class ToMemoryConfigOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<
          mlir::tt::ttnn::ToMemoryConfigOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::ToMemoryConfigOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::ToMemoryConfigOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::ToMemoryConfigOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getMemoryConfig()) |
            emitter.getMemoryConfig(srcOp.getResult()),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// ArangeOp conversion pattern
//
namespace {
class ArangeOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::ArangeOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::ArangeOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::ArangeOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::ArangeOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getStart()),
        emitter.emit(srcOp.getEnd()),
        emitter.emit(srcOp.getStep()),
        emitter.emit(srcOp.getDtype(), "dtype"),
        emitter.emit(srcOp.getDevice(), "device"),
        emitter.emit(srcOp.getLayout(), "layout"),
        emitter.emit(srcOp.getMemoryConfig() |
                         emitter.getMemoryConfig(srcOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// EmptyOp conversion pattern
//
namespace {
class EmptyOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::EmptyOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::EmptyOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::EmptyOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::EmptyOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getShape()),
        emitter.emit(srcOp.getDtype()),
        emitter.emit(srcOp.getLayout()),
        emitter.emit(srcOp.getDevice()),
        emitter.emit(srcOp.getMemoryConfig()) |
            emitter.getMemoryConfig(srcOp.getResult()),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// FullOp conversion pattern
//
namespace {
class FullOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::FullOp> {
public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::FullOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::FullOp fullOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (auto fillValueAttr = mlir::dyn_cast<FloatAttr>(fullOp.getFillValue())) {
      auto fillValue = fillValueAttr.getValue().convertToDouble();
      return matchAndRewriteImpl(fullOp, fillValue, adaptor, rewriter);
    }

    if (auto fillValueAttr =
            mlir::dyn_cast<IntegerAttr>(fullOp.getFillValue())) {
      auto fillValue =
          static_cast<int32_t>(fillValueAttr.getValue().getSExtValue());
      return matchAndRewriteImpl(fullOp, fillValue, adaptor, rewriter);
    }
    return failure();
  }

private:
  template <
      typename FillValueT,
      typename = std::void_t<llvm::is_one_of<FillValueT, double, int32_t>>>
  LogicalResult matchAndRewriteImpl(mlir::tt::ttnn::FullOp fullOp,
                                    FillValueT fillValue, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::FullOp> emitter(
        fullOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(fullOp.getShape(), "shape"),
        emitter.emit(fillValue, "fill_value"),
        emitter.emit(fullOp.getDtype(), "dtype"),
        emitter.emit(fullOp.getLayout(), "layout"),
        emitter.emit(fullOp.getDevice(), "device"),
        emitter.emit(fullOp.getMemoryConfig() |
                         emitter.getMemoryConfig(fullOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// NamedFullOp conversion pattern for operations like ttnn::zeros or ttnn::ones
//
namespace {
template <typename SourceOp>
class NamedFullOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<SourceOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      SourceOp>::TTNNToEmitPyBaseOpConversionPattern;
  using Adaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp namedFullOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<SourceOp> emitter(namedFullOp, adaptor,
                                                        rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(namedFullOp.getShape(), "shape"),
        emitter.emit(namedFullOp.getDtype(), "dtype"),
        emitter.emit(namedFullOp.getLayout(), "layout"),
        emitter.emit(namedFullOp.getDevice(), "device"),
        /*  emitter.emit(namedFullOp.getMemoryConfig() |
                          emitter.getMemoryConfig(namedFullOp.getResult()),
                      "memory_config"), */
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// Rand op conversion pattern
//
namespace {
class RandOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<tt::ttnn::RandOp> {
public:
  using TTNNToEmitPyBaseOpConversionPattern<
      tt::ttnn::RandOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ttnn::RandOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<tt::ttnn::RandOp> emitter(srcOp, adaptor,
                                                                rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getSize()),
        emitter.emit(srcOp.getDevice()),
        emitter.emit(srcOp.getDtype(), "dtype"),
        emitter.emit(srcOp.getLayout(), "layout"),
        emitter.emit(srcOp.getMemoryConfig(), "memory_config"),
        emitter.emit(srcOp.getLow(), "low"),
        emitter.emit(srcOp.getHigh(), "high"),
        emitter.emit(srcOp.getSeed(), "seed"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// Prod op conversion pattern
//
namespace {
class ProdOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::ProdOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::ProdOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::ProdOp srcOp,
                  mlir::tt::ttnn::ProdOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::ProdOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getDimArg()),
        emitter.emit(srcOp.getKeepDim()),
        emitter.emit(srcOp.getMemoryConfig() |
                         emitter.getMemoryConfig(srcOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// ReductionOp conversion pattern
//
namespace {
template <typename ReductionOp>
class ReductionOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<ReductionOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      ReductionOp>::TTNNToEmitPyBaseOpConversionPattern;
  using Adaptor = typename ReductionOp::Adaptor;

  LogicalResult
  matchAndRewrite(ReductionOp reductionOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<ReductionOp> emitter(reductionOp, adaptor,
                                                           rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(reductionOp.getInput()),
        emitter.template emit<::ttsl::SmallVector<int32_t>>(
            reductionOp.getDimArg()),
        emitter.emit(reductionOp.getKeepDim()),
        emitter.emit(std::nullopt |
                         emitter.getMemoryConfig(reductionOp.getResult()),
                     "memory_config")};

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// Argmax op conversion pattern
//
namespace {
class ArgMaxOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::ArgMaxOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::ArgMaxOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::ArgMaxOp srcOp,
                  mlir::tt::ttnn::ArgMaxOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::ArgMaxOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getDim()),
        emitter.emit(srcOp.getKeepDim()),
        emitter.emit(std::nullopt, "sub_core_grids"),
        emitter.emit(srcOp.getUseMulticore(), "use_multicore"),
        emitter.emit(srcOp.getMemoryConfig() |
                         emitter.getMemoryConfig(srcOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// Conv2dOp conversion pattern
//
namespace {
class Conv2dOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::Conv2dOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::Conv2dOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::Conv2dOp conv2dOp,
                  mlir::tt::ttnn::Conv2dOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::Conv2dOp> emitter(
        conv2dOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(conv2dOp.getInput(), "input_tensor"),
        emitter.emit(conv2dOp.getWeight(), "weight_tensor"),
        emitter.emit(conv2dOp.getDevice(), "device"),
        emitter.emit(conv2dOp.getInChannels(), "in_channels"),
        emitter.emit(conv2dOp.getOutChannels(), "out_channels"),
        emitter.emit(conv2dOp.getBatchSize(), "batch_size"),
        emitter.emit(conv2dOp.getInputHeight(), "input_height"),
        emitter.emit(conv2dOp.getInputWidth(), "input_width"),
        emitter.template emit<std::vector<uint32_t>>(
            conv2dOp.getKernelSizeAttr(), "kernel_size"),
        emitter.template emit<std::vector<uint32_t>>(conv2dOp.getStrideAttr(),
                                                     "stride"),
        emitter.template emit<std::vector<uint32_t>>(conv2dOp.getPaddingAttr(),
                                                     "padding"),
        emitter.template emit<std::vector<uint32_t>>(conv2dOp.getDilationAttr(),
                                                     "dilation"),
        emitter.emit(conv2dOp.getGroups(), "groups"),
        emitter.emit(conv2dOp.getBias(), "bias_tensor"),
        emitter.emit(conv2dOp.getConv2dConfig(), "conv_config"),
        emitter.emit(std::nullopt, "compute_config"),
        emitter.emit(std::nullopt |
                         emitter.getMemoryConfig(conv2dOp.getResult()),
                     "memory_config"),
    };
    /* llvm::SmallVector<mlir::StringRef> keyword_args{
        "input_tensor",   "weight_tensor", "bias_tensor", "device",
        "in_channels",    "out_channels",  "batch_size",  "input_height",
        "input_width",    "kernel_size",   "stride",      "padding",
        "dilation",       "groups",        "dtype",       "conv_config",
        "compute_config", "memory_config"};
 */
    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// ConvTranspose2d op conversion pattern
//
namespace {
class ConvTranspose2dOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<
          mlir::tt::ttnn::ConvTranspose2dOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::ConvTranspose2dOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::ConvTranspose2dOp srcOp,
                  mlir::tt::ttnn::ConvTranspose2dOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::ConvTranspose2dOp>
        emitter(srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput(), "input_tensor"),
        emitter.emit(srcOp.getWeight(), "weight_tensor"),
        emitter.emit(srcOp.getDevice(), "device"),
        emitter.emit(srcOp.getInChannels(), "in_channels"),
        emitter.emit(srcOp.getOutChannels(), "out_channels"),
        emitter.emit(srcOp.getBatchSize(), "batch_size"),
        emitter.emit(srcOp.getInputHeight(), "input_height"),
        emitter.emit(srcOp.getInputWidth(), "input_width"),
        emitter.emit<std::array<uint32_t, 2>>(srcOp.getKernelSizeAttr(),
                                              "kernel_size"),
        emitter.emit<std::array<uint32_t, 2>>(srcOp.getStrideAttr(), "stride"),
        emitter.emit<std::array<uint32_t, 2>>(srcOp.getPaddingAttr(),
                                              "padding"),
        emitter.emit<std::array<uint32_t, 2>>(srcOp.getOutputPaddingAttr(),
                                              "output_padding"),
        emitter.emit<std::array<uint32_t, 2>>(srcOp.getDilationAttr(),
                                              "dilation"),
        emitter.emit(srcOp.getGroups(), "groups"),
        emitter.emit(srcOp.getDtype(), "dtype"),
        emitter.emit(srcOp.getBias(), "bias_tensor"),
        emitter.emit(srcOp.getConv2dConfig(), "conv_config"),
        emitter.emit(std::nullopt, "compute_config"),
        emitter.emit(srcOp.getMemoryConfig() |
                         emitter.getMemoryConfig(srcOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// PrepareConv2dWeights op conversion pattern
//
namespace {
class PrepareConv2dWeightsOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<
          mlir::tt::ttnn::PrepareConv2dWeightsOp> {

private:
  std::string getPrefixSearchPattern() const override {
    return "ttnn.prepare_conv2d_weights";
  }
  std::string getPrefixSwapPattern() const override {
    return "ttnn.prepare_conv_weights";
  }

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::PrepareConv2dWeightsOp>::
      TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::PrepareConv2dWeightsOp srcOp,
                  mlir::tt::ttnn::PrepareConv2dWeightsOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::PrepareConv2dWeightsOp>
        emitter(srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getWeightTensor(), "weight_tensor"),
        emitter.emit(srcOp.getInputMemoryConfig(), "input_memory_config"),
        emitter.emit(srcOp.getInputTensorLayout(), "input_layout"),
        emitter.emit(srcOp.getWeightsFormat(), "weights_format"),
        emitter.emit(srcOp.getInChannels(), "in_channels"),
        emitter.emit(srcOp.getOutChannels(), "out_channels"),
        emitter.emit(srcOp.getBatchSize(), "batch_size"),
        emitter.emit(srcOp.getInputHeight(), "input_height"),
        emitter.emit(srcOp.getInputWidth(), "input_width"),
        emitter.emit<std::array<uint32_t, 2>>(srcOp.getKernelSizeAttr(),
                                              "kernel_size"),
        emitter.emit<std::array<uint32_t, 2>>(srcOp.getStrideAttr(), "stride"),
        emitter.emit<
            std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>>>(
            srcOp.getPaddingAttr(), "padding"),
        emitter.emit<std::array<uint32_t, 2>>(srcOp.getDilationAttr(),
                                              "dilation"),
        emitter.emit(srcOp.getHasBias(), "has_bias"),
        emitter.emit(srcOp.getGroups(), "groups"),
        emitter.emit(srcOp.getDevice(), "device"),
        emitter.emit(srcOp.getInputDtype(), "input_dtype"),
        emitter.emit(srcOp.getOutputDtype(), "output_dtype"),
        emitter.emit(srcOp.getConv2dConfig(), "conv_config"),
        emitter.emit(std::nullopt, "compute_config"),
        emitter.emit(std::nullopt, "slice_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// PrepareConv2dBias op conversion pattern
//
namespace {
class PrepareConv2dBiasOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<
          mlir::tt::ttnn::PrepareConv2dBiasOp> {

private:
  std::string getPrefixSearchPattern() const override {
    return "ttnn.prepare_conv2d_bias";
  }
  std::string getPrefixSwapPattern() const override {
    return "ttnn.prepare_conv_bias";
  }

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::PrepareConv2dBiasOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::PrepareConv2dBiasOp srcOp,
                  mlir::tt::ttnn::PrepareConv2dBiasOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::PrepareConv2dBiasOp>
        emitter(srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getBiasTensor(), "bias_tensor"),
        emitter.emit(srcOp.getInputMemoryConfig(), "input_memory_config"),
        emitter.emit(srcOp.getInputTensorLayout(), "input_layout"),
        emitter.emit(srcOp.getInChannels(), "in_channels"),
        emitter.emit(srcOp.getOutChannels(), "out_channels"),
        emitter.emit(srcOp.getBatchSize(), "batch_size"),
        emitter.emit(srcOp.getInputHeight(), "input_height"),
        emitter.emit(srcOp.getInputWidth(), "input_width"),
        emitter.emit<std::array<uint32_t, 2>>(srcOp.getKernelSizeAttr(),
                                              "kernel_size"),
        emitter.emit<std::array<uint32_t, 2>>(srcOp.getStrideAttr(), "stride"),
        emitter.emit<
            std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>>>(
            srcOp.getPaddingAttr(), "padding"),
        emitter.emit<std::array<uint32_t, 2>>(srcOp.getDilationAttr(),
                                              "dilation"),
        emitter.emit(srcOp.getGroups(), "groups"),
        emitter.emit(srcOp.getDevice(), "device"),
        emitter.emit(srcOp.getInputDtype(), "input_dtype"),
        emitter.emit(srcOp.getOutputDtype(), "output_dtype"),
        emitter.emit(srcOp.getConv2dConfig(), "conv_config"),
        emitter.emit(std::nullopt, "compute_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// PadOp conversion pattern
//
namespace {
class PadOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::PadOp> {
public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::PadOp>::TTNNToEmitPyBaseOpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::PadOp padOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::PadOp> emitter(
        padOp, adaptor, rewriter);

    // Convert padding from flat array to ArrayAttr of DenseI32ArrayAttr pairs
    auto paddingArray = padOp.getPadding();
    llvm::SmallVector<mlir::Attribute, 4> paddingPairs;
    for (size_t i = 0; i < paddingArray.size(); i += 2) {
      mlir::SmallVector<int32_t, 2> paddingPair = {paddingArray[i],
                                                   paddingArray[i + 1]};

      paddingPairs.push_back(
          mlir::DenseI32ArrayAttr::get(rewriter.getContext(), paddingPair));
    }

    mlir::ArrayAttr paddingPairsArrayAttr =
        mlir::ArrayAttr::get(rewriter.getContext(), paddingPairs);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(padOp.getInput()),
        emitter.emit<std::vector<std::array<int32_t, 2>>>(
            paddingPairsArrayAttr),
        emitter.emit(padOp.getValue()),
        emitter.emit(padOp.getUseMulticore(), "use_multicore"),
        emitter.emit(padOp.getMemoryConfig() |
                         emitter.getMemoryConfig(padOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// ReshapeOp conversion pattern
//
namespace {
class ReshapeOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::ReshapeOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::ReshapeOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::ReshapeOp reshapeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::ReshapeOp> emitter(
        reshapeOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(reshapeOp.getInput()),
        emitter.emit<std::vector<int32_t>>(reshapeOp.getShape()),
        emitter.emit(reshapeOp.getMemoryConfig() |
                         emitter.getMemoryConfig(reshapeOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// RepeatOp conversion pattern
//
namespace {
class RepeatOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::RepeatOp> {
public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::RepeatOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::RepeatOp repeatOp,
                  mlir::tt::ttnn::RepeatOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::RepeatOp> emitter(
        repeatOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(repeatOp.getInput()),
        emitter.emit(repeatOp.getRepeatDims()),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// RepeatInterleave op conversion pattern
//
namespace {
class RepeatInterleaveOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<
          mlir::tt::ttnn::RepeatInterleaveOp> {
public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::RepeatInterleaveOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::RepeatInterleaveOp srcOp,
                  mlir::tt::ttnn::RepeatInterleaveOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::RepeatInterleaveOp>
        emitter(srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getRepeats()),
        emitter.emit(srcOp.getDim()),
        emitter.emit(srcOp.getMemoryConfig() |
                         emitter.getMemoryConfig(srcOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// Sort op conversion pattern
//
namespace {
class SortOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<tt::ttnn::SortOp> {
public:
  using TTNNToEmitPyBaseOpConversionPattern<
      tt::ttnn::SortOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ttnn::SortOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<tt::ttnn::SortOp> emitter(srcOp, adaptor,
                                                                rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getDim()),
        emitter.emit(srcOp.getDescending()),
        emitter.emit(srcOp.getStable()),
        emitter.emit(std::nullopt | emitter.getMemoryConfig(srcOp.getValues()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// TransposeOp conversion pattern
//
namespace {
class TransposeOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::TransposeOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::TransposeOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::TransposeOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::TransposeOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getDim0()),
        emitter.emit(srcOp.getDim1()),
        emitter.emit(std::nullopt | emitter.getMemoryConfig(srcOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// ConcatOp conversion pattern
//
namespace {
class ConcatOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::ConcatOp> {
public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::ConcatOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::ConcatOp concatOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::ConcatOp> emitter(
        concatOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(concatOp.getInputs()),
        emitter.emit(concatOp.getDim()),
        emitter.emit(std::nullopt |
                         emitter.getMemoryConfig(concatOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// PermuteOp conversion pattern
//
namespace {
class PermuteOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::PermuteOp> {
public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::PermuteOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::PermuteOp permuteOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::PermuteOp> emitter(
        permuteOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(permuteOp.getInput()),
        emitter.emit(permuteOp.getPermutation()),
        emitter.emit(std::nullopt |
                         emitter.getMemoryConfig(permuteOp.getResult()),
                     "memory_config"),
        emitter.emit(permuteOp.getPadValue(), "pad_value"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// EmbeddingOp conversion pattern
//
namespace {
class EmbeddingOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::EmbeddingOp> {
public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::EmbeddingOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::EmbeddingOp embeddingOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::EmbeddingOp> emitter(
        embeddingOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(embeddingOp.getInput()),
        emitter.emit(embeddingOp.getWeight()),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// EmbeddingBackward conversion pattern
//
namespace {
class EmbeddingBackwardOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<
          mlir::tt::ttnn::EmbeddingBackwardOp> {
public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::EmbeddingBackwardOp>::TTNNToEmitPyBaseOpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::EmbeddingBackwardOp srcOp,
                  mlir::tt::ttnn::EmbeddingBackwardOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::EmbeddingBackwardOp>
        emitter(srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getWeight()),
        emitter.emit(srcOp.getInGradient()),
        emitter.emit(srcOp.getDtype(), "dtype"),
        emitter.emit(srcOp.getMemoryConfig() |
                         emitter.getMemoryConfig(srcOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// SliceDynamicOp conversion pattern
//
namespace {
class SliceDynamicOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<
          mlir::tt::ttnn::SliceDynamicOp> {
private:
  std::string getPrefixSearchPattern() const override {
    return "ttnn.slice_dynamic";
  }
  std::string getPrefixSwapPattern() const override { return "ttnn.slice"; }

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::SliceDynamicOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::SliceDynamicOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::SliceDynamicOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getBegins()),
        emitter.emit(srcOp.getEnds()),
        emitter.template emit<::ttsl::SmallVector<int32_t>>(srcOp.getStep()),
        emitter.emit(std::nullopt | emitter.getMemoryConfig(srcOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// SliceStaticOp conversion pattern
//
namespace {
class SliceStaticOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<
          mlir::tt::ttnn::SliceStaticOp> {
private:
  std::string getPrefixSearchPattern() const override {
    return "ttnn.slice_static";
  }
  std::string getPrefixSwapPattern() const override { return "ttnn.slice"; }

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::SliceStaticOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::SliceStaticOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::SliceStaticOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.template emit<::ttsl::SmallVector<int32_t>>(srcOp.getBegins()),
        emitter.template emit<::ttsl::SmallVector<int32_t>>(srcOp.getEnds()),
        emitter.template emit<::ttsl::SmallVector<int32_t>>(srcOp.getStep()),
        emitter.emit(std::nullopt | emitter.getMemoryConfig(srcOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

namespace {
class ConcatenateHeadsOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<
          mlir::tt::ttnn::ConcatenateHeadsOp> {
private:
  std::string getPrefixSearchPattern() const override {
    return "ttnn.concatenate_heads";
  }
  std::string getPrefixSwapPattern() const override {
    return "ttnn.transformer.concatenate_heads";
  }

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::ConcatenateHeadsOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::ConcatenateHeadsOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::ConcatenateHeadsOp>
        emitter(srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(std::nullopt | emitter.getMemoryConfig(srcOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// FillCacheOp
//
namespace {
class FillCacheOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::FillCacheOp> {
public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::FillCacheOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::FillCacheOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::FillCacheOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getCache()), emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getBatchOffset())};

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// UpdateCacheOp
//
namespace {
class UpdateCacheOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<
          mlir::tt::ttnn::UpdateCacheOp> {
public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::UpdateCacheOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::UpdateCacheOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::UpdateCacheOp> emitter(
        srcOp, adaptor, rewriter);

    // The `update_index` is modeled as a tensor in the IR, but the
    // `ttnn.update_cache` expects a `int` scalar.
    auto updateIndex = rewriter
                           .create<emitpy::CallOpaqueOp>(
                               srcOp.getLoc(), rewriter.getI32Type(),
                               ttnn_to_emitpy::kGetScalarFromTensorFunctionName,
                               adaptor.getUpdateIndex(),
                               /*args=*/nullptr,
                               /*keyword_args=*/nullptr)
                           .getResult(0);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getCache()), emitter.emit(srcOp.getInput()),
        emitter.emit(updateIndex, /*attrName=*/"", /*index=*/2),
        emitter.emit(srcOp.getBatchOffset(), "batch_offset")};

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// GetTupleElementOp conversion pattern
//
namespace {
class GetTupleElementOpConversionPattern
    : public OpConversionPattern<mlir::tt::ttcore::GetTupleElementOp> {

public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttcore::GetTupleElementOp getTupleElementOp,
                  mlir::tt::ttcore::GetTupleElementOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // SubscriptOp requires a Value object as index, which is created by
    // invoking the emitpy::LiteralOp.
    //
    Value indexAsVal = rewriter.create<emitpy::LiteralOp>(
        getTupleElementOp->getLoc(), rewriter.getIndexType(),
        std::to_string(adaptor.getIndex()));

    rewriter.replaceOpWithNewOp<emitpy::SubscriptOp>(
        getTupleElementOp,
        this->getTypeConverter()->convertType(getTupleElementOp.getType()),
        adaptor.getOperand(), indexAsVal);

    return success();
  }
};
} // namespace

// TupleOp conversion pattern
//
namespace {
class TupleOpConversionPattern
    : public OpConversionPattern<mlir::tt::ttcore::TupleOp> {

public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttcore::TupleOp tupleOp,
                  mlir::tt::ttcore::TupleOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        tupleOp, this->getTypeConverter()->convertType(tupleOp.getType()),
        "util_create_list", adaptor.getOperands());
    return success();
  }
};
} // namespace

// LoadCached Op conversion pattern
//
// This op is worked around - it only calls the consteval fn, but there is no
// caching.
// TODO (4936): https://github.com/tenstorrent/tt-mlir/issues/4936
//
namespace {
class LoadCachedOpConversionPattern
    : public OpConversionPattern<mlir::tt::ttcore::LoadCachedOp> {

public:
  using OpConversionPattern<
      mlir::tt::ttcore::LoadCachedOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttcore::LoadCachedOp loadCachedOp,
                  mlir::tt::ttcore::LoadCachedOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Create list of tensors.
    //
    llvm::SmallVector<Value> operands;
    if (loadCachedOp.getInputs().size() > 0) {
      mlir::Value tensorsInList =
          rewriter
              .create<emitpy::CallOpaqueOp>(
                  loadCachedOp.getLoc(),
                  emitpy::OpaqueType::get(rewriter.getContext(),
                                          "[ttnn.Tensor]"),
                  ttnn_to_emitpy::kCreateListFunctionName,
                  adaptor.getOperands(), nullptr)
              ->getResult(0);
      operands.push_back(tensorsInList);
    }

    // Call into the callee, no caching mechanism.
    //
    emitpy::OpaqueType resultType =
        emitpy::OpaqueType::get(rewriter.getContext(), "[ttnn.Tensor]");
    auto cacheOp = rewriter.create<func::CallOp>(
        loadCachedOp.getLoc(), resultType, loadCachedOp.getCallee(), operands);

    // Unpack list of tensors.
    //
    llvm::SmallVector<Value> results;
    for (unsigned i = 0; i < loadCachedOp.getNumResults(); ++i) {
      // Create index value.
      //
      auto indexType = rewriter.getIndexType();
      auto indexOp = rewriter.create<emitpy::LiteralOp>(
          loadCachedOp.getLoc(), indexType, std::to_string(i));
      Value indexVal = indexOp.getResult();

      // Get reference to the i-th element in the result.
      //
      auto subscriptOp = rewriter.create<emitpy::SubscriptOp>(
          loadCachedOp.getLoc(),
          emitpy::OpaqueType::get(rewriter.getContext(), "ttnn.Tensor"),
          cacheOp->getResult(0), indexVal);

      results.push_back(subscriptOp.getResult());
    }

    // Replace the original op with the extracted results.
    //
    rewriter.replaceOp(loadCachedOp, results);

    return success();
  }
};
} // namespace

// ModuleOp conversion pattern
//
// This conversion pattern removes attributes from the ModuleOp.
//
namespace {
class ModuleOpConversionPattern : public OpConversionPattern<mlir::ModuleOp> {

public:
  using OpConversionPattern<mlir::ModuleOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::ModuleOp moduleOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(moduleOp, [&]() {
      for (const NamedAttribute &attr : moduleOp->getAttrs()) {
        moduleOp->removeAttr(attr.getName());
      }
    });

    return success();
  }
};
} // namespace

namespace {
class DumpTensorOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::DumpTensorOp> {
public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::DumpTensorOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::DumpTensorOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::DumpTensorOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getFilePath()),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

namespace {
class LoadTensorOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::LoadTensorOp> {
public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::LoadTensorOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::LoadTensorOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::LoadTensorOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getFilePath()),
        emitter.emit(srcOp.getDevice(), "device"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// NLPConcatHeadsOp conversion pattern
//
namespace {
class NLPConcatHeadsOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<
          mlir::tt::ttnn::NLPConcatHeadsOp> {
private:
  std::string getPrefixSearchPattern() const override {
    return "ttnn.nlp_concat_heads";
  }
  std::string getPrefixSwapPattern() const override {
    return "ttnn.experimental.nlp_concat_heads";
  }

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::NLPConcatHeadsOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::NLPConcatHeadsOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::NLPConcatHeadsOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getMemoryConfig() |
                         emitter.getMemoryConfig(srcOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// NLPConcatHeadsDecodeOp conversion pattern
//
namespace {
class NLPConcatHeadsDecodeOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<
          mlir::tt::ttnn::NLPConcatHeadsDecodeOp> {
private:
  std::string getPrefixSearchPattern() const override {
    return "ttnn.nlp_concat_heads_decode";
  }
  std::string getPrefixSwapPattern() const override {
    return "ttnn.experimental.nlp_concat_heads_decode";
  }

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::NLPConcatHeadsDecodeOp>::
      TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::NLPConcatHeadsDecodeOp srcOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::NLPConcatHeadsDecodeOp>
        emitter(srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getNumHeads(), "num_heads"),
        emitter.emit(srcOp.getMemoryConfig() |
                         emitter.getMemoryConfig(srcOp.getResult()),
                     "memory_config")};

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// BatchNormOp conversion pattern
//
namespace {
class BatchNormOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::BatchNormOp> {
public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::BatchNormOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::BatchNormOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::BatchNormOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getRunningMean(), "running_mean"),
        emitter.emit(srcOp.getRunningVar(), "running_var"),
        emitter.emit(srcOp.getTraining(), "training"),
        emitter.emit(srcOp.getEpsilon(), "eps"),
        emitter.emit(srcOp.getMomentum(), "momentum"),
        emitter.emit(srcOp.getWeight(), "weight"),
        emitter.emit(srcOp.getBias(), "bias"),
        emitter.emit(srcOp.getMemoryConfig() |
                         emitter.getMemoryConfig(srcOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// RMSNormOp conversion pattern
//
namespace {
class RMSNormOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::RMSNormOp> {
public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::RMSNormOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::RMSNormOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::RMSNormOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getEpsilon(), "epsilon"),
        emitter.emit(srcOp.getWeight(), "weight"),
        emitter.emit(srcOp.getBias(), "bias"),
        emitter.emit(std::nullopt, "residual_input_tensor"),
        emitter.emit(srcOp.getMemoryConfig() |
                         emitter.getMemoryConfig(srcOp.getResult()),
                     "memory_config"),
        emitter.emit(std::nullopt, "program_config"),
        emitter.emit(std::nullopt, "compute_kernel_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// NLPCreateQKVHeadsDecodeOp conversion pattern
namespace {
class NLPCreateQKVHeadsDecodeOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<
          mlir::tt::ttnn::NLPCreateQKVHeadsDecodeOp> {
private:
  std::string getPrefixSearchPattern() const override {
    return "ttnn.nlp_create_qkv_heads_decode";
  }
  std::string getPrefixSwapPattern() const override {
    return "ttnn.experimental.nlp_create_qkv_heads_decode";
  }

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::NLPCreateQKVHeadsDecodeOp>::
      TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::NLPCreateQKVHeadsDecodeOp srcOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::NLPCreateQKVHeadsDecodeOp>
        emitter(srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getNumHeads(), "num_heads"),
        emitter.emit(srcOp.getNumKvHeads(), "num_kv_heads"),
        emitter.emit(srcOp.getOverlapQkCoregrid(), "overlap_qk_coregrid"),
        emitter.emit(srcOp.getBatchOffset(), "batch_offset"),
        emitter.emit(srcOp.getSliceSize(), "slice_size"),
        emitter.emit(srcOp.getMemoryConfig(), "memory_config")};

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// SplitQueryKeyValueAndSplitHeadsOp conversion pattern
namespace {
class SplitQueryKeyValueAndSplitHeadsOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<
          mlir::tt::ttnn::SplitQueryKeyValueAndSplitHeadsOp> {
private:
  std::string getPrefixSearchPattern() const override {
    return "ttnn.split_query_key_value_and_split_heads";
  }
  std::string getPrefixSwapPattern() const override {
    return "ttnn.transformer.split_query_key_value_and_split_heads";
  }

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::SplitQueryKeyValueAndSplitHeadsOp>::
      TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::SplitQueryKeyValueAndSplitHeadsOp srcOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<
        mlir::tt::ttnn::SplitQueryKeyValueAndSplitHeadsOp>
        emitter(srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInputTensor()),
        emitter.emit(srcOp.getKvInputTensor()),
        emitter.emit(srcOp.getNumHeads(), "num_heads"),
        emitter.emit(srcOp.getNumKvHeads(), "num_kv_heads"),
        emitter.emit(srcOp.getTransposeKey(), "transpose_key"),
        emitter.emit(srcOp.getMemoryConfig(), "memory_config")};

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// RotaryEmbeddingLlama conversion pattern
//
namespace {
class RotaryEmbeddingLlamaOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<
          mlir::tt::ttnn::RotaryEmbeddingLlamaOp> {
private:
  std::string getPrefixSearchPattern() const override {
    return "ttnn.rotary_embedding_llama";
  }
  std::string getPrefixSwapPattern() const override {
    return "ttnn.experimental.rotary_embedding_llama";
  }

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::RotaryEmbeddingLlamaOp>::
      TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::RotaryEmbeddingLlamaOp srcOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::RotaryEmbeddingLlamaOp>
        emitter(srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getCosCache()),
        emitter.emit(srcOp.getSinCache()),
        emitter.emit(srcOp.getTransMat()),
        emitter.emit(srcOp.getIsDecodeMode(), "is_decode_mode"),
        emitter.emit(srcOp.getMemoryConfig() |
                         emitter.getMemoryConfig(srcOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// ScaledDotProductAttentionOp conversion pattern
//
namespace {
class ScaledDotProductAttentionOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<
          mlir::tt::ttnn::ScaledDotProductAttentionOp> {

private:
  std::string getPrefixSearchPattern() const override {
    return "ttnn.scaled_dot_product_attention";
  }
  std::string getPrefixSwapPattern() const override {
    return "ttnn.transformer.scaled_dot_product_attention";
  }

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::ScaledDotProductAttentionOp>::
      TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::ScaledDotProductAttentionOp srcOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<
        mlir::tt::ttnn::ScaledDotProductAttentionOp>
        emitter(srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getQuery()),
        emitter.emit(srcOp.getKey()),
        emitter.emit(srcOp.getValue()),
        emitter.emit(srcOp.getAttentionMask(), "attn_mask"),
        emitter.emit(srcOp.getIsCausal(), "is_causal"),
        emitter.emit<float>(srcOp.getScaleAttr(), "scale"),
        emitter.emit(srcOp.getMemoryConfig() |
                         emitter.getMemoryConfig(srcOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// ScaledDotProductAttentionDecodeOp conversion pattern
//
namespace {
class ScaledDotProductAttentionDecodeOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<
          mlir::tt::ttnn::ScaledDotProductAttentionDecodeOp> {

private:
  std::string getPrefixSearchPattern() const override {
    return "ttnn.scaled_dot_product_attention_decode";
  }
  std::string getPrefixSwapPattern() const override {
    return "ttnn.transformer.scaled_dot_product_attention_decode";
  }

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::ScaledDotProductAttentionDecodeOp>::
      TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::ScaledDotProductAttentionDecodeOp srcOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<
        mlir::tt::ttnn::ScaledDotProductAttentionDecodeOp>
        emitter(srcOp, adaptor, rewriter);

    // NOLINTBEGIN(clang-analyzer-cplusplus.NewDelete)
    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getQuery()),
        emitter.emit(srcOp.getKey()),
        emitter.emit(srcOp.getValue()),
        emitter.emit(srcOp.getIsCausal(), "is_causal"),
        emitter.emit(srcOp.getAttentionMask(), "attn_mask"),
        emitter.emit(srcOp.getCurPosTensor(), "cur_pos_tensor"),
        emitter.emit(srcOp.getAttentionSink(), "attention_sink"),
        emitter.emit(srcOp.getScale(), "scale"),
        emitter.emit(srcOp.getMemoryConfig() |
                         emitter.getMemoryConfig(srcOp.getResult()),
                     "memory_config"),
    };
    // NOLINTEND(clang-analyzer-cplusplus.NewDelete)

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// Quantization ops conversion pattern
//
template <typename OpType>
class QuantizationOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<OpType> {
public:
  using TTNNToEmitPyBaseOpConversionPattern<
      OpType>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(OpType srcOp, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ttnn_to_emitpy::EmitPyTTNNEmitter<OpType> emitter(srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getScale()),
        emitter.emit(srcOp.getZeroPoint()),
        emitter.emit(srcOp.getAxis(), "axis"),
        emitter.emit(srcOp.getOutputDtype(), "dtype"),
        emitter.emit(srcOp.getMemoryConfig() |
                         emitter.getMemoryConfig(srcOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);
    return success();
  }
};

// Requantization conversion pattern
//
namespace {
class RequantizeOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::RequantizeOp> {
public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::RequantizeOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::RequantizeOp srcOp,
                  mlir::tt::ttnn::RequantizeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::RequantizeOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getInScale()),
        emitter.emit(srcOp.getInZeroPoint()),
        emitter.emit(srcOp.getOutScale()),
        emitter.emit(srcOp.getOutZeroPoint()),
        emitter.emit(srcOp.getAxis(), "axis"),
        emitter.emit(srcOp.getOutputDtype(), "dtype"),
        emitter.emit(srcOp.getMemoryConfig() |
                         emitter.getMemoryConfig(srcOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);
    return success();
  }
};
} // namespace

namespace mlir::tt {

void populateTTNNToEmitPyPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                  TypeConverter &typeConverter) {
  // Device ops
  //
  // clang-format off
  patterns.add<GetDeviceOpConversionPattern,
               TTDeviceOpConversionPattern
              >(typeConverter, ctx);
  // clang-format on

  // Tensor ops
  //
  // clang-format off
  patterns.add<ArangeOpConversionPattern,
               EmptyOpConversionPattern,
               FullOpConversionPattern,
               NamedFullOpConversionPattern<mlir::tt::ttnn::OnesOp>,
               NamedFullOpConversionPattern<mlir::tt::ttnn::ZerosOp>,
               RandOpConversionPattern>(typeConverter, ctx);
  // clang-format on

  // Arith ops
  //
  // clang-format off
  patterns.add<ArithConstantOpConversionPattern>(typeConverter, ctx);
  // clang-format on

  // Matmul ops
  //
  // clang-format off
  patterns.add<MatmulOpConversionPattern, LinearOpConversionPattern>(typeConverter, ctx);
  // clang-format on

  // Reduction ops
  //
  // clang-format off
  patterns.add<ProdOpConversionPattern,
               ReductionOpConversionPattern<mlir::tt::ttnn::MaxOp>,
               ReductionOpConversionPattern<mlir::tt::ttnn::MeanOp>,
               ReductionOpConversionPattern<mlir::tt::ttnn::MinOp>,
               ReductionOpConversionPattern<mlir::tt::ttnn::SumOp>,
               ArgMaxOpConversionPattern>(typeConverter, ctx);
  // clang-format on

  // Eltwise unary ops
  //
  // clang-format off
  patterns.add<ClampOpConversionPattern<::mlir::tt::ttnn::ClampScalarOp>,
               ClampOpConversionPattern<mlir::tt::ttnn::ClampTensorOp>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::AbsOp>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::AtanOp>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::BitwiseNotOp>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::CbrtOp>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::CeilOp>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::CosOp>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::Expm1Op>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::FloorOp>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::IsFiniteOp>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::LogicalNotOp>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::NegOp>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::ReciprocalOp>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::ReluOp>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::Relu6Op>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::RsqrtOp>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::SignOp>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::SiluOp>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::SinOp>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::SqrtOp>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::TanOp>,
               EltwiseUnaryWithFastAndApproximateModeOpConversionPattern<mlir::tt::ttnn::GeluOp>,
               EltwiseUnaryWithFastAndApproximateModeOpConversionPattern<mlir::tt::ttnn::ExpOp>,
               EltwiseUnaryWithFastAndApproximateModeOpConversionPattern<mlir::tt::ttnn::ErfOp>,
               EltwiseUnaryWithFastAndApproximateModeOpConversionPattern<mlir::tt::ttnn::ErfcOp>,
               EltwiseUnaryWithFastAndApproximateModeOpConversionPattern<mlir::tt::ttnn::LogOp>,
               EltwiseUnaryWithFastAndApproximateModeOpConversionPattern<mlir::tt::ttnn::Log1pOp>,
               EltwiseUnaryWithFastAndApproximateModeOpConversionPattern<mlir::tt::ttnn::TanhOp>,
               EltwiseUnaryWithFloatParameterOpConversionPattern<mlir::tt::ttnn::LeakyReluOp>,
               EltwiseUnaryWithVectorAndFastAndApproximateModeOpConversionPattern<mlir::tt::ttnn::SigmoidOp>>(typeConverter, ctx);
  // clang-format on

  // Eltwise binary ops
  //
  // clang-format off
  patterns.add<EltwiseBinaryOpConversionPattern<mlir::tt::ttnn::AddOp>,
               EltwiseBinaryOpConversionPattern<mlir::tt::ttnn::DivideOp>,
               EltwiseBinaryOpConversionPattern<mlir::tt::ttnn::EqualOp>,
               EltwiseBinaryOpConversionPattern<mlir::tt::ttnn::GreaterEqualOp>,
               EltwiseBinaryOpConversionPattern<mlir::tt::ttnn::GreaterThanOp>,
               EltwiseBinaryOpConversionPattern<mlir::tt::ttnn::LessEqualOp>,
               EltwiseBinaryOpConversionPattern<mlir::tt::ttnn::LessThanOp>,
               EltwiseBinaryOpConversionPattern<mlir::tt::ttnn::LogicalAndOp>,
               EltwiseBinaryOpConversionPattern<mlir::tt::ttnn::LogicalOrOp>,
               EltwiseBinaryOpConversionPattern<mlir::tt::ttnn::LogicalXorOp>,
               EltwiseBinaryOpConversionPattern<mlir::tt::ttnn::MultiplyOp>,
               EltwiseBinaryOpConversionPattern<mlir::tt::ttnn::NotEqualOp>,
               EltwiseBinaryOpConversionPattern<mlir::tt::ttnn::SubtractOp>,
               EltwiseBinaryCompositeOpConversionPattern<mlir::tt::ttnn::BitwiseAndOp>,
               EltwiseBinaryCompositeOpConversionPattern<mlir::tt::ttnn::BitwiseOrOp>,
               EltwiseBinaryCompositeOpConversionPattern<mlir::tt::ttnn::BitwiseXorOp>,
               EltwiseBinaryCompositeOpConversionPattern<mlir::tt::ttnn::LogicalLeftShiftOp>,
               EltwiseBinaryOpConversionPattern<mlir::tt::ttnn::LogicalRightShiftOp>,
               EltwiseBinaryCompositeOpConversionPattern<mlir::tt::ttnn::RemainderOp>,
               EltwiseBinaryCompositeOpConversionPattern<mlir::tt::ttnn::Atan2Op>,
               EltwiseBinaryCompositeWithDTypeOpConversionPattern<mlir::tt::ttnn::PowTensorOp>,
               EltwiseBinaryCompositeWithDTypeOpConversionPattern<mlir::tt::ttnn::MinimumOp>,
               EltwiseBinaryCompositeWithDTypeOpConversionPattern<mlir::tt::ttnn::MaximumOp>,
               PowScalarOpConversionPattern>(typeConverter, ctx);
  // clang-format on

  patterns.add<EltwiseTernaryOpConversionPattern<ttnn::WhereOp>>(typeConverter,
                                                                 ctx);

  // Tensor manipulation ops
  //
  // clang-format off
  patterns.add<ConcatOpConversionPattern,
               PermuteOpConversionPattern,
               PadOpConversionPattern,
               ReshapeOpConversionPattern,
               RepeatInterleaveOpConversionPattern,
               RepeatOpConversionPattern,
               SliceDynamicOpConversionPattern,
               SliceStaticOpConversionPattern,
               SortOpConversionPattern,
               TransposeOpConversionPattern
              >(typeConverter, ctx);
  // clang-format on

  // Memory ops
  //
  // clang-format off
  patterns.add<DeallocateOpConversionPattern,
               FromDeviceOpConversionPattern,
               ToDeviceOpConversionPattern,
               ToDTypeOpConversionPattern,
               ToLayoutOpConversionPattern,
               ToMemoryConfigOpConversionPattern,
               TypecastOpConversionPattern
              >(typeConverter, ctx);
  // clang-format on

  // Pooling ops
  //
  // clang-format off
  patterns.add<AvgPool2dOpConversionPattern,
    MaxPool2dOpConversionPattern,
    UpsampleOpConversionPattern>(typeConverter, ctx);
  // clang-format on

  // Convolution ops
  //
  patterns.add<Conv2dOpConversionPattern, ConvTranspose2dOpConversionPattern,
               PrepareConv2dWeightsOpConversionPattern,
               PrepareConv2dBiasOpConversionPattern>(typeConverter, ctx);

  // Normalization ops
  //
  patterns.add<BatchNormOpConversionPattern, RMSNormOpConversionPattern>(
      typeConverter, ctx);

  // Transformers ops
  //
  patterns.add<ConcatenateHeadsOpConversionPattern>(typeConverter, ctx);

  // Other ops
  //
  // clang-format off
  patterns.add<EmbeddingOpConversionPattern, EmbeddingBackwardOpConversionPattern, MorehCumSumOpConversionPattern, SoftmaxOpConversionPattern>(typeConverter, ctx);
  // clang-format on

  // Tensor serialization ops
  //
  // clang-format off
  patterns.add<DumpTensorOpConversionPattern,
               LoadTensorOpConversionPattern>(typeConverter, ctx);
  // clang-format on

  // Tuple ops
  //
  // clang-format off
  patterns.add<GetTupleElementOpConversionPattern,
               TupleOpConversionPattern>(typeConverter, ctx);
  // clang-format on

  // Constant op
  //
  patterns.add<ConstantOpConversionPattern>(typeConverter, ctx);

  // KV Cache ops
  //
  patterns.add<FillCacheOpConversionPattern>(typeConverter, ctx);
  patterns.add<UpdateCacheOpConversionPattern>(typeConverter, ctx);

  // Quantization ops.
  //
  patterns.add<QuantizationOpConversionPattern<mlir::tt::ttnn::QuantizeOp>,
               QuantizationOpConversionPattern<mlir::tt::ttnn::DequantizeOp>,
               RequantizeOpConversionPattern>(typeConverter, ctx);

  // Consteval ops
  patterns.add<LoadCachedOpConversionPattern>(typeConverter, ctx);

  // Module op
  //
  // clang-format off
  patterns.add<ModuleOpConversionPattern>(typeConverter, ctx);
  // clang-format on

  patterns.add<NLPConcatHeadsOpConversionPattern>(typeConverter, ctx);
  patterns.add<NLPConcatHeadsDecodeOpConversionPattern>(typeConverter, ctx);
  patterns.add<NLPCreateQKVHeadsDecodeOpConversionPattern>(typeConverter, ctx);
  patterns.add<SplitQueryKeyValueAndSplitHeadsOpConversionPattern>(
      typeConverter, ctx);
  patterns.add<RotaryEmbeddingLlamaOpConversionPattern>(typeConverter, ctx);
  patterns.add<ScaledDotProductAttentionOpConversionPattern>(typeConverter,
                                                             ctx);
  patterns.add<ScaledDotProductAttentionDecodeOpConversionPattern>(
      typeConverter, ctx);
}

} // namespace mlir::tt
