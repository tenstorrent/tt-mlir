// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitC/TTNNToEmitC.h"

#include "ttmlir/Conversion/TTNNToEmitC/EmitCConversion.h"
#include "ttmlir/Conversion/TTNNToEmitC/Utils.h"
#include "ttmlir/Dialect/TT/IR/TTOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"

#define GET_OP_CLASSES
#include "ttmlir/Dialect/TT/IR/TTOpsDialect.h.inc"

using namespace mlir;
using namespace mlir::tt;

emitc::OpaqueAttr createNullDevicePointer(Builder &builder) {
  return builder.getType<emitc::OpaqueAttr>(
      "static_cast<::ttnn::distributed::MeshDevice *>(nullptr)");
}

// Base class for TTNN to EmitC OpConversionPattern.
//
namespace {
template <typename SourceOp>
class TTNNToEmitCBaseOpConversionPattern
    : public OpConversionPattern<SourceOp> {
public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using Adaptor = typename SourceOp::Adaptor;

private:
  std::string virtual getPrefixSearchPattern() const { return "ttnn."; }
  std::string virtual getPrefixSwapPattern() const { return "ttnn::"; }

public:
  // Converts op name by removing the dialect prefix ("ttnn.") and replacing
  // with namespace prefix ("ttnn::")
  //
  std::string convertOpName(SourceOp op) const {
    auto name = op.getOperationName();
    assert(
        name.starts_with(getPrefixSearchPattern()) &&
        "DefaultOpConversionPattern only supports ops from the TTNN dialect");

    return name.str().replace(0, getPrefixSearchPattern().size(),
                              getPrefixSwapPattern());
  }
};
} // namespace

// Default op conversion pattern, used to convert most ops.
//
namespace {
template <typename SourceOp>
class DefaultOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<SourceOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      SourceOp>::TTNNToEmitCBaseOpConversionPattern;
  using Adaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    int numReturnTypes = srcOp->getResultTypes().size();
    assert(numReturnTypes <= 1 &&
           "DefaultOpConversionPattern does not support multiple return types");

    // If srcOp has a return type, cast it before converting.
    //
    if (numReturnTypes == 1) {
      auto resultTy = cast<emitc::OpaqueType>(
          this->getTypeConverter()->convertType(srcOp->getResult(0).getType()));
      rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
          srcOp, resultTy, this->convertOpName(srcOp), nullptr, nullptr,
          adaptor.getOperands());
    } else {
      // No return type, only convert the op.
      //
      rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
          srcOp, srcOp->getResultTypes(), this->convertOpName(srcOp), nullptr,
          nullptr, adaptor.getOperands());
    }

    return success();
  }
};
} // namespace

// Eltwise Unary op conversion pattern
//
// Currently, it has to insert nullopts for some parameters that are not
// modelled in the dialect (memcfg).
//
namespace {
template <typename SourceOp>
class EltwiseUnaryOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<SourceOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      SourceOp>::TTNNToEmitCBaseOpConversionPattern;
  using Adaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<SourceOp> emitter(srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(std::nullopt) | emitter.getMemoryConfig(srcOp.getResult()),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// EltwiseUnaryWithFastAndApproximateModeOp conversion pattern
//
// Currently, it has to insert nullopts for some parameters that are not
// modelled in the dialect (parameter, memcfg).
//
namespace {
template <typename SourceOp>
class EltwiseUnaryWithFastAndApproximateModeOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<SourceOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      SourceOp>::TTNNToEmitCBaseOpConversionPattern;
  using Adaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<SourceOp> emitter(srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        /*parameter=*/emitter.emit(false),
        emitter.emit(std::nullopt) | emitter.getMemoryConfig(srcOp.getResult()),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// EltwiseUnaryWithVectorAndFastAndApproximateModeOp conversion pattern
//
// Currently, it has to insert nullopts for some parameters that are not
// modelled in the dialect (parameter, memcfg).
//
namespace {
template <typename SourceOp>
class EltwiseUnaryWithVectorAndFastAndApproximateModeOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<SourceOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      SourceOp>::TTNNToEmitCBaseOpConversionPattern;
  using Adaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<SourceOp> emitter(srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(static_cast<int>(::ttnn::operations::unary::VecMode::RC)),
        /*parameter=*/emitter.emit(false),
        emitter.emit(std::nullopt) | emitter.getMemoryConfig(srcOp.getResult()),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// ElementwiseUnaryWithFloatParameterOp conversion pattern
//
// Currently, it has to insert nullopts for some parameters that are not
// modelled in the dialect (parameter, memcfg).
//
namespace {
template <typename SourceOp>
class ElementwiseUnaryWithFloatParameterOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<SourceOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      SourceOp>::TTNNToEmitCBaseOpConversionPattern;
  using Adaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<SourceOp> emitter(srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        /*parameter=*/emitter.emit(srcOp.getParameter()),
        emitter.emit(std::nullopt) | emitter.getMemoryConfig(srcOp.getResult()),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// EltwiseUnaryCompositeOp conversion pattern
//
// Currently, it has to insert nullopts for some parameters that are not
// modelled in the dialect (memcfg).
//
namespace {
template <typename SourceOp>
class EltwiseUnaryCompositeOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<SourceOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      SourceOp>::TTNNToEmitCBaseOpConversionPattern;
  using Adaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<SourceOp> emitter(srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(std::nullopt) | emitter.getMemoryConfig(srcOp.getResult()),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// ClampOpConversionPattern conversion pattern
//
// Currently, it has to insert nullopts for some parameters that are not
// modelled in the dialect (memcfg).
//
namespace {
template <typename SourceOp>
class ClampOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<SourceOp> {
private:
  std::string getPrefixSearchPattern() const override {
    if constexpr (std::is_same_v<SourceOp, tt::ttnn::ClampScalarOp>) {
      return "ttnn.clamp_scalar";
    } else if constexpr (std::is_same_v<SourceOp, tt::ttnn::ClampTensorOp>) {
      return "ttnn.clamp_tensor";
    }

    llvm_unreachable("Operation not supported.");
  }

  std::string getPrefixSwapPattern() const override { return "ttnn::clamp"; }

public:
  using TTNNToEmitCBaseOpConversionPattern<
      SourceOp>::TTNNToEmitCBaseOpConversionPattern;
  using Adaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<SourceOp> emitter(srcOp, adaptor, rewriter);
    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getMin()),
        emitter.emit(srcOp.getMax()),
        emitter.emit(std::nullopt) | emitter.getMemoryConfig(srcOp.getResult()),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// Eltwise Binary op conversion pattern
//
// Currently, it has to insert nullopts for some parameters that are not
// modelled in the dialect (output dtype, memcfg).
//
namespace {
template <typename SourceOp>
class EltwiseBinaryOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<SourceOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      SourceOp>::TTNNToEmitCBaseOpConversionPattern;
  using Adaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<SourceOp> emitter(srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getLhs()),
        emitter.emit(srcOp.getRhs()),
        /*dtype=*/emitter.emit(std::nullopt),
        emitter.emit(std::nullopt) | emitter.getMemoryConfig(srcOp.getResult()),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// Eltwise Binary Composite op conversion pattern
//
// Currently, it has to insert nullopts for some parameters that are not
// modelled in the dialect (memcfg).
//
namespace {
template <typename SourceOp>
class EltwiseBinaryCompositeOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<SourceOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      SourceOp>::TTNNToEmitCBaseOpConversionPattern;
  using Adaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<SourceOp> emitter(srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getLhs()),
        emitter.emit(srcOp.getRhs()),
        emitter.emit(std::nullopt) | emitter.getMemoryConfig(srcOp.getResult()),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// Eltwise Binary NG Composite op conversion pattern
//
// Currently, it has to insert nullopts for some parameters that are not
// modelled in the dialect (memcfg).
//
namespace {
template <typename SourceOp>
class EltwiseBinaryNGCompositeOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<SourceOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      SourceOp>::TTNNToEmitCBaseOpConversionPattern;
  using Adaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<SourceOp> emitter(srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getLhs()),
        emitter.emit(srcOp.getRhs()),
        emitter.emit(std::nullopt),
        emitter.emit(std::nullopt) | emitter.getMemoryConfig(srcOp.getResult()),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// Eltwise Ternary op conversion pattern
//
// Currently, it has to insert nullopts for some parameters that are not
// modelled in the dialect (memcfg).
//
namespace {
template <typename SourceOp>
class EltwiseTernaryOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<SourceOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      SourceOp>::TTNNToEmitCBaseOpConversionPattern;
  using Adaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<SourceOp> emitter(srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getFirst()),
        emitter.emit(srcOp.getSecond()),
        emitter.emit(srcOp.getThird()),
        emitter.emit(std::nullopt) | emitter.getMemoryConfig(srcOp.getResult()),
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
    : public TTNNToEmitCBaseOpConversionPattern<tt::ttnn::LinearOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      tt::ttnn::LinearOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ttnn::LinearOp srcOp, tt::ttnn::LinearOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<tt::ttnn::LinearOp> emitter(srcOp, adaptor,
                                                                rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getA()),
        emitter.emit(srcOp.getB()),
        emitter.emit(srcOp.getBias()),
        emitter.emit(srcOp.getTransposeA()),
        emitter.emit(srcOp.getTransposeB()),
        emitter.emit(std::nullopt) | emitter.getMemoryConfig(srcOp.getResult()),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// Matmul op conversion pattern
//
// ANCHOR: adding_an_op_matmul_op_rewriter_emitc
namespace {
class MatmulOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<tt::ttnn::MatmulOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      tt::ttnn::MatmulOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ttnn::MatmulOp srcOp, tt::ttnn::MatmulOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<tt::ttnn::MatmulOp> emitter(srcOp, adaptor,
                                                                rewriter);

    // ANCHOR: adding_an_op_matmul_ttnn_to_emitc_array_attrs
    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getA()),
        emitter.emit(srcOp.getB()),
        emitter.emit(srcOp.getTransposeA()),
        emitter.emit(srcOp.getTransposeB()),
        emitter.emit(std::nullopt) | emitter.getMemoryConfig(srcOp.getResult()),
    };
    // ANCHOR_END: adding_an_op_matmul_ttnn_to_emitc_array_attrs

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace
// ANCHOR_END: adding_an_op_matmul_op_rewriter_emitc

// Generic 2d pooling op conversion pattern. Coverts maxpool_2d and avgpool_2d.
//
namespace {
template <typename Pooling2dOp>
class Pooling2dOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<Pooling2dOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      Pooling2dOp>::TTNNToEmitCBaseOpConversionPattern;
  using Adaptor = typename Pooling2dOp::Adaptor;

  LogicalResult
  matchAndRewrite(Pooling2dOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<Pooling2dOp> emitter(srcOp, adaptor,
                                                         rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getBatchSize()),
        emitter.emit(srcOp.getInputHeight()),
        emitter.emit(srcOp.getInputWidth()),
        emitter.emit(srcOp.getChannels()),
        emitter.template emit<std::array<uint32_t, 2>>(
            srcOp.getKernelSizeAttr()),
        emitter.template emit<std::array<uint32_t, 2>>(srcOp.getStrideAttr()),
        emitter.template emit<std::array<uint32_t, 2>>(srcOp.getPaddingAttr()),
        emitter.template emit<std::array<uint32_t, 2>>(srcOp.getDilationAttr()),
        emitter.getMemoryConfig(srcOp.getResult()),
        emitter.emit(srcOp.getAppliedShardScheme()),
        emitter.emit(srcOp.getCeilMode()),
        emitter.emit(srcOp.getInPlaceHalo()),
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
    : public TTNNToEmitCBaseOpConversionPattern<tt::ttnn::UpsampleOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      tt::ttnn::UpsampleOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ttnn::UpsampleOp srcOp,
                  tt::ttnn::UpsampleOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<tt::ttnn::UpsampleOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit<int32_t>(srcOp.getScaleFactor()) |
            emitter.emit<std::array<uint32_t, 2>>(srcOp.getScaleFactor()),
        emitter.emit(srcOp.getMode()),
        emitter.emit(srcOp.getMemoryConfig()) |
            emitter.getMemoryConfig(srcOp.getResult()),
    };

    emitter.replaceOp(*this, args);
    return success();
  }
};

// Quantization ops conversion pattern
//
template <typename OpType>
class QuantizationOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<OpType> {
public:
  using TTNNToEmitCBaseOpConversionPattern<
      OpType>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ttnn_to_emitc::EmitCTTNNEmitter<OpType> emitter(op, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(op.getInput()),
        emitter.emit(op.getScale()),
        emitter.emit(op.getZeroPoint()),
        emitter.emit(op.getAxis()),
        emitter.emit(op.getOutputDtype()),
        emitter.emit(std::nullopt) | emitter.emit(op.getMemoryConfig()),
    };

    emitter.replaceOp(*this, args);
    return success();
  }
};

class RequantizeOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<tt::ttnn::RequantizeOp> {
public:
  using TTNNToEmitCBaseOpConversionPattern<
      tt::ttnn::RequantizeOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ttnn::RequantizeOp op,
                  tt::ttnn::RequantizeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ttnn_to_emitc::EmitCTTNNEmitter<tt::ttnn::RequantizeOp> emitter(op, adaptor,
                                                                    rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(op.getInput()),
        emitter.emit(op.getInScale()),
        emitter.emit(op.getInZeroPoint()),
        emitter.emit(op.getOutScale()),
        emitter.emit(op.getOutZeroPoint()),
        emitter.emit(op.getAxis()),
        emitter.emit(op.getOutputDtype()),
        emitter.emit(std::nullopt) | emitter.emit(op.getMemoryConfig()),
    };

    emitter.replaceOp(*this, args);
    return success();
  }
};
} // namespace

// Softmax op conversion pattern
//
namespace {
class SoftmaxOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<tt::ttnn::SoftmaxOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      tt::ttnn::SoftmaxOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ttnn::SoftmaxOp srcOp,
                  tt::ttnn::SoftmaxOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<tt::ttnn::SoftmaxOp> emitter(srcOp, adaptor,
                                                                 rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getDimension()),
        emitter.emit(std::nullopt) | emitter.getMemoryConfig(srcOp.getResult()),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// Embedding op conversion pattern
//
namespace {
class EmbeddingOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<tt::ttnn::EmbeddingOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      tt::ttnn::EmbeddingOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ttnn::EmbeddingOp embeddingOp,
                  tt::ttnn::EmbeddingOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<tt::ttnn::EmbeddingOp> emitter(
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

// Moreh CumSum op conversion pattern
//
namespace {
class MorehCumSumOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<tt::ttnn::MorehCumSumOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      tt::ttnn::MorehCumSumOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ttnn::MorehCumSumOp srcOp,
                  tt::ttnn::MorehCumSumOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<tt::ttnn::MorehCumSumOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getDim()),
        /*output=*/emitter.emit(std::nullopt),
        emitter.emit(srcOp.getMemoryConfig()) |
            emitter.getMemoryConfig(srcOp.getResult()),
        /*compute_kernel_config=*/emitter.emit(std::nullopt),
    };

    emitter.replaceOp(*this, args);
    return success();
  }
};
} // namespace

// Reduction ops conversion pattern
//
namespace {
template <typename ReductionOp>
class ReductionOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<ReductionOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      ReductionOp>::TTNNToEmitCBaseOpConversionPattern;
  using Adaptor = typename ReductionOp::Adaptor;

  LogicalResult
  matchAndRewrite(ReductionOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<ReductionOp> emitter(srcOp, adaptor,
                                                         rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.template emit<::ttnn::SmallVector<int32_t>>(srcOp.getDimArg()),
        emitter.emit(srcOp.getKeepDim()),
        emitter.emit(std::nullopt) | emitter.getMemoryConfig(srcOp.getResult()),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// Argmax op conversion pattern
//
namespace {
class ArgMaxOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<tt::ttnn::ArgMaxOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      tt::ttnn::ArgMaxOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ttnn::ArgMaxOp srcOp, tt::ttnn::ArgMaxOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<tt::ttnn::ArgMaxOp> emitter(srcOp, adaptor,
                                                                rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getDim()),
        /*keepdim=*/emitter.emit(srcOp.getKeepDim()),
        /*sub_core_grids=*/emitter.emit(std::nullopt),
        emitter.emit(srcOp.getUseMulticore()),
        emitter.emit(srcOp.getMemoryConfig()) |
            emitter.getMemoryConfig(srcOp.getResult()),
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
    : public TTNNToEmitCBaseOpConversionPattern<tt::ttnn::ProdOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      tt::ttnn::ProdOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ttnn::ProdOp srcOp, tt::ttnn::ProdOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<tt::ttnn::ProdOp> emitter(srcOp, adaptor,
                                                              rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getDimArg()),
        emitter.emit(srcOp.getKeepDim()),
        emitter.emit(srcOp.getMemoryConfig()) |
            emitter.getMemoryConfig(srcOp.getResult()),
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
    : public TTNNToEmitCBaseOpConversionPattern<
          tt::ttnn::PrepareConv2dWeightsOp> {

private:
  std::string getPrefixSearchPattern() const override {
    return "ttnn.prepare_conv2d_weights";
  }
  std::string getPrefixSwapPattern() const override {
    return "ttnn::operations::conv::conv2d::prepare_conv_weights";
  }

public:
  using TTNNToEmitCBaseOpConversionPattern<
      tt::ttnn::PrepareConv2dWeightsOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ttnn::PrepareConv2dWeightsOp srcOp,
                  tt::ttnn::PrepareConv2dWeightsOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<tt::ttnn::PrepareConv2dWeightsOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getWeightTensor()),
        emitter.emit(srcOp.getInputMemoryConfig()),
        emitter.emit(srcOp.getInputTensorLayout()),
        emitter.emit(srcOp.getWeightsFormat()),
        emitter.emit(srcOp.getInChannels()),
        emitter.emit(srcOp.getOutChannels()),
        emitter.emit(srcOp.getBatchSize()),
        emitter.emit(srcOp.getInputHeight()),
        emitter.emit(srcOp.getInputWidth()),
        emitter.emit<std::array<uint32_t, 2>>(srcOp.getKernelSizeAttr()),
        emitter.emit<std::array<uint32_t, 2>>(srcOp.getStrideAttr()),
        emitter.emit<std::array<uint32_t, 2>>(srcOp.getPaddingAttr()),
        emitter.emit<std::array<uint32_t, 2>>(srcOp.getDilationAttr()),
        emitter.emit(srcOp.getHasBias()),
        emitter.emit(srcOp.getGroups()),
        emitter.emit(srcOp.getDevice()),
        /*conv2d_config=*/emitter.emit(std::nullopt),
        /*compute_config_=*/emitter.emit(std::nullopt),
        /*dram_slice_config=*/emitter.emit(std::nullopt),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// Conv2d op conversion pattern
//
namespace {
class Conv2dOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<tt::ttnn::Conv2dOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      tt::ttnn::Conv2dOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ttnn::Conv2dOp srcOp, tt::ttnn::Conv2dOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<tt::ttnn::Conv2dOp> emitter(srcOp, adaptor,
                                                                rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getWeight()),
        emitter.emit(srcOp.getDevice()),
        emitter.emit(srcOp.getInChannels()),
        emitter.emit(srcOp.getOutChannels()),
        emitter.emit(srcOp.getBatchSize()),
        emitter.emit(srcOp.getInputHeight()),
        emitter.emit(srcOp.getInputWidth()),
        emitter.emit<std::array<uint32_t, 2>>(srcOp.getKernelSizeAttr()),
        emitter.emit<std::array<uint32_t, 2>>(srcOp.getStrideAttr()),
        emitter.emit<std::array<uint32_t, 2>>(srcOp.getPaddingAttr()),
        emitter.emit<std::array<uint32_t, 2>>(srcOp.getDilationAttr()),
        emitter.emit(srcOp.getGroups()),
        emitter.emit(srcOp.getBias()),
        emitter.emit(std::nullopt) |
            emitter.getConv2dConfig(srcOp.getInput(), srcOp.getWeight()),
        /*compute_config=*/emitter.emit(std::nullopt),
        emitter.emit(std::nullopt) | emitter.getMemoryConfig(srcOp.getResult()),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// ConvTranspose2d op conversion pattern
//
namespace {
class ConvTranspose2dOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<tt::ttnn::ConvTranspose2dOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      tt::ttnn::ConvTranspose2dOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ttnn::ConvTranspose2dOp srcOp,
                  tt::ttnn::ConvTranspose2dOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<tt::ttnn::ConvTranspose2dOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getWeight()),
        emitter.emit(srcOp.getDevice()),
        emitter.emit(srcOp.getInChannels()),
        emitter.emit(srcOp.getOutChannels()),
        emitter.emit(srcOp.getBatchSize()),
        emitter.emit(srcOp.getInputHeight()),
        emitter.emit(srcOp.getInputWidth()),
        emitter.emit<std::array<uint32_t, 2>>(srcOp.getKernelSizeAttr()),
        emitter.emit<std::array<uint32_t, 2>>(srcOp.getStrideAttr()),
        emitter.emit<std::array<uint32_t, 2>>(srcOp.getPaddingAttr()),
        emitter.emit<std::array<uint32_t, 2>>(srcOp.getOutputPaddingAttr()),
        emitter.emit<std::array<uint32_t, 2>>(srcOp.getDilationAttr()),
        emitter.emit(srcOp.getGroups()),
        emitter.emit(srcOp.getBias()),
        emitter.emit(std::nullopt) |
            emitter.getConv2dConfig(srcOp.getInput(), srcOp.getWeight()),
        /*compute_config=*/emitter.emit(std::nullopt),
        emitter.emit(std::nullopt) | emitter.getMemoryConfig(srcOp.getResult()),
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
    : public TTNNToEmitCBaseOpConversionPattern<tt::ttnn::ReshapeOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      tt::ttnn::ReshapeOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ttnn::ReshapeOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<tt::ttnn::ReshapeOp> emitter(srcOp, adaptor,
                                                                 rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit<std::vector<int32_t>>(srcOp.getShape()),
        emitter.emit(srcOp.getMemoryConfig()),
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
    : public TTNNToEmitCBaseOpConversionPattern<tt::ttnn::TransposeOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      tt::ttnn::TransposeOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ttnn::TransposeOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<tt::ttnn::TransposeOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getDim0()),
        emitter.emit(srcOp.getDim1()),
        emitter.emit(std::nullopt) | emitter.getMemoryConfig(srcOp.getResult()),
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
    : public TTNNToEmitCBaseOpConversionPattern<tt::ttnn::ConcatOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      tt::ttnn::ConcatOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ttnn::ConcatOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<tt::ttnn::ConcatOp> emitter(srcOp, adaptor,
                                                                rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInputs()),
        emitter.emit(srcOp.getDim()),
        emitter.emit(std::nullopt) | emitter.getMemoryConfig(srcOp.getResult()),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// Repeat op conversion pattern
//
namespace {
class RepeatOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<tt::ttnn::RepeatOp> {
public:
  using TTNNToEmitCBaseOpConversionPattern<
      tt::ttnn::RepeatOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ttnn::RepeatOp srcOp, tt::ttnn::RepeatOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<tt::ttnn::RepeatOp> emitter(srcOp, adaptor,
                                                                rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getRepeatDims()),
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
    : public TTNNToEmitCBaseOpConversionPattern<tt::ttnn::RepeatInterleaveOp> {
public:
  using TTNNToEmitCBaseOpConversionPattern<
      tt::ttnn::RepeatInterleaveOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ttnn::RepeatInterleaveOp repeatInterleaveOp,
                  tt::ttnn::RepeatInterleaveOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<tt::ttnn::RepeatInterleaveOp> emitter(
        repeatInterleaveOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(repeatInterleaveOp.getInput()),
        emitter.emit(repeatInterleaveOp.getRepeats()),
        emitter.emit(repeatInterleaveOp.getDim()),
        emitter.emit(repeatInterleaveOp.getMemoryConfig()) |
            emitter.getMemoryConfig(repeatInterleaveOp.getResult()),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// tt::DeviceOp conversion pattern
//
namespace {
struct TTDeviceOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<tt::DeviceOp> {
  using TTNNToEmitCBaseOpConversionPattern<
      tt::DeviceOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::DeviceOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(srcOp);
    return success();
  }
};
} // namespace

// GetDeviceOp conversion pattern
//
namespace {
class GetDeviceOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<tt::ttnn::GetDeviceOp> {

private:
  std::string getPrefixSearchPattern() const override {
    return "ttnn.get_device";
  }
  std::string getPrefixSwapPattern() const override {
    return "ttnn::DeviceGetter::getInstance";
  }

public:
  using TTNNToEmitCBaseOpConversionPattern<
      tt::ttnn::GetDeviceOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ttnn::GetDeviceOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<tt::ttnn::GetDeviceOp> emitter(
        srcOp, adaptor, rewriter);

    emitter.replaceOp(*this, {});

    return success();
  }
};
} // namespace

// ToDeviceOp conversion pattern
//
namespace {
class ToDeviceOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<tt::ttnn::ToDeviceOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      tt::ttnn::ToDeviceOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ttnn::ToDeviceOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<tt::ttnn::ToDeviceOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getDevice()),
        emitter.emit(srcOp.getMemoryConfig()) |
            emitter.getMemoryConfig(srcOp.getResult()),
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
    : public TTNNToEmitCBaseOpConversionPattern<tt::ttnn::FromDeviceOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      tt::ttnn::FromDeviceOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ttnn::FromDeviceOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<tt::ttnn::FromDeviceOp> emitter(
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
    : public TTNNToEmitCBaseOpConversionPattern<tt::ttnn::TypecastOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      tt::ttnn::TypecastOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ttnn::TypecastOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<tt::ttnn::TypecastOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getDtype()),
        emitter.emit(std::nullopt) | emitter.getMemoryConfig(srcOp.getResult()),
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
    : public TTNNToEmitCBaseOpConversionPattern<tt::ttnn::ToDTypeOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      tt::ttnn::ToDTypeOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ttnn::ToDTypeOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<tt::ttnn::ToDTypeOp> emitter(srcOp, adaptor,
                                                                 rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getDtype()),
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
    : public TTNNToEmitCBaseOpConversionPattern<tt::ttnn::ToMemoryConfigOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      tt::ttnn::ToMemoryConfigOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ttnn::ToMemoryConfigOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<tt::ttnn::ToMemoryConfigOp> emitter(
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

// ToLayoutOp conversion pattern
//
namespace {
class ToLayoutOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<tt::ttnn::ToLayoutOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      tt::ttnn::ToLayoutOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ttnn::ToLayoutOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<tt::ttnn::ToLayoutOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getLayout()),
        emitter.emit(srcOp.getDtype()),
        emitter.emit(srcOp.getMemoryConfig()) |
            emitter.getMemoryConfig(srcOp.getResult()),
        emitter.emit(srcOp.getDevice()) |
            emitter.emit<::ttnn::distributed::MeshDevice>(nullptr),
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
    : public TTNNToEmitCBaseOpConversionPattern<tt::ttnn::EmptyOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      tt::ttnn::EmptyOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ttnn::EmptyOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<tt::ttnn::EmptyOp> emitter(srcOp, adaptor,
                                                               rewriter);

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

// Named FullOp conversion pattern for operations like ttnn::zeros or ttnn::ones
//
namespace {
template <typename SourceOp>
class NamedFullOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<SourceOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      SourceOp>::TTNNToEmitCBaseOpConversionPattern;
  using Adaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<SourceOp> emitter(srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getShape()),
        emitter.emit(srcOp.getDtype()),
        emitter.emit(srcOp.getLayout()),
        emitter.template emit<
            ::ttnn::operations::creation::detail::OptionalMeshDevice>(
            srcOp.getDevice()),
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
    : public TTNNToEmitCBaseOpConversionPattern<tt::ttnn::FullOp> {
public:
  using TTNNToEmitCBaseOpConversionPattern<
      tt::ttnn::FullOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ttnn::FullOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (auto fillValueAttr = mlir::dyn_cast<FloatAttr>(srcOp.getFillValue())) {
      auto fillValue = fillValueAttr.getValue().convertToFloat();
      return matchAndRewriteImpl(srcOp, fillValue, adaptor, rewriter);
    }
    if (auto fillValueAttr =
            mlir::dyn_cast<IntegerAttr>(srcOp.getFillValue())) {
      auto fillValue =
          static_cast<int32_t>(fillValueAttr.getValue().getSExtValue());
      return matchAndRewriteImpl(srcOp, fillValue, adaptor, rewriter);
    }
    return failure();
  }

private:
  template <typename FillValueT,
            typename = std::void_t<llvm::is_one_of<FillValueT, float, int32_t>>>
  LogicalResult matchAndRewriteImpl(tt::ttnn::FullOp srcOp,
                                    FillValueT fillValue, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
    ttnn_to_emitc::EmitCTTNNEmitter<tt::ttnn::FullOp> emitter(srcOp, adaptor,
                                                              rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getShape()),
        emitter.emit(fillValue),
        emitter.emit(srcOp.getDtype()),
        emitter.emit(srcOp.getLayout()),
        emitter.emit<::ttnn::operations::creation::detail::OptionalMeshDevice>(
            srcOp.getDevice()),
        emitter.emit(srcOp.getMemoryConfig()) |
            emitter.getMemoryConfig(srcOp.getResult()),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// DeallocateOp conversion pattern
//
namespace {
class DeallocateOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<tt::ttnn::DeallocateOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      tt::ttnn::DeallocateOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ttnn::DeallocateOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<tt::ttnn::DeallocateOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getForce()),
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

    rewriter.replaceOpWithNewOp<emitc::ConstantOp>(constOp, newTy,
                                                   adaptor.getValue());
    return success();
  }
};
} // namespace

namespace {
class GetTupleElementOpConversionPattern
    : public OpConversionPattern<tt::GetTupleElementOp> {

public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::GetTupleElementOp getTupleElementOp,
                  tt::GetTupleElementOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // SubscriptOp requires a Value object as index, which is created by
    // invoking the emitc::LiteralOp.
    //
    Value indexAsVal = rewriter.create<emitc::LiteralOp>(
        getTupleElementOp->getLoc(), rewriter.getIndexType(),
        std::to_string(adaptor.getIndex()));

    // SubscriptOp also returns an emitc::LValueType, so we wrap the
    // OpaqueType with LValueType.
    //
    emitc::LValueType lvalueReturnType =
        emitc::LValueType::get(emitc::OpaqueType::get(
            rewriter.getContext(), ttnn_to_emitc::TypeNameV<::ttnn::Tensor>));
    Value subscript = rewriter.create<emitc::SubscriptOp>(
        getTupleElementOp->getLoc(), lvalueReturnType, adaptor.getOperand(),
        indexAsVal);

    // As SubscriptOp returns an LValueType, we need to convert it to an
    // OpaqueType - this is done by invoking the emitc::LoadOp.
    //
    rewriter.replaceOpWithNewOp<emitc::LoadOp>(
        getTupleElementOp,
        emitc::OpaqueType::get(getContext(),
                               ttnn_to_emitc::TypeNameV<::ttnn::Tensor>),
        subscript);
    return success();
  }
};
} // namespace

namespace {
class TupleOpConversionPattern : public OpConversionPattern<tt::TupleOp> {

public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::TupleOp tupleOp, tt::TupleOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // EmitC doesn't offer a way to create a vector from a list of values, so
    // we need to create a utility function that does this. This is achieved
    // by using EmitC's VerbatimOp.

    // Try to find if utility vec creation function is already defined in the
    // module. If not, insert it.
    //
    tt::ttnn_to_emitc::utils::insertVecCreateFnIfNotExists(rewriter, tupleOp);

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        tupleOp, this->getTypeConverter()->convertType(tupleOp.getType()),
        tt::ttnn_to_emitc::utils::kCreateVectorFunctionName, nullptr, nullptr,
        adaptor.getOperands());
    return success();
  }
};
} // namespace

// LoadCached Op conversion pattern
//
// This is crude solution to use a static vector of results s.t. we only execute
// the subgraphs once.  No support for tensor dirtying atm.
//
namespace {
class LoadCachedOpConversionPattern
    : public OpConversionPattern<tt::LoadCachedOp> {

public:
  using OpConversionPattern<tt::LoadCachedOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::LoadCachedOp srcOp, tt::LoadCachedOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the callee function
    llvm::StringRef callee = srcOp.getCallee();

    // Try to find if utility vec creation function is already defined in the
    // module. If not, insert it.
    tt::ttnn_to_emitc::utils::insertVecCreateFnIfNotExists(rewriter, srcOp);

    // Create a tuple of all input tensors
    auto tupleType = emitc::OpaqueType::get(rewriter.getContext(),
                                            "::std::vector<::ttnn::Tensor>");

    // Convert result types
    SmallVector<Type> resultTypes;
    for (auto type : srcOp.getResultTypes()) {
      resultTypes.push_back(getTypeConverter()->convertType(type));
    }

    // Generate a unique name for our global variable
    std::string globalVarName = "g_cached_result_" + callee.str();
    FlatSymbolRefAttr globalSym =
        SymbolRefAttr::get(rewriter.getContext(), globalVarName);

    // Insert a global variable declaration before the current function
    // This ensures it comes after the header include in the generated C++ code
    auto funcOp = srcOp->getParentOfType<func::FuncOp>();
    auto currentInsertionPoint = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(funcOp);

    // Create the global variable using EmitC's GlobalOp
    rewriter.create<emitc::GlobalOp>(
        srcOp.getLoc(), StringAttr::get(rewriter.getContext(), globalVarName),
        TypeAttr::get(tupleType),
        /*initialValue=*/nullptr,
        /*extern_specifier=*/UnitAttr(),
        /*static_specifier=*/UnitAttr::get(rewriter.getContext()),
        /*const_specifier=*/UnitAttr());

    // Restore the insertion point to continue with the function
    rewriter.restoreInsertionPoint(currentInsertionPoint);

    // Create the function pointer type
    auto funcPtrType = emitc::OpaqueType::get(
        rewriter.getContext(), "::std::function<::std::vector<::ttnn::Tensor>(:"
                               ":std::vector<::ttnn::Tensor>)>");
    auto addressAttr =
        emitc::OpaqueAttr::get(rewriter.getContext(), "&" + callee.str());
    auto funcPtrValue = rewriter.create<emitc::ConstantOp>(
        srcOp.getLoc(), funcPtrType, addressAttr);

    auto tupleOp = rewriter.create<emitc::CallOpaqueOp>(
        srcOp.getLoc(), tupleType,
        tt::ttnn_to_emitc::utils::kCreateVectorFunctionName, nullptr, nullptr,
        adaptor.getInputs());
    Value tupleValue = tupleOp.getResult(0);

    // Get a reference to the global variable using GetGlobalOp
    auto globalVar = rewriter.create<emitc::GetGlobalOp>(
        srcOp.getLoc(), emitc::LValueType::get(tupleType), globalSym);

    // Create a pointer type for the output parameter
    auto ptrType = emitc::PointerType::get(rewriter.getContext(), tupleType);

    // Get the address of the global variable
    auto addressOfOp = rewriter.create<emitc::ApplyOp>(srcOp.getLoc(), ptrType,
                                                       "&", globalVar);

    // Call the wrapper function with the pointer
    rewriter.create<emitc::CallOpaqueOp>(
        srcOp.getLoc(), TypeRange{}, "ttnn::constEvalFuncWrapper",
        ValueRange{funcPtrValue, tupleValue, addressOfOp}, ArrayAttr{});

    // Load the value from the global variable
    auto resultVar =
        rewriter.create<emitc::LoadOp>(srcOp.getLoc(), tupleType, globalVar);

    // Unpack the tuple result - extract each element from the tuple
    SmallVector<Value> results;

    for (unsigned i = 0; i < srcOp.getNumResults(); ++i) {
      // Create index value
      auto indexType = rewriter.getIndexType();
      auto indexOp = rewriter.create<emitc::LiteralOp>(
          srcOp.getLoc(), indexType, std::to_string(i));
      Value indexVal = indexOp.getResult();

      // Create LValue type for the tensor reference
      auto lvalueType = emitc::LValueType::get(
          emitc::OpaqueType::get(rewriter.getContext(), "::ttnn::Tensor"));

      // Get reference to the i-th element in the static cache result
      // Use the variable that references our global result
      auto subscriptOp = rewriter.create<emitc::SubscriptOp>(
          srcOp.getLoc(), lvalueType, resultVar.getResult(), indexVal);

      // Load the actual tensor value from the reference
      auto loadOp = rewriter.create<emitc::LoadOp>(
          srcOp.getLoc(),
          emitc::OpaqueType::get(rewriter.getContext(), "::ttnn::Tensor"),
          subscriptOp.getResult());
      results.push_back(loadOp.getResult());
    }

    // Replace the original op with the extracted results
    rewriter.replaceOp(srcOp, results);

    return success();
  }
};
} // namespace

// Module Op conversion pattern
//
// This conversion pattern removes attributes from the ModuleOp. Previously,
// ttmlir-translate would complain when translating to C++ if there were any
// attributes from "unregistered" dialects.
//
namespace {
class ModuleOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<mlir::ModuleOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      mlir::ModuleOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::ModuleOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.modifyOpInPlace(srcOp, [&]() {
      for (const NamedAttribute &attr : srcOp->getAttrs()) {
        srcOp->removeAttr(attr.getName());
      }
    });

    return success();
  }
};
} // namespace

// MeshShardOp conversion pattern
//
namespace {
class MeshShardOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<tt::ttnn::MeshShardOp> {
public:
  using TTNNToEmitCBaseOpConversionPattern<
      tt::ttnn::MeshShardOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ttnn::MeshShardOp srcOp,
                  tt::ttnn::MeshShardOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.create<emitc::VerbatimOp>(
        srcOp.getLoc(),
        "assert(0 && \"Mesh shard operation is "
        "not supported in emitc yet.\"); // ::ttnn::mesh_shard");
    ttnn_to_emitc::EmitCTTNNEmitter<tt::ttnn::MeshShardOp> emitter(
        srcOp, adaptor, rewriter);
    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
    };
    emitter.replaceOp(*this, args);
    return success();
  }
};
} // namespace

// AllGatherOp conversion pattern
//
namespace {
class AllGatherOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<tt::ttnn::AllGatherOp> {
public:
  using TTNNToEmitCBaseOpConversionPattern<
      tt::ttnn::AllGatherOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ttnn::AllGatherOp srcOp,
                  tt::ttnn::AllGatherOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ttnn_to_emitc::EmitCTTNNEmitter<tt::ttnn::AllGatherOp> emitter(
        srcOp, adaptor, rewriter);
    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getAllGatherDim()),
        emitter.emit(srcOp.getClusterAxis()),
        emitter.emit(srcOp.getDevice()),
        /*numLinks=*/emitter.emit(1),
        emitter.emit(std::nullopt) | emitter.getMemoryConfig(srcOp.getResult()),
        /*numWorkers=*/emitter.emit(std::nullopt),
        /*numBuffersPerChannel=*/emitter.emit(std::nullopt),
        /*ttnn::ccl::Topology=*/
        rewriter.getType<emitc::OpaqueAttr>("::ttnn::ccl::Topology::Linear"),
    };

    emitter.replaceOp(*this, args);
    return success();
  }
};
} // namespace

// ReduceScatterOp conversion pattern
//
namespace {
class ReduceScatterOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<tt::ttnn::ReduceScatterOp> {
public:
  using TTNNToEmitCBaseOpConversionPattern<
      tt::ttnn::ReduceScatterOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ttnn::ReduceScatterOp srcOp,
                  tt::ttnn::ReduceScatterOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ttnn_to_emitc::EmitCTTNNEmitter<tt::ttnn::ReduceScatterOp> emitter(
        srcOp, adaptor, rewriter);
    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getScatterDim()),
        emitter.emit(srcOp.getClusterAxis()),
        emitter.emit(srcOp.getDevice()),
        mlir::tt::ttnn_to_emitc::utils::convertReduceType(
            rewriter, srcOp.getReduceType()),
        /*numLinks=*/emitter.emit(1),
        emitter.emit(std::nullopt) | emitter.getMemoryConfig(srcOp.getResult()),
        /*ttnn::ccl::Topology=*/
        rewriter.getType<emitc::OpaqueAttr>("::ttnn::ccl::Topology::Linear"),
        /*userDefinedNumWorkers=*/emitter.emit(std::nullopt),
        /*userDefinedNumBuffersPerChannel=*/emitter.emit(std::nullopt),
    };

    emitter.replaceOp(*this, args);
    return success();
  }
};
} // namespace

// CollectivePermuteOp conversion pattern
//
namespace {
class CollectivePermuteOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<tt::ttnn::CollectivePermuteOp> {
public:
  using TTNNToEmitCBaseOpConversionPattern<
      tt::ttnn::CollectivePermuteOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ttnn::CollectivePermuteOp srcOp,
                  tt::ttnn::CollectivePermuteOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.create<emitc::VerbatimOp>(
        srcOp.getLoc(),
        "assert(0 && \"Collective permute operation is "
        "not supported in emitc yet.\"); // ::ttnn::collective_permute");
    ttnn_to_emitc::EmitCTTNNEmitter<tt::ttnn::CollectivePermuteOp> emitter(
        srcOp, adaptor, rewriter);
    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
    };
    emitter.replaceOp(*this, args);
    return success();
  }
};
} // namespace

// SliceOp conversion pattern
//
namespace {
class SliceOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<tt::ttnn::SliceOp> {
public:
  using TTNNToEmitCBaseOpConversionPattern<
      tt::ttnn::SliceOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ttnn::SliceOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<tt::ttnn::SliceOp> emitter(srcOp, adaptor,
                                                               rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit<::ttnn::SmallVector<int32_t>>(srcOp.getBegins()),
        emitter.emit<::ttnn::SmallVector<int32_t>>(srcOp.getEnds()),
        emitter.emit<::ttnn::SmallVector<int32_t>>(srcOp.getStep()),
        emitter.emit(std::nullopt) | emitter.getMemoryConfig(srcOp.getResult()),
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
    : public TTNNToEmitCBaseOpConversionPattern<tt::ttnn::PermuteOp> {
public:
  using TTNNToEmitCBaseOpConversionPattern<
      tt::ttnn::PermuteOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ttnn::PermuteOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitc::EmitCTTNNEmitter<tt::ttnn::PermuteOp> emitter(srcOp, adaptor,
                                                                 rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit<::ttnn::SmallVector<int64_t>>(srcOp.getPermutation()),
        emitter.emit(std::nullopt) | emitter.getMemoryConfig(srcOp.getResult()),
        emitter.emit(srcOp.getPadValue()),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

namespace mlir::tt {

// ANCHOR: op_rewriter_pattern_set_emitc
void populateTTNNToEmitCPatterns(mlir::MLIRContext *ctx,
                                 mlir::RewritePatternSet &patterns,
                                 TypeConverter &typeConverter) {
  // Device ops
  //
  patterns.add<TTDeviceOpConversionPattern>(typeConverter, ctx);
  patterns.add<GetDeviceOpConversionPattern>(typeConverter, ctx);

  // Memory ops
  //
  // clang-format off
  patterns.add<ToLayoutOpConversionPattern,
               ToMemoryConfigOpConversionPattern,
               ToDTypeOpConversionPattern,
               TypecastOpConversionPattern,
               ToDeviceOpConversionPattern,
               FromDeviceOpConversionPattern,
               DeallocateOpConversionPattern>(typeConverter, ctx);
  // clang-format on

  // Tensor ops
  //
  // clang-format off
  patterns.add<EmptyOpConversionPattern,
               NamedFullOpConversionPattern<tt::ttnn::ZerosOp>,
               NamedFullOpConversionPattern<tt::ttnn::OnesOp>,
               FullOpConversionPattern,
               DefaultOpConversionPattern<tt::ttnn::ArangeOp>,
               DefaultOpConversionPattern<tt::ttnn::ConstantOp>>(typeConverter, ctx);
  // clang-format on

  // Eltwise unary ops
  //
  patterns
      .add<EltwiseUnaryOpConversionPattern<tt::ttnn::AbsOp>,
           EltwiseUnaryCompositeOpConversionPattern<tt::ttnn::CbrtOp>,
           ClampOpConversionPattern<tt::ttnn::ClampScalarOp>,
           ClampOpConversionPattern<tt::ttnn::ClampTensorOp>,
           EltwiseUnaryOpConversionPattern<tt::ttnn::FloorOp>,
           EltwiseUnaryOpConversionPattern<tt::ttnn::IsFiniteOp>,
           EltwiseUnaryOpConversionPattern<tt::ttnn::LogicalNotOp>,
           EltwiseUnaryOpConversionPattern<tt::ttnn::BitwiseNotOp>,
           EltwiseUnaryOpConversionPattern<tt::ttnn::NegOp>,
           EltwiseUnaryOpConversionPattern<tt::ttnn::ReluOp>,
           ElementwiseUnaryWithFloatParameterOpConversionPattern<
               tt::ttnn::LeakyReluOp>,
           EltwiseUnaryWithFastAndApproximateModeOpConversionPattern<
               tt::ttnn::GeluOp>,
           EltwiseUnaryOpConversionPattern<tt::ttnn::SqrtOp>,
           EltwiseUnaryWithFastAndApproximateModeOpConversionPattern<
               tt::ttnn::RsqrtOp>,
           EltwiseUnaryOpConversionPattern<tt::ttnn::SignOp>,
           EltwiseUnaryWithVectorAndFastAndApproximateModeOpConversionPattern<
               tt::ttnn::SigmoidOp>,
           EltwiseUnaryCompositeOpConversionPattern<tt::ttnn::Log1pOp>,
           EltwiseUnaryOpConversionPattern<tt::ttnn::ReciprocalOp>,
           EltwiseUnaryWithFastAndApproximateModeOpConversionPattern<
               tt::ttnn::ExpOp>,
           EltwiseUnaryOpConversionPattern<tt::ttnn::CeilOp>,
           EltwiseUnaryOpConversionPattern<tt::ttnn::SinOp>,
           EltwiseUnaryOpConversionPattern<tt::ttnn::CosOp>,
           EltwiseUnaryOpConversionPattern<tt::ttnn::Expm1Op>,
           EltwiseUnaryOpConversionPattern<tt::ttnn::TanOp>,
           EltwiseUnaryOpConversionPattern<tt::ttnn::TanhOp>,
           EltwiseUnaryOpConversionPattern<tt::ttnn::AtanOp>,
           EltwiseUnaryOpConversionPattern<tt::ttnn::LogOp>>(typeConverter,
                                                             ctx);

  // Eltwise binary ops
  //
  patterns
      .add<EltwiseBinaryOpConversionPattern<tt::ttnn::AddOp>,
           EltwiseBinaryOpConversionPattern<tt::ttnn::SubtractOp>,
           EltwiseBinaryOpConversionPattern<tt::ttnn::MultiplyOp>,
           EltwiseBinaryOpConversionPattern<tt::ttnn::LogicalAndOp>,
           EltwiseBinaryOpConversionPattern<tt::ttnn::LogicalOrOp>,
           EltwiseBinaryOpConversionPattern<tt::ttnn::LogicalXorOp>,
           EltwiseBinaryCompositeOpConversionPattern<tt::ttnn::BitwiseAndOp>,
           EltwiseBinaryCompositeOpConversionPattern<tt::ttnn::BitwiseOrOp>,
           EltwiseBinaryCompositeOpConversionPattern<tt::ttnn::BitwiseXorOp>,
           EltwiseBinaryOpConversionPattern<tt::ttnn::EqualOp>,
           EltwiseBinaryOpConversionPattern<tt::ttnn::NotEqualOp>,
           EltwiseBinaryOpConversionPattern<tt::ttnn::GreaterEqualOp>,
           EltwiseBinaryOpConversionPattern<tt::ttnn::GreaterThanOp>,
           EltwiseBinaryOpConversionPattern<tt::ttnn::LessEqualOp>,
           EltwiseBinaryOpConversionPattern<tt::ttnn::LessThanOp>,
           EltwiseBinaryNGCompositeOpConversionPattern<tt::ttnn::MaximumOp>,
           EltwiseBinaryNGCompositeOpConversionPattern<tt::ttnn::MinimumOp>,
           EltwiseBinaryOpConversionPattern<tt::ttnn::DivideOp>,
           EltwiseBinaryCompositeOpConversionPattern<tt::ttnn::ScatterOp>,
           EltwiseBinaryCompositeOpConversionPattern<tt::ttnn::RemainderOp>,
           EltwiseBinaryNGCompositeOpConversionPattern<tt::ttnn::PowOp>,
           EltwiseBinaryCompositeOpConversionPattern<tt::ttnn::Atan2Op>>(
          typeConverter, ctx);

  // Eltwise ternary ops
  //
  patterns.add<EltwiseTernaryOpConversionPattern<tt::ttnn::WhereOp>>(
      typeConverter, ctx);

  // Tensor manipulation ops
  //
  patterns.add<TransposeOpConversionPattern, ConcatOpConversionPattern,
               ReshapeOpConversionPattern, RepeatOpConversionPattern,
               RepeatInterleaveOpConversionPattern, SliceOpConversionPattern,
               PermuteOpConversionPattern,
               DefaultOpConversionPattern<tt::ttnn::PadOp>>(typeConverter, ctx);

  // Quantization ops.
  //
  patterns.add<QuantizationOpConversionPattern<tt::ttnn::QuantizeOp>,
               QuantizationOpConversionPattern<tt::ttnn::DequantizeOp>,
               RequantizeOpConversionPattern>(typeConverter, ctx);

  // Matmul ops
  //
  patterns.add<LinearOpConversionPattern, MatmulOpConversionPattern>(
      typeConverter, ctx);

  // Reduction ops
  //
  patterns.add<ReductionOpConversionPattern<tt::ttnn::SumOp>,
               ReductionOpConversionPattern<tt::ttnn::MeanOp>,
               ReductionOpConversionPattern<tt::ttnn::MaxOp>,
               ReductionOpConversionPattern<tt::ttnn::MinOp>,
               ProdOpConversionPattern, ArgMaxOpConversionPattern>(
      typeConverter, ctx);

  // Pooling ops
  //
  patterns.add<Pooling2dOpConversionPattern<tt::ttnn::AvgPool2dOp>,
               Pooling2dOpConversionPattern<tt::ttnn::MaxPool2dOp>>(
      typeConverter, ctx);
  patterns.add<UpsampleOpConversionPattern>(typeConverter, ctx);

  // Convolution ops
  //
  patterns.add<PrepareConv2dWeightsOpConversionPattern>(typeConverter, ctx);
  patterns.add<Conv2dOpConversionPattern>(typeConverter, ctx);
  patterns.add<ConvTranspose2dOpConversionPattern>(typeConverter, ctx);

  // Other ops
  //
  patterns.add<SoftmaxOpConversionPattern, EmbeddingOpConversionPattern,
               DefaultOpConversionPattern<tt::ttnn::EmbeddingBackwardOp>,
               MorehCumSumOpConversionPattern>(typeConverter, ctx);

  // CCL ops
  //
  patterns.add<AllGatherOpConversionPattern>(typeConverter, ctx);
  patterns.add<ReduceScatterOpConversionPattern>(typeConverter, ctx);
  patterns.add<CollectivePermuteOpConversionPattern>(typeConverter, ctx);
  patterns.add<MeshShardOpConversionPattern>(typeConverter, ctx);

  // KV Cache ops
  //
  patterns.add<DefaultOpConversionPattern<tt::ttnn::UpdateCacheOp>>(
      typeConverter, ctx);
  patterns.add<DefaultOpConversionPattern<tt::ttnn::FillCacheOp>>(typeConverter,
                                                                  ctx);

  // Arith ops
  //
  patterns.add<ArithConstantOpConversionPattern>(typeConverter, ctx);

  // Tuple ops
  //
  patterns.add<GetTupleElementOpConversionPattern>(typeConverter, ctx);
  patterns.add<TupleOpConversionPattern>(typeConverter, ctx);

  // LoadCached op
  //
  patterns.add<LoadCachedOpConversionPattern>(typeConverter, ctx);

  // Module op
  //
  patterns.add<ModuleOpConversionPattern>(typeConverter, ctx);
}
// ANCHOR_END: op_rewriter_pattern_set_emitc

} // namespace mlir::tt
