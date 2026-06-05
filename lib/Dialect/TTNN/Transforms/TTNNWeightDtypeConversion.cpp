// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/BFPDtypeParser.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNWEIGHTDTYPECONVERSION
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

// Check if the given DataType is a legal per-arg weight dtype override.
static bool isLegalWeightDtype(ttcore::DataType dtype) {
  return dtype == ttcore::DataType::BFP_BFloat8 ||
         dtype == ttcore::DataType::BFP_BFloat4 ||
         dtype == ttcore::DataType::BFloat16;
}

// Template pattern to rewrite Matmul/Linear operations to use a specified
// weight dtype. This pattern matches ops where the weight (B operand) is
// bf16/f32, and inserts a typecast operation to convert it to the target dtype.
//
// The effective dtype is resolved per-op: if the op carries a
// "ttcore.weight_dtype" discardable attribute (propagated from function args
// by the TTIRPropagateWeightDtype pass), that per-op dtype takes priority.
// Otherwise the global targetDtype (from the pass option) is used.
template <typename OpTy>
class WeightDtypeConversionPattern : public mlir::OpRewritePattern<OpTy> {
public:
  WeightDtypeConversionPattern(mlir::MLIRContext *ctx,
                               std::optional<ttcore::DataType> targetDtype)
      : mlir::OpRewritePattern<OpTy>(ctx), targetDtype(targetDtype) {}

  mlir::LogicalResult
  matchAndRewrite(OpTy op, mlir::PatternRewriter &rewriter) const override {
    // Get the weight operand (B operand).
    auto weight = op.getB();

    // Check if weight traces to constant/parameter arguments.
    if (!ttcore::valueTracesToConstantArgs(weight)) {
      return mlir::failure();
    }

    // Resolve effective dtype: per-op annotation takes priority over global.
    std::optional<ttcore::DataType> effectiveDtype = targetDtype;
    if (auto dtypeStrAttr = op->template getAttrOfType<mlir::StringAttr>(
            "ttcore.weight_dtype")) {
      auto perOpDtype = ttcore::DataTypeStringToEnum(dtypeStrAttr.getValue());
      if (perOpDtype && isLegalWeightDtype(*perOpDtype)) {
        effectiveDtype = perOpDtype;
      } else {
        op.emitWarning("ignoring invalid ttcore.weight_dtype \"")
            << dtypeStrAttr.getValue()
            << "\"; legal values are: bfp_bf8, bfp_bf4, bf16";
      }
    }

    // No per-op annotation and no global dtype.
    if (!effectiveDtype) {
      return mlir::failure();
    }

    ttcore::DataType dtype = *effectiveDtype;
    auto weightType = weight.getType();

    // Check if weight is already the target dtype or is convertible (bf16/f32).
    mlir::Type elType = weightType.getElementType();
    if (auto tileType = mlir::dyn_cast<ttcore::TileType>(elType)) {
      if (tileType.getDataType() == dtype) {
        return mlir::failure();
      }
      if (tileType.getDataType() != ttcore::DataType::BFloat16 &&
          tileType.getDataType() != ttcore::DataType::Float32) {
        return mlir::failure();
      }
    } else {
      if (!mlir::isa<mlir::BFloat16Type, mlir::Float32Type>(elType)) {
        return mlir::failure();
      }
      // Already the target dtype — no conversion needed.
      if ((dtype == ttcore::DataType::BFloat16 &&
           mlir::isa<mlir::BFloat16Type>(elType)) ||
          (dtype == ttcore::DataType::Float32 &&
           mlir::isa<mlir::Float32Type>(elType))) {
        return mlir::failure();
      }
    }

    // Create new tensor type with target element type.
    auto newWeightType =
        ttnn::utils::RankedTensorTypeFactory::create(weightType, dtype);

    // Blockfloat targets (BFP_BFloat4, BFP_BFloat8) go through a host-side
    // typecast via: from_device → typecast (host) → to_device.
    // The host typecast dispatches to tt-metal's host packer for BFP
    // formats. Const-eval results are cached at compile time, so the
    // host roundtrip is paid once per cached weight per program.
    //
    // Other targets (e.g. BFloat16 per-tensor override) keep the existing
    // single-`typecast` device codepath.
    mlir::Value newWeight;
    if (dtype == ttcore::DataType::BFP_BFloat4 ||
        dtype == ttcore::DataType::BFP_BFloat8) {
      auto hostInputType = ttnn::utils::RankedTensorTypeFactory::create(
          mlir::cast<RankedTensorType>(weight.getType()),
          ttnn::BufferType::SystemMemory);
      auto fromDevOp =
          rewriter.create<FromDeviceOp>(op.getLoc(), hostInputType, weight);

      auto hostOutputType =
          ttnn::utils::RankedTensorTypeFactory::create(hostInputType, dtype);
      auto typecastOp = rewriter.create<TypecastOp>(op.getLoc(), hostOutputType,
                                                    fromDevOp.getResult());

      mlir::Value device = ttnn::utils::getOrInsertDevice(rewriter, op);
      auto toDevOp = rewriter.create<ToDeviceOp>(
          op.getLoc(), newWeightType, typecastOp.getResult(), device);
      newWeight = toDevOp.getResult();
    } else {
      // Single device typecast for non-blockfloat targets.
      auto typecastOp =
          rewriter.create<TypecastOp>(op.getLoc(), newWeightType, weight);
      newWeight = typecastOp.getResult();
    }

    // Update op to use the new weight. The per-op "ttcore.weight_dtype"
    // attribute is intentionally kept: the greedy rewriter may re-match this
    // op, and without the attribute it would fall back to the global default,
    // defeating per-op overrides (e.g. bf16 overridden by global bfp_bf8).
    // The attribute is discardable and harmless in the final IR.
    rewriter.modifyOpInPlace(op, [&]() { op.getBMutable().assign(newWeight); });

    return mlir::success();
  }

private:
  std::optional<ttcore::DataType> targetDtype;
};

// Weight-dtype conversion for the opaque ttnn.tt_lang_op (the streaming
// selective-experts matmul). Unlike Matmul/Linear, tt_lang_op has no `getB()`
// and the ttcore.weight_dtype annotation is never propagated onto it, so we
// trace operand 1 (in1 = expert weights, per the stream kernel's arg_roles)
// back to its originating func arg and read the arg's ttcore.weight_dtype
// directly. If a block format is requested, insert from_device -> host typecast
// -> to_device and rebind operand 1 so the weights stream as bfp8/bfp4.
class TtLangWeightDtypeConversionPattern
    : public mlir::OpRewritePattern<TtLangOp> {
public:
  TtLangWeightDtypeConversionPattern(mlir::MLIRContext *ctx,
                                     std::optional<ttcore::DataType> targetDtype)
      : mlir::OpRewritePattern<TtLangOp>(ctx), targetDtype(targetDtype) {}

  mlir::LogicalResult
  matchAndRewrite(TtLangOp op, mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 2) {
      return mlir::failure();
    }
    mlir::Value weight = op->getOperand(1);

    // The per-op ttcore.weight_dtype is set by TTIRPropagateWeightDtype (which
    // traces in1 back to the func arg through TM/CCL/shard ops) and forwarded
    // across the TTIR->TTNN conversion. Global dtype is the fallback.
    std::optional<ttcore::DataType> effectiveDtype = targetDtype;
    if (auto dtypeAttr =
            op->getAttrOfType<mlir::StringAttr>("ttcore.weight_dtype")) {
      auto perOp = ttcore::DataTypeStringToEnum(dtypeAttr.getValue());
      if (perOp && isLegalWeightDtype(*perOp)) {
        effectiveDtype = perOp;
      }
    }
    if (!effectiveDtype) {
      return mlir::failure();
    }
    ttcore::DataType dtype = *effectiveDtype;

    auto weightType = mlir::cast<RankedTensorType>(weight.getType());
    mlir::Type elType = weightType.getElementType();
    if (auto tileType = mlir::dyn_cast<ttcore::TileType>(elType)) {
      if (tileType.getDataType() == dtype) {
        return mlir::failure();
      }
      if (tileType.getDataType() != ttcore::DataType::BFloat16 &&
          tileType.getDataType() != ttcore::DataType::Float32) {
        return mlir::failure();
      }
    } else if (!mlir::isa<mlir::BFloat16Type, mlir::Float32Type>(elType)) {
      return mlir::failure();
    }

    auto newWeightType =
        ttnn::utils::RankedTensorTypeFactory::create(weightType, dtype);
    mlir::Value newWeight;
    if (dtype == ttcore::DataType::BFP_BFloat4 ||
        dtype == ttcore::DataType::BFP_BFloat8) {
      auto hostInputType = ttnn::utils::RankedTensorTypeFactory::create(
          weightType, ttnn::BufferType::SystemMemory);
      auto fromDevOp =
          rewriter.create<FromDeviceOp>(op.getLoc(), hostInputType, weight);
      auto hostOutputType =
          ttnn::utils::RankedTensorTypeFactory::create(hostInputType, dtype);
      auto typecastOp = rewriter.create<TypecastOp>(op.getLoc(), hostOutputType,
                                                    fromDevOp.getResult());
      mlir::Value device = ttnn::utils::getOrInsertDevice(rewriter, op);
      auto toDevOp = rewriter.create<ToDeviceOp>(
          op.getLoc(), newWeightType, typecastOp.getResult(), device);
      newWeight = toDevOp.getResult();
    } else {
      auto typecastOp =
          rewriter.create<TypecastOp>(op.getLoc(), newWeightType, weight);
      newWeight = typecastOp.getResult();
    }

    rewriter.modifyOpInPlace(op, [&]() { op->setOperand(1, newWeight); });
    return mlir::success();
  }

private:
  std::optional<ttcore::DataType> targetDtype;
};

// Only block formats (bfp_bf8, bfp_bf4) are supported as global overrides
// because models already arrive from frontends in bf16 — there is no need for a
// global bf16 override. Per-tensor overrides (via ttcore.weight_dtype op
// attribute) do support bf16 to let individual weights opt out of a
// global block-format conversion.
class TTNNWeightDtypeConversionPass
    : public impl::TTNNWeightDtypeConversionBase<
          TTNNWeightDtypeConversionPass> {
public:
  using impl::TTNNWeightDtypeConversionBase<
      TTNNWeightDtypeConversionPass>::TTNNWeightDtypeConversionBase;

  void runOnOperation() final {
    // Resolve global target dtype (std::nullopt if not specified).
    std::optional<ttcore::DataType> globalDtype;
    if (targetDtype != BFPDtype::None) {
      globalDtype = bfpDtypeToDataType(targetDtype);
    }

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<WeightDtypeConversionPattern<MatmulOp>,
                 WeightDtypeConversionPattern<LinearOp>,
                 WeightDtypeConversionPattern<SparseMatmulOp>,
                 TtLangWeightDtypeConversionPattern>(&getContext(), globalDtype);

    if (failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

} // namespace mlir::tt::ttnn
