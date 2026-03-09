// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"

#ifdef TTMLIR_ENABLE_OPMODEL
#include "ttmlir/Dialect/TTNN/Transforms/Fusing/FusionValidator.h"
#include "ttmlir/Dialect/TTNN/Transforms/Fusing/RoPEFusingPattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Fusing/SDPAFusingPattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Fusing/TopKFusingPattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/NLPConcatHeadsDecodeInputRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"
#endif

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNFUSING
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

template <typename ActivationOp>
class TTNNConv2dWithActivation : public mlir::OpRewritePattern<Conv2dOp> {
  using TTNNConv2dWithActivation::OpRewritePattern<Conv2dOp>::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(Conv2dOp srcOp, mlir::PatternRewriter &rewriter) const final {
    if (!isFusable(srcOp)) {
      return failure();
    }

    ActivationOp activationOp = getActivationOp(srcOp);
    Value activationInput = activationOp.getInput();

    auto activation = getActivationOpType(rewriter);

    ttcore::DataType weightDtype = ttcore::elementTypeToDataType(
        srcOp.getWeight().getType().getElementType());
    Conv2dConfigAttr conv2dConfigAttr =
        srcOp.getConv2dConfigAttr()
            ? srcOp.getConv2dConfigAttr()
            : Conv2dConfigAttr::get(rewriter.getContext());
    conv2dConfigAttr = conv2dConfigAttr.withActivation(activation)
                           .withWeightsDtype(weightDtype);

    rewriter.modifyOpInPlace(
        srcOp, [&]() { srcOp.setConv2dConfigAttr(conv2dConfigAttr); });

    // Replace the activation op uses with either conv2d or reshape
    // depending on if reshape was present.
    rewriter.replaceAllUsesWith(activationOp, activationInput);

    return mlir::success();
  }

private:
  ActivationOp getActivationOp(Conv2dOp srcOp) const {
    assert((ttmlir::utils::allUsersOfType<ReshapeOp, ActivationOp>(srcOp)) &&
           "Conv2d should have either activation or Reshape as user.");

    if (ttmlir::utils::allUsersOfType<ActivationOp>(srcOp)) {
      return mlir::cast<ActivationOp>(*srcOp.getResult().getUsers().begin());
    }

    ReshapeOp reshapeOp =
        mlir::cast<ReshapeOp>(*srcOp.getResult().getUsers().begin());

    assert(reshapeOp.getResult().hasOneUse() &&
           ttmlir::utils::allUsersOfType<ActivationOp>(reshapeOp) &&
           "Reshape should have only one user and that user should be "
           "activation.");
    return mlir::cast<ActivationOp>(*reshapeOp.getResult().getUsers().begin());
  }

  ttnn::UnaryOpType getActivationOpType(mlir::PatternRewriter &rewriter) const {
    // Extract op name from full operation name (e.g., "ttnn.relu" -> "relu")
    // and convert to enum
    llvm::StringLiteral fullOpName = ActivationOp::getOperationName();
    llvm::StringRef opName = fullOpName.rsplit('.').second;
    auto activation = ttnn::symbolizeUnaryOpType(opName);
    assert(activation.has_value() && "Unsupported activation op");
    return activation.value();
  }

  bool isFusable(Conv2dOp srcOp) const {
    if (srcOp.getConv2dConfig() && srcOp.getConv2dConfig()->hasActivation()) {
      return false;
    }

    // Conv2d has multiple uses so we cannot fuse.
    if (!srcOp.getResult().hasOneUse()) {
      return false;
    }

    // Conv2d only user is activation so we can fuse.
    if (ttmlir::utils::allUsersOfType<ActivationOp>(srcOp)) {
      return true;
    }

    // Since window flattening will add reshape after conv we need to check
    // if there is reshape right after conv2d.
    if (!ttmlir::utils::allUsersOfType<ReshapeOp>(srcOp)) {
      return false;
    }

    ReshapeOp reshapeOp =
        mlir::cast<ReshapeOp>(*srcOp.getResult().getUsers().begin());

    // If we want to fuse activation to conv we need to make sure that reshape
    // has only one user and that user is activation.
    return reshapeOp.getResult().hasOneUse() &&
           ttmlir::utils::allUsersOfType<ActivationOp>(reshapeOp);
  }
};

template <typename SrcOp, typename ActivationOp>
class TTNNMatmulAndLinearWithActivation : public mlir::OpRewritePattern<SrcOp> {
  using TTNNMatmulAndLinearWithActivation::template OpRewritePattern<
      SrcOp>::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(SrcOp srcOp, mlir::PatternRewriter &rewriter) const final {
    if (!isFusable(srcOp)) {
      return failure();
    }

    ActivationOp activationOp =
        mlir::cast<ActivationOp>(*srcOp.getResult().getUsers().begin());
    Value activationInput = activationOp.getInput();
    auto activationStr = getActivationString();

    rewriter.modifyOpInPlace(srcOp, [&]() {
      srcOp.setActivationAttr(rewriter.getStringAttr(activationStr));
    });

    rewriter.replaceAllUsesWith(activationOp, activationInput);
    return mlir::success();
  }

private:
  // After tt-metal resolves this issue:
  // https://github.com/tenstorrent/tt-metal/issues/31393, we can use the
  // UnaryWithParam enum directly instead of string.
  std::string getActivationString() const {
    llvm::StringLiteral fullOpName = ActivationOp::getOperationName();
    llvm::StringRef opName = fullOpName.rsplit('.').second;
    return opName.str();
  }

  bool isFusable(SrcOp srcOp) const {
    if (srcOp.getActivation()) {
      return false;
    }

    if (!srcOp.getResult().hasOneUse()) {
      return false;
    }

    if (ttmlir::utils::allUsersOfType<ActivationOp>(srcOp)) {
      return true;
    }

    return false;
  }
};

#ifdef TTMLIR_ENABLE_OPMODEL

// Extract a scalar float constant from a value by looking through transparent
// TTNN ops and optional const-eval load_cached wrappers.
static std::optional<float> extractFloatConstant(Value v) {
  v = ttmlir::utils::lookThrough<ToLayoutOp, ToMemoryConfigOp, TypecastOp>(v);

  if (auto fullOp = v.getDefiningOp<FullOp>()) {
    if (auto attr = mlir::dyn_cast<FloatAttr>(fullOp.getFillValue())) {
      return attr.getValue().convertToFloat();
    }
  }

  if (auto loadCached = v.getDefiningOp<ttcore::LoadCachedOp>()) {
    auto callee = loadCached.getCallee();
    auto moduleOp = loadCached->getParentOfType<ModuleOp>();
    if (!moduleOp) {
      return std::nullopt;
    }

    auto funcOp = moduleOp.lookupSymbol<func::FuncOp>(callee);
    if (!funcOp) {
      return std::nullopt;
    }

    std::optional<float> result;
    funcOp.walk([&](FullOp fullOp) {
      if (auto attr = mlir::dyn_cast<FloatAttr>(fullOp.getFillValue())) {
        result = attr.getValue().convertToFloat();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return result;
  }

  return std::nullopt;
}

// ============================================================================
// Distributed RMSNorm Fusing
// ============================================================================
//
// This matcher is split into two logical parts:
// - matchRMSNormPostAllGather: matches the normalization tail:
//   add(multiply(stats, scale), epsilon) -> rsqrt -> multiply(x, rsqrt) ->
//   typecast -> multiply(weight, ...)
// - matchRMSNormPreAllGather: matches local stats production:
//   sum(pow_scalar(x, 2), dim=[2]) [with optional reshapes]
//
// Depending on whether there is a collective between these parts:
// - all_reduce(sum) between pre/post + valid distributed input shape
//   => fuse to ttnn.distributed_rms_norm
// - no collective (single-chip/local stats path)
//   => fuse to ttnn.rms_norm
//
// TODO: Add dedicated fusing to explicitly produce
// rms_norm_pre_all_gather and rms_norm_post_all_gather once those staged
// ops/rewrite boundaries are available in TTNN.
class DistributedRMSNormFusing : public mlir::OpRewritePattern<MultiplyOp> {
  using DistributedRMSNormFusing::OpRewritePattern<MultiplyOp>::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(MultiplyOp srcOp, mlir::PatternRewriter &rewriter) const final {
    RMSNormMatch match;
    if (failed(matchRMSNormPostAllGather(srcOp, match))) {
      llvm::errs() << "[RMSNormFusing] post-all-gather match failed: " << srcOp
                   << "\n";
      return failure();
    }

    Value statsProducer =
        ttmlir::utils::lookThrough<ToLayoutOp, ToMemoryConfigOp, TypecastOp>(
            match.statsValue);

    // Distributed path: local stats are reduced across devices before
    // normalization. This corresponds to the current distributed_rms_norm
    // decomposition.
    if (auto allReduce = statsProducer.getDefiningOp<AllReduceOp>()) {
      if (allReduce.getReduceType() != ttcore::ReduceType::Sum) {
        llvm::errs() << "[RMSNormFusing] all_reduce is not Sum: " << srcOp
                     << "\n";
        return failure();
      }
      if (failed(matchRMSNormPreAllGather(allReduce.getInput(), match))) {
        llvm::errs()
            << "[RMSNormFusing] pre-all-gather match failed after all_reduce: "
            << srcOp << "\n";
        return failure();
      }
      return rewriteToDistributedRMSNorm(srcOp, allReduce, match, rewriter);
    }

    // Single-chip/local path: no collective between pre/post pieces.
    //
    // TODO: If this is a staged distributed decomposition that does not match
    // all_reduce(sum), split it into explicit rms_norm_pre_all_gather and
    // rms_norm_post_all_gather fusing once those boundaries are materialized.
    if (failed(matchRMSNormPreAllGather(statsProducer, match))) {
      llvm::errs() << "[RMSNormFusing] pre-all-gather match failed on local "
                      "path: "
                   << srcOp << "\n";
      return failure();
    }
    return rewriteToRMSNorm(srcOp, match, rewriter);
  }

private:
  mlir::LogicalResult failWithReason(llvm::StringRef reason) const {
    llvm::errs() << "[RMSNormFusing] " << reason << "\n";
    return failure();
  }

  struct RMSNormMatch {
    Value weight;
    Value normalizedSourceF32;
    Value statsValue;
    std::optional<float> epsilon;
    std::optional<float> scale;

    SumOp sumOp = nullptr;
    MeanOp meanOp = nullptr;
    Value bf16Input;
  };

  mlir::LogicalResult matchRMSNormPostAllGather(MultiplyOp srcOp,
                                                RMSNormMatch &match) const {
    // Root multiply must have users.
    if (!srcOp || srcOp->getNumResults() != 1 ||
        ttmlir::utils::countUsers(srcOp->getResult(0)) == 0) {
      llvm::errs()
          << "[RMSNormFusing] root multiply has no users or invalid result arity\n";
      return failure();
    }

    // root = multiply(weight, normalized_bf16) (operand order is commutative)
    TypecastOp normalizedTypecast = nullptr;
    if (auto lhsTc = srcOp.getLhs().getDefiningOp<TypecastOp>()) {
      normalizedTypecast = lhsTc;
      match.weight = srcOp.getRhs();
    } else if (auto rhsTc = srcOp.getRhs().getDefiningOp<TypecastOp>()) {
      normalizedTypecast = rhsTc;
      match.weight = srcOp.getLhs();
    } else {
      return failWithReason("root multiply does not have typecast operand");
    }

    if (!normalizedTypecast || normalizedTypecast->getNumResults() != 1 ||
        ttmlir::utils::countUsers(normalizedTypecast->getResult(0)) != 1) {
      return failWithReason(
          "normalized typecast must have exactly one result and one user");
    }

    auto normalizedPreCastType =
        mlir::cast<RankedTensorType>(normalizedTypecast.getInput().getType());
    auto normalizedType =
        mlir::cast<RankedTensorType>(normalizedTypecast.getType());
    if (!normalizedPreCastType.getElementType().isF32() ||
        !normalizedType.getElementType().isBF16()) {
      return failWithReason("normalized branch requires f32 -> bf16 typecast");
    }

    auto normalizedMul =
        normalizedTypecast.getInput().getDefiningOp<MultiplyOp>();
    if (!normalizedMul || normalizedMul->getNumResults() != 1 ||
        ttmlir::utils::countUsers(normalizedMul->getResult(0)) != 1) {
      return failWithReason(
          "normalized pre-cast producer must be single-use multiply");
    }

    // normalizedMul = multiply([reshape](source_f32), [reshape](rsqrt(...)))
    // Look through layout/memory/reshape wrappers on both branches.
    RsqrtOp rsqrtOp = nullptr;
    Value lhsCandidate =
        ttmlir::utils::lookThrough<ToLayoutOp, ToMemoryConfigOp, ReshapeOp>(
            normalizedMul.getLhs());
    Value rhsCandidate =
        ttmlir::utils::lookThrough<ToLayoutOp, ToMemoryConfigOp, ReshapeOp>(
            normalizedMul.getRhs());

    if (auto lhsRsqrt = lhsCandidate.getDefiningOp<RsqrtOp>()) {
      rsqrtOp = lhsRsqrt;
      match.normalizedSourceF32 = rhsCandidate;
    } else if (auto rhsRsqrt = rhsCandidate.getDefiningOp<RsqrtOp>()) {
      rsqrtOp = rhsRsqrt;
      match.normalizedSourceF32 = lhsCandidate;
    } else {
      return failWithReason("normalized multiply has no rsqrt operand");
    }

    if (!rsqrtOp || !match.normalizedSourceF32 || rsqrtOp->getNumResults() != 1 ||
        ttmlir::utils::countUsers(rsqrtOp->getResult(0)) != 1) {
      return failWithReason("rsqrt branch is invalid or has non-single use");
    }

    auto addOp = rsqrtOp.getInput().getDefiningOp<AddOp>();
    if (!addOp || addOp->getNumResults() != 1 ||
        ttmlir::utils::countUsers(addOp->getResult(0)) != 1) {
      return failWithReason("rsqrt input must be single-use add");
    }

    // add = multiply(stats, scale) + epsilon_const
    //    or stats + epsilon_const (for mean(x^2) lowering)
    MultiplyOp meanScaleMul = nullptr;
    if (auto lhsMul = addOp.getLhs().getDefiningOp<MultiplyOp>()) {
      meanScaleMul = lhsMul;
      match.epsilon = extractConstant(addOp.getRhs());
    } else if (auto rhsMul = addOp.getRhs().getDefiningOp<MultiplyOp>()) {
      meanScaleMul = rhsMul;
      match.epsilon = extractConstant(addOp.getLhs());
    }

    ReshapeOp statsReshape = nullptr;
    if (meanScaleMul) {
      if (!match.epsilon || meanScaleMul->getNumResults() != 1 ||
          ttmlir::utils::countUsers(meanScaleMul->getResult(0)) != 1) {
        return failWithReason(
            "mean scale multiply must be single-use with epsilon const");
      }

      // meanScaleMul = multiply([reshape](stats), scale_const)
      if (auto lhsReshape = meanScaleMul.getLhs().getDefiningOp<ReshapeOp>()) {
        statsReshape = lhsReshape;
        match.statsValue = statsReshape.getInput();
        match.scale = extractConstant(meanScaleMul.getRhs());
      } else if (auto rhsReshape =
                     meanScaleMul.getRhs().getDefiningOp<ReshapeOp>()) {
        statsReshape = rhsReshape;
        match.statsValue = statsReshape.getInput();
        match.scale = extractConstant(meanScaleMul.getLhs());
      } else {
        // Some TTNN pipelines simplify away the reshape before scale multiply.
        if (extractConstant(meanScaleMul.getRhs())) {
          match.statsValue = meanScaleMul.getLhs();
          match.scale = extractConstant(meanScaleMul.getRhs());
        } else if (extractConstant(meanScaleMul.getLhs())) {
          match.statsValue = meanScaleMul.getRhs();
          match.scale = extractConstant(meanScaleMul.getLhs());
        }
      }
    } else {
      // Direct mean-like form: add(stats, epsilon_const)
      auto lhsConst = extractConstant(addOp.getLhs());
      auto rhsConst = extractConstant(addOp.getRhs());
      if (lhsConst && !rhsConst) {
        match.epsilon = lhsConst;
        match.statsValue = addOp.getRhs();
        match.scale = 1.0f;
      } else if (rhsConst && !lhsConst) {
        match.epsilon = rhsConst;
        match.statsValue = addOp.getLhs();
        match.scale = 1.0f;
      } else {
        return failWithReason(
            "add must be stats(+optional scale) plus epsilon constant");
      }
    }

    if (!match.statsValue || !match.scale || match.epsilon.value() <= 0.0f) {
      return failWithReason(
          "failed to recover stats/scale/epsilon or epsilon <= 0");
    }
    if (statsReshape &&
        (statsReshape->getNumResults() != 1 ||
         ttmlir::utils::countUsers(statsReshape->getResult(0)) != 1)) {
      return failWithReason("stats reshape before scale is not single-use");
    }
    return success();
  }

  mlir::LogicalResult matchRMSNormPreAllGather(Value statsInput,
                                               RMSNormMatch &match) const {
    auto sumInputReshape = statsInput.getDefiningOp<ReshapeOp>();
    Value sumCandidate = statsInput;
    if (sumInputReshape) {
      if (sumInputReshape->getNumResults() != 1 ||
          ttmlir::utils::countUsers(sumInputReshape->getResult(0)) != 1) {
        return failWithReason("stats input reshape is not single-use");
      }
      sumCandidate = sumInputReshape.getInput();
    }

    auto getActualDim = [](int64_t dim, int64_t rank) {
      return dim < 0 ? rank + dim : dim;
    };

    auto sumOp = sumCandidate.getDefiningOp<SumOp>();
    auto meanOp = sumCandidate.getDefiningOp<MeanOp>();
    Value reduceInput;
    if (sumOp) {
      if (sumOp->getNumResults() != 1 ||
          ttmlir::utils::countUsers(sumOp->getResult(0)) != 1 ||
          sumOp.getKeepDim()) {
        return failWithReason("sum reduce must be single-use with keep_dim=false");
      }
      if (!sumOp.getDimArg()) {
        return failWithReason("sum reduce has no dim_arg");
      }
      auto dims = ttmlir::utils::getIntegerVector<int64_t>(*sumOp.getDimArg());
      if (!dims || dims->size() != 1) {
        return failWithReason("sum reduce must reduce exactly one dim");
      }
      auto reduceInputType =
          mlir::cast<RankedTensorType>(sumOp.getInput().getType());
      if (getActualDim(dims->front(), reduceInputType.getRank()) !=
          reduceInputType.getRank() - 1) {
        return failWithReason("sum reduce dim must be the last dim");
      }
      reduceInput = sumOp.getInput();
      match.sumOp = sumOp;
      match.meanOp = nullptr;
    } else if (meanOp) {
      if (meanOp->getNumResults() != 1 ||
          ttmlir::utils::countUsers(meanOp->getResult(0)) != 1 ||
          !meanOp.getKeepDim()) {
        return failWithReason(
            "mean reduce must be single-use with keep_dim=true");
      }
      if (!meanOp.getDimArg()) {
        return failWithReason("mean reduce has no dim_arg");
      }
      auto dims = ttmlir::utils::getIntegerVector<int64_t>(*meanOp.getDimArg());
      if (!dims || dims->size() != 1) {
        return failWithReason("mean reduce must reduce exactly one dim");
      }
      auto reduceInputType =
          mlir::cast<RankedTensorType>(meanOp.getInput().getType());
      if (getActualDim(dims->front(), reduceInputType.getRank()) !=
          reduceInputType.getRank() - 1) {
        return failWithReason("mean reduce dim must be the last dim");
      }
      reduceInput = meanOp.getInput();
      match.meanOp = meanOp;
      match.sumOp = nullptr;
    } else {
      return failWithReason("stats producer is neither sum nor mean");
    }

    auto powOp = reduceInput.getDefiningOp<PowScalarOp>();
    if (!powOp || powOp->getNumResults() != 1 ||
        ttmlir::utils::countUsers(powOp->getResult(0)) != 1 || !isPow2(powOp)) {
      return failWithReason(
          "reduce input must be single-use pow_scalar(x, 2.0)");
    }

    auto powInputReshape = powOp.getLhs().getDefiningOp<ReshapeOp>();
    if (powInputReshape &&
        (powInputReshape->getNumResults() != 1 ||
         ttmlir::utils::countUsers(powInputReshape->getResult(0)) != 1)) {
      return failWithReason("pow input reshape is not single-use");
    }

    Value sourceF32 =
        powInputReshape ? powInputReshape.getInput() : powOp.getLhs();
    if (sourceF32 != match.normalizedSourceF32) {
      return failWithReason("pow source does not match normalized source");
    }

    auto sourceTypecast = sourceF32.getDefiningOp<TypecastOp>();
    if (!sourceTypecast) {
      return failWithReason("normalized source is not produced by typecast");
    }

    auto sourceType = mlir::cast<RankedTensorType>(sourceTypecast.getType());
    auto sourceInputType =
        mlir::cast<RankedTensorType>(sourceTypecast.getInput().getType());
    if (!sourceType.getElementType().isF32() ||
        !sourceInputType.getElementType().isBF16()) {
      return failWithReason("normalized source typecast must be bf16 -> f32");
    }

    match.bf16Input = sourceTypecast.getInput();
    return success();
  }

  mlir::LogicalResult rewriteToDistributedRMSNorm(
      MultiplyOp srcOp, AllReduceOp allReduce, RMSNormMatch &match,
      mlir::PatternRewriter &rewriter) const {
    Value bf16Input = match.bf16Input;

    op_model::ScopedSingletonDeviceGuard deviceGuard(srcOp.getOperation());
    auto device = utils::getOrInsertDevice(rewriter, srcOp);

    constexpr int64_t kExpectedInputRank = 4;
    constexpr int64_t kExpectedDim2 = 32;
    auto fusedInputType = mlir::cast<RankedTensorType>(bf16Input.getType());
    if (fusedInputType.getRank() > 0 &&
        fusedInputType.getRank() < kExpectedInputRank) {
      SmallVector<int64_t> expandedShape;
      expandedShape.reserve(kExpectedInputRank);
      for (int64_t i = 0; i < kExpectedInputRank - fusedInputType.getRank();
           ++i) {
        expandedShape.push_back(1);
      }
      llvm::append_range(expandedShape, fusedInputType.getShape());
      auto expandedType =
          utils::RankedTensorTypeFactory::create(fusedInputType, expandedShape);

      SmallVector<int32_t> expandedShapeI32;
      expandedShapeI32.reserve(expandedShape.size());
      for (int64_t dim : expandedShape) {
        expandedShapeI32.push_back(static_cast<int32_t>(dim));
      }

      bf16Input = rewriter
                      .create<ReshapeOp>(srcOp.getLoc(), expandedType, bf16Input,
                                         rewriter.getI32ArrayAttr(expandedShapeI32),
                                         ttnn::MemoryConfigAttr())
                      .getResult();
      fusedInputType = mlir::cast<RankedTensorType>(bf16Input.getType());
    }

    auto inputShape = fusedInputType.getShape();
    if (fusedInputType.getRank() != kExpectedInputRank || inputShape[0] != 1 ||
        inputShape[1] != 1 || inputShape[2] != kExpectedDim2 ||
        ShapedType::isDynamic(inputShape[3]) ||
        inputShape[3] % kExpectedDim2 != 0) {
      return rewriter.notifyMatchFailure(
          srcOp,
          "distributed_rms_norm expects input shape (1, 1, 32, M), where M is "
          "static and divisible by 32");
    }

    Value normalizedWeight = nullptr;
    if (failed(normalizeWeightForRMSNorm(
            srcOp.getLoc(), match.weight, bf16Input, rewriter, normalizedWeight))) {
      return failure();
    }

    auto distResultType = fusedInputType;
    auto distResultLayout =
        mlir::dyn_cast_or_null<TTNNLayoutAttr>(distResultType.getEncoding());
    auto distMemoryConfig =
        distResultLayout
            ? MemoryConfigAttr::get(distResultLayout, distResultLayout.getGrid())
            : MemoryConfigAttr();

    auto distRMSNormOp = rewriter.create<DistributedRMSNormOp>(
        srcOp.getLoc(), distResultType, bf16Input, normalizedWeight,
        /*residual=*/nullptr,
        /*stats=*/nullptr, device.getResult(), allReduce.getClusterAxis(),
        llvm::APFloat(match.epsilon.value()),
        /*sub_device_id=*/allReduce.getSubDeviceIdAttr(),
        /*memory_config=*/distMemoryConfig,
        /*num_links=*/allReduce.getNumLinksAttr(),
        /*topology=*/allReduce.getTopologyAttr(),
        /*compute_config=*/match.sumOp ? match.sumOp.getComputeConfigAttr()
                                       : match.meanOp.getComputeConfigAttr(),
        /*program_config=*/nullptr);

    rewriter.replaceOp(srcOp,
                       reshapeResultIfNeeded(rewriter, srcOp, distRMSNormOp));
    return success();
  }

  mlir::LogicalResult rewriteToRMSNorm(MultiplyOp srcOp, RMSNormMatch &match,
                                       mlir::PatternRewriter &rewriter) const {
    auto outputType = mlir::cast<RankedTensorType>(match.bf16Input.getType());
    auto outputLayout =
        mlir::dyn_cast_or_null<TTNNLayoutAttr>(outputType.getEncoding());
    auto memoryConfig =
        outputLayout ? MemoryConfigAttr::get(outputLayout, outputLayout.getGrid())
                     : MemoryConfigAttr();

    Value normalizedWeight = nullptr;
    if (failed(normalizeWeightForRMSNorm(srcOp.getLoc(), match.weight,
                                         match.bf16Input, rewriter,
                                         normalizedWeight))) {
      return failure();
    }

    auto rmsNorm = rewriter.create<RMSNormOp>(
        srcOp.getLoc(), outputType, match.bf16Input, normalizedWeight,
        /*bias=*/nullptr, llvm::APFloat(match.epsilon.value()), memoryConfig,
        match.sumOp ? match.sumOp.getComputeConfigAttr()
                    : match.meanOp.getComputeConfigAttr());

    rewriter.replaceOp(srcOp, reshapeResultIfNeeded(rewriter, srcOp, rmsNorm));
    return success();
  }

  mlir::LogicalResult
  normalizeWeightForRMSNorm(Location loc, Value weight, Value input,
                            mlir::PatternRewriter &rewriter,
                            Value &normalizedWeight) const {
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    if (inputType.getRank() == 0 ||
        ShapedType::isDynamic(inputType.getShape().back())) {
      return failWithReason(
          "rms_norm input must have static last dimension to normalize weight");
    }
    int64_t normalizedDim = inputType.getShape().back();

    auto weightType = mlir::dyn_cast<RankedTensorType>(weight.getType());
    if (!weightType) {
      return failWithReason("rms_norm weight must be ranked tensor");
    }

    auto isCompatibleLastDim = [&](int64_t dim) {
      return ShapedType::isDynamic(dim) || dim == normalizedDim;
    };

    auto isSqueezableWeightShape = [&](RankedTensorType type) {
      if (type.getRank() < 1) {
        return false;
      }
      auto shape = type.getShape();
      for (int64_t i = 0; i < type.getRank() - 1; ++i) {
        if (ShapedType::isDynamic(shape[i]) || shape[i] != 1) {
          return false;
        }
      }
      return isCompatibleLastDim(shape.back());
    };

    // TTIR-style behavior: try to recover a canonical gamma source by looking
    // through broadcast/reshape chains before creating a new reshape.
    Value baseWeight = weight;
    while (true) {
      if (auto toLayout = baseWeight.getDefiningOp<ToLayoutOp>()) {
        baseWeight = toLayout.getInput();
        continue;
      }
      if (auto toMemoryConfig = baseWeight.getDefiningOp<ToMemoryConfigOp>()) {
        baseWeight = toMemoryConfig.getInput();
        continue;
      }
      if (auto reshape = baseWeight.getDefiningOp<ReshapeOp>()) {
        auto reshapeInputType =
            mlir::cast<RankedTensorType>(reshape.getInput().getType());
        if (!isSqueezableWeightShape(reshapeInputType)) {
          break;
        }
        baseWeight = reshape.getInput();
        continue;
      }
      break;
    }

    auto baseWeightType = mlir::dyn_cast<RankedTensorType>(baseWeight.getType());
    if (!baseWeightType) {
      return failWithReason("rms_norm canonicalized weight must be ranked tensor");
    }

    if (baseWeightType.getRank() == 1) {
      if (!isCompatibleLastDim(baseWeightType.getShape().back())) {
        return failWithReason("1D rms_norm weight size mismatches input last dim");
      }
      normalizedWeight = baseWeight;
      return success();
    }

    return failWithReason(
        "rms_norm weight must already be recoverable as 1D; no new reshape "
        "is created by fusion");
  }

  template <typename OpTy>
  Value reshapeResultIfNeeded(mlir::PatternRewriter &rewriter, MultiplyOp srcOp,
                              OpTy fusedOp) const {
    Value result = fusedOp.getResult();
    if (result.getType() == srcOp.getType()) {
      return result;
    }

    auto targetType = mlir::cast<RankedTensorType>(srcOp.getType());
    SmallVector<int32_t> shapeI32;
    shapeI32.reserve(targetType.getShape().size());
    for (int64_t dim : targetType.getShape()) {
      shapeI32.push_back(static_cast<int32_t>(dim));
    }

    return rewriter
        .create<ReshapeOp>(srcOp.getLoc(), srcOp.getType(), result,
                           rewriter.getI32ArrayAttr(shapeI32),
                           ttnn::MemoryConfigAttr())
        .getResult();
  }

  std::optional<float> extractConstant(Value v) const {
    return extractFloatConstant(v);
  }

  bool isPow2(PowScalarOp powOp) const {
    auto rhs = mlir::dyn_cast<FloatAttr>(powOp.getRhs());
    return rhs && rhs.getValue().convertToFloat() == 2.0f;
  }
};

// ============================================================================
// NLP Concat Heads Decode Fusing
// ============================================================================
//
// Matches the decode-phase concat-heads pattern that appears after
// scaled_dot_product_attention_decode in LLMs:
//
//   permute([1, 2, 0, 3])  :  [S, B, H, D] -> [B, H, S, D]
//   reshape                 :  [B, H, S, D] -> [B, H*D]  (or similar collapse)
//
// This sequence shuffles the multi-head attention output back into a single
// hidden dimension. It is replaced by the optimized hardware op
// nlp_concat_heads_decode which performs:
//
//   [S, B, H_padded, D] -> [S, 1, B, num_heads * D]
//
// followed by a reshape to match the original output shape.
//
class NLPConcatHeadsDecodeFusing : public mlir::OpRewritePattern<ReshapeOp> {
  using NLPConcatHeadsDecodeFusing::OpRewritePattern<
      ReshapeOp>::OpRewritePattern;

  // Permutation that converts [S, B, H, D] -> [B, H, S, D].
  static constexpr std::array<int64_t, 4> kConcatHeadsDecodePermutation = {
      1, 2, 0, 3};

public:
  mlir::LogicalResult
  matchAndRewrite(ReshapeOp reshapeOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto permuteOp = reshapeOp.getInput().getDefiningOp<PermuteOp>();
    if (!permuteOp) {
      return failure();
    }

    // Check permutation is [1, 2, 0, 3].
    auto permutation = permuteOp.getPermutation();
    if (!llvm::equal(permutation,
                     ArrayRef<int64_t>(kConcatHeadsDecodePermutation))) {
      return failure();
    }

    Value input = permuteOp.getInput();
    auto inputType = mlir::cast<RankedTensorType>(input.getType());

    auto inputShape = inputType.getShape();
    int64_t seqLen = inputShape[0];
    int64_t batchSize = inputShape[1];
    int64_t numHeads = inputShape[2];
    int64_t headDim = inputShape[3];

    // NLP concat heads decode is specifically for decode phase (seq_len == 1).
    if (seqLen != 1) {
      return failure();
    }

    // TODO(vkovacevic): https://github.com/tenstorrent/tt-metal/issues/38992
    // The tt-metal nlp_concat_heads_decode op computes its output logical shape
    // from the input's padded shape. If head_dim or batch aren't tile-aligned,
    // the output logical shape will differ from what our IR expects, causing a
    // volume mismatch in the subsequent reshape at runtime.
    constexpr int64_t kTileSize = 32;
    if (headDim % kTileSize != 0 || batchSize % kTileSize != 0) {
      return failure();
    }

    SmallVector<int64_t> concatHeadsOutputShape = {seqLen, 1, batchSize,
                                                   numHeads * headDim};
    auto concatHeadsResultType = utils::RankedTensorTypeFactory::create(
        inputType, concatHeadsOutputShape);

    op_model::ScopedSingletonDeviceGuard deviceGuard(reshapeOp);

    auto nlpConcatHeadsDecodeOp = rewriter.create<NLPConcatHeadsDecodeOp>(
        reshapeOp.getLoc(), concatHeadsResultType, input,
        rewriter.getUI32IntegerAttr(static_cast<uint32_t>(numHeads)),
        /*memory_config=*/MemoryConfigAttr());

    // Validate the fused op. The op requires height-sharded L1 input, so
    // try the workaround-sharded version since the workaround pass hasn't
    // run yet.
    auto workaround = workarounds::decomposition::getWorkaroundedInput(
        nlpConcatHeadsDecodeOp, rewriter);
    if (workaround) {
      auto shardedInputType =
          mlir::cast<RankedTensorType>(workaround->getType());
      auto shardedResultType = utils::RankedTensorTypeFactory::create(
          shardedInputType, concatHeadsOutputShape);

      auto validationOp = rewriter.create<NLPConcatHeadsDecodeOp>(
          reshapeOp.getLoc(), shardedResultType, workaround->getResult(),
          rewriter.getUI32IntegerAttr(static_cast<uint32_t>(numHeads)),
          /*memory_config=*/MemoryConfigAttr());

      std::vector<TTNNLayoutAttr> inputLayouts =
          utils::extractInputLayouts(validationOp.getOperation());
      OpConfig config(
          mlir::cast<TTNNLayoutAttr>(shardedResultType.getEncoding()));
      auto validationResult = op_constraint_validation::validateOperation(
          validationOp.getOperation(), inputLayouts, config);

      rewriter.eraseOp(validationOp);
      rewriter.eraseOp(*workaround);

      if (!validationResult.isSuccess()) {
        rewriter.eraseOp(nlpConcatHeadsDecodeOp);
        return failure();
      }
    }

    rewriter.setInsertionPointAfter(nlpConcatHeadsDecodeOp);

    auto newReshapeOp = rewriter.create<ReshapeOp>(
        reshapeOp.getLoc(), reshapeOp.getType(),
        nlpConcatHeadsDecodeOp.getResult(), reshapeOp.getShapeAttr(),
        /*memory_config=*/MemoryConfigAttr());

    rewriter.replaceOp(reshapeOp, newReshapeOp.getResult());
    return mlir::success();
  }
};

#endif // TTMLIR_ENABLE_OPMODEL

class TTNNFusingPass : public impl::TTNNFusingBase<TTNNFusingPass> {
public:
  using impl::TTNNFusingBase<TTNNFusingPass>::TTNNFusingBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    // TODO(mvasiljevic): Add HardsigmoidOp once tt-metal issue is resolved
    // https://github.com/tenstorrent/tt-metal/issues/30973
    patterns.add<
        TTNNConv2dWithActivation<ReluOp>, TTNNConv2dWithActivation<Relu6Op>,
        TTNNConv2dWithActivation<SiluOp>, TTNNConv2dWithActivation<SigmoidOp>,
        TTNNMatmulAndLinearWithActivation<MatmulOp, SigmoidOp>,
        TTNNMatmulAndLinearWithActivation<LinearOp, SigmoidOp>,
        TTNNMatmulAndLinearWithActivation<MatmulOp, SiluOp>,
        TTNNMatmulAndLinearWithActivation<LinearOp, SiluOp>,
        TTNNMatmulAndLinearWithActivation<MatmulOp, GeluOp>,
        TTNNMatmulAndLinearWithActivation<LinearOp, GeluOp>>(&getContext());

#ifdef TTMLIR_ENABLE_OPMODEL
    if (enableOpConstraints) {
      FusionValidationConfig validationConfig;
      validationConfig.maxFallbackAttempts = maxFallbackAttempts;

      patterns.add<fusing::RoPEFusing>(&getContext());
      patterns.add<fusing::RoPEDecodeFusing>(&getContext());
      patterns.add<DistributedRMSNormFusing>(&getContext());
      patterns.add<fusing::TopKFusing>(&getContext(), validationConfig);
      patterns.add<fusing::SDPAFusing>(&getContext(), validationConfig);
      patterns.add<NLPConcatHeadsDecodeFusing>(&getContext());
    }
#endif // TTMLIR_ENABLE_OPMODEL

    // Add TypecastOp canonicalization patterns to fold consecutive typecasts
    // (e.g. bf16->f32->bf16) that appear after SDPA fusing, enabling
    // patterns like NLPConcatHeadsDecodeFusing to match cleanly.
    TypecastOp::getCanonicalizationPatterns(patterns, &getContext());

    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace

} // namespace mlir::tt::ttnn
