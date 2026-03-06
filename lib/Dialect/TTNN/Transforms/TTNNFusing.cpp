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
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

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

// ============================================================================
// Distributed RMSNorm Fusing
// ============================================================================
//
// Matches the distributed RMSNorm decomposition pattern:
//
//   x_bf16
//     -> typecast(f32)
//     -> reshape
//     -> pow_scalar(2)
//     -> sum(dim=2)
//     -> reshape
//     -> reduce_scatter(sum)
//     -> all_gather
//     -> reshape
//     -> multiply(scale)
//     -> add(epsilon)
//     -> rsqrt
//     -> multiply(x_reshaped_f32, rsqrt)
//     -> typecast(bf16)
//     -> multiply(weight)
//
// Rewrites to:
//   distributed_rms_norm(x_bf16, weight, cluster_axis, epsilon)
//   [optional reshape to preserve the original output rank/shape]
//
class DistributedRMSNormFusing : public mlir::OpRewritePattern<MultiplyOp> {
  using DistributedRMSNormFusing::OpRewritePattern<MultiplyOp>::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(MultiplyOp srcOp, mlir::PatternRewriter &rewriter) const final {
    log(0, "matchAndRewrite: start");
    MatchState state;
    if (!matchStructure(srcOp, state)) {
      log(0, "matchAndRewrite: pattern did not match");
      return failure();
    }
    log(0, "matchAndRewrite: pattern matched, creating distributed_rms_norm");

    op_model::ScopedSingletonDeviceGuard deviceGuard(srcOp.getOperation());
    auto device = utils::getOrInsertDevice(rewriter, srcOp);
    auto distResultType = mlir::cast<RankedTensorType>(state.bf16Input.getType());
    auto distResultLayout =
        mlir::dyn_cast_or_null<TTNNLayoutAttr>(distResultType.getEncoding());
    auto distMemoryConfig =
        distResultLayout
            ? MemoryConfigAttr::get(distResultLayout, distResultLayout.getGrid())
            : MemoryConfigAttr();
    if (distResultLayout) {
      log(1, "using fused op memory_config from result layout");
    } else {
      log(1, "fused result has no TTNNLayout encoding, leaving memory_config null");
    }

    auto distRMSNormOp = rewriter.create<DistributedRMSNormOp>(
        srcOp.getLoc(), distResultType, state.bf16Input,
        state.weight,
        /*residual=*/nullptr,
        /*stats=*/nullptr, device.getResult(), state.clusterAxis,
        llvm::APFloat(state.epsilon),
        /*sub_device_id=*/state.subDeviceId,
        /*memory_config=*/distMemoryConfig,
        /*num_links=*/state.numLinks,
        /*topology=*/state.topology,
        /*compute_config=*/state.computeConfig,
        /*program_config=*/nullptr);
    log(1, "created ttnn.distributed_rms_norm");

    Value replacement = distRMSNormOp.getResult();
    if (replacement.getType() != srcOp.getType()) {
      log(1, "output type differs from original, inserting reshape");
      auto targetType = mlir::cast<RankedTensorType>(srcOp.getType());
      SmallVector<int32_t> shapeI32;
      shapeI32.reserve(targetType.getShape().size());
      for (int64_t dim : targetType.getShape()) {
        shapeI32.push_back(static_cast<int32_t>(dim));
      }
      replacement = rewriter
                        .create<ReshapeOp>(srcOp.getLoc(), srcOp.getType(),
                                           replacement,
                                           rewriter.getI32ArrayAttr(shapeI32),
                                           ttnn::MemoryConfigAttr())
                        .getResult();
      log(2, "reshape inserted");
    }

    rewriter.replaceOp(srcOp, replacement);
    log(0, "matchAndRewrite: success");
    return success();
  }

private:
  struct MatchState {
    Value bf16Input;
    Value weight;
    float epsilon = 0.0f;
    uint32_t clusterAxis = 0;
    mlir::IntegerAttr subDeviceId;
    mlir::IntegerAttr numLinks;
    ttcore::TopologyAttr topology;
    DeviceComputeKernelConfigAttr computeConfig;
  };

  static void log(unsigned indentLevel, const llvm::Twine &message) {
    llvm::errs() << "[DistributedRMSNormFusing] ";
    for (unsigned i = 0; i < indentLevel; ++i) {
      llvm::errs() << "  ";
    }
    llvm::errs() << message << "\n";
  }

  static Value skipTypecasts(Value v) {
    while (auto typecast = v.getDefiningOp<TypecastOp>()) {
      v = typecast.getInput();
    }
    return v;
  }

  std::optional<float> extractConstant(Value v) const {
    v = skipTypecasts(v);

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

  bool hasExactlyOneUse(Operation *op) const {
    if (!op || op->getNumResults() != 1) {
      return false;
    }
    size_t nonDeallocateUses = 0;
    for (OpOperand &use : op->getResult(0).getUses()) {
      if (!mlir::isa<DeallocateOp>(use.getOwner())) {
        ++nonDeallocateUses;
        if (nonDeallocateUses > 1) {
          return false;
        }
      }
    }
    return nonDeallocateUses == 1;
  }

  bool hasNonDeallocateUse(Operation *op) const {
    if (!op || op->getNumResults() != 1) {
      return false;
    }
    for (OpOperand &use : op->getResult(0).getUses()) {
      if (!mlir::isa<DeallocateOp>(use.getOwner())) {
        return true;
      }
    }
    return false;
  }

  bool isPow2(PowScalarOp powOp) const {
    auto rhs = mlir::dyn_cast<FloatAttr>(powOp.getRhs());
    return rhs && rhs.getValue().convertToFloat() == 2.0f;
  }

  bool matchStructure(MultiplyOp root, MatchState &state) const {
    log(1, "matchStructure: enter");
    if (!hasNonDeallocateUse(root)) {
      log(2, "fail: root multiply has no non-deallocate uses");
      return false;
    }
    log(2, "ok: root multiply has non-deallocate use(s)");

    // root = multiply(weight, normalized_bf16) (operand order is commutative)
    TypecastOp normalizedTypecast = nullptr;
    Value weightCandidate;
    if (auto lhsTc = root.getLhs().getDefiningOp<TypecastOp>()) {
      normalizedTypecast = lhsTc;
      weightCandidate = root.getRhs();
    } else if (auto rhsTc = root.getRhs().getDefiningOp<TypecastOp>()) {
      normalizedTypecast = rhsTc;
      weightCandidate = root.getLhs();
    } else {
      log(2, "fail: neither root operand is typecast(normalized)");
      return false;
    }
    log(2, "ok: found normalized typecast branch and weight branch");

    if (!hasExactlyOneUse(normalizedTypecast)) {
      log(2, "fail: normalized typecast does not have exactly one use");
      return false;
    }
    log(2, "ok: normalized typecast has one use");

    auto normalizedPreCastType =
        mlir::cast<RankedTensorType>(normalizedTypecast.getInput().getType());
    auto normalizedType =
        mlir::cast<RankedTensorType>(normalizedTypecast.getType());
    if (!normalizedPreCastType.getElementType().isF32() ||
        !normalizedType.getElementType().isBF16()) {
      log(2, "fail: normalized typecast is not f32 -> bf16");
      return false;
    }
    log(2, "ok: normalized typecast is f32 -> bf16");

    auto normalizedMul =
        normalizedTypecast.getInput().getDefiningOp<MultiplyOp>();
    if (!normalizedMul || !hasExactlyOneUse(normalizedMul)) {
      log(2, "fail: normalized pre-cast input is not single-use multiply");
      return false;
    }
    log(2, "ok: normalized pre-cast input is single-use multiply");

    // normalizedMul = multiply([reshape](source_f32), rsqrt(...))
    RsqrtOp rsqrtOp = nullptr;
    ReshapeOp dataReshape = nullptr;
    Value normalizedSourceF32;
    if (auto lhsRsqrt = normalizedMul.getLhs().getDefiningOp<RsqrtOp>()) {
      rsqrtOp = lhsRsqrt;
      Value dataBranch = normalizedMul.getRhs();
      dataReshape = dataBranch.getDefiningOp<ReshapeOp>();
      normalizedSourceF32 = dataReshape ? dataReshape.getInput() : dataBranch;
    } else if (auto rhsRsqrt = normalizedMul.getRhs().getDefiningOp<RsqrtOp>()) {
      rsqrtOp = rhsRsqrt;
      Value dataBranch = normalizedMul.getLhs();
      dataReshape = dataBranch.getDefiningOp<ReshapeOp>();
      normalizedSourceF32 = dataReshape ? dataReshape.getInput() : dataBranch;
    } else {
      log(2, "fail: normalized multiply does not contain rsqrt branch");
      return false;
    }

    if (!rsqrtOp || !normalizedSourceF32 || !hasExactlyOneUse(rsqrtOp)) {
      log(2, "fail: rsqrt/data branch validation failed");
      return false;
    }
    if (dataReshape && !hasExactlyOneUse(dataReshape)) {
      log(2, "fail: optional data reshape does not have exactly one non-deallocate use");
      return false;
    }
    log(2, "ok: found rsqrt and data branch (reshape optional)");

    auto addOp = rsqrtOp.getInput().getDefiningOp<AddOp>();
    if (!addOp || !hasExactlyOneUse(addOp)) {
      log(2, "fail: rsqrt input is not single-use add");
      return false;
    }
    log(2, "ok: rsqrt input is single-use add");

    // add = multiply(stats, scale) + epsilon_const
    MultiplyOp meanScaleMul = nullptr;
    std::optional<float> epsilon;
    if (auto lhsMul = addOp.getLhs().getDefiningOp<MultiplyOp>()) {
      meanScaleMul = lhsMul;
      epsilon = extractConstant(addOp.getRhs());
    } else if (auto rhsMul = addOp.getRhs().getDefiningOp<MultiplyOp>()) {
      meanScaleMul = rhsMul;
      epsilon = extractConstant(addOp.getLhs());
    } else {
      log(2, "fail: add op does not contain multiply(stats, scale) branch");
      return false;
    }

    if (!meanScaleMul || !epsilon || !hasExactlyOneUse(meanScaleMul)) {
      log(2, "fail: meanScaleMul missing/single-use failure/epsilon not const");
      return false;
    }
    log(2, "ok: extracted epsilon and meanScaleMul");

    // meanScaleMul = multiply([reshape](all_gather(...)), scale_const)
    ReshapeOp statsReshape = nullptr;
    Value gatheredStats;
    std::optional<float> scale;
    if (auto lhsReshape = meanScaleMul.getLhs().getDefiningOp<ReshapeOp>()) {
      statsReshape = lhsReshape;
      gatheredStats = statsReshape.getInput();
      scale = extractConstant(meanScaleMul.getRhs());
    } else if (auto rhsReshape =
                   meanScaleMul.getRhs().getDefiningOp<ReshapeOp>()) {
      statsReshape = rhsReshape;
      gatheredStats = statsReshape.getInput();
      scale = extractConstant(meanScaleMul.getLhs());
    } else {
      // Some TTNN pipelines simplify away the reshape before scale multiply.
      // Accept direct all_gather output as well.
      if (extractConstant(meanScaleMul.getRhs())) {
        gatheredStats = meanScaleMul.getLhs();
        scale = extractConstant(meanScaleMul.getRhs());
      } else if (extractConstant(meanScaleMul.getLhs())) {
        gatheredStats = meanScaleMul.getRhs();
        scale = extractConstant(meanScaleMul.getLhs());
      }
    }

    if (!gatheredStats) {
      log(2, "fail: meanScaleMul stats branch missing");
      return false;
    }

    if (statsReshape && !hasExactlyOneUse(statsReshape)) {
      log(2, "fail: optional stats reshape does not have exactly one non-deallocate use");
      return false;
    }

    if (!scale) {
      log(2, "fail: meanScaleMul scale is not constant");
      return false;
    }

    if (statsReshape) {
      log(2, "ok: extracted scale and stats reshape");
    } else {
      log(2, "ok: extracted scale and stats branch (reshape elided)");
    }

    Value statsProducer = skipTypecasts(gatheredStats);
    if (auto toLayout = statsProducer.getDefiningOp<ToLayoutOp>()) {
      statsProducer = skipTypecasts(toLayout.getInput());
    } else if (auto toMemCfg = statsProducer.getDefiningOp<ToMemoryConfigOp>()) {
      statsProducer = skipTypecasts(toMemCfg.getInput());
    }

    auto allGather = statsProducer.getDefiningOp<AllGatherOp>();
    auto allReduce = statsProducer.getDefiningOp<AllReduceOp>();
    ReduceScatterOp reduceScatter = nullptr;
    Value statsCommInput;
    bool usesAllReducePath = false;

    if (allGather) {
      log(2, "ok: found all_gather");
      reduceScatter = allGather.getInput().getDefiningOp<ReduceScatterOp>();
      if (!reduceScatter) {
        log(2, "fail: all_gather input is not reduce_scatter");
        return false;
      }
      log(2, "ok: found reduce_scatter");

      if (reduceScatter.getReduceType() != ttcore::ReduceType::Sum) {
        log(2, "fail: reduce_scatter reduce_type is not sum");
        return false;
      }
      log(2, "ok: reduce_scatter reduce_type is sum");
      if (allGather.getClusterAxis() != reduceScatter.getClusterAxis() ||
          allGather.getAllGatherDim() != reduceScatter.getScatterDim()) {
        log(2, "fail: all_gather/reduce_scatter axis or dim mismatch");
        return false;
      }
      log(2, "ok: all_gather/reduce_scatter axis and dim match");
      statsCommInput = reduceScatter.getInput();
    } else if (allReduce) {
      if (allReduce.getReduceType() != ttcore::ReduceType::Sum) {
        log(2, "fail: all_reduce reduce_type is not sum");
        return false;
      }
      usesAllReducePath = true;
      statsCommInput = allReduce.getInput();
      log(2, "ok: found all_reduce(sum) stats path");
    } else {
      if (Operation *statsProducerOp = statsProducer.getDefiningOp()) {
        llvm::errs() << "[DistributedRMSNormFusing]     fail detail: stats "
                        "branch producer after peeling is op '"
                     << statsProducerOp->getName() << "'\n";
        llvm::errs() << "[DistributedRMSNormFusing]       op: "
                     << *statsProducerOp << "\n";
      } else {
        llvm::errs() << "[DistributedRMSNormFusing]     fail detail: stats "
                        "branch producer after peeling is a block argument/value "
                        "without defining op\n";
      }
      log(2, "fail: stats branch producer is neither all_gather nor all_reduce");
      return false;
    }

    auto sumInputReshape = statsCommInput.getDefiningOp<ReshapeOp>();
    Value sumCandidate = statsCommInput;
    if (sumInputReshape) {
      if (!hasExactlyOneUse(sumInputReshape)) {
        log(2, "fail: optional sum-input reshape does not have exactly one non-deallocate use");
        return false;
      }
      sumCandidate = sumInputReshape.getInput();
      log(2, "ok: found sum-input reshape");
    } else {
      log(2, "ok: sum-input reshape is elided");
    }

    auto sumOp = sumCandidate.getDefiningOp<SumOp>();
    if (!sumOp || !hasExactlyOneUse(sumOp) || sumOp.getKeepDim()) {
      log(2, "fail: sum candidate is not single-use sum or keep_dim=true");
      return false;
    }
    log(2, "ok: found sum with keep_dim=false");

    if (!sumOp.getDimArg()) {
      log(2, "fail: sum has no dim_arg");
      return false;
    }
    auto dims = ttmlir::utils::getIntegerVector<int64_t>(*sumOp.getDimArg());
    if (!dims || dims->size() != 1 || dims->front() != 2) {
      log(2, "fail: sum dim_arg is not exactly [2]");
      return false;
    }
    log(2, "ok: sum dim_arg is [2]");

    auto powOp = sumOp.getInput().getDefiningOp<PowScalarOp>();
    if (!powOp || !hasExactlyOneUse(powOp) || !isPow2(powOp)) {
      log(2, "fail: sum input is not single-use pow_scalar(rhs=2)");
      return false;
    }
    log(2, "ok: found pow_scalar(rhs=2)");

    auto powInputReshape = powOp.getLhs().getDefiningOp<ReshapeOp>();
    if (powInputReshape && !hasExactlyOneUse(powInputReshape)) {
      log(2, "fail: optional pow-input reshape does not have exactly one non-deallocate use");
      return false;
    }
    log(2, "ok: found pow-input branch (reshape optional)");

    Value sourceF32 = powInputReshape ? powInputReshape.getInput() : powOp.getLhs();
    if (sourceF32 != normalizedSourceF32) {
      log(2, "fail: source f32 mismatch between pow branch and normalization branch");
      return false;
    }
    log(2, "ok: source f32 is shared by both branches");

    auto sourceTypecast = sourceF32.getDefiningOp<TypecastOp>();
    if (!sourceTypecast) {
      log(2, "fail: shared source is not defined by typecast");
      return false;
    }

    auto sourceType = mlir::cast<RankedTensorType>(sourceTypecast.getType());
    auto sourceInputType =
        mlir::cast<RankedTensorType>(sourceTypecast.getInput().getType());
    if (!sourceType.getElementType().isF32() ||
        !sourceInputType.getElementType().isBF16()) {
      log(2, "fail: source typecast is not bf16 -> f32");
      return false;
    }
    log(2, "ok: source typecast is bf16 -> f32");

    if (epsilon.value() <= 0.0f) {
      log(2, "fail: epsilon must be positive");
      return false;
    }
    log(2, "ok: epsilon is positive");

    state.bf16Input = sourceTypecast.getInput();
    state.weight = weightCandidate;
    state.epsilon = epsilon.value();
    if (usesAllReducePath) {
      state.clusterAxis = allReduce.getClusterAxis();
      state.subDeviceId = allReduce.getSubDeviceIdAttr();
      state.numLinks = allReduce.getNumLinksAttr();
      state.topology = allReduce.getTopologyAttr();
      state.computeConfig = sumOp.getComputeConfigAttr();
    } else {
      state.clusterAxis = reduceScatter.getClusterAxis();
      state.subDeviceId = reduceScatter.getSubDeviceIdAttr();
      state.numLinks = reduceScatter.getNumLinksAttr();
      state.topology = reduceScatter.getTopologyAttr();
      state.computeConfig = reduceScatter.getComputeConfigAttr()
                                ? reduceScatter.getComputeConfigAttr()
                                : sumOp.getComputeConfigAttr();
    }
    log(1, "matchStructure: success");
    return true;
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
