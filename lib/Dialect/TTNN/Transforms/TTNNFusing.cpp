// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#ifdef TTMLIR_ENABLE_OPMODEL
#include "ttmlir/Dialect/TTNN/Transforms/Fusing/RoPEFusingPattern.h"
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
    Value fusedInput = state.bf16Input;
    auto fusedInputType = mlir::cast<RankedTensorType>(fusedInput.getType());
    if (fusedInputType.getRank() > 0 && fusedInputType.getRank() < 4) {
      SmallVector<int64_t> expandedShape;
      expandedShape.reserve(4);
      for (int64_t i = 0; i < 4 - fusedInputType.getRank(); ++i) {
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
      fusedInput = rewriter
                       .create<ReshapeOp>(srcOp.getLoc(), expandedType, fusedInput,
                                          rewriter.getI32ArrayAttr(expandedShapeI32),
                                          ttnn::MemoryConfigAttr())
                       .getResult();
      log(1, "expanded fused input rank to 4 for distributed_rms_norm");
      fusedInputType = mlir::cast<RankedTensorType>(fusedInput.getType());
    }

    auto distResultType = fusedInputType;
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
        srcOp.getLoc(), distResultType, fusedInput,
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
// SDPA Fusing
// ============================================================================
//
// Matches Scaled Dot Product Attention:
//   Attention(Q, K, V) = softmax((Q @ K^T) * scale + mask) @ V
//
// Anchors on the final matmul (attention @ V) and walks backward:
//
//   matmul (attention @ V)
//      |
//   [where]          <- optional causal masking
//      |
//   softmax
//      |
//   [add(mask)]      <- optional attention mask
//      |
//   [multiply(scale) | divide(scale)] <- optional scaling factor
//      |
//   matmul (Q @ K^T)
//
// Uses skipTransparent() to handle type conversions and layout ops that don't
// change semantics, making the pattern robust to variations in the IR.
//
class SDPAFusing : public mlir::OpRewritePattern<MatmulOp> {
  using SDPAFusing::OpRewritePattern<MatmulOp>::OpRewritePattern;

  // SDPA Query, Key, Value tensors have shape [B, H, S, D]
  // (Batch, NumHeads, SeqLen, HeadDim).
  static constexpr int64_t kNumHeadsDim = 1;
  static constexpr int64_t kSeqLenDim = 2;

  // Permutation to convert query from [B, H, S, D] -> [S, B, H, D] for SDPA
  // decode op.
  static constexpr std::array<int64_t, 4> kToDecodePermutation = {2, 0, 1, 3};

  // Permutation to un-transpose key from [B, H, D, S] -> [B, H, S, D].
  // Used when key comes from SplitQueryKeyValueAndSplitHeadsOp with
  // transpose_key=true.
  static constexpr std::array<int64_t, 4> kUnTransposeKeyPermutation = {0, 1, 3,
                                                                        2};

public:
  mlir::LogicalResult
  matchAndRewrite(MatmulOp srcOp,
                  mlir::PatternRewriter &rewriter) const override {
    SDPAComponents c;
    c.attentionMatmul = srcOp;
    c.value = srcOp.getB();

    // Match: matmul -> [where] -> softmax -> score
    if (!matchSoftmaxPath(srcOp.getA(), c)) {
      return failure();
    }

    if (!matchScoreComputation(c.softmax.getInput(), c)) {
      return failure();
    }

    // Validate semantic constraints (single-use of intermediate ops, valid
    // scale range) before modifying the IR.
    if (!validateSemantics(c)) {
      return failure();
    }

    // Prepare inputs for SDPA: normalize Q/K/V/mask by skipping transparent ops
    // and dropping matmul-only transforms (e.g. K^T permute, GQA head
    // expansion). Key un-transpose for SDPA op legality is handled during input
    // canonicalization (see unTransposeKeyIfNeeded()).
    prepareInputsForSDPA(c, rewriter);

    return createSDPAOp(rewriter, c);
  }

private:
  struct SDPAComponents {
    Value query, key, value, mask;
    std::optional<float> scale;
    MatmulOp attentionMatmul;
    SoftmaxOp softmax;
    Operation *scoreOp = nullptr;
  };

  // ============================================================================
  // Transparent Op Utilities
  // ============================================================================

  // Operations that don't change semantic meaning - can be traced through.
  static bool isTransparentOp(Operation *op) {
    return isa<ToLayoutOp, ToMemoryConfigOp, TypecastOp>(op);
  }

  // Skip through transparent ops to find the semantic operation.
  Value skipTransparent(Value v) const {
    while (Operation *defOp = v.getDefiningOp()) {
      if (!isTransparentOp(defOp)) {
        break;
      }
      v = defOp->getOperand(0);
    }
    return v;
  }

  // ============================================================================
  // Layout / Transpose Utilities
  // ============================================================================

  // Check if a permutation is a transpose on the last two dimensions.
  // For a 4D tensor [B, H, S, D], a transpose permutation would be [0, 1, 3,
  // 2]. This is the typical transpose used before matrix multiplication.
  static bool isTransposeOnLastTwoDims(ArrayRef<int64_t> perm) {
    if (perm.size() < 2) {
      return false;
    }

    size_t n = perm.size();
    // Check that all dimensions except the last two are identity.
    for (size_t i = 0; i < n - 2; ++i) {
      if (perm[i] != static_cast<int64_t>(i)) {
        return false;
      }
    }

    // Check that the last two dimensions are swapped.
    return perm[n - 2] == static_cast<int64_t>(n - 1) &&
           perm[n - 1] == static_cast<int64_t>(n - 2);
  }

  // Check if key is transposed by looking at its source operation or shape.
  // Returns true if:
  // 1. Key came from SplitQueryKeyValueAndSplitHeadsOp with transpose_key=true
  // 2. Key shape suggests transposition: K[B, H, D, S] where D matches Q's
  //    head_dim and S matches V's seq_len
  bool isKeyTransposed(Value key, Value query, Value value) const {
    // Check explicit source first
    Operation *defOp = key.getDefiningOp();
    if (auto splitOp =
            dyn_cast_or_null<SplitQueryKeyValueAndSplitHeadsOp>(defOp)) {
      return splitOp.getTransposeKey();
    }

    // Shape-based detection for keys transposed via permute operations
    auto kType = mlir::dyn_cast<RankedTensorType>(key.getType());
    auto qType = mlir::dyn_cast<RankedTensorType>(query.getType());
    auto vType = mlir::dyn_cast<RankedTensorType>(value.getType());

    if (!kType || !qType || !vType || kType.getRank() != 4 ||
        qType.getRank() != 4 || vType.getRank() != 4) {
      return false;
    }

    auto kShape = kType.getShape();
    auto qShape = qType.getShape();
    auto vShape = vType.getShape();

    // Q: [B, H, S_q, head_dim], K_normal: [B, H, S_k, head_dim]
    // K_transposed: [B, H, head_dim, S_k]
    int64_t qHeadDim = qShape[3];
    int64_t vSeqLen = vShape[kSeqLenDim];

    // If K's dim[2] matches Q's head_dim and K's dim[3] matches V's seq_len,
    // then K is transposed: [B, H, head_dim, seq_k]
    bool kDim2MatchesHeadDim = kShape[2] == qHeadDim;
    bool kDim3MatchesSeqLen = kShape[3] == vSeqLen;

    return kDim2MatchesHeadDim && kDim3MatchesSeqLen;
  }

  // ============================================================================
  // Constant Extraction
  // ============================================================================

  std::optional<float> extractConstant(Value v) const {
    // Skip transparent ops to find the actual constant.
    v = skipTransparent(v);

    // Direct FullOp.
    if (auto fullOp = v.getDefiningOp<FullOp>()) {
      if (auto attr = mlir::dyn_cast<FloatAttr>(fullOp.getFillValue())) {
        return attr.getValue().convertToFloat();
      }
    }

    // Try load_cached - look up the const_eval function and find FullOp inside.
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

      // Walk the function body to find a FullOp.
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
  // Q/K Extraction with Scale Handling
  // ============================================================================

  // Extract tensor and its scale. Checks if skipping transparent ops leads to a
  // multiply with a constant scale. If so, extracts the scale and returns the
  // tensor input. Otherwise returns the original value unchanged.
  std::pair<Value, std::optional<float>> extractTensorWithScale(Value v) const {
    std::optional<float> scale;

    // Check if transparent ops lead to a multiply (scale applied to tensor).
    Value skipped = skipTransparent(v);
    if (auto mulOp = skipped.getDefiningOp<MultiplyOp>()) {
      if (auto s = extractConstant(mulOp.getRhs())) {
        scale = s;
        return {mulOp.getLhs(), scale};
      }
      if (auto s = extractConstant(mulOp.getLhs())) {
        scale = s;
        return {mulOp.getRhs(), scale};
      }
    }

    // No multiply found - return original value unchanged.
    return {v, scale};
  }

  // Returns false if we find both post-matmul scaling AND pre-scaling on Q/K,
  // which would indicate this is likely not a standard SDPA pattern.
  // Also rejects if Q or K comes from a LoadCachedOp (const-eval function).
  bool extractQKWithScales(Value a, Value b, SDPAComponents &c) const {
    auto [query, qScale] = extractTensorWithScale(a);
    auto [key, kScale] = extractTensorWithScale(b);

    // Reject if we found both post-matmul scale and pre-scaling on Q/K.
    // Standard SDPA uses one or the other, not both.
    bool hasPostMatmulScale = c.scale.has_value();
    bool hasPreScale = qScale.has_value() || kScale.has_value();
    if (hasPostMatmulScale && hasPreScale) {
      return false;
    }

    c.query = query;
    c.key = key;

    // Combine pre-scales if present: Q*s and K*s → combined scale = s*s.
    if (hasPreScale) {
      float qs = qScale.value_or(1.0f);
      float ks = kScale.value_or(1.0f);
      c.scale = qs * ks;
    }
    return true;
  }

  // ============================================================================
  // Pattern Matching with Backtracking
  // ============================================================================

  // Match: [Typecast] -> [where(cond, zeros, softmax)] -> softmax
  bool matchSoftmaxPath(Value v, SDPAComponents &c) const {
    v = skipTransparent(v);

    // Try where(cond, zeros, softmax) pattern first
    if (auto whereOp = v.getDefiningOp<WhereOp>()) {
      Value softmaxCandidate = skipTransparent(whereOp.getThird());
      if (auto softmax = softmaxCandidate.getDefiningOp<SoftmaxOp>()) {
        c.softmax = softmax;
        return true;
      }
    }

    // Direct softmax
    if (auto softmax = v.getDefiningOp<SoftmaxOp>()) {
      c.softmax = softmax;
      return true;
    }

    return false;
  }

  // Match score computation with backtracking for different orderings.
  // Patterns (in order of priority):
  //   1. [transparent] -> linear(Q_scaled, K_scaled, mask)
  //   2. [transparent] -> add(score_chain, mask)
  //   3. [transparent] -> score_chain (no mask)
  bool matchScoreComputation(Value v, SDPAComponents &c) const {
    v = skipTransparent(v);

    // Try linear(Q_scaled, K_scaled, mask) first
    if (auto linearOp = v.getDefiningOp<LinearOp>()) {
      c.scoreOp = linearOp;
      if (!extractQKWithScales(linearOp.getA(), linearOp.getB(), c)) {
        return false;
      }
      if (linearOp.getBias()) {
        c.mask = linearOp.getBias();
      }
      return true;
    }

    // Try add(score, mask) with both operand orderings
    if (auto addOp = v.getDefiningOp<AddOp>()) {
      // Try lhs as score, rhs as mask
      if (matchScoreChain(addOp.getLhs(), c)) {
        c.mask = addOp.getRhs();
        return true;
      }
      // Try rhs as score, lhs as mask
      if (matchScoreChain(addOp.getRhs(), c)) {
        c.mask = addOp.getLhs();
        return true;
      }
      return false;
    }

    // No add - try direct score chain (no mask)
    return matchScoreChain(v, c);
  }

  // Match: [transparent] -> [multiply(*, scale) | divide(*, scale)] ->
  //        [transparent] -> matmul
  // Extracts scale if present, then matches the Q@K matmul.
  bool matchScoreChain(Value v, SDPAComponents &c) const {
    v = skipTransparent(v);

    // Optional multiply for scale (post-matmul scaling)
    if (auto mulOp = v.getDefiningOp<MultiplyOp>()) {
      if (auto scale = extractConstant(mulOp.getRhs())) {
        c.scale = scale;
        v = skipTransparent(mulOp.getLhs());
      } else if (auto scale = extractConstant(mulOp.getLhs())) {
        c.scale = scale;
        v = skipTransparent(mulOp.getRhs());
      }
    }

    // Optional divide for scale (post-matmul scaling, e.g. SegFormer style)
    // Division by X is equivalent to multiply by 1/X.
    else if (auto divOp = v.getDefiningOp<DivideOp>()) {
      if (auto divisor = extractConstant(divOp.getRhs())) {
        if (*divisor != 0.0f) {
          c.scale = 1.0f / *divisor;
          v = skipTransparent(divOp.getLhs());
        }
      }
    }

    // Must end with matmul (different from attention matmul)
    if (auto matmul = v.getDefiningOp<MatmulOp>()) {
      if (matmul != c.attentionMatmul) {
        c.scoreOp = matmul;
        if (!extractQKWithScales(matmul.getA(), matmul.getB(), c)) {
          return false;
        }
        return true;
      }
    }

    return false;
  }

  // ============================================================================
  // Input Canonicalization (dtype/TM/mask)
  // ============================================================================

  // TODO(tt-metal): SDPA should natively support f32 inputs. Currently
  // tt-metal's SDPA only accepts bf16/bfp8_b/bfp4_b, so we insert a typecast
  // when the input is f32. Remove this once tt-metal adds f32 support.
  static Value castToBF16IfNeeded(Value v, PatternRewriter &rewriter) {
    auto vType = cast<RankedTensorType>(v.getType());
    if (!vType.getElementType().isF32()) {
      return v;
    }

    auto dataType = ttcore::DataType::BFloat16;
    auto castType = utils::RankedTensorTypeFactory::create(vType, dataType);
    return rewriter.create<TypecastOp>(
        v.getLoc(), castType, v,
        ttcore::DataTypeAttr::get(rewriter.getContext(), dataType));
  }

  static Value restoreElementTypeIfNeeded(Value v, Type elementType,
                                          PatternRewriter &rewriter) {
    auto vType = cast<RankedTensorType>(v.getType());
    if (vType.getElementType() == elementType) {
      return v;
    }

    // Convert MLIR element type to ttcore::DataType.
    auto dataType = ttcore::elementTypeToDataType(elementType);

    // Create new tensor type with correctly updated encoding.
    auto castType = utils::RankedTensorTypeFactory::create(vType, dataType);
    return rewriter.create<TypecastOp>(
        v.getLoc(), castType, v,
        ttcore::DataTypeAttr::get(rewriter.getContext(), dataType));
  }

  // Find the element type at the end of a "TM chain" (typecast/reshape/permute/
  // repeat_interleave), without allocating a temporary vector. This keeps dtype
  // expectations stable when we later drop some of these ops for SDPA inputs.
  static Type getTargetElementType(Value v) {
    Type lastSeen = cast<RankedTensorType>(v.getType()).getElementType();
    while (Operation *defOp = v.getDefiningOp()) {
      if (isa<TypecastOp, ReshapeOp, PermuteOp, RepeatInterleaveOp>(defOp)) {
        v = defOp->getOperand(0);
        lastSeen = cast<RankedTensorType>(v.getType()).getElementType();
        continue;
      }

      break;
    }
    return lastSeen;
  }

  std::pair<Value, Type> analyzeQ(Value v) const {
    if (auto typecastOp = v.getDefiningOp<TypecastOp>()) {
      v = typecastOp.getInput();
    }

    // If Q comes from load_cached, trace through const-eval function to find
    // the original dtype before any f32 conversions.
    if (auto loadCached = v.getDefiningOp<ttcore::LoadCachedOp>()) {
      auto funcOp = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          loadCached, loadCached.getCalleeAttr());
      if (funcOp) {
        // Find the return op and get the value corresponding to this result.
        unsigned resultIdx = cast<OpResult>(v).getResultNumber();
        for (auto &block : funcOp.getBody()) {
          if (auto returnOp = dyn_cast<func::ReturnOp>(block.getTerminator())) {
            Value innerV = returnOp.getOperand(resultIdx);
            // Extract original tensor before scaling to get the true dtype.
            auto [originalTensor, scale] = extractTensorWithScale(innerV);
            return {v, getTargetElementType(originalTensor)};
          }
        }
      }
    }

    return {v, getTargetElementType(v)};
  }

  // Analyze K tensor: trace through TMs we can drop,
  // and track whether we skipped a K^T permute.
  std::tuple<Value, Type, bool> analyzeK(Value v) const {
    Type targetDtype = getTargetElementType(v);
    bool skippedTranspose = false;

    while (Operation *defOp = v.getDefiningOp()) {
      if (isa<TypecastOp>(defOp)) {
        v = defOp->getOperand(0);
        continue;
      }

      if (auto repeatOp = dyn_cast<RepeatInterleaveOp>(defOp)) {
        // Only skip if it's GQA head expansion (on dim 1 in [B,H,S,D])
        if (repeatOp.getDim() == kNumHeadsDim) {
          v = repeatOp.getInput();
          continue;
        }
        break;
      }

      if (auto permuteOp = dyn_cast<PermuteOp>(defOp)) {
        // Only skip if it's a transpose on last two dims (K^T for matmul)
        if (isTransposeOnLastTwoDims(permuteOp.getPermutation())) {
          v = permuteOp.getInput();
          skippedTranspose = true;
          continue;
        }
      }

      break;
    }

    return {v, targetDtype, skippedTranspose};
  }

  // Analyze V tensor: trace through TMs we can drop.
  std::pair<Value, Type> analyzeV(Value v) const {
    Type targetDtype = getTargetElementType(v);

    while (Operation *defOp = v.getDefiningOp()) {
      if (isa<TypecastOp>(defOp)) {
        v = defOp->getOperand(0);
        continue;
      }

      if (auto repeatOp = dyn_cast<RepeatInterleaveOp>(defOp)) {
        // Only skip if it's GQA head expansion (on dim 1 in [B,H,S,D])
        if (repeatOp.getDim() == kNumHeadsDim) {
          v = repeatOp.getInput();
          continue;
        }
        break;
      }

      break;
    }

    return {v, targetDtype};
  }

  // Trace mask back through broadcast materialization ops (e.g. RepeatOp).
  //
  // Many frontends materialize attention mask broadcasts early (often via
  // `ttnn.repeat`) to match the score tensor shape. For SDPA we prefer to keep
  // the original mask and let broadcastMaskForSDPA() re-broadcast to the exact
  // shape required by the fused op.
  Value prepareMask(Value v) const {
    while (Operation *defOp = v.getDefiningOp()) {
      if (isa<TypecastOp>(defOp)) {
        v = defOp->getOperand(0);
        continue;
      }

      if (auto repeatOp = dyn_cast<RepeatOp>(defOp)) {
        v = repeatOp.getInput();
        continue;
      }

      break;
    }
    return v;
  }

  // Slice mask on head dimension if it was broadcasted.
  //
  // TTNN SDPA expects mask with shape [B, 1, S_q, S_kv], but some frontends
  // may broadcast the mask to [B, H, S_q, S_kv] matching Q's num_heads.
  // We slice to [B, 1, S_q, S_kv] which SDPA can then broadcast internally.
  Value sliceMaskOnHeadDimIfNeeded(Value mask, PatternRewriter &rewriter,
                                   Location loc) const {
    auto maskType = mlir::cast<RankedTensorType>(mask.getType());
    auto maskShape = maskType.getShape();

    // Only handle 4D masks.
    if (maskShape.size() != 4) {
      return mask;
    }

    // If head dim (dim 1) is already 1, no slicing needed.
    if (maskShape[1] == 1) {
      return mask;
    }

    // Slice to get [B, 1, S_q, S_kv].
    SmallVector<int32_t> begins = {0, 0, 0, 0};
    SmallVector<int32_t> ends = {static_cast<int32_t>(maskShape[0]), 1,
                                 static_cast<int32_t>(maskShape[2]),
                                 static_cast<int32_t>(maskShape[3])};
    SmallVector<int32_t> steps = {1, 1, 1, 1};

    SmallVector<int64_t> resultShape = {maskShape[0], 1, maskShape[2],
                                        maskShape[3]};
    auto resultType =
        utils::RankedTensorTypeFactory::create(maskType, resultShape);

    return rewriter.create<SliceStaticOp>(
        loc, resultType, mask, rewriter.getI32ArrayAttr(begins),
        rewriter.getI32ArrayAttr(ends), rewriter.getI32ArrayAttr(steps));
  }

  // Prepare matched inputs for SDPA operation.
  //
  // This normalizes inputs while keeping the pattern robust to frontend
  // variations:
  // - Skip transparent ops (ToLayout, ToMemoryConfig, Typecast)
  // - Drop matmul-only transforms on K/V (K^T permute, typecast wrappers, GQA
  //   head expansion via repeat_interleave)
  // - Trace mask through broadcast materialization (RepeatOp) to recover the
  //   original mask and let broadcastMaskForSDPA() re-broadcast precisely
  //
  // Each preparation step is only committed if shapes remain SDPA-legal.
  void prepareInputsForSDPA(SDPAComponents &c,
                            PatternRewriter &rewriter) const {
    // Analyze all inputs upfront before committing any changes.
    // This ensures K and V are validated together (important for GQA where
    // both may need repeat_interleave traced through).
    auto [preparedQ, preparedQElementType] = analyzeQ(c.query);
    auto [preparedK, preparedKElementType, skippedKTranspose] = analyzeK(c.key);
    auto [preparedV, preparedVElementType] = analyzeV(c.value);

    // Validate and commit Q.
    if (validateShapes(preparedQ, c.key, c.value)) {
      c.query =
          restoreElementTypeIfNeeded(preparedQ, preparedQElementType, rewriter);
    } else {
      c.query =
          restoreElementTypeIfNeeded(c.query, preparedQElementType, rewriter);
    }

    // Validate K and V together - both must be prepared or neither.
    // This handles GQA where K and V are both traced through repeat_interleave.
    if (validateShapes(c.query, preparedK, preparedV)) {
      c.key = preparedK;
      c.value = preparedV;
    }

    // If key is still in a transposed form, materialize an un-transpose so the
    // fused SDPA op sees the expected [B, H, S, D] shape. Do this before
    // restoring element type so the permute operates on the traced-back value.
    //  Only do this if we didn't already skip a K^T permute during
    // tracing, to avoid adding a unneeded transpose when shapes are ambiguous
    // (e.g., when seq_k == head_dim).
    if (!skippedKTranspose) {
      c.key = unTransposeKeyIfNeeded(c.query, c.key, c.value, rewriter,
                                     c.attentionMatmul.getLoc());
    }

    // Restore element types for K and V after any shape transformations.
    c.key = restoreElementTypeIfNeeded(c.key, preparedKElementType, rewriter);
    c.value =
        restoreElementTypeIfNeeded(c.value, preparedVElementType, rewriter);

    if (c.mask) {
      c.mask = prepareMask(c.mask);

      // If mask is broadcasted on head dimension (dim 1), slice it to
      // [B, 1, S_q, S_kv] since TTNN SDPA doesn't support this broadcast.
      c.mask = sliceMaskOnHeadDimIfNeeded(c.mask, rewriter,
                                          c.attentionMatmul.getLoc());

      // The mask should have the same element type as the qkv tensors.
      c.mask =
          restoreElementTypeIfNeeded(c.mask, preparedQElementType, rewriter);
    }
  }

  // ============================================================================
  // Key Un-transpose
  // ============================================================================

  // If key appears transposed (via source op or shape heuristic), generate a
  // permute to restore the expected shape [B, H, S, D] for SDPA.
  Value unTransposeKeyIfNeeded(Value query, Value key, Value value,
                               mlir::PatternRewriter &rewriter,
                               Location loc) const {
    if (!isKeyTransposed(key, query, value)) {
      return key;
    }

    // Generate permute to un-transpose: [B, H, D, S] -> [B, H, S, D]
    return ttir_to_ttnn::utils::generatePermute(
        mlir::cast<TypedValue<RankedTensorType>>(key),
        llvm::to_vector(kUnTransposeKeyPermutation), rewriter, loc);
  }

  // ============================================================================
  // Validation
  // ============================================================================

  // Check if an SDPA validation error can be recovered by TTNNWorkarounds pass.
  // These errors are handled by
  // ScaledDotProductAttentionPadTileDimsRewritePattern which pads:
  // - sequence dimensions to be divisible by chunk size (32) when mask is
  //   present
  // - head dimensions to be divisible by tile width (32) always
  static bool isRecoverableSDPAError(const std::string &errorMessage) {
    // Q sequence length not divisible by q_chunk_size (default 32)
    if (errorMessage.find(
            "Q sequence length must be divisible by q_chunk_size") !=
        std::string::npos) {
      return true;
    }
    // K sequence length not divisible by k_chunk_size (default 32)
    if (errorMessage.find(
            "K sequence length must be divisible by k_chunk_size") !=
        std::string::npos) {
      return true;
    }
    // Head dimension not tile-aligned (requires padding)
    if (errorMessage.find("Padding is not supported on the head_dim") !=
        std::string::npos) {
      return true;
    }
    return false;
  }

  bool validateShapes(Value query, Value key, Value value) const {
    if (!query || !key || !value) {
      return false;
    }

    auto qType = mlir::dyn_cast<RankedTensorType>(query.getType());
    auto kType = mlir::dyn_cast<RankedTensorType>(key.getType());
    auto vType = mlir::dyn_cast<RankedTensorType>(value.getType());

    if (!qType || !kType || !vType) {
      return false;
    }

    // Validate shapes: Q, K, V should be 4D tensors.
    // Q shape: [batch, num_heads, seq_q, head_dim]
    // K shape: [batch, num_kv_heads, seq_k, head_dim] or
    //          [batch, num_kv_heads, head_dim, seq_k] if transposed
    // V shape: [batch, num_kv_heads, seq_v, head_dim]
    auto qShape = qType.getShape();
    auto kShape = kType.getShape();
    auto vShape = vType.getShape();

    if (qShape.size() != 4 || kShape.size() != 4 || vShape.size() != 4) {
      return false;
    }

    int64_t qHeadDim = qShape[3];
    int64_t vSeqLen = vShape[kSeqLenDim];
    int64_t vHeadDim = vShape[3];

    bool keyTransposed = isKeyTransposed(key, query, value);
    int64_t kSeqLen = keyTransposed ? kShape[3] : kShape[kSeqLenDim];
    int64_t kHeadDim = keyTransposed ? kShape[kSeqLenDim] : kShape[3];

    // Key and Value must have the same sequence length.
    if (kSeqLen != vSeqLen) {
      return false;
    }

    // Head dimensions must match across Q, K, V.
    if (qHeadDim != kHeadDim || kHeadDim != vHeadDim) {
      return false;
    }

    // Batch dimensions must match.
    if (qShape[0] != kShape[0] || kShape[0] != vShape[0]) {
      return false;
    }

    // Validate num_heads:
    // - K and V must have the same num_heads (num_kv_heads)
    // - Q's num_heads must be divisible by num_kv_heads (for GQA/MQA support)
    int64_t qNumHeads = qShape[kNumHeadsDim];
    int64_t kNumHeads = kShape[kNumHeadsDim];
    int64_t vNumHeads = vShape[kNumHeadsDim];

    if (kNumHeads != vNumHeads) {
      return false;
    }

    if (qNumHeads % kNumHeads != 0) {
      return false;
    }

    return true;
  }

  bool validateSemantics(const SDPAComponents &c) const {
    if (!c.query || !c.key || !c.value || !c.softmax || !c.scoreOp) {
      return false;
    }

    if (!validateShapes(c.query, c.key, c.value)) {
      return false;
    }

    if (!c.softmax->hasOneUse()) {
      return false;
    }

    // If softmax feeds into a typecast, verify the typecast also has one use.
    if (auto *softmaxUser = *c.softmax->getUsers().begin()) {
      if (isa<TypecastOp>(softmaxUser) && !softmaxUser->hasOneUse()) {
        return false;
      }
    }

    if (c.scale.has_value() && (*c.scale <= 0.0f || *c.scale > 1.0f)) {
      return false;
    }

    return true;
  }

  // Broadcast attention mask to the required shape for SDPA operations.
  //
  // For regular SDPA:
  //   Target mask shape: [batch, 1, query_seq, key_seq]
  //   - Dimension 1 (heads) stays as 1
  //
  // For decode SDPA:
  //   Target mask shape: [batch, 1, num_heads, key_seq]
  //   - Dimension 1 is query seq (always 1 for decode)
  //   - Dimension 2 must explicitly match num_heads
  Value broadcastMaskForSDPA(Value mask, RankedTensorType qType,
                             RankedTensorType kType, bool isDecode,
                             mlir::PatternRewriter &rewriter,
                             Location loc) const {
    if (!mask) {
      return mask;
    }

    auto maskType = mlir::cast<RankedTensorType>(mask.getType());
    auto qShape = qType.getShape();
    auto kShape = kType.getShape();

    // Compute target mask shape based on SDPA variant.
    // Q shape: [batch, num_heads, seq_len, head_dim]
    // K shape: [batch, num_heads, key_seq, head_dim]
    SmallVector<int64_t> targetShape;
    if (isDecode) {
      // Decode: [batch, 1, num_heads, key_seq]
      targetShape = {qShape[0], 1, qShape[kNumHeadsDim], kShape[kSeqLenDim]};
    } else {
      // Regular: [batch, 1, query_seq, key_seq]
      targetShape = {qShape[0], 1, qShape[kSeqLenDim], kShape[kSeqLenDim]};
    }

    // Check if broadcast is needed.
    if (llvm::equal(maskType.getShape(), targetShape)) {
      return mask;
    }

    auto broadcastType =
        utils::RankedTensorTypeFactory::create(maskType, targetShape);
    auto broadcastDims = ttmlir::utils::getBroadcastDimensions<int64_t>(
        maskType.getShape(), targetShape);
    auto shapeAttr = ShapeAttr::get(rewriter.getContext(), broadcastDims);

    return rewriter.create<RepeatOp>(loc, broadcastType, mask, shapeAttr);
  }

  mlir::LogicalResult createSDPAOp(mlir::PatternRewriter &rewriter,
                                   SDPAComponents &c) const {
    op_model::ScopedSingletonDeviceGuard deviceGuard(
        c.attentionMatmul.getOperation());

    // When no scale is found in the pattern, explicitly set scale=1.0 to
    // prevent tt-metal from applying the default 1/sqrt(head_dim) scaling.
    float scale = c.scale.value_or(1.0f);
    FloatAttr scaleAttr = rewriter.getF32FloatAttr(scale);

    // Capture original output element type to restore after SDPA if needed.
    auto originalOutputType =
        mlir::cast<RankedTensorType>(c.attentionMatmul.getResult().getType());
    Type originalElementType = originalOutputType.getElementType();

    // Cast inputs to bf16 if they are f32, since tt-metal SDPA only supports
    // bf16/bfp8_b/bfp4_b. The output will be cast back to the original dtype.
    // TODO(tt-metal): Remove this once tt-metal adds f32 support.
    // tt-metal issue: https://github.com/tenstorrent/tt-metal/issues/36717
    c.query = castToBF16IfNeeded(c.query, rewriter);
    c.key = castToBF16IfNeeded(c.key, rewriter);
    c.value = castToBF16IfNeeded(c.value, rewriter);
    if (c.mask) {
      c.mask = castToBF16IfNeeded(c.mask, rewriter);
    }

    auto qType = mlir::cast<RankedTensorType>(c.query.getType());
    auto qShape = qType.getShape();
    auto kType = mlir::cast<RankedTensorType>(c.key.getType());

    // Check if this is decode mode (query seq_len == 1)
    // Query shape: [batch x num_heads x seq_len x head_size]
    bool isDecode = qShape.size() == 4 && qShape[kSeqLenDim] == 1;
    // Broadcast mask to the required shape for the SDPA variant.
    Value attentionMask = broadcastMaskForSDPA(
        c.mask, qType, kType, isDecode, rewriter, c.attentionMatmul.getLoc());
    if (isDecode) {
      // Permute query: [B, H, 1, D] -> [1, B, H, D]
      Value permutedQuery = ttir_to_ttnn::utils::generatePermute(
          mlir::cast<TypedValue<RankedTensorType>>(c.query),
          llvm::to_vector(kToDecodePermutation), rewriter,
          c.attentionMatmul.getLoc());

      auto decodeOp = rewriter.create<ScaledDotProductAttentionDecodeOp>(
          c.attentionMatmul.getLoc(), permutedQuery.getType(), permutedQuery,
          c.key, c.value,
          /*is_causal=*/rewriter.getBoolAttr(false), attentionMask,
          /*cur_pos_tensor=*/Value(),
          /*attention_sink=*/Value(), scaleAttr,
          /*memory_config=*/MemoryConfigAttr(),
          /*program_config=*/SDPAProgramConfigAttr());

      // Validate the operation using op constraint validation
      std::vector<TTNNLayoutAttr> inputLayouts =
          utils::extractInputLayouts(decodeOp.getOperation());

      auto resultType =
          mlir::cast<RankedTensorType>(decodeOp.getResult().getType());
      OpConfig config(mlir::cast<TTNNLayoutAttr>(resultType.getEncoding()));
      auto result = op_constraint_validation::validateOperation(
          decodeOp.getOperation(), inputLayouts, config);

      if (!result.isSuccess() && !isRecoverableSDPAError(result.errorMessage)) {
        rewriter.eraseOp(decodeOp);
        return failure();
      }

      // Permute result back: [1, B, H, D] -> [B, H, 1, D].
      Value finalResult = ttir_to_ttnn::utils::generatePermute(
          decodeOp.getResult(),
          ttmlir::utils::inversePermutation(kToDecodePermutation), rewriter,
          c.attentionMatmul.getLoc());

      // Restore original element type if SDPA produced a different dtype.
      finalResult = restoreElementTypeIfNeeded(finalResult, originalElementType,
                                               rewriter);

      rewriter.replaceOp(c.attentionMatmul, finalResult);
    } else {
      auto sdpaOp = rewriter.create<ScaledDotProductAttentionOp>(
          c.attentionMatmul.getLoc(), c.query.getType(), c.query, c.key,
          c.value, attentionMask,
          /*is_causal=*/rewriter.getBoolAttr(false), scaleAttr,
          /*sliding_window_size=*/IntegerAttr(),
          /*memory_config=*/MemoryConfigAttr());

      // Validate the operation using op constraint validation
      std::vector<TTNNLayoutAttr> inputLayouts =
          utils::extractInputLayouts(sdpaOp.getOperation());

      auto resultType =
          mlir::cast<RankedTensorType>(sdpaOp.getResult().getType());
      OpConfig config(mlir::cast<TTNNLayoutAttr>(resultType.getEncoding()));
      auto result = op_constraint_validation::validateOperation(
          sdpaOp.getOperation(), inputLayouts, config);

      if (!result.isSuccess() && !isRecoverableSDPAError(result.errorMessage)) {
        rewriter.eraseOp(sdpaOp);
        return failure();
      }

      // Restore original element type if SDPA produced a different dtype.
      Value finalResult = restoreElementTypeIfNeeded(
          sdpaOp.getResult(), originalElementType, rewriter);

      rewriter.replaceOp(c.attentionMatmul, finalResult);
    }

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
      patterns.add<fusing::RoPEFusing>(&getContext());
      patterns.add<fusing::RoPEDecodeFusing>(&getContext());
      patterns.add<DistributedRMSNormFusing>(&getContext());
      patterns.add<SDPAFusing>(&getContext());
    }
#endif // TTMLIR_ENABLE_OPMODEL

    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace

} // namespace mlir::tt::ttnn
