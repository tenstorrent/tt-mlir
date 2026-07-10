// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/StableHLOToTTIR/StableHLOLegalizeComposite.h"

#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"
#include "ttmlir/Dialect/StableHLO/Utils/StableHLOUtils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"

#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"

#include <limits>

using namespace mlir;
using namespace mlir::tt;
using namespace mlir::tt::stablehlo::utils;

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_LEGALIZESTABLEHLOCOMPOSITETOTTIR
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::ttir

namespace {

template <typename TargetOp>
class StableHLOToTTIRCompositeOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CompositeOp> {

  using OpConversionPattern<mlir::stablehlo::CompositeOp>::OpConversionPattern;

public:
  StableHLOToTTIRCompositeOpConversionPattern(MLIRContext *context,
                                              llvm::StringRef opName)
      : OpConversionPattern<mlir::stablehlo::CompositeOp>(context),
        opName(opName) {}

  LogicalResult
  matchAndRewrite(mlir::stablehlo::CompositeOp srcOp,
                  mlir::stablehlo::CompositeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (srcOp.getName() != opName) {
      return rewriter.notifyMatchFailure(
          srcOp, ("CompositeOp must be " + std::string(opName) + ".").c_str());
    }
    if (srcOp.getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp, "CompositeOp must have exactly one result.");
    }

    auto outputType =
        mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

    auto compositeAttrs = srcOp.getCompositeAttributes();
    SmallVector<NamedAttribute> namedAttrs;
    if (compositeAttrs) {
      for (const auto &attr : compositeAttrs) {
        namedAttrs.push_back(attr);
      }
    }

    rewriter.replaceOpWithNewOp<TargetOp>(srcOp, outputType,
                                          adaptor.getOperands(), namedAttrs);
    return success();
  }

private:
  std::string opName;
};

// Special handling for all tenstorrent.topk* composite ops.
// Three different composite ops are supported:
// - tenstorrent.topk: generated when both the values and indices are needed in
// the graph.
// - tenstorrent.topk_indices: generated when only the indices are needed in the
// graph.
// - tenstorrent.topk_values: generated when only the values are needed in the
// graph.
class TenstorrentTopKConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CompositeOp> {
public:
  TenstorrentTopKConversionPattern(MLIRContext *context)
      : OpConversionPattern<mlir::stablehlo::CompositeOp>(context) {}

  LogicalResult
  matchAndRewrite(mlir::stablehlo::CompositeOp srcOp,
                  mlir::stablehlo::CompositeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!supportedOpNames.contains(srcOp.getName())) {
      return failure();
    }
    if (adaptor.getOperands().size() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp, srcOp.getName() +
                     " composite op must have exactly one input operand.");
    }

    bool isTopKWithValues =
        srcOp.getName() == kTTTopKValuesCustomCallTargetName;
    bool isTopKWithIndices =
        srcOp.getName() == kTTTopKIndicesCustomCallTargetName;
    bool isTopKWithBoth = srcOp.getName() == kTTTopKCustomCallTargetName;

    if (isTopKWithBoth && srcOp->getNumResults() != 2) {
      return rewriter.notifyMatchFailure(
          srcOp,
          "tenstorrent.topk composite op must have exactly two results.");
    }
    if (isTopKWithValues && srcOp->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp,
          "tenstorrent.topk_values composite op must have exactly one result.");
    }
    if (isTopKWithIndices && srcOp->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(srcOp,
                                         "tenstorrent.topk_indices composite "
                                         "op must have exactly one result.");
    }

    SmallVector<RankedTensorType, 2> resultTypes;
    if (isTopKWithBoth) {
      resultTypes = {
          mlir::cast<RankedTensorType>(srcOp.getResult(0).getType()),
          mlir::cast<RankedTensorType>(srcOp.getResult(1).getType())};
    } else {
      resultTypes = {
          mlir::cast<RankedTensorType>(srcOp.getResult(0).getType())};
    }

    DictionaryAttr compositeAttrs = srcOp.getCompositeAttributes();
    IntegerAttr kAttr = IntegerAttr::get(rewriter.getIntegerType(32), 1);
    IntegerAttr dimAttr = IntegerAttr::get(rewriter.getIntegerType(32), -1);
    BoolAttr sortedAttr = BoolAttr::get(rewriter.getContext(), false);
    BoolAttr largestAttr = BoolAttr::get(rewriter.getContext(), true);

    if (compositeAttrs) {
      if (auto attr = compositeAttrs.getAs<IntegerAttr>("k")) {
        int64_t val = attr.getInt();
        if (!llvm::isInt<32>(val)) {
          return rewriter.notifyMatchFailure(
              srcOp, "k value is too large for i32: " + Twine(val));
        }
        kAttr = rewriter.getI32IntegerAttr(static_cast<int32_t>(val));
      }
      if (auto attr = compositeAttrs.getAs<IntegerAttr>("dim")) {
        int64_t val = attr.getInt();
        if (!llvm::isInt<32>(val)) {
          return rewriter.notifyMatchFailure(
              srcOp, "dim value is too large for i32: " + Twine(val));
        }
        dimAttr = rewriter.getI32IntegerAttr(static_cast<int32_t>(val));
      }
      if (auto attr = compositeAttrs.getAs<BoolAttr>("largest")) {
        largestAttr = attr;
      }
      if (auto attr = compositeAttrs.getAs<BoolAttr>("sorted")) {
        sortedAttr = attr;
      }
    } else {
      return rewriter.notifyMatchFailure(
          srcOp,
          "tenstorrent.topk composite op must have composite_attributes.");
    }

    auto inputType =
        mlir::cast<RankedTensorType>(adaptor.getOperands()[0].getType());
    RankedTensorType valuesType, indicesType;

    if (isTopKWithBoth) {
      valuesType = resultTypes[0];
      indicesType = resultTypes[1];
    } else if (isTopKWithValues) {
      valuesType = resultTypes[0];
      indicesType =
          RankedTensorType::get(valuesType.getShape(), rewriter.getI32Type());
    } else {
      auto indicesResultType = resultTypes[0];
      valuesType = RankedTensorType::get(indicesResultType.getShape(),
                                         inputType.getElementType());
      indicesType = indicesResultType;
    }

    auto input = adaptor.getOperands()[0];
    auto topKOp = rewriter.create<ttir::TopKOp>(
        srcOp.getLoc(), valuesType, indicesType, input, kAttr, dimAttr,
        largestAttr, sortedAttr);

    if (isTopKWithBoth) {
      rewriter.replaceOp(srcOp, {topKOp.getValues(), topKOp.getIndices()});
    } else if (isTopKWithValues) {
      rewriter.replaceOp(srcOp, {topKOp.getValues()});
    } else {
      rewriter.replaceOp(srcOp, {topKOp.getIndices()});
    }

    return success();
  }

private:
  llvm::SmallSet<llvm::StringRef, 3> supportedOpNames = {
      kTTTopKCustomCallTargetName,
      kTTTopKIndicesCustomCallTargetName,
      kTTTopKValuesCustomCallTargetName,
  };
};

// Converts stablehlo.composite @tenstorrent.argmax -> ttir.argmax.
// Used in the non-sharded path where composites are not converted to
// custom_calls.
class TenstorrentArgMaxConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CompositeOp> {
public:
  TenstorrentArgMaxConversionPattern(MLIRContext *context)
      : OpConversionPattern<mlir::stablehlo::CompositeOp>(context) {}

  LogicalResult
  matchAndRewrite(mlir::stablehlo::CompositeOp srcOp,
                  mlir::stablehlo::CompositeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (srcOp.getName() != kTTArgMaxCustomCallTargetName) {
      return failure();
    }
    if (adaptor.getOperands().size() != 1 || srcOp.getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp, "tenstorrent.argmax must have 1 operand and 1 result");
    }

    auto outputType =
        mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

    DictionaryAttr compositeAttrs = srcOp.getCompositeAttributes();
    int32_t dim = -1;
    bool keepdim = false;
    if (compositeAttrs) {
      if (auto attr = compositeAttrs.getAs<IntegerAttr>("dim")) {
        dim = static_cast<int32_t>(attr.getInt());
      }
      if (auto attr = compositeAttrs.getAs<BoolAttr>("keepdim")) {
        keepdim = attr.getValue();
      }
    }

    auto dimArg = rewriter.getI32ArrayAttr({dim});
    rewriter.replaceOpWithNewOp<ttir::ArgMaxOp>(
        srcOp, outputType, adaptor.getOperands()[0], keepdim, dimArg);
    return success();
  }
};

// Lowers tenstorrent.argmax custom_call ops that were converted from composites
// by FlattenOrConvertCompositesPass (kCompositesWithCustomSharding path).
//
// Two execution paths:
//   Single-device (not inside sdy.manual_computation, or reduction dim
//   replicated):
//     → plain ttir.argmax.
//
//   Distributed (inside sdy.manual_computation with reduction dim sharded):
//     Each chip holds a local shard of the reduction dim. Uses O(n) max
//     reduction + argmax instead of O(n log²n) bitonic sort from topk.
//     The distributed sequence is:
//       1. ttir.max (local)       → [batch, 1] local max values
//       2. ttir.argmax (local)    → [batch, 1] local indices
//       3. ttir.all_gather        → [batch, N] values + indices
//       4. add compile-time shard offset → [batch, N] global indices
//       5. ttir.argmax (merge)    → [batch, 1] which shard won
//       6. ttir.gather            → [batch, 1] aligned global index
class CustomCallArgMaxConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CustomCallOp> {
public:
  CustomCallArgMaxConversionPattern(MLIRContext *context)
      : OpConversionPattern<mlir::stablehlo::CustomCallOp>(context) {}

  LogicalResult
  matchAndRewrite(mlir::stablehlo::CustomCallOp srcOp,
                  mlir::stablehlo::CustomCallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (srcOp.getCallTargetName() != kTTArgMaxCustomCallTargetName ||
        !srcOp->hasAttr(kHasCustomShardingAttr)) {
      return failure();
    }
    if (adaptor.getOperands().size() != 1 || srcOp.getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp, "argmax custom_call must have 1 operand and 1 result");
    }

    auto compositeAttrs = mlir::dyn_cast_or_null<DictionaryAttr>(
        srcOp->getDiscardableAttr(kCustomCallCompositeAttrsKey));
    if (!compositeAttrs) {
      return rewriter.notifyMatchFailure(srcOp, "missing composite attributes");
    }

    int32_t dim = -1;
    bool keepdim = false;
    if (auto attr = compositeAttrs.getAs<IntegerAttr>("dim")) {
      dim = static_cast<int32_t>(attr.getInt());
    }
    if (auto attr = compositeAttrs.getAs<BoolAttr>("keepdim")) {
      keepdim = attr.getValue();
    }

    Value input = adaptor.getOperands()[0];
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    int64_t rank = inputType.getRank();
    int64_t batch = inputType.getShape()[0];
    mlir::Type elemType = inputType.getElementType();

    int64_t reductionDim = (dim >= 0) ? dim : rank + dim;
    int64_t numItemsLocal = inputType.getShape()[reductionDim];

    auto resultType =
        mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());
    auto loc = srcOp.getLoc();

    // Distributed iff inside sdy.manual_computation with the reduction dim
    // sharded. Same walk-back logic as CustomCallTopKConversionPattern.
    int64_t numShards = 0;
    uint32_t clusterAxis = 0;

    if (srcOp->getParentOfType<mlir::sdy::ManualComputationOp>()) {
      auto moduleOp = srcOp->getParentOfType<mlir::ModuleOp>();
      auto meshOps = shardy_utils::getMeshOps(moduleOp);
      if (!meshOps.empty()) {
        Value shardingSource = srcOp.getOperand(0);
        int64_t trackedDim = reductionDim;
        auto findReshapeMappedDim =
            [](llvm::ArrayRef<int64_t> outShape,
               llvm::ArrayRef<int64_t> inShape,
               int64_t outDim) -> std::optional<int64_t> {
          if (outDim < 0 || outDim >= static_cast<int64_t>(outShape.size())) {
            return std::nullopt;
          }
          if (outShape[outDim] == 1) {
            return std::nullopt;
          }
          int64_t nonUnitPos = 0;
          for (int64_t i = 0; i < outDim; ++i) {
            if (outShape[i] != 1) {
              nonUnitPos++;
            }
          }
          int64_t count = 0;
          for (int64_t i = 0; i < static_cast<int64_t>(inShape.size()); ++i) {
            if (inShape[i] == 1) {
              continue;
            }
            if (count == nonUnitPos) {
              return inShape[i] == outShape[outDim] ? std::optional<int64_t>(i)
                                                    : std::nullopt;
            }
            count++;
          }
          return std::nullopt;
        };
        mlir::sdy::TensorShardingAttr resolvedSharding = nullptr;
        bool walkOk = true;
        while (walkOk) {
          mlir::Operation *def = shardingSource.getDefiningOp();
          if (!def) {
            break;
          }
          if (auto reshape = llvm::dyn_cast<mlir::stablehlo::ReshapeOp>(def)) {
            auto outShape =
                mlir::cast<mlir::RankedTensorType>(reshape.getType())
                    .getShape();
            auto inShape = mlir::cast<mlir::RankedTensorType>(
                               reshape.getOperand().getType())
                               .getShape();
            auto mapped = findReshapeMappedDim(outShape, inShape, trackedDim);
            if (!mapped) {
              walkOk = false;
              break;
            }
            trackedDim = *mapped;
            shardingSource = reshape.getOperand();
            continue;
          }
          if (auto transpose =
                  llvm::dyn_cast<mlir::stablehlo::TransposeOp>(def)) {
            auto perm = transpose.getPermutation();
            if (trackedDim < 0 ||
                trackedDim >= static_cast<int64_t>(perm.size())) {
              walkOk = false;
              break;
            }
            trackedDim = perm[trackedDim];
            shardingSource = transpose.getOperand();
            continue;
          }
          if (auto sel = llvm::dyn_cast<mlir::stablehlo::SelectOp>(def)) {
            shardingSource = sel.getOnTrue();
            continue;
          }
          if (auto dot = llvm::dyn_cast<mlir::stablehlo::DotGeneralOp>(def)) {
            auto dnums = dot.getDotDimensionNumbers();
            auto batchLhs = dnums.getLhsBatchingDimensions();
            auto batchRhs = dnums.getRhsBatchingDimensions();
            auto contractLhs = dnums.getLhsContractingDimensions();
            auto contractRhs = dnums.getRhsContractingDimensions();
            auto lhsShape =
                mlir::cast<mlir::RankedTensorType>(dot.getLhs().getType())
                    .getShape();
            auto rhsShape =
                mlir::cast<mlir::RankedTensorType>(dot.getRhs().getType())
                    .getShape();
            int64_t numBatch = batchLhs.size();
            if (trackedDim < numBatch) {
              trackedDim = batchLhs[trackedDim];
              shardingSource = dot.getLhs();
              continue;
            }
            int64_t pos = trackedDim - numBatch;
            llvm::SmallVector<int64_t> lhsOther;
            for (int64_t i = 0; i < static_cast<int64_t>(lhsShape.size());
                 ++i) {
              if (!llvm::is_contained(batchLhs, i) &&
                  !llvm::is_contained(contractLhs, i)) {
                lhsOther.push_back(i);
              }
            }
            if (pos < static_cast<int64_t>(lhsOther.size())) {
              trackedDim = lhsOther[pos];
              shardingSource = dot.getLhs();
              continue;
            }
            llvm::SmallVector<int64_t> rhsOther;
            for (int64_t i = 0; i < static_cast<int64_t>(rhsShape.size());
                 ++i) {
              if (!llvm::is_contained(batchRhs, i) &&
                  !llvm::is_contained(contractRhs, i)) {
                rhsOther.push_back(i);
              }
            }
            int64_t rhsPos = pos - lhsOther.size();
            if (rhsPos < static_cast<int64_t>(rhsOther.size())) {
              trackedDim = rhsOther[rhsPos];
              shardingSource = dot.getRhs();
              continue;
            }
            walkOk = false;
            break;
          }
          // sdy collective ops
          // (all_slice/all_gather/reshard/sharding_constraint) are emitted as
          // stablehlo.composite carrying the authoritative result sharding in
          // composite_attributes["out_sharding"]. Read it directly instead of
          // walking further back (e.g. to the lm_head weight, whose sharding
          // may name a different axis than the resharded logits).
          if (auto composite =
                  llvm::dyn_cast<mlir::stablehlo::CompositeOp>(def)) {
            if (auto attrs = composite.getCompositeAttributes()) {
              resolvedSharding =
                  attrs.getAs<mlir::sdy::TensorShardingAttr>("out_sharding");
            }
            break;
          }
          break;
        }
        // Prefer the operand sharding from the walk; it resolves the correct
        // axis on 2D meshes. Fall back to the sdy composite's out_sharding only
        // when the operand sharding names no axis on the reduction dim (the
        // all_slice grammar path, where it is a replicated/open default that
        // yields numShards=0).
        if (walkOk && shardingSource.use_empty()) {
          walkOk = false;
        }
        mlir::sdy::TensorShardingAttr fallbackSharding =
            walkOk ? shardy_utils::getOperandShardingAttr(
                         *shardingSource.use_begin(), meshOps[0])
                   : shardy_utils::getOperandShardingAttr(
                         srcOp->getOpOperand(0), meshOps[0]);
        int64_t fallbackQueryDim = walkOk ? trackedDim : reductionDim;
        // The distributed lowering assumes a single shard axis on the reduction
        // dim (offset = shard * numItemsLocal, one all_gather cluster axis).
        // Sharding the reduction dim across >1 mesh axis is unsupported — fail
        // loudly instead of silently taking only the first axis.
        if (fallbackSharding) {
          auto fbDims = fallbackSharding.getDimShardings();
          if (fallbackQueryDim < static_cast<int64_t>(fbDims.size()) &&
              fbDims[fallbackQueryDim].getAxes().size() > 1) {
            return rewriter.notifyMatchFailure(
                srcOp, "distributed argmax: reduction dim sharded across "
                       "multiple mesh axes is unsupported");
          }
        }
        auto resolveNumShards = [&](mlir::sdy::TensorShardingAttr sharding,
                                    int64_t queryDim) -> bool {
          if (!sharding) {
            return false;
          }
          auto dimShardings = sharding.getDimShardings();
          if (queryDim < static_cast<int64_t>(dimShardings.size()) &&
              !dimShardings[queryDim].getAxes().empty()) {
            llvm::StringRef axisName =
                dimShardings[queryDim].getAxes()[0].getName();
            for (auto [idx, axis] :
                 llvm::enumerate(meshOps[0].getMeshAttr().getAxes())) {
              if (axis.getName() == axisName) {
                numShards = axis.getSize();
                clusterAxis = static_cast<uint32_t>(idx);
                return true;
              }
            }
          }
          return false;
        };
        if (!resolveNumShards(fallbackSharding, fallbackQueryDim)) {
          resolveNumShards(resolvedSharding, trackedDim);
        }
      }
    }

    bool isDistributed = numShards > 1;

    if (!isDistributed) {
      auto dimArg =
          rewriter.getI32ArrayAttr({static_cast<int32_t>(reductionDim)});
      rewriter.replaceOpWithNewOp<ttir::ArgMaxOp>(srcOp, resultType, input,
                                                  keepdim, dimArg);
      return success();
    }

    // NOTE (known runtime bug — path currently unused by tt-xla): the i32
    // index all_gather below corrupts the tiny [batch, 1] local-argmax tensor
    // in a full multi-collective model graph (it reads tile padding, e.g.
    // yielding 32), while the bf16 value all_gather beside it is fine. The
    // sequence is correct in isolation (unit tests and isolated probes pass)
    // but returns wrong indices end-to-end, so tt-xla's vocab-sharded greedy
    // sampling does NOT use composite_argmax — it gathers logits to full and
    // runs a plain argmax instead. Routing the index through f32 only reduces
    // the error to off-by-one, so the real fix is the small/sub-tile
    // all_gather in tt-metal; re-enable this path once that is fixed.
    // Distributed argmax: local max/argmax → all_gather → shard offset →
    // merge argmax → gather.
    auto dimArgAttr =
        rewriter.getI32ArrayAttr({static_cast<int32_t>(reductionDim)});
    auto dimI32 =
        rewriter.getI32IntegerAttr(static_cast<int32_t>(reductionDim));

    // 1. Local max + argmax (O(n) scan — no bitonic sort).
    auto localMaxType = RankedTensorType::get({batch, 1}, elemType);
    auto localIdxType =
        RankedTensorType::get({batch, 1}, rewriter.getI32Type());
    Value localMax = rewriter.create<ttir::MaxOp>(
        loc, localMaxType, input, true /*keep_dim*/, dimArgAttr);
    Value localIdx = rewriter.create<ttir::ArgMaxOp>(
        loc, localIdxType, input, true /*keep_dim*/, dimArgAttr);

    // 2. All_gather: [batch, 1] → [batch, numShards].
    auto gatheredMaxType = RankedTensorType::get({batch, numShards}, elemType);
    auto gatheredIdxType =
        RankedTensorType::get({batch, numShards}, rewriter.getI32Type());
    Value gatheredMax = rewriter.create<ttir::AllGatherOp>(
        loc, gatheredMaxType, localMax, static_cast<int32_t>(reductionDim),
        clusterAxis);
    Value gatheredIdx = rewriter.create<ttir::AllGatherOp>(
        loc, gatheredIdxType, localIdx, static_cast<int32_t>(reductionDim),
        clusterAxis);

    // 3. Add shard offsets to make indices global.
    SmallVector<int32_t> offsetValues;
    offsetValues.reserve(batch * numShards);
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t shard = 0; shard < numShards; ++shard) {
        offsetValues.push_back(static_cast<int32_t>(shard * numItemsLocal));
      }
    }
    auto offsetAttr = DenseElementsAttr::get(
        gatheredIdxType, llvm::ArrayRef<int32_t>(offsetValues));
    Value offsetConst =
        rewriter.create<ttir::ConstantOp>(loc, gatheredIdxType, offsetAttr);
    Value globalIdx = rewriter.create<ttir::AddOp>(loc, gatheredIdxType,
                                                   gatheredIdx, offsetConst);

    // 4. Merge: argmax on gathered values to find which shard won.
    auto mergeIdxType =
        RankedTensorType::get({batch, 1}, rewriter.getI32Type());
    Value mergeIdx = rewriter.create<ttir::ArgMaxOp>(
        loc, mergeIdxType, gatheredMax, true /*keep_dim*/, dimArgAttr);

    // 5. Gather the global index of the winning shard.
    auto ui32Type =
        RankedTensorType::get({batch, 1}, rewriter.getIntegerType(32, false));
    Value mergeIdxU32 =
        rewriter.create<ttir::TypecastOp>(loc, ui32Type, mergeIdx);
    auto finalIdxType =
        RankedTensorType::get({batch, 1}, rewriter.getI32Type());
    Value result = rewriter.create<ttir::GatherOp>(loc, finalIdxType, globalIdx,
                                                   mergeIdxU32, dimI32);

    // 6. Handle keepdim and type casting.
    if (!keepdim) {
      auto reshapedType = RankedTensorType::get({batch}, rewriter.getI32Type());
      result = rewriter.create<ttir::ReshapeOp>(
          loc, reshapedType, result,
          rewriter.getI32ArrayAttr({static_cast<int32_t>(batch)}));
    }

    if (resultType.getElementType() != rewriter.getI32Type()) {
      result = rewriter.create<ttir::TypecastOp>(loc, resultType, result);
    }

    rewriter.replaceOp(srcOp, {result});
    return success();
  }
};

// Lowers tenstorrent.topk* custom_call ops that were converted from composites
// by FlattenOrConvertCompositesPass (kCompositesWithCustomSharding path).
//
// Two execution paths:
//   Single-device (not inside sdy.manual_computation, or topk dim replicated):
//     → plain ttir.topk, same as TenstorrentTopKConversionPattern.
//
//   Distributed (inside sdy.manual_computation with topk dim sharded):
//     Each chip holds a local shard of the topk dim. The distributed sequence
//     is:
//       1. ttir.topk locally  → [batch, k] values + local indices
//       2. ttir.all_gather    → [batch, k*N] values + local indices
//       3. add compile-time shard offset → [batch, k*N] global indices
//       4. ttir.topk (merge)  → [batch, k] merged values + sort order
//       5. ttir.gather        → [batch, k] aligned global indices
class CustomCallTopKConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CustomCallOp> {
public:
  CustomCallTopKConversionPattern(MLIRContext *context)
      : OpConversionPattern<mlir::stablehlo::CustomCallOp>(context) {}

  LogicalResult
  matchAndRewrite(mlir::stablehlo::CustomCallOp srcOp,
                  mlir::stablehlo::CustomCallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!supportedOpNames.contains(srcOp.getCallTargetName()) ||
        !srcOp->hasAttr(kHasCustomShardingAttr)) {
      return failure();
    }
    if (adaptor.getOperands().size() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp, "topk custom_call must have exactly one input operand.");
    }

    bool isTopKWithBoth =
        srcOp.getCallTargetName() == kTTTopKCustomCallTargetName;
    bool isTopKWithValues =
        srcOp.getCallTargetName() == kTTTopKValuesCustomCallTargetName;

    auto compositeAttrs = mlir::dyn_cast_or_null<DictionaryAttr>(
        srcOp->getDiscardableAttr(kCustomCallCompositeAttrsKey));
    if (!compositeAttrs) {
      return rewriter.notifyMatchFailure(srcOp, "missing composite attributes");
    }

    IntegerAttr kAttr = rewriter.getI32IntegerAttr(1);
    IntegerAttr dimAttr = rewriter.getI32IntegerAttr(-1);
    BoolAttr sortedAttr = BoolAttr::get(rewriter.getContext(), true);
    BoolAttr largestAttr = BoolAttr::get(rewriter.getContext(), true);

    if (auto attr = compositeAttrs.getAs<IntegerAttr>("k")) {
      int64_t val = attr.getInt();
      if (!llvm::isInt<32>(val)) {
        return rewriter.notifyMatchFailure(srcOp, "k too large for i32");
      }
      kAttr = rewriter.getI32IntegerAttr(static_cast<int32_t>(val));
    }
    if (auto attr = compositeAttrs.getAs<IntegerAttr>("dim")) {
      int64_t val = attr.getInt();
      if (!llvm::isInt<32>(val)) {
        return rewriter.notifyMatchFailure(srcOp, "dim too large for i32");
      }
      dimAttr = rewriter.getI32IntegerAttr(static_cast<int32_t>(val));
    }
    if (auto attr = compositeAttrs.getAs<BoolAttr>("sorted")) {
      sortedAttr = attr;
    }
    if (auto attr = compositeAttrs.getAs<BoolAttr>("largest")) {
      largestAttr = attr;
    }

    Value input = adaptor.getOperands()[0];
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    int64_t rank = inputType.getRank();
    int64_t batch = inputType.getShape()[0];
    mlir::Type elemType = inputType.getElementType();
    int32_t k = kAttr.getInt();

    int64_t d = dimAttr.getInt();
    int64_t topkDim = (d >= 0) ? d : rank + d;
    int64_t numItemsLocal = inputType.getShape()[topkDim];

    RankedTensorType valuesType, indicesType;
    if (isTopKWithBoth) {
      valuesType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());
      indicesType = mlir::cast<RankedTensorType>(srcOp.getResult(1).getType());
    } else if (isTopKWithValues) {
      valuesType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());
      indicesType =
          RankedTensorType::get(valuesType.getShape(), rewriter.getI32Type());
    } else {
      indicesType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());
      valuesType = RankedTensorType::get(indicesType.getShape(), elemType);
    }

    auto loc = srcOp.getLoc();

    // Distributed iff inside sdy.manual_computation with the topk dim sharded.
    int64_t numShards = 0;
    uint32_t clusterAxis = 0;

    if (srcOp->getParentOfType<mlir::sdy::ManualComputationOp>()) {
      auto moduleOp = srcOp->getParentOfType<mlir::ModuleOp>();
      auto meshOps = shardy_utils::getMeshOps(moduleOp);
      if (!meshOps.empty()) {
        // Walk back through pass-through ops to find a value whose sharding
        // can be resolved, tracking which dim of the walked-back value
        // corresponds to the original topk dim. Frontends (e.g. torch_xla)
        // emit reshape/transpose/dot_general chains between sharded block
        // args and composites; querying the immediate operand returns the
        // replicated default and silently disables the distributed lowering.
        Value shardingSource = srcOp.getOperand(0);
        int64_t trackedDim = topkDim;
        auto findReshapeMappedDim =
            [](llvm::ArrayRef<int64_t> outShape,
               llvm::ArrayRef<int64_t> inShape,
               int64_t outDim) -> std::optional<int64_t> {
          // Handle unit-dim noise (e.g. (B,V) <-> (1,B,V)) by matching
          // non-unit dims by ordinal. Bail on non-trivial merges/splits.
          if (outDim < 0 || outDim >= static_cast<int64_t>(outShape.size())) {
            return std::nullopt;
          }
          if (outShape[outDim] == 1) {
            return std::nullopt;
          }
          int64_t nonUnitPos = 0;
          for (int64_t i = 0; i < outDim; ++i) {
            if (outShape[i] != 1) {
              nonUnitPos++;
            }
          }
          int64_t count = 0;
          for (int64_t i = 0; i < static_cast<int64_t>(inShape.size()); ++i) {
            if (inShape[i] == 1) {
              continue;
            }
            if (count == nonUnitPos) {
              return inShape[i] == outShape[outDim] ? std::optional<int64_t>(i)
                                                    : std::nullopt;
            }
            count++;
          }
          return std::nullopt;
        };
        mlir::sdy::TensorShardingAttr resolvedSharding = nullptr;
        bool walkOk = true;
        while (walkOk) {
          mlir::Operation *def = shardingSource.getDefiningOp();
          if (!def) {
            break; // Reached a BlockArgument; query sharding from manual
                   // computation's in_shardings.
          }
          if (auto reshape = llvm::dyn_cast<mlir::stablehlo::ReshapeOp>(def)) {
            auto outShape =
                mlir::cast<mlir::RankedTensorType>(reshape.getType())
                    .getShape();
            auto inShape = mlir::cast<mlir::RankedTensorType>(
                               reshape.getOperand().getType())
                               .getShape();
            auto mapped = findReshapeMappedDim(outShape, inShape, trackedDim);
            if (!mapped) {
              walkOk = false;
              break;
            }
            trackedDim = *mapped;
            shardingSource = reshape.getOperand();
            continue;
          }
          if (auto transpose =
                  llvm::dyn_cast<mlir::stablehlo::TransposeOp>(def)) {
            auto perm = transpose.getPermutation();
            if (trackedDim < 0 ||
                trackedDim >= static_cast<int64_t>(perm.size())) {
              walkOk = false;
              break;
            }
            trackedDim = perm[trackedDim];
            shardingSource = transpose.getOperand();
            continue;
          }
          if (auto sel = llvm::dyn_cast<mlir::stablehlo::SelectOp>(def)) {
            // select(cond, on_true, on_false): on_true and on_false have the
            // same shape and same sharding implications for the result.
            // Walk through on_true (operand 1), skipping the condition.
            shardingSource = sel.getOnTrue();
            continue;
          }
          if (auto dot = llvm::dyn_cast<mlir::stablehlo::DotGeneralOp>(def)) {
            auto dnums = dot.getDotDimensionNumbers();
            auto batchLhs = dnums.getLhsBatchingDimensions();
            auto batchRhs = dnums.getRhsBatchingDimensions();
            auto contractLhs = dnums.getLhsContractingDimensions();
            auto contractRhs = dnums.getRhsContractingDimensions();
            auto lhsShape =
                mlir::cast<mlir::RankedTensorType>(dot.getLhs().getType())
                    .getShape();
            auto rhsShape =
                mlir::cast<mlir::RankedTensorType>(dot.getRhs().getType())
                    .getShape();
            int64_t numBatch = batchLhs.size();
            // Output layout: [batch_dims, lhs_other_dims, rhs_other_dims].
            if (trackedDim < numBatch) {
              trackedDim = batchLhs[trackedDim];
              shardingSource = dot.getLhs();
              continue;
            }
            int64_t pos = trackedDim - numBatch;
            llvm::SmallVector<int64_t> lhsOther;
            for (int64_t i = 0; i < static_cast<int64_t>(lhsShape.size());
                 ++i) {
              if (!llvm::is_contained(batchLhs, i) &&
                  !llvm::is_contained(contractLhs, i)) {
                lhsOther.push_back(i);
              }
            }
            if (pos < static_cast<int64_t>(lhsOther.size())) {
              trackedDim = lhsOther[pos];
              shardingSource = dot.getLhs();
              continue;
            }
            llvm::SmallVector<int64_t> rhsOther;
            for (int64_t i = 0; i < static_cast<int64_t>(rhsShape.size());
                 ++i) {
              if (!llvm::is_contained(batchRhs, i) &&
                  !llvm::is_contained(contractRhs, i)) {
                rhsOther.push_back(i);
              }
            }
            int64_t rhsPos = pos - lhsOther.size();
            if (rhsPos < static_cast<int64_t>(rhsOther.size())) {
              trackedDim = rhsOther[rhsPos];
              shardingSource = dot.getRhs();
              continue;
            }
            walkOk = false;
            break;
          }
          // sdy collective ops
          // (all_slice/all_gather/reshard/sharding_constraint) are emitted as
          // stablehlo.composite carrying the authoritative result sharding in
          // composite_attributes["out_sharding"]. Read it directly instead of
          // walking further back (e.g. to the lm_head weight, whose sharding
          // may name a different axis than the resharded logits).
          if (auto composite =
                  llvm::dyn_cast<mlir::stablehlo::CompositeOp>(def)) {
            if (auto attrs = composite.getCompositeAttributes()) {
              resolvedSharding =
                  attrs.getAs<mlir::sdy::TensorShardingAttr>("out_sharding");
            }
            break;
          }
          // Unknown op — stop and query sharding on what we have.
          break;
        }
        // Prefer the operand sharding from the walk; it resolves the correct
        // axis on 2D meshes. Fall back to the sdy composite's out_sharding only
        // when the operand sharding names no axis on the topk dim (the
        // all_slice grammar path, where it is a replicated/open default that
        // yields numShards=0).
        if (walkOk && shardingSource.use_empty()) {
          walkOk = false;
        }
        mlir::sdy::TensorShardingAttr fallbackSharding =
            walkOk ? shardy_utils::getOperandShardingAttr(
                         *shardingSource.use_begin(), meshOps[0])
                   : shardy_utils::getOperandShardingAttr(
                         srcOp->getOpOperand(0), meshOps[0]);
        int64_t fallbackQueryDim = walkOk ? trackedDim : topkDim;
        // The distributed lowering assumes a single shard axis on the topk dim
        // (offset = shard * numItemsLocal, one all_gather cluster axis).
        // Sharding the topk dim across >1 mesh axis is unsupported — fail
        // loudly instead of silently taking only the first axis.
        if (fallbackSharding) {
          auto fbDims = fallbackSharding.getDimShardings();
          if (fallbackQueryDim < static_cast<int64_t>(fbDims.size()) &&
              fbDims[fallbackQueryDim].getAxes().size() > 1) {
            return rewriter.notifyMatchFailure(
                srcOp,
                "distributed topk: topk dim sharded across multiple mesh "
                "axes is unsupported");
          }
        }
        auto resolveNumShards = [&](mlir::sdy::TensorShardingAttr sharding,
                                    int64_t queryDim) -> bool {
          if (!sharding) {
            return false;
          }
          auto dimShardings = sharding.getDimShardings();
          if (queryDim < static_cast<int64_t>(dimShardings.size()) &&
              !dimShardings[queryDim].getAxes().empty()) {
            llvm::StringRef axisName =
                dimShardings[queryDim].getAxes()[0].getName();
            for (auto [idx, axis] :
                 llvm::enumerate(meshOps[0].getMeshAttr().getAxes())) {
              if (axis.getName() == axisName) {
                numShards = axis.getSize();
                clusterAxis = static_cast<uint32_t>(idx);
                return true;
              }
            }
          }
          return false;
        };
        if (!resolveNumShards(fallbackSharding, fallbackQueryDim)) {
          resolveNumShards(resolvedSharding, trackedDim);
        }
      }
    }

    bool isDistributed = numShards > 1;

    if (!isDistributed) {
      auto topKOp = rewriter.create<ttir::TopKOp>(loc, valuesType, indicesType,
                                                  input, kAttr, dimAttr,
                                                  largestAttr, sortedAttr);
      if (isTopKWithBoth) {
        rewriter.replaceOp(srcOp, {topKOp.getValues(), topKOp.getIndices()});
      } else if (isTopKWithValues) {
        rewriter.replaceOp(srcOp, {topKOp.getValues()});
      } else {
        rewriter.replaceOp(srcOp, {topKOp.getIndices()});
      }
      return success();
    }

    // Multi-core eligibility: ttnn.topk only uses its multi-core program
    // factory when k <= 64 AND the topk-dim width is a power of 2 (it needs an
    // even power-of-2 core split); otherwise it falls back to a single core
    // (~10x slower, e.g. ~14.7ms vs ~1.3ms for a 16k-wide shard). So the
    // per-shard LOCAL topk must keep k small (localK <= 64) and run on a
    // power-of-2 width, while a separate MERGE topk produces the user's k from
    // the gathered candidates. Decoupling localK (per-shard) from outK (output)
    // is what keeps the local topk multi-core; the merge runs over a tiny width
    // (numShards*localK), so its single-core cost is negligible.
    constexpr int64_t kTileWidth = 32;
    constexpr int64_t kMaxMultiCoreK = 64;
    auto roundUpTo = [](int64_t v, int64_t m) { return ((v + m - 1) / m) * m; };

    // Output/merge candidate count: tile-aligned, >= k.
    int64_t outK = std::max(kTileWidth, roundUpTo(k, kTileWidth));
    // Per-shard local candidate count: smallest tile-multiple with
    // localK*numShards >= outK, capped at kMaxMultiCoreK for multi-core.
    int64_t localK =
        std::min(kMaxMultiCoreK,
                 roundUpTo((outK + numShards - 1) / numShards, kTileWidth));
    localK = std::min(localK, numItemsLocal);
    // If numShards*localK still can't cover outK (very large k), grow localK
    // even if it costs single-core — correctness over speed.
    while (localK * numShards < outK && localK < numItemsLocal) {
      localK = std::min(localK + kTileWidth, numItemsLocal);
    }
    int64_t totalCandidates = localK * numShards;
    outK = std::min(outK, totalCandidates);
    bool padded = outK != k;
    auto dimI32 = rewriter.getI32IntegerAttr(static_cast<int32_t>(topkDim));
    auto localKAttr = rewriter.getI32IntegerAttr(static_cast<int32_t>(localK));
    auto outKAttr = rewriter.getI32IntegerAttr(static_cast<int32_t>(outK));

    // Pad the local shard width up to a power of 2 so the local topk is
    // multi-core eligible. -inf padding can never win, so the returned local
    // indices stay in [0, numItemsLocal) and need no offset correction.
    Value localInput = input;
    int64_t pow2Width = 1;
    while (pow2Width < numItemsLocal) {
      pow2Width <<= 1;
    }
    if (pow2Width != numItemsLocal) {
      SmallVector<int64_t> padShape(inputType.getShape().begin(),
                                    inputType.getShape().end());
      padShape[topkDim] = pow2Width;
      SmallVector<int32_t> padCfg(2 * rank, 0);
      padCfg[2 * topkDim + 1] = static_cast<int32_t>(pow2Width - numItemsLocal);
      localInput = rewriter.create<ttir::PadOp>(
          loc, RankedTensorType::get(padShape, elemType), input,
          rewriter.getDenseI32ArrayAttr(padCfg),
          rewriter.getF32FloatAttr(-std::numeric_limits<float>::infinity()));
    }

    auto localValType = RankedTensorType::get({batch, localK}, elemType);
    auto localIndType =
        RankedTensorType::get({batch, localK}, rewriter.getI32Type());
    auto localTopK = rewriter.create<ttir::TopKOp>(
        loc, localValType, localIndType, localInput, localKAttr, dimI32,
        largestAttr, BoolAttr::get(rewriter.getContext(), false));

    auto gatheredValType =
        RankedTensorType::get({batch, totalCandidates}, elemType);
    auto gatheredIndType =
        RankedTensorType::get({batch, totalCandidates}, rewriter.getI32Type());

    Value gatheredVals = rewriter.create<ttir::AllGatherOp>(
        loc, gatheredValType, localTopK.getValues(),
        static_cast<int32_t>(topkDim), clusterAxis);
    Value gatheredInds = rewriter.create<ttir::AllGatherOp>(
        loc, gatheredIndType, localTopK.getIndices(),
        static_cast<int32_t>(topkDim), clusterAxis);

    // After all_gather, block [shard*localK : (shard+1)*localK] holds
    // chip `shard`'s localK candidates with local indices in
    // [0, numItemsLocal). Adding shard*numItemsLocal makes them global.
    SmallVector<int32_t> offsetValues;
    offsetValues.reserve(batch * totalCandidates);
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t shard = 0; shard < numShards; ++shard) {
        int32_t shardOffset = static_cast<int32_t>(shard * numItemsLocal);
        for (int32_t j = 0; j < localK; ++j) {
          offsetValues.push_back(shardOffset);
        }
      }
    }
    auto offsetType = gatheredIndType;
    auto offsetAttr = DenseElementsAttr::get(
        offsetType, llvm::ArrayRef<int32_t>(offsetValues));
    Value offsetConst =
        rewriter.create<ttir::ConstantOp>(loc, offsetType, offsetAttr);
    Value globalInds = rewriter.create<ttir::AddOp>(loc, gatheredIndType,
                                                    gatheredInds, offsetConst);

    // Merge with outK (tile-aligned >= k) over the small gathered candidate
    // set; slice down to the user's k after the gather. (If padded==false this
    // is the same as the original merge with K=k.)
    auto mergedValType = RankedTensorType::get({batch, outK}, elemType);
    auto sortOrderType =
        RankedTensorType::get({batch, outK}, rewriter.getI32Type());
    auto mergeTopK = rewriter.create<ttir::TopKOp>(
        loc, mergedValType, sortOrderType, gatheredVals, outKAttr, dimI32,
        largestAttr, BoolAttr::get(rewriter.getContext(), true));

    // GatherOp requires unsigned indices.
    auto ui32Type = RankedTensorType::get({batch, outK},
                                          rewriter.getIntegerType(32, false));
    Value sortOrderUi32 = rewriter.create<ttir::TypecastOp>(
        loc, ui32Type, mergeTopK.getIndices());
    Value alignedInds = rewriter.create<ttir::GatherOp>(
        loc, RankedTensorType::get({batch, outK}, rewriter.getI32Type()),
        globalInds, sortOrderUi32, dimI32);

    Value finalValues = mergeTopK.getValues();
    Value finalInds = alignedInds;
    if (padded) {
      // Slice down to the user-visible k along the topk dim.
      SmallVector<int64_t> sliceStarts(2, 0);
      SmallVector<int64_t> sliceEnds = {batch, k};
      SmallVector<int64_t> sliceSteps(2, 1);
      auto slicedValType = RankedTensorType::get({batch, k}, elemType);
      auto slicedIndType =
          RankedTensorType::get({batch, k}, rewriter.getI32Type());
      auto toI32Array = [&](ArrayRef<int64_t> v) {
        return rewriter.getI32ArrayAttr(
            SmallVector<int32_t>(v.begin(), v.end()));
      };
      finalValues = rewriter.create<ttir::SliceStaticOp>(
          loc, slicedValType, finalValues, toI32Array(sliceStarts),
          toI32Array(sliceEnds), toI32Array(sliceSteps));
      finalInds = rewriter.create<ttir::SliceStaticOp>(
          loc, slicedIndType, finalInds, toI32Array(sliceStarts),
          toI32Array(sliceEnds), toI32Array(sliceSteps));
    }

    // Cast i32 indices to the declared index type if it differs.
    auto castIfNeeded = [&](Value inds, RankedTensorType targetType) -> Value {
      if (targetType.getElementType() == rewriter.getI32Type()) {
        return inds;
      }
      return rewriter.create<ttir::TypecastOp>(loc, targetType, inds);
    };

    if (isTopKWithBoth) {
      rewriter.replaceOp(srcOp,
                         {finalValues, castIfNeeded(finalInds, indicesType)});
    } else if (isTopKWithValues) {
      rewriter.replaceOp(srcOp, {finalValues});
    } else {
      rewriter.replaceOp(srcOp, {castIfNeeded(finalInds, indicesType)});
    }
    return success();
  }

private:
  llvm::SmallSet<llvm::StringRef, 3> supportedOpNames = {
      kTTTopKCustomCallTargetName,
      kTTTopKValuesCustomCallTargetName,
      kTTTopKIndicesCustomCallTargetName,
  };
};

// Special handling for tenstorrent.uniform -> ttir.rand, as
// it requires extracting values from operands and translating them to
// attributes, and because ttir.rand is a non-DPS op.
class TenstorrentUniformToRandConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CompositeOp> {

public:
  TenstorrentUniformToRandConversionPattern(MLIRContext *context)
      : OpConversionPattern<mlir::stablehlo::CompositeOp>(context) {}

  LogicalResult
  matchAndRewrite(mlir::stablehlo::CompositeOp srcOp,
                  mlir::stablehlo::CompositeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (srcOp.getName() != "tenstorrent.uniform") {
      return failure();
    }

    auto outputType =
        mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

    DictionaryAttr compositeAttrs = srcOp.getCompositeAttributes();

    // Extract shape attribute.
    auto sizeAttr = mlir::cast<ArrayAttr>(compositeAttrs.get("shape"));

    // Extract low and high from constant operands.
    auto lowOp =
        adaptor.getOperands()[1].getDefiningOp<mlir::stablehlo::ConstantOp>();
    auto highOp =
        adaptor.getOperands()[2].getDefiningOp<mlir::stablehlo::ConstantOp>();

    assert(lowOp && "low operand must be a ConstantOp");
    assert(highOp && "high operand must be a ConstantOp");

    auto lowValue = mlir::cast<DenseFPElementsAttr>(lowOp.getValue());
    auto highValue = mlir::cast<DenseFPElementsAttr>(highOp.getValue());

    assert(lowValue.getNumElements() == 1 &&
           "Expected low operand to be a scalar constant");
    assert(highValue.getNumElements() == 1 &&
           "Expected high operand to be a scalar constant");

    auto lowAttr = rewriter.getF32FloatAttr(lowValue.getValues<float>()[0]);
    auto highAttr = rewriter.getF32FloatAttr(highValue.getValues<float>()[0]);

    // Proceed with default seed = 0 for now, because in tt-metal it will
    // actually generate different random numbers on each execution, which we
    // agreed is acceptable for training use cases for now. This workaround is
    // needed because seed is of tensor type in StableHLO, but float in tt-metal
    // and actual conversion can't be done.
    auto seedAttr = rewriter.getUI32IntegerAttr(0);

    rewriter.replaceOpWithNewOp<ttir::RandOp>(srcOp, outputType, sizeAttr,
                                              lowAttr, highAttr, seedAttr);
    return success();
  }
};

// Shared helper: extract normalized_shape and epsilon from a DictionaryAttr,
// build the named attributes and operand segment sizes, and replace srcOp
// with ttir::RMSNormOp.
static LogicalResult convertToTTIRRMSNorm(mlir::Operation *srcOp,
                                          mlir::ValueRange operands,
                                          DictionaryAttr compositeAttrs,
                                          ConversionPatternRewriter &rewriter) {
  auto outputType = mlir::cast<RankedTensorType>(srcOp->getResult(0).getType());

  auto normalizedShapeAttr = compositeAttrs.get("normalized_shape");
  SmallVector<int64_t> normalizedShapeVec;

  if (auto denseAttr =
          mlir::dyn_cast<DenseIntElementsAttr>(normalizedShapeAttr)) {
    for (auto val : denseAttr.getValues<int64_t>()) {
      normalizedShapeVec.push_back(val);
    }
  } else if (auto arrayAttr = mlir::dyn_cast<ArrayAttr>(normalizedShapeAttr)) {
    for (auto attr : arrayAttr) {
      normalizedShapeVec.push_back(mlir::cast<IntegerAttr>(attr).getInt());
    }
  } else {
    return rewriter.notifyMatchFailure(
        srcOp, "normalized_shape must be a dense tensor or array attribute");
  }

  auto normalizedShapeDenseAttr =
      rewriter.getDenseI64ArrayAttr(normalizedShapeVec);

  auto epsilonAttr = compositeAttrs.get("epsilon");

  SmallVector<NamedAttribute> namedAttrs;
  namedAttrs.push_back(
      rewriter.getNamedAttr("normalized_shape", normalizedShapeDenseAttr));
  if (epsilonAttr) {
    namedAttrs.push_back(rewriter.getNamedAttr("epsilon", epsilonAttr));
  }

  // ttir.rms_norm has AttrSizedOperandSegments: [input, weight, bias]
  size_t numOperands = operands.size();
  SmallVector<int32_t> segmentSizes;
  if (numOperands == 3) { // input, weight, bias
    segmentSizes = {1, 1, 1};
  } else if (numOperands == 2) { // input, weight
    segmentSizes = {1, 1, 0};
  } else { // input
    segmentSizes = {1, 0, 0};
  }

  namedAttrs.push_back(rewriter.getNamedAttr(
      "operandSegmentSizes", rewriter.getDenseI32ArrayAttr(segmentSizes)));

  rewriter.replaceOpWithNewOp<ttir::RMSNormOp>(srcOp, outputType, operands,
                                               namedAttrs);
  return success();
}

// Special handling for tenstorrent.moe_gpt_decode -> ttir.moe_gpt_decode.
//
// The composite carries 11 operands (5 routing/activation tensors, 4 unfused
// gate_up/down expert weights, and the 2 mandatory fused 6D kernel weights).
// ttir.moe_gpt_decode is fused-only, so this dedicated pattern selects the 7
// fused-only operands (dropping the unfused weights) and strips the legacy
// `has_fused_weights` attribute the op does not declare.
class TenstorrentMoeGPTDecodeConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CompositeOp> {
public:
  TenstorrentMoeGPTDecodeConversionPattern(MLIRContext *context)
      : OpConversionPattern<mlir::stablehlo::CompositeOp>(context) {}

  LogicalResult
  matchAndRewrite(mlir::stablehlo::CompositeOp srcOp,
                  mlir::stablehlo::CompositeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (srcOp.getName() != "tenstorrent.moe_gpt_decode") {
      return failure();
    }
    if (srcOp.getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp,
          "tenstorrent.moe_gpt_decode composite must have exactly one result.");
    }

    // The frontend composite carries 11 operands so the flatten -> shard ->
    // re-outline path keeps a contiguous group (operands 0-4
    // routing/activation, 5-8 the unfused gate_up/down expert weights, 9-10 the
    // fused 6D kernel weights). ttir.moe_gpt_decode is fused-only (7 operands),
    // so we drop the 4 unfused weights here and let them DCE: the kernel
    // computes on the fused weights and the decomposition derives the
    // per-device expert count from fused_w0_w1 dim 2.
    ValueRange operands = adaptor.getOperands();
    size_t numOperands = operands.size();
    if (numOperands != 11) {
      return rewriter.notifyMatchFailure(
          srcOp, "tenstorrent.moe_gpt_decode composite expects 11 operands "
                 "(5 routing/activation + 4 unfused expert weights + "
                 "fused_w0_w1 + fused_w2).");
    }
    SmallVector<Value> fusedOnlyOperands = {
        operands[0], operands[1], operands[2], operands[3],
        operands[4], operands[9], operands[10]};

    auto outputType =
        mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

    // Copy the composite attributes (num_devices, cluster_axis, num_experts,
    // num_experts_per_tok, intermediate_size, alpha, limit) to the TTIR op.
    // Drop the legacy `has_fused_weights` marker if present: the fused weights
    // are now mandatory operands, so it carries no information and the op does
    // not declare it.
    SmallVector<NamedAttribute> namedAttrs;
    if (auto compositeAttrs = srcOp.getCompositeAttributes()) {
      for (const auto &attr : compositeAttrs) {
        if (attr.getName() == "has_fused_weights") {
          continue;
        }
        namedAttrs.push_back(attr);
      }
    }

    rewriter.replaceOpWithNewOp<ttir::MoeGPTDecodeOp>(
        srcOp, outputType, fusedOnlyOperands, namedAttrs);
    return success();
  }
};

// Converts stablehlo.composite @tenstorrent.rms_norm -> ttir.rms_norm.
// Used in the non-sharded path where composites are not converted to
// custom_calls.
class TenstorrentRMSNormConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CompositeOp> {

public:
  TenstorrentRMSNormConversionPattern(MLIRContext *context)
      : OpConversionPattern<mlir::stablehlo::CompositeOp>(context) {}

  LogicalResult
  matchAndRewrite(mlir::stablehlo::CompositeOp srcOp,
                  mlir::stablehlo::CompositeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (srcOp.getName() != "tenstorrent.rms_norm") {
      return failure();
    }
    if (srcOp.getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp, "CompositeOp must have exactly one result.");
    }
    return convertToTTIRRMSNorm(srcOp, adaptor.getOperands(),
                                srcOp.getCompositeAttributes(), rewriter);
  }
};

// Converts stablehlo.custom_call @tenstorrent.rms_norm -> ttir.rms_norm.
// Used in the sharded path where composites with custom sharding rules
// were converted to custom_calls by FlattenOrConvertCompositesPass.
class CustomCallRMSNormConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CustomCallOp> {

public:
  CustomCallRMSNormConversionPattern(MLIRContext *context)
      : OpConversionPattern<mlir::stablehlo::CustomCallOp>(context) {}

  LogicalResult
  matchAndRewrite(mlir::stablehlo::CustomCallOp srcOp,
                  mlir::stablehlo::CustomCallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getCallTargetNameAttr() != kTTRMSNormCustomCallTargetName ||
        !srcOp->hasAttr(kHasCustomShardingAttr)) {
      return failure();
    }
    if (srcOp.getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp, "CustomCallOp must have exactly one result.");
    }
    auto compositeAttrs = mlir::dyn_cast_or_null<DictionaryAttr>(
        srcOp->getDiscardableAttr(kCustomCallCompositeAttrsKey));
    if (!compositeAttrs) {
      return rewriter.notifyMatchFailure(
          srcOp, "missing attributes on converted custom_call op");
    }
    return convertToTTIRRMSNorm(srcOp, adaptor.getOperands(), compositeAttrs,
                                rewriter);
  }
};

// Converts stablehlo.custom_call @tenstorrent.distributed_rms_norm ->
// ttir.distributed_rms_norm.
// This handles custom_calls created by FuseDistributedCustomCallsPass when
// fusing all_gather + rms_norm + all_slice into a distributed variant.
// Attributes (cluster_axis, epsilon) are metadata read from an attribute.
// Operands: input, optional weight.
class CustomCallDistributedRMSNormConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CustomCallOp> {

public:
  CustomCallDistributedRMSNormConversionPattern(MLIRContext *context)
      : OpConversionPattern<mlir::stablehlo::CustomCallOp>(context) {}

  LogicalResult
  matchAndRewrite(mlir::stablehlo::CustomCallOp srcOp,
                  mlir::stablehlo::CustomCallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getCallTargetNameAttr() !=
        mlir::tt::stablehlo::utils::kDistributedRmsNormTargetName) {
      return failure();
    }

    if (srcOp.getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp, "CustomCallOp must have exactly one result.");
    }

    auto outputType =
        mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

    auto compositeAttrs =
        mlir::dyn_cast_or_null<DictionaryAttr>(srcOp->getDiscardableAttr(
            mlir::tt::stablehlo::utils::kCustomCallCompositeAttrsKey));
    if (!compositeAttrs) {
      return rewriter.notifyMatchFailure(
          srcOp, "missing attributes on converted custom_call op");
    }

    // Extract cluster_axis attribute and convert to UI32.
    auto clusterAxisAttr = compositeAttrs.get("cluster_axis");
    if (!clusterAxisAttr) {
      return rewriter.notifyMatchFailure(
          srcOp, "distributed_rms_norm requires cluster_axis attribute");
    }
    auto intAttr = mlir::dyn_cast<IntegerAttr>(clusterAxisAttr);
    if (!intAttr) {
      return rewriter.notifyMatchFailure(
          srcOp, "cluster_axis must be an integer attribute");
    }
    auto ui32ClusterAxisAttr =
        rewriter.getUI32IntegerAttr(static_cast<uint32_t>(intAttr.getInt()));

    SmallVector<NamedAttribute> namedAttrs;
    namedAttrs.push_back(
        rewriter.getNamedAttr("cluster_axis", ui32ClusterAxisAttr));

    auto epsilonAttr = compositeAttrs.get("epsilon");
    if (epsilonAttr) {
      namedAttrs.push_back(rewriter.getNamedAttr("epsilon", epsilonAttr));
    }

    // ttir.distributed_rms_norm has AttrSizedOperandSegments:
    //   [input, weight, residual]
    size_t numOperands = adaptor.getOperands().size();
    SmallVector<int32_t> segmentSizes;
    if (numOperands == 3) { // input, weight, residual
      segmentSizes = {1, 1, 1};
    } else if (numOperands == 2) { // input, weight
      segmentSizes = {1, 1, 0};
    } else { // input only
      segmentSizes = {1, 0, 0};
    }

    namedAttrs.push_back(rewriter.getNamedAttr(
        "operandSegmentSizes", rewriter.getDenseI32ArrayAttr(segmentSizes)));

    rewriter.replaceOpWithNewOp<ttir::DistributedRMSNormOp>(
        srcOp, outputType, adaptor.getOperands(), namedAttrs);
    return success();
  }
};

// Special handling for tenstorrent.layer_norm -> ttir.layer_norm
// Converts normalized_shape tensor attribute to DenseI64ArrayAttr
// and sets operandSegmentSizes for AttrSizedOperandSegments
class TenstorrentLayerNormConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CompositeOp> {

public:
  TenstorrentLayerNormConversionPattern(MLIRContext *context)
      : OpConversionPattern<mlir::stablehlo::CompositeOp>(context) {}

  LogicalResult
  matchAndRewrite(mlir::stablehlo::CompositeOp srcOp,
                  mlir::stablehlo::CompositeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (srcOp.getName() != "tenstorrent.layer_norm") {
      return failure();
    }

    if (srcOp.getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp, "CompositeOp must have exactly one result.");
    }

    auto outputType =
        mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

    DictionaryAttr compositeAttrs = srcOp.getCompositeAttributes();

    auto normalizedShapeAttr = compositeAttrs.get("normalized_shape");
    SmallVector<int64_t> normalizedShapeVec;

    if (auto denseAttr =
            mlir::dyn_cast<DenseIntElementsAttr>(normalizedShapeAttr)) {
      for (auto val : denseAttr.getValues<int64_t>()) {
        normalizedShapeVec.push_back(val);
      }
    } else if (auto arrayAttr =
                   mlir::dyn_cast<ArrayAttr>(normalizedShapeAttr)) {
      for (auto attr : arrayAttr) {
        normalizedShapeVec.push_back(mlir::cast<IntegerAttr>(attr).getInt());
      }
    } else {
      return rewriter.notifyMatchFailure(
          srcOp, "normalized_shape must be a dense tensor or array attribute");
    }

    auto normalizedShapeDenseAttr =
        rewriter.getDenseI64ArrayAttr(normalizedShapeVec);

    auto epsilonAttr = compositeAttrs.get("epsilon");

    SmallVector<NamedAttribute> namedAttrs;
    namedAttrs.push_back(
        rewriter.getNamedAttr("normalized_shape", normalizedShapeDenseAttr));
    if (epsilonAttr) {
      namedAttrs.push_back(rewriter.getNamedAttr("epsilon", epsilonAttr));
    }

    // ttir.layer_norm has AttrSizedOperandSegments: [input, weight, bias]
    size_t numOperands = adaptor.getOperands().size();
    SmallVector<int32_t> segmentSizes;
    if (numOperands == 3) { // input, weight, bias
      segmentSizes = {1, 1, 1};
    } else if (numOperands == 2) { // input, weight
      segmentSizes = {1, 1, 0};
    } else { // input only
      segmentSizes = {1, 0, 0};
    }

    namedAttrs.push_back(rewriter.getNamedAttr(
        "operandSegmentSizes", rewriter.getDenseI32ArrayAttr(segmentSizes)));

    rewriter.replaceOpWithNewOp<ttir::LayerNormOp>(
        srcOp, outputType, adaptor.getOperands(), namedAttrs);
    return success();
  }
};

class ShardyAllSliceToTTIRMeshPartitionConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CompositeOp> {
public:
  ShardyAllSliceToTTIRMeshPartitionConversionPattern(MLIRContext *context)
      : OpConversionPattern<mlir::stablehlo::CompositeOp>(context) {}

  LogicalResult
  matchAndRewrite(mlir::stablehlo::CompositeOp srcOp,
                  mlir::stablehlo::CompositeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (srcOp.getName() != "sdy.all_slice") {
      return failure();
    }

    if (srcOp->getNumOperands() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp,
          "sdy.all_slice composite op must have exactly one input operand");
    }

    DictionaryAttr compositeAttrs = srcOp.getCompositeAttributes();
    auto maybeOutShardingAttr = compositeAttrs.get("out_sharding");
    if (!maybeOutShardingAttr) {
      return rewriter.notifyMatchFailure(srcOp,
                                         "out_sharding attribute is required");
    }
    // Extract the out_sharding attribute
    mlir::ModuleOp moduleOp = srcOp->getParentOfType<mlir::ModuleOp>();
    mlir::sdy::MeshOp globalMeshOp = shardy_utils::getMeshOps(moduleOp)[0];
    mlir::sdy::TensorShardingAttr outShardingAttr =
        mlir::cast<mlir::sdy::TensorShardingAttr>(maybeOutShardingAttr);

    // Calculate the attributes for the ttir.mesh_shard op.
    llvm::Expected<mlir::tt::shardy_utils::ShardyMeshSharding>
        shardyMeshSharding =
            mlir::tt::shardy_utils::ShardyMeshSharding::generate(
                globalMeshOp.getMeshAttr(), outShardingAttr,
                mlir::tt::ttcore::ShardStatus::Unsharded,
                ttcore::MeshShardDirection::FullToShard);
    if (auto err = shardyMeshSharding.takeError()) {
      return rewriter.notifyMatchFailure(
          srcOp, "Error trying to parse shardy annotation when legalizing "
                 "sdy.all_slice composite op.");
    }
    auto shardDims = shardyMeshSharding->getShardDims();
    llvm::SmallVector<int32_t> tensorDims;
    llvm::SmallVector<uint32_t> clusterAxes;
    for (auto [dimIdx, dim] : llvm::enumerate(shardDims)) {
      if (dim >= 0) {
        tensorDims.push_back(static_cast<int32_t>(dim));
        clusterAxes.push_back(static_cast<uint32_t>(dimIdx));
      }
    }
    rewriter.setInsertionPoint(srcOp);

    mlir::Value currInput = adaptor.getOperands().front();
    auto meshShape = shardyMeshSharding->getMeshShape();
    // Replace the composite op with 1 or more ttir.mesh_partition ops.
    for (size_t i = 0; i < tensorDims.size(); ++i) {
      auto currInputType = mlir::cast<RankedTensorType>(currInput.getType());
      llvm::SmallVector<int64_t> newShape(currInputType.getShape().begin(),
                                          currInputType.getShape().end());
      if (static_cast<size_t>(tensorDims[i]) >= newShape.size()) {
        return rewriter.notifyMatchFailure(
            srcOp, "Invalid mesh partition dimension index.");
      }
      if (static_cast<size_t>(clusterAxes[i]) >= meshShape.size()) {
        return rewriter.notifyMatchFailure(srcOp, "Invalid mesh axis index.");
      }
      // Compute new shape for the result tensor, with original dimension
      // divided by mesh axis size
      int64_t meshAxisSize = meshShape[clusterAxes[i]];
      if (newShape[tensorDims[i]] == ShapedType::kDynamic ||
          newShape[tensorDims[i]] % meshAxisSize != 0) {
        return rewriter.notifyMatchFailure(
            srcOp,
            "Dimension size must be static and divisible by mesh axis size.");
      }
      newShape[tensorDims[i]] = newShape[tensorDims[i]] / meshAxisSize;
      auto resultType =
          mlir::RankedTensorType::get(newShape, currInputType.getElementType(),
                                      currInputType.getEncoding());
      currInput = rewriter.create<ttir::MeshPartitionOp>(
          srcOp->getLoc(), resultType, currInput,
          rewriter.getSI32IntegerAttr(tensorDims[i]),
          rewriter.getUI32IntegerAttr(clusterAxes[i]));
    }
    rewriter.replaceOp(srcOp, currInput);
    return success();
  }
};

// Special handling for tenstorrent.group_norm -> ttir.group_norm
// Extracts num_groups (int) and epsilon from composite attributes
// and sets operandSegmentSizes for AttrSizedOperandSegments
class TenstorrentGroupNormConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CompositeOp> {

public:
  TenstorrentGroupNormConversionPattern(MLIRContext *context)
      : OpConversionPattern<mlir::stablehlo::CompositeOp>(context) {}

  LogicalResult
  matchAndRewrite(mlir::stablehlo::CompositeOp srcOp,
                  mlir::stablehlo::CompositeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (srcOp.getName() != "tenstorrent.group_norm") {
      return failure();
    }

    if (srcOp.getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp, "CompositeOp must have exactly one result.");
    }

    auto outputType =
        mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

    DictionaryAttr compositeAttrs = srcOp.getCompositeAttributes();

    // Extract num_groups as a scalar integer attribute.
    auto numGroupsAttr = compositeAttrs.get("num_groups");
    IntegerAttr numGroupsIntAttr;
    if (auto intAttr = mlir::dyn_cast_or_null<IntegerAttr>(numGroupsAttr)) {
      numGroupsIntAttr = rewriter.getI64IntegerAttr(intAttr.getInt());
    } else if (auto denseAttr = mlir::dyn_cast_or_null<DenseIntElementsAttr>(
                   numGroupsAttr)) {
      // Handle case where num_groups comes as a single-element dense tensor.
      assert(denseAttr.getNumElements() == 1 &&
             "Expected num_groups to be a single-element dense tensor");
      numGroupsIntAttr =
          rewriter.getI64IntegerAttr((*denseAttr.getValues<int64_t>().begin()));
    } else {
      return rewriter.notifyMatchFailure(
          srcOp, "num_groups must be an integer attribute");
    }

    auto epsilonAttr = compositeAttrs.get("epsilon");

    auto channelDimAttr = compositeAttrs.get("channel_dim");
    assert(channelDimAttr && "channel_dim must be present");

    SmallVector<NamedAttribute> namedAttrs;
    namedAttrs.push_back(rewriter.getNamedAttr("num_groups", numGroupsIntAttr));
    if (epsilonAttr) {
      namedAttrs.push_back(rewriter.getNamedAttr("epsilon", epsilonAttr));
    }
    if (channelDimAttr) {
      namedAttrs.push_back(
          rewriter.getNamedAttr("channel_dim", channelDimAttr));
    }

    // ttir.group_norm has AttrSizedOperandSegments:
    //   [input, input_mask, weight, bias]
    // From StableHLO composite, we expect operands in order:
    //   input, [weight], [bias] (no input_mask from frontend).
    size_t numOperands = adaptor.getOperands().size();
    SmallVector<int32_t> segmentSizes;
    if (numOperands == 3) { // input, weight, bias
      segmentSizes = {1, 0, 1, 1};
    } else if (numOperands == 2) { // input, weight
      segmentSizes = {1, 0, 1, 0};
    } else { // input only
      segmentSizes = {1, 0, 0, 0};
    }

    namedAttrs.push_back(rewriter.getNamedAttr(
        "operandSegmentSizes", rewriter.getDenseI32ArrayAttr(segmentSizes)));

    // ttir.group_norm requires rank >= 4; reshape rank<4 to 4D by appending
    // trailing unit dims (keeps channel_dim at the same index), then
    // reshape the result back to the original rank.
    Value input = adaptor.getOperands()[0];
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    int64_t inputRank = inputType.getRank();
    if (inputRank > 5) {
      return rewriter.notifyMatchFailure(srcOp, "input rank must be <= 5");
    }

    if (inputRank >= 4) {
      rewriter.replaceOpWithNewOp<ttir::GroupNormOp>(
          srcOp, outputType, adaptor.getOperands(), namedAttrs);
      return success();
    }

    SmallVector<int64_t> paddedShape(inputType.getShape().begin(),
                                     inputType.getShape().end());
    paddedShape.append(4 - inputRank, 1);
    auto paddedType =
        RankedTensorType::get(paddedShape, inputType.getElementType());

    SmallVector<Value> gnOperands(adaptor.getOperands().begin(),
                                  adaptor.getOperands().end());
    gnOperands[0] = rewriter.create<ttir::ReshapeOp>(
        srcOp.getLoc(), paddedType, input,
        rewriter.getI32ArrayAttr(llvm::to_vector_of<int32_t>(paddedShape)));

    Value gnResult = rewriter.create<ttir::GroupNormOp>(
        srcOp.getLoc(), paddedType, gnOperands, namedAttrs);

    rewriter.replaceOpWithNewOp<ttir::ReshapeOp>(
        srcOp, outputType, gnResult,
        rewriter.getI32ArrayAttr(
            llvm::to_vector_of<int32_t>(outputType.getShape())));
    return success();
  }
};

// Lowers `stablehlo.composite "tenstorrent.scaled_dot_product_attention"` to
// `ttir.scaled_dot_product_attention`. The composite is emitted by the
// frontend (tt-xla) wrapper around
// `torch.nn.functional.scaled_dot_product_attention`.
//
// Operand/attribute mapping:
//   - Operands: [query, key, value] or [query, key, value, attention_mask].
//     Shapes are passed through unchanged; the TTIR generic verifier enforces
//     [B, Hq, Sq, D] / [B, Hkv, Sk, D] / [1|B, 1|Hq, Sq, Sk] layouts.
//   - Attributes: only `is_causal` and `scale` are forwarded from the
//     composite. Other attributes that PyTorch SDPA may set
//     (e.g. `sliding_window_size`, `enable_gqa`) are not propagated.
//
// Policy decisions baked in below:
//   1. The 4th operand (`attention_mask`) is forwarded only when
//      `is_causal == false`, matching the kernel's mutual-exclusivity rule.
//   2. `attention_sink` is hard-coded to operand-segment size 0; the frontend
//      cannot supply it yet. Tracked by
//      https://github.com/tenstorrent/tt-xla/issues/4030.
// Shared lowering of scaled_dot_product_attention (composite or custom_call
// form) to ttir.scaled_dot_product_attention. Operands are [query, key, value]
// and optionally a 4th attention_mask; is_causal/scale come from the composite
// attribute dictionary.
static LogicalResult convertToTTIRScaledDotProductAttention(
    mlir::Operation *srcOp, mlir::ValueRange operands,
    DictionaryAttr compositeAttrs, RankedTensorType outputType,
    ConversionPatternRewriter &rewriter) {
  size_t numOperands = operands.size();
  if (numOperands < 3) {
    return rewriter.notifyMatchFailure(
        srcOp, "scaled_dot_product_attention must have at least 3 operands "
               "(query, key, value).");
  }

  // For now, frontend composite doesnt support attention_sink.
  // Issue: https://github.com/tenstorrent/tt-xla/issues/4030
  if (numOperands > 4) {
    return rewriter.notifyMatchFailure(
        srcOp, "scaled_dot_product_attention must have at most 4 operands "
               "(query, key, value, attention_mask). "
               "Attention sink is not supported yet.");
  }

  SmallVector<NamedAttribute> namedAttrs;

  bool isCausal = true;
  if (compositeAttrs) {
    if (auto attr = compositeAttrs.getAs<BoolAttr>("is_causal")) {
      isCausal = attr.getValue();
      namedAttrs.push_back(rewriter.getNamedAttr("is_causal", attr));
    }
    if (auto attr = compositeAttrs.getAs<FloatAttr>("scale")) {
      namedAttrs.push_back(rewriter.getNamedAttr(
          "scale", rewriter.getF32FloatAttr(
                       static_cast<float>(attr.getValueAsDouble()))));
    }
  }

  // The first 3 operands are always query, key, value. A 4th operand
  // (attention_mask) is present only when is_causal is false.
  bool hasAttnMask = !isCausal && numOperands == 4;
  SmallVector<Value> sdpaOperands = {operands[0], operands[1], operands[2]};
  if (hasAttnMask) {
    // Left-pad mask with unit dims to 4D — matches PyTorch broadcast semantics.
    Value mask = operands[3];
    auto maskType = mlir::cast<RankedTensorType>(mask.getType());
    int64_t maskRank = maskType.getRank();
    if (maskRank > 4) {
      return rewriter.notifyMatchFailure(srcOp,
                                         "attention_mask rank must be <= 4");
    }
    if (maskRank < 4) {
      SmallVector<int64_t> paddedShape(4 - maskRank, 1);
      paddedShape.append(maskType.getShape().begin(),
                         maskType.getShape().end());
      auto paddedType =
          RankedTensorType::get(paddedShape, maskType.getElementType());
      mask = rewriter.create<ttir::ReshapeOp>(
          srcOp->getLoc(), paddedType, mask,
          rewriter.getI32ArrayAttr(llvm::to_vector_of<int32_t>(paddedShape)));
    }
    sdpaOperands.push_back(mask);
  }

  // ttir.scaled_dot_product_attention has AttrSizedOperandSegments:
  //   [query, key, value, attention_mask, attention_sink]
  SmallVector<int32_t> segmentSizes = {1, 1, 1, hasAttnMask ? 1 : 0, 0};
  namedAttrs.push_back(rewriter.getNamedAttr(
      "operandSegmentSizes", rewriter.getDenseI32ArrayAttr(segmentSizes)));

  rewriter.replaceOpWithNewOp<ttir::ScaledDotProductAttentionOp>(
      srcOp, outputType, sdpaOperands, namedAttrs);
  return success();
}

class TenstorrentScaledDotProductAttentionConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CompositeOp> {

public:
  TenstorrentScaledDotProductAttentionConversionPattern(MLIRContext *context)
      : OpConversionPattern<mlir::stablehlo::CompositeOp>(context) {}

  LogicalResult
  matchAndRewrite(mlir::stablehlo::CompositeOp srcOp,
                  mlir::stablehlo::CompositeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (srcOp.getName() != "tenstorrent.scaled_dot_product_attention") {
      return failure();
    }

    if (srcOp.getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp, "CompositeOp must have exactly one result.");
    }

    auto outputType =
        mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

    return convertToTTIRScaledDotProductAttention(
        srcOp, adaptor.getOperands(), srcOp.getCompositeAttributes(),
        outputType, rewriter);
  }
};

// Converts the sharded form, stablehlo.custom_call
// @tenstorrent.scaled_dot_product_attention, to
// ttir.scaled_dot_product_attention. FlattenOrConvertCompositesPass converts
// the composite into this custom_call so Shardy can propagate the head
// sharding; this lowers it back, mirroring CustomCallRMSNormConversionPattern.
class CustomCallScaledDotProductAttentionConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CustomCallOp> {

public:
  CustomCallScaledDotProductAttentionConversionPattern(MLIRContext *context)
      : OpConversionPattern<mlir::stablehlo::CustomCallOp>(context) {}

  LogicalResult
  matchAndRewrite(mlir::stablehlo::CustomCallOp srcOp,
                  mlir::stablehlo::CustomCallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getCallTargetNameAttr() != kTTSDPACompositeName ||
        !srcOp->hasAttr(kHasCustomShardingAttr)) {
      return failure();
    }
    if (srcOp.getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp, "CustomCallOp must have exactly one result.");
    }
    auto compositeAttrs = mlir::dyn_cast_or_null<DictionaryAttr>(
        srcOp->getDiscardableAttr(kCustomCallCompositeAttrsKey));
    if (!compositeAttrs) {
      return rewriter.notifyMatchFailure(
          srcOp, "missing attributes on converted custom_call op");
    }
    auto outputType =
        mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());
    return convertToTTIRScaledDotProductAttention(
        srcOp, adaptor.getOperands(), compositeAttrs, outputType, rewriter);
  }
};

// Shared helper: builds a ttir.gather from an (input, index) pair, casting the
// index to UInt32 when needed.
static LogicalResult convertToTTIRGather(mlir::Operation *srcOp,
                                         mlir::Value input, mlir::Value index,
                                         RankedTensorType outputType,
                                         int32_t dim,
                                         ConversionPatternRewriter &rewriter) {
  // Cast the index tensor to UInt32 type if it isn't already UInt32 or UInt16.
  auto indexType = mlir::cast<RankedTensorType>(index.getType());
  if (!indexType.getElementType().isInteger()) {
    return rewriter.notifyMatchFailure(
        srcOp, "Index tensor must be of an integer type");
  }
  if (!indexType.getElementType().isUnsignedInteger(32) &&
      !indexType.getElementType().isUnsignedInteger(16)) {
    auto ui32Type = RankedTensorType::get(indexType.getShape(),
                                          rewriter.getIntegerType(32, false));
    index = rewriter.create<ttir::TypecastOp>(srcOp->getLoc(), ui32Type, index);
    indexType = mlir::cast<RankedTensorType>(index.getType());
  }

  auto inputType = mlir::cast<RankedTensorType>(input.getType());
  int64_t inputRank = inputType.getRank();
  int64_t indexRank = indexType.getRank();
  if (inputRank != indexRank) {
    return rewriter.notifyMatchFailure(
        srcOp, "Index and Input tensors must have same rank");
  }
  if (inputRank == 0) {
    return rewriter.notifyMatchFailure(
        srcOp, "0-rank tensors (scalars) are not supported.");
  }

  rewriter.replaceOpWithNewOp<ttir::GatherOp>(srcOp, outputType, input, index,
                                              rewriter.getI32IntegerAttr(dim));
  return success();
}

class TenstorrentGatherConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CompositeOp> {
public:
  TenstorrentGatherConversionPattern(MLIRContext *context)
      : OpConversionPattern<mlir::stablehlo::CompositeOp>(context) {}

  LogicalResult
  matchAndRewrite(mlir::stablehlo::CompositeOp srcOp,
                  mlir::stablehlo::CompositeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (srcOp.getName() != kTTGatherCustomCallTargetName &&
        srcOp.getName() != kTTGatherDimCustomCallTargetName) {
      return failure();
    }

    llvm::StringRef name = srcOp.getName();
    if (adaptor.getOperands().size() != 2) {
      return rewriter.notifyMatchFailure(
          srcOp, llvm::Twine(name) + " must have exactly 2 operands");
    }

    if (srcOp.getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp, llvm::Twine(name) + " must have exactly one result");
    }

    auto outputType =
        mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

    DictionaryAttr compositeAttrs = srcOp.getCompositeAttributes();
    auto dimIntAttr =
        compositeAttrs ? compositeAttrs.getAs<IntegerAttr>("dim") : nullptr;
    if (!dimIntAttr) {
      return rewriter.notifyMatchFailure(
          srcOp, llvm::Twine(name) + " requires integer 'dim' attribute");
    }
    int32_t dim = static_cast<int32_t>(dimIntAttr.getInt());

    return convertToTTIRGather(srcOp, adaptor.getOperands()[0],
                               adaptor.getOperands()[1], outputType, dim,
                               rewriter);
  }
};

// Converts stablehlo.custom_call @tenstorrent.gather_dim -> ttir.gather.
// This handles custom_calls created by FlattenOrConvertCompositesPass from the
// tenstorrent.gather composite (so Shardy can propagate shardings through it).
class CustomCallGatherConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CustomCallOp> {

public:
  CustomCallGatherConversionPattern(MLIRContext *context)
      : OpConversionPattern<mlir::stablehlo::CustomCallOp>(context) {}

  LogicalResult
  matchAndRewrite(mlir::stablehlo::CustomCallOp srcOp,
                  mlir::stablehlo::CustomCallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if ((adaptor.getCallTargetNameAttr() != kTTGatherCustomCallTargetName &&
         adaptor.getCallTargetNameAttr() != kTTGatherDimCustomCallTargetName) ||
        !srcOp->hasAttr(kHasCustomShardingAttr)) {
      return failure();
    }
    llvm::StringRef targetName = adaptor.getCallTargetNameAttr().getValue();
    if (adaptor.getOperands().size() != 2) {
      return rewriter.notifyMatchFailure(
          srcOp, llvm::Twine(targetName) + " must have exactly 2 operands");
    }
    if (srcOp.getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp, llvm::Twine(targetName) + " must have exactly one result");
    }

    auto outputType =
        mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

    auto compositeAttrs = mlir::dyn_cast_or_null<DictionaryAttr>(
        srcOp->getDiscardableAttr(kCustomCallCompositeAttrsKey));
    if (!compositeAttrs) {
      return rewriter.notifyMatchFailure(
          srcOp, "missing attributes on converted custom_call op");
    }
    auto dimIntAttr = compositeAttrs.getAs<IntegerAttr>("dim");
    if (!dimIntAttr) {
      return rewriter.notifyMatchFailure(
          srcOp, llvm::Twine(targetName) + " requires integer 'dim' attribute");
    }
    int32_t dim = static_cast<int32_t>(dimIntAttr.getInt());

    return convertToTTIRGather(srcOp, adaptor.getOperands()[0],
                               adaptor.getOperands()[1], outputType, dim,
                               rewriter);
  }
};

struct LegalizeStableHLOCompositeToTTIR
    : public ttir::impl::LegalizeStableHLOCompositeToTTIRBase<
          LegalizeStableHLOCompositeToTTIR> {
  void runOnOperation() final {
    MLIRContext *context = &getContext();

    ConversionTarget target(*context);
    target.addLegalDialect<ttir::TTIRDialect>();
    // StableHLO is intentionally not marked as either legal or illegal.

    RewritePatternSet patterns(context);
    populateStableHLOCompositeLegalizationPatterns(context, patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>>
createLegalizeStableHLOCompositeToTTIRPass() {
  return std::make_unique<LegalizeStableHLOCompositeToTTIR>();
}

void populateStableHLOCompositeLegalizationPatterns(
    MLIRContext *context, RewritePatternSet &patterns) {
  patterns.add<StableHLOToTTIRCompositeOpConversionPattern<ttir::GeluOp>>(
      context, "tenstorrent.gelu");
  patterns.add<StableHLOToTTIRCompositeOpConversionPattern<ttir::GeluOp>>(
      context, "tenstorrent.gelu_tanh");
  patterns.add<TenstorrentMoeGPTDecodeConversionPattern>(context);
  patterns.add<TenstorrentRMSNormConversionPattern>(context);
  patterns.add<CustomCallRMSNormConversionPattern>(context);
  patterns.add<CustomCallDistributedRMSNormConversionPattern>(context);
  patterns.add<TenstorrentLayerNormConversionPattern>(context);
  patterns.add<TenstorrentGroupNormConversionPattern>(context);
  patterns.add<TenstorrentUniformToRandConversionPattern>(context);
  patterns.add<TenstorrentTopKConversionPattern>(context);
  patterns.add<CustomCallTopKConversionPattern>(context);
  patterns.add<TenstorrentArgMaxConversionPattern>(context);
  patterns.add<CustomCallArgMaxConversionPattern>(context);
  patterns.add<TenstorrentScaledDotProductAttentionConversionPattern>(context);
  patterns.add<CustomCallScaledDotProductAttentionConversionPattern>(context);
  patterns.add<TenstorrentGatherConversionPattern>(context);
  patterns.add<CustomCallGatherConversionPattern>(context);
  patterns.add<ShardyAllSliceToTTIRMeshPartitionConversionPattern>(context);
}
} // namespace mlir::tt
