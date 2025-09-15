// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Transforms/ShardyCCLToStableHLOCCL.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"
#include "ttmlir/Utils.h"

#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::tt::stablehlo {

// Create a dense attribute from replica groups.
static mlir::DenseIntElementsAttr createDenseAttrFromReplicaGroups(
    MLIRContext *context,
    const llvm::SmallVector<llvm::SmallVector<int64_t>> &replicaGroups) {
  int64_t rows = replicaGroups.size();
  int64_t cols = replicaGroups.empty() ? 0 : replicaGroups[0].size();
  llvm::SmallVector<int64_t> flattenedValues(
      ::ttmlir::utils::flatten(replicaGroups));
  mlir::RankedTensorType type = mlir::RankedTensorType::get(
      {rows, cols}, mlir::IntegerType::get(context, 64));
  return mlir::DenseIntElementsAttr::get(type, flattenedValues);
}

// Generate a default replica groups 2D matrix. This is needed to figure out the
// devices that will participate in the collective operation with each other.
// For example, if the mesh shape = [2, 4], we generate [[0,1,2,3],[4,5,6,7]].
static llvm::SmallVector<llvm::SmallVector<int64_t>>
generateDefaultReplicaGroups(const llvm::SmallVector<int64_t> &meshShape) {
  assert(meshShape.size() == 2 && "Mesh shape must have 2 elements when trying "
                                  "to generate default replica groups\n.");
  llvm::SmallVector<llvm::SmallVector<int64_t>> replicaGroups(
      meshShape[0], llvm::SmallVector<int64_t>(meshShape[1]));
  int64_t value = 0;
  for (uint32_t i = 0; i < meshShape[0]; i++) {
    for (uint32_t j = 0; j < meshShape[1]; j++) {
      replicaGroups[i][j] = value;
      value++;
    }
  }

  return replicaGroups;
}

// Transpose a replica groups. This is needed in case we have a different
// cluster axis for which the collective operation is performing on than the
// default replica groups. For example, if the mesh shape = [2, 4], we generate
// a default replica groups as: [[0,1,2,3],[4,5,6,7]]. Then we transpose it
// [[0,4],[1,5],[2,6],[3,7]].
static llvm::SmallVector<llvm::SmallVector<int64_t>> transposeReplicaGroups(
    const llvm::SmallVector<llvm::SmallVector<int64_t>> &replicaGroups) {
  assert(!replicaGroups.empty() &&
         "Replica groups cannot be empty when trying to transpose it\n.");
  llvm::SmallVector<llvm::SmallVector<int64_t>> transposedReplicaGroups(
      replicaGroups[0].size(),
      llvm::SmallVector<int64_t>(replicaGroups.size()));
  for (uint32_t i = 0; i < replicaGroups.size(); i++) {
    for (uint32_t j = 0; j < replicaGroups[0].size(); j++) {
      transposedReplicaGroups[j][i] = replicaGroups[i][j];
    }
  }

  return transposedReplicaGroups;
}

// Get the correct replica groups for the given mesh and mesh axis.
static llvm::SmallVector<llvm::SmallVector<int64_t>>
populateReplicaGroups(mlir::tt::shardy_utils::MeshMap meshMap,
                      mlir::StringRef meshAxis) {
  assert(meshMap.size() == 2 && "Meshmap must have exactly 2 elements when "
                                "trying to populate replica groups\n.");
  auto it = meshMap.begin();
  llvm::SmallVector<int64_t> meshShape = {it->second, (it + 1)->second};
  llvm::SmallVector<llvm::SmallVector<int64_t>> replicaGroups =
      generateDefaultReplicaGroups(meshShape);

  if (it->first != meshAxis) {
    return replicaGroups;
  }

  return transposeReplicaGroups(replicaGroups);
}

template <typename SrcOp>
static mlir::tt::shardy_utils::MeshMap getMeshMap(SrcOp &srcOp) {
  // Get the mesh attr that relates to the sdy.all_gather.
  mlir::sdy::MeshAttr meshAttr;
  mlir::Attribute attr = srcOp.getOutSharding().getMeshOrRef();

  // Check if it's a symbol reference or a meshAttr.
  if (mlir::isa<mlir::sdy::MeshAttr>(attr)) {
    meshAttr = mlir::cast<mlir::sdy::MeshAttr>(attr);
  } else if (mlir::isa<mlir::SymbolRefAttr>(attr)) {
    mlir::SymbolRefAttr symbolRefAttr = mlir::cast<mlir::SymbolRefAttr>(attr);
    auto *symbolOp = mlir::SymbolTable::lookupSymbolIn(
        srcOp->template getParentOfType<ModuleOp>(), symbolRefAttr);
    mlir::sdy::MeshOp meshOp = mlir::cast<mlir::sdy::MeshOp>(symbolOp);
    meshAttr = meshOp.getMesh();
  }
  mlir::tt::shardy_utils::MeshMap meshMap =
      mlir::tt::shardy_utils::createMeshMapFromMeshAttr(meshAttr);

  return meshMap;
}

template <typename SrcOp, typename ReductionOp>
static void addReductionBlock(PatternRewriter &rewriter, SrcOp &srcOp,
                              mlir::RankedTensorType outputType) {
  mlir::Location loc = srcOp.getLoc();

  // Reduction type for stablehlo reduce scatter and all reduce requires an
  // empty tensor shape.
  mlir::RankedTensorType reductionType =
      mlir::RankedTensorType::get({}, outputType.getElementType());
  mlir::Block *block =
      rewriter.createBlock(&srcOp.getRegion(), /*insertPt*/ {},
                           {reductionType, reductionType}, {loc, loc});
  ReductionOp reductionOp = rewriter.create<ReductionOp>(
      loc, block->getArgument(0), block->getArgument(1));
  rewriter.create<mlir::stablehlo::ReturnOp>(loc, reductionOp.getResult());
}

// AllGatherOp
class ShardyToStableHLOAllGatherOpRewritePattern
    : public OpRewritePattern<mlir::sdy::AllGatherOp> {
  using OpRewritePattern::OpRewritePattern;

public:
  LogicalResult matchAndRewrite(mlir::sdy::AllGatherOp srcOp,
                                PatternRewriter &rewriter) const override {
    MLIRContext *context = getContext();

    // Set a default channel handle attr since we don't use it in tt-mlir stack
    // but stablehlo::AllGatherOp rewriter requires it.
    mlir::stablehlo::ChannelHandleAttr channelHandleAttr =
        mlir::stablehlo::ChannelHandleAttr::get(context, /*handle=*/1,
                                                /*type=*/1);

    // Iterate through gathering axes and create an all gather operation for
    // each axes that needs to be gathered.
    mlir::tt::shardy_utils::MeshMap meshMap =
        getMeshMap<mlir::sdy::AllGatherOp>(srcOp);

    Value result = srcOp.getOperand();
    for (auto [allGatherDim, axisRefListAttr] :
         llvm::enumerate(srcOp.getGatheringAxes())) {
      llvm::ArrayRef<mlir::sdy::AxisRefAttr> axisRefList =
          axisRefListAttr.getValue();

      // If the tensor dimension doesn't have any sharding on it, skip it.
      if (axisRefList.size() == 0) {
        continue;
      }

      // Currently we don't support multi-sharding on a single tensor dimension
      // across different mesh axes.
      if (axisRefList.size() > 1) {
        return rewriter.notifyMatchFailure(
            srcOp, "AllGatherOp does not support multi-sharding on a single "
                   "tensor dimension across different mesh axes.");
      }

      // Determine the output type based on the previous operand input and the
      // all gather dimension.
      mlir::StringRef meshAxis = axisRefList[0].getName();
      mlir::RankedTensorType prevOutputType =
          mlir::cast<mlir::RankedTensorType>(result.getType());
      llvm::SmallVector<int64_t> newShape =
          llvm::SmallVector<int64_t>(prevOutputType.getShape());
      newShape[allGatherDim] *= meshMap[meshAxis.str()];
      mlir::RankedTensorType newOutputType = mlir::RankedTensorType::get(
          newShape, prevOutputType.getElementType());

      // Create new chained all gather ops depending on the number of shardings.
      mlir::stablehlo::AllGatherOp allGatherOp =
          rewriter.create<mlir::stablehlo::AllGatherOp>(
              srcOp.getLoc(), newOutputType, result, allGatherDim,
              createDenseAttrFromReplicaGroups(
                  context, populateReplicaGroups(meshMap, meshAxis)),
              channelHandleAttr);
      result = allGatherOp.getResult(0);
    }

    rewriter.replaceAllUsesWith(srcOp, result);
    srcOp->erase();
    return success();
  }
};

// ReduceScatterOp
class ShardyToStableHLOReduceScatterOpRewritePattern
    : public OpRewritePattern<mlir::sdy::ReduceScatterOp> {
  using OpRewritePattern::OpRewritePattern;

public:
  LogicalResult matchAndRewrite(mlir::sdy::ReduceScatterOp srcOp,
                                PatternRewriter &rewriter) const override {
    MLIRContext *context = getContext();

    // Set a default channel handle attr since we don't use it in tt-mlir stack
    // but stablehlo::ReduceScatterOp rewriter requires it.
    mlir::stablehlo::ChannelHandleAttr channelHandleAttr =
        mlir::stablehlo::ChannelHandleAttr::get(context, /*handle*/ 1,
                                                /*type*/ 1);

    // Iterate through reduce scatter axes and create a reduce scatter operation
    // for each axes that needs to be reduced.
    mlir::tt::shardy_utils::MeshMap meshMap =
        getMeshMap<mlir::sdy::ReduceScatterOp>(srcOp);

    Value result = srcOp.getOperand();
    for (auto [reduceScatterDim, axisRefListAttr] :
         llvm::enumerate(srcOp.getReduceScatterAxes())) {
      llvm::ArrayRef<mlir::sdy::AxisRefAttr> axisRefList =
          axisRefListAttr.getValue();

      // If the tensor dimension doesn't have any sharding on it, skip it.
      if (axisRefList.size() == 0) {
        continue;
      }

      // Currently we don't support multi-sharding on a single tensor dimension
      // across different mesh axes.
      if (axisRefList.size() > 1) {
        return rewriter.notifyMatchFailure(
            srcOp,
            "ReduceScatterOp does not support multi-sharding on a single "
            "tensor dimension across different mesh axes.");
      }

      // Determine the output type based on the previous operand input and the
      // reduce scatter dimension.
      mlir::StringRef meshAxis = axisRefList[0].getName();
      mlir::RankedTensorType prevOutputType =
          mlir::cast<mlir::RankedTensorType>(result.getType());
      llvm::SmallVector<int64_t> newShape =
          llvm::SmallVector<int64_t>(prevOutputType.getShape());
      newShape[reduceScatterDim] /= meshMap[meshAxis.str()];
      mlir::RankedTensorType newOutputType = mlir::RankedTensorType::get(
          newShape, prevOutputType.getElementType());

      // Create new chained reduce scatter ops depending on the number of
      // shardings.
      mlir::stablehlo::ReduceScatterOp reduceScatterOp =
          rewriter.create<mlir::stablehlo::ReduceScatterOp>(
              srcOp.getLoc(), newOutputType, result, reduceScatterDim,
              createDenseAttrFromReplicaGroups(
                  context, populateReplicaGroups(meshMap, meshAxis)),
              channelHandleAttr);
      result = reduceScatterOp.getResult();

      // Create a single block because stablehlo.reduce_scatter op is a region
      // based op. Default the reduction type to sum since shardy does not have
      // support for custom reduction types.
      addReductionBlock<mlir::stablehlo::ReduceScatterOp,
                        mlir::stablehlo::AddOp>(rewriter, reduceScatterOp,
                                                newOutputType);
    }

    rewriter.replaceAllUsesWith(srcOp, result);
    srcOp->erase();
    return success();
  }
};

// AllReduceOp
class ShardyToStableHLOAllReduceOpRewritePattern
    : public OpRewritePattern<mlir::sdy::AllReduceOp> {
  using OpRewritePattern::OpRewritePattern;

public:
  LogicalResult matchAndRewrite(mlir::sdy::AllReduceOp srcOp,
                                PatternRewriter &rewriter) const override {
    MLIRContext *context = getContext();

    // Set a default channel handle attr since we don't use it in tt-mlir stack
    // but stablehlo::AllReduceOp rewriter requires it.
    mlir::stablehlo::ChannelHandleAttr channelHandleAttr =
        mlir::stablehlo::ChannelHandleAttr::get(context, /*handle*/ 1,
                                                /*type*/ 1);

    // Iterate through reduction axes and create a all reduce operation for
    // each axes that needs to be reduced.
    mlir::tt::shardy_utils::MeshMap meshMap =
        getMeshMap<mlir::sdy::AllReduceOp>(srcOp);
    Value result = srcOp.getOperand();

    for (auto reductionAxis : srcOp.getReductionAxes()) {
      // Create new chained all reduce ops depending on the number of shardings.
      // The shape of the all reduce will not change.
      mlir::StringRef meshAxis = reductionAxis.getName();
      mlir::RankedTensorType newOutputType =
          mlir::cast<mlir::RankedTensorType>(result.getType());
      mlir::stablehlo::AllReduceOp allReduceOp =
          rewriter.create<mlir::stablehlo::AllReduceOp>(
              srcOp.getLoc(), newOutputType, result,
              createDenseAttrFromReplicaGroups(
                  context, populateReplicaGroups(meshMap, meshAxis)),
              channelHandleAttr);
      result = allReduceOp.getResult(0);

      // Create a single block because stablehlo.all_reduce op is a region based
      // op. Default the reduction type to sum since shardy does not have
      // support for custom reduction types.
      addReductionBlock<mlir::stablehlo::AllReduceOp, mlir::stablehlo::AddOp>(
          rewriter, allReduceOp, newOutputType);
    }

    rewriter.replaceAllUsesWith(srcOp, result);
    srcOp->erase();
    return success();
  }
};

// AllToAllOp
class ShardyToStableHLOAllToAllOpRewritePattern
    : public OpRewritePattern<mlir::sdy::AllToAllOp> {
  using OpRewritePattern::OpRewritePattern;

public:
  LogicalResult matchAndRewrite(mlir::sdy::AllToAllOp srcOp,
                                PatternRewriter &rewriter) const override {
    MLIRContext *context = getContext();

    // Set a default channel handle attr since we don't use it in tt-mlir stack
    // but stablehlo::AllToAllOp rewriter requires it.
    mlir::stablehlo::ChannelHandleAttr channelHandleAttr =
        mlir::stablehlo::ChannelHandleAttr::get(context, /*handle*/ 1,
                                                /*type*/ 1);

    // Iterate through all all to all parameter attributes and insert a new all
    // to all op for each axis to perform the operation on.
    mlir::tt::shardy_utils::MeshMap meshMap =
        getMeshMap<mlir::sdy::AllToAllOp>(srcOp);
    Value result = srcOp.getOperand();

    for (mlir::sdy::AllToAllParamAttr allToAllParamAttr : srcOp.getParams()) {
      uint64_t sliceDim = allToAllParamAttr.getTgtDim();
      uint64_t concatDim = allToAllParamAttr.getSrcDim();
      llvm::ArrayRef<mlir::sdy::AxisRefAttr> axisRefList =
          allToAllParamAttr.getAxes();

      // If the tensor dimension doesn't have any sharding on it, skip it.
      if (axisRefList.size() == 0) {
        continue;
      }

      // Currently we don't support multi-sharding on a single tensor dimension
      // across different mesh axes.
      if (axisRefList.size() > 1) {
        return rewriter.notifyMatchFailure(
            srcOp, "AllToAllOp does not support multi-sharding on a single "
                   "tensor dimension across different mesh axes.");
      }

      // Create new chained all to all ops depending on the number of shardings.
      // Calculate new output type based on split and concat dims.
      mlir::StringRef meshAxis = axisRefList[0].getName();
      mlir::RankedTensorType prevOutputType =
          mlir::cast<mlir::RankedTensorType>(result.getType());
      llvm::SmallVector<int64_t> newShape =
          llvm::SmallVector<int64_t>(prevOutputType.getShape());
      newShape[sliceDim] /= meshMap[meshAxis.str()];
      newShape[concatDim] *= meshMap[meshAxis.str()];
      mlir::RankedTensorType newOutputType = mlir::RankedTensorType::get(
          newShape, prevOutputType.getElementType());
      mlir::stablehlo::AllToAllOp allToAllOp =
          rewriter.create<mlir::stablehlo::AllToAllOp>(
              srcOp.getLoc(), newOutputType, result, sliceDim, concatDim,
              meshMap[meshAxis.str()],
              createDenseAttrFromReplicaGroups(
                  context, populateReplicaGroups(meshMap, meshAxis)),
              channelHandleAttr);
      result = allToAllOp.getResult(0);
    }

    rewriter.replaceAllUsesWith(srcOp, result);
    srcOp->erase();
    return success();
  }
};

// AllSliceOp
class ShardyToStableHLOAllSliceOpRewritePattern
    : public OpRewritePattern<mlir::sdy::AllSliceOp> {
  using OpRewritePattern::OpRewritePattern;

public:
  LogicalResult
  matchAndRewrite(mlir::sdy::AllSliceOp srcOp,
                  mlir::PatternRewriter &rewriter) const override {
    MLIRContext *context = getContext();
    llvm::outs() << "[HET DEBUG] AllSliceOp rewrite called\n";

    // Mesh axis -> parts mapping.
    mlir::tt::shardy_utils::MeshMap meshMap =
        getMeshMap<mlir::sdy::AllSliceOp>(srcOp);

    // Channel handle used by AllToAll (opaque in this context).
    auto ch = mlir::stablehlo::ChannelHandleAttr::get(context, /*handle=*/1,
                                                      /*type=*/1);

    // Current tensor value/type threaded through the loop.
    mlir::Value result = srcOp.getOperand();
    auto prevType = mlir::cast<mlir::RankedTensorType>(result.getType());

    // Per-dimension slicing axes (we support at most one mesh axis per tensor
    // dim).
    auto axesPerDim = srcOp.getSlicingAxes();

    for (auto it = axesPerDim.begin(), e = axesPerDim.end(); it != e; ++it) {
      int64_t sliceDim =
          static_cast<int64_t>(std::distance(axesPerDim.begin(), it));
      llvm::ArrayRef<mlir::sdy::AxisRefAttr> axisRefs = it->getValue();

      // No sharding/slicing on this dimension → skip.
      if (axisRefs.empty()) {
        continue;
      }

      // We don't support multiple mesh axes mapped to a single tensor
      // dimension.
      if (axisRefs.size() > 1) {
        return rewriter.notifyMatchFailure(
            srcOp, "all_slice with multi-axis on a single tensor dimension is "
                   "not supported.");
      }

      // Resolve parts for the target mesh axis.
      mlir::StringRef meshAxis = axisRefs.front().getName();
      auto found = meshMap.find(meshAxis.str());
      if (found == meshMap.end()) {
        return rewriter.notifyMatchFailure(srcOp,
                                           "mesh axis not found in mesh map.");
      }
      int64_t parts = found->second;

      // Validate shape divisibility by parts and compute chunk length.
      llvm::SmallVector<int64_t> shape(prevType.getShape().begin(),
                                       prevType.getShape().end());
      if (shape[sliceDim] < 0 || shape[sliceDim] % parts != 0) {
        return rewriter.notifyMatchFailure(
            srcOp, "dimension size must be static and divisible by mesh axis "
                   "size (parts).");
      }
      int64_t fullLen = shape[sliceDim];
      int64_t chunkLen = fullLen / parts;

      // 1) Reshape: [..., N, ...] → [..., parts, chunkLen, ...]
      llvm::SmallVector<int64_t> reshapeShape;
      reshapeShape.reserve(shape.size() + 1);
      for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); ++i) {
        if (i == sliceDim) {
          reshapeShape.push_back(parts);    // inserted "parts" axis
          reshapeShape.push_back(chunkLen); // following "chunk" axis
        } else {
          reshapeShape.push_back(shape[i]);
        }
      }
      auto reshapedType =
          mlir::RankedTensorType::get(reshapeShape, prevType.getElementType());
      auto reshaped = rewriter.create<mlir::stablehlo::ReshapeOp>(
          srcOp.getLoc(), reshapedType, result);

      // 2) Replica groups for the target mesh axis.
      auto groups = createDenseAttrFromReplicaGroups(
          context, populateReplicaGroups(meshMap, meshAxis));

      // 3) AllToAll along the inserted "parts" axis (split and concat on same
      // axis).
      int64_t partsDim = sliceDim; // "parts" axis is inserted at sliceDim
      auto allToAll = rewriter.create<mlir::stablehlo::AllToAllOp>(
          srcOp.getLoc(),
          reshapedType, // result type is identical to input
          reshaped.getResult(),
          /*split_dimension=*/partsDim,
          /*concat_dimension=*/partsDim,
          /*split_count=*/parts,
          /*replica_groups=*/groups,
          /*channel_handle=*/ch);

      // 4) Static slice the "parts" axis to take index 0 only.
      llvm::SmallVector<int64_t> startIdx(reshapeShape.size(), 0);
      llvm::SmallVector<int64_t> limitIdx(reshapeShape.begin(),
                                          reshapeShape.end());
      llvm::SmallVector<int64_t> strides(reshapeShape.size(), 1);
      limitIdx[partsDim] =
          1; // take only the first entry along the "parts" axis

      auto startAttr = rewriter.getDenseI64ArrayAttr(startIdx);
      auto limitAttr = rewriter.getDenseI64ArrayAttr(limitIdx);
      auto stridesAttr = rewriter.getDenseI64ArrayAttr(strides);

      auto slicedShape = reshapeShape;
      slicedShape[partsDim] = 1;
      auto slicedType =
          mlir::RankedTensorType::get(slicedShape, prevType.getElementType());

      mlir::Value allToAllOut = allToAll.getResult(0);
      auto slice = rewriter.create<mlir::stablehlo::SliceOp>(
          srcOp.getLoc(), slicedType, allToAllOut, startAttr, limitAttr,
          stridesAttr);

      // 5) Remove the singleton "parts" axis → final shape with chunkLen at
      // sliceDim.
      shape[sliceDim] = chunkLen;
      auto finalType =
          mlir::RankedTensorType::get(shape, prevType.getElementType());
      auto squeezed = rewriter.create<mlir::stablehlo::ReshapeOp>(
          srcOp.getLoc(), finalType, slice.getResult());

      // Thread through for the next dimension (if any).
      result = squeezed.getResult();
      prevType = finalType;
    }

    rewriter.replaceOp(srcOp, result);
    return mlir::success();
  }
};

// CollectivePermuteOp
class ShardyToStableHLOCollectivePermuteOpRewritePattern
    : public OpRewritePattern<mlir::sdy::CollectivePermuteOp> {
  using OpRewritePattern::OpRewritePattern;

public:
  LogicalResult matchAndRewrite(mlir::sdy::CollectivePermuteOp srcOp,
                                PatternRewriter &rewriter) const override {
    srcOp.emitError() << "ShardyToStableHLO lowering for CollectivePermuteOp "
                         "is not implemented yet: "
                         "https://github.com/tenstorrent/tt-mlir/issues/3370.";
    return failure();
  }
};

void populateShardyCCLToStableHLOCCLPatterns(MLIRContext *ctx,
                                             RewritePatternSet &patterns) {
  patterns.add<ShardyToStableHLOAllGatherOpRewritePattern>(ctx);
  patterns.add<ShardyToStableHLOAllReduceOpRewritePattern>(ctx);
  patterns.add<ShardyToStableHLOReduceScatterOpRewritePattern>(ctx);
  patterns.add<ShardyToStableHLOAllToAllOpRewritePattern>(ctx);
  patterns.add<ShardyToStableHLOAllSliceOpRewritePattern>(ctx);
  patterns.add<ShardyToStableHLOCollectivePermuteOpRewritePattern>(ctx);
}

} // namespace mlir::tt::stablehlo
