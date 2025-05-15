// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Transforms/ShardyCCLToStableHLOCCL.h"
#include "ttmlir/Dialect/StableHLO/Transforms/ShardyUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include <iostream>

namespace mlir::tt::stablehlo {

// Create a dense attribute from replica groups.
static inline mlir::DenseIntElementsAttr createDenseAttrFromReplicaGroups(
    MLIRContext *context,
    const llvm::SmallVector<llvm::SmallVector<int64_t>> &replicaGroups) {
  int64_t rows = replicaGroups.size();
  int64_t cols = replicaGroups.empty() ? 0 : replicaGroups[0].size();
  llvm::SmallVector<int64_t> flattenedValues;
  flattenedValues.reserve(rows * cols);

  for (const auto &row : replicaGroups) {
    llvm::append_range(flattenedValues, row);
  }

  mlir::RankedTensorType type = mlir::RankedTensorType::get(
      {rows, cols}, mlir::IntegerType::get(context, 64));
  return mlir::DenseIntElementsAttr::get(type, flattenedValues);
}

/*
Generate a default replica groups 2D matrix. This is needed to figure out the
devices that will participate in the collective operation with each other. For
example, if the mesh shape = [2, 4], we generate 0 1 2 3 4 5 6 7
*/
static inline llvm::SmallVector<llvm::SmallVector<int64_t>>
generateDefaultReplicaGroups(const llvm::SmallVector<int64_t> &meshShape) {
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

/*
Transpose a replica groups. This is needed in case we have a different cluster
axis for which the collective operation is performing on than the default
replica groups. For example, if the mesh shape = [2, 4], we generate a default
replica groups as: 0 1 2 3 4 5 6 7

And then transpose it
0 4
1 5
2 6
3 7
*/
static inline llvm::SmallVector<llvm::SmallVector<int64_t>>
transposeReplicaGroups(
    const llvm::SmallVector<llvm::SmallVector<int64_t>> &replicaGroups) {
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
static inline llvm::SmallVector<llvm::SmallVector<int64_t>>
populateReplicaGroups(mlir::tt::sdy_utils::MeshMap meshMap,
                      mlir::StringRef meshAxis) {
  auto it = meshMap.begin();
  llvm::SmallVector<int64_t> meshShape = {it->second, (it + 1)->second};
  llvm::SmallVector<llvm::SmallVector<int64_t>> replicaGroups =
      generateDefaultReplicaGroups(meshShape);

  if (it->first != meshAxis) {
    return replicaGroups;
  }

  return transposeReplicaGroups(replicaGroups);
}

class ShardyToStableHLOAllGatherOpConversionPattern
    : public OpConversionPattern<mlir::sdy::AllGatherOp> {
  using OpConversionPattern<mlir::sdy::AllGatherOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::sdy::AllGatherOp srcOp,
                  mlir::sdy::AllGatherOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *context = getContext();
    mlir::OpBuilder builder(context);

    // Set a default channel handle attr since we don't use it in tt-mlir stack
    // but stablehlo::AllGatherOp builder requires it.
    mlir::stablehlo::ChannelHandleAttr channelHandleAttr =
        mlir::stablehlo::ChannelHandleAttr::get(context, /*handle*/ 1,
                                                /*type*/ 1);

    // Get the mesh attr that relates to the sdy.all_gather
    mlir::sdy::MeshAttr meshAttr;
    mlir::Attribute attr = srcOp.getOutSharding().getMeshOrRef();

    // Check if it's a symbol reference or a meshAttr
    if (mlir::isa<mlir::sdy::MeshAttr>(attr)) {
      meshAttr = mlir::dyn_cast<mlir::sdy::MeshAttr>(attr);
    } else if (mlir::isa<mlir::SymbolRefAttr>(attr)) {
      mlir::SymbolRefAttr symbolRefAttr =
          mlir::dyn_cast<mlir::SymbolRefAttr>(attr);
      auto symbolOp = mlir::SymbolTable::lookupSymbolIn(
          srcOp->getParentOfType<ModuleOp>(), symbolRefAttr);
      mlir::sdy::MeshOp meshOp = mlir::dyn_cast<mlir::sdy::MeshOp>(symbolOp);
      meshAttr = meshOp.getMesh();
    }
    mlir::tt::sdy_utils::MeshMap meshMap =
        mlir::tt::sdy_utils::createMeshMapFromMeshAttr(meshAttr);

    // Iterate through gathering axes and create an all gather operation for
    // each axes that needs to be gathered.
    mlir::Operation *prevOp = srcOp.getOperation();
    builder.setInsertionPoint(srcOp);

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

      // We want to apply the correct operand to the all gather op in case we
      // have to insert multiple all gathers. The first all gather will take the
      // operand of the sdy.all_gather it is replacing. All subsequent all
      // gathers will inherit the previous all gather.
      auto operand = prevOp == srcOp.getOperation() ? prevOp->getOperands()[0]
                                                    : prevOp->getResult(0);

      // Determine the output type based on the previous operand input and the
      // all gather dimension.
      mlir::StringRef meshAxis = axisRefList[0].getName();
      mlir::RankedTensorType prevOutputType =
          mlir::cast<mlir::RankedTensorType>(
              getTypeConverter()->convertType(operand.getType()));
      llvm::SmallVector<int64_t> newShape =
          llvm::SmallVector<int64_t>(prevOutputType.getShape());
      newShape[allGatherDim] *= meshMap[meshAxis];
      llvm::SmallVector<mlir::Type> newOutputTypes = {
          mlir::RankedTensorType::get(newShape,
                                      prevOutputType.getElementType())};

      // Create new all gather op and replace the previous op's uses with this
      // new op.
      mlir::stablehlo::AllGatherOp allGatherOp =
          rewriter.create<mlir::stablehlo::AllGatherOp>(
              srcOp.getLoc(), newOutputTypes, operand, allGatherDim,
              createDenseAttrFromReplicaGroups(
                  context, populateReplicaGroups(meshMap, meshAxis)),
              channelHandleAttr);
      builder.setInsertionPoint(allGatherOp);
      prevOp->replaceUsesWithIf(allGatherOp,
                                [&allGatherOp](mlir::OpOperand &use) {
                                  return use.getOwner() != allGatherOp;
                                });
      prevOp = allGatherOp.getOperation();
    }

    srcOp->erase();
    return success();
  }
};

void populateShardyCCLToStableHLOCCLPatterns(MLIRContext *ctx,
                                             RewritePatternSet &patterns,
                                             TypeConverter &typeConverter) {
  patterns.add<ShardyToStableHLOAllGatherOpConversionPattern>(typeConverter,
                                                              ctx);
}

} // namespace mlir::tt::stablehlo
