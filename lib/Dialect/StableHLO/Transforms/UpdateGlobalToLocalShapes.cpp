// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "ttmlir/Dialect/StableHLO/Transforms/ShardyCCLToStableHLOCCL.h"
#include "ttmlir/Dialect/StableHLO/Utils/GSPMDUtils.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_UPDATEGLOBALTOLOCALSHAPESPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

// Helper function to determine if a ScatterOp represents a safe sharded scatter
//  that doesn't require attr rewriting when running
//  UpdateGlobalToLocalShapesPass. This requires the scatter to have inputs that
//  are either unsharded, or jointly sharded along the same axis, which must
//  differ from the scatter axis:
// i.e. both insertedWindowDims and scatterDimsToOperandDims must be orthogonal
//  to the update/input sharding dims.
static bool isSafeShardedScatter(mlir::stablehlo::ScatterOp scatterOp,
                                 mlir::sdy::MeshOp &globalMeshOp) {
  mlir::stablehlo::ScatterDimensionNumbersAttr scatterDimensionNumbers =
      scatterOp.getScatterDimensionNumbers();
  llvm::ArrayRef<int64_t> insertedWindowDims =
      scatterDimensionNumbers.getInsertedWindowDims();
  llvm::ArrayRef<int64_t> scatterDimsToOperandDims =
      scatterDimensionNumbers.getScatterDimsToOperandDims();

  // Get sharding info for inputs.
  llvm::ArrayRef<mlir::sdy::DimensionShardingAttr> inputDimShardings =
      shardy_utils::getOperandShardingAttr(
          scatterOp.getOperation()->getOpOperand(0), globalMeshOp)
          .getDimShardings();

  // Get sharding info for updates.
  llvm::ArrayRef<mlir::sdy::DimensionShardingAttr> updateDimShardings =
      shardy_utils::getOperandShardingAttr(
          scatterOp.getOperation()->getOpOperand(2), globalMeshOp)
          .getDimShardings();

  // Early exit conditions - return false if we can't analyze safely.
  if (inputDimShardings.empty() || updateDimShardings.empty()) {
    return false;
  }

  mlir::OperandRange scatterInputs = scatterOp.getInputs();
  mlir::OperandRange scatterUpdates = scatterOp.getUpdates();

  // SHLO scatter accepts a variadic number of inputs / updates tensors, we
  // expect only 1.
  if (scatterInputs.size() != 1 || scatterUpdates.size() != 1) {
    return false;
  }

  // Expect the inputs and updates to have the same size.
  mlir::RankedTensorType scatterInputType =
      mlir::dyn_cast<mlir::RankedTensorType>(scatterInputs.front().getType());
  mlir::RankedTensorType scatterUpdateType =
      mlir::dyn_cast<mlir::RankedTensorType>(scatterUpdates.front().getType());
  if (scatterInputType.getShape().size() !=
      scatterUpdateType.getShape().size()) {
    return false;
  }

  // insertedWindowDims / scatterDimsToOperandDims must be a single dim and
  // equal for a cache update.
  if (insertedWindowDims.size() != 1 || scatterDimsToOperandDims.size() != 1 ||
      (insertedWindowDims.front() != scatterDimsToOperandDims.front())) {
    return false;
  }

  int64_t scatterAxis = insertedWindowDims.front();

  // Determine the "sharding axis"
  // input and updates must be sharded equivalently.
  if (!inputDimShardings.equals(updateDimShardings)) {
    return false;
  }

  llvm::SmallVector<uint8_t> shardingAxes = {};

  for (auto [dimIndex, sharding] : llvm::enumerate(inputDimShardings)) {
    if (!sharding.getAxes().empty()) {
      shardingAxes.push_back(dimIndex);
    }
  }

  // We expect the sharding axes and scatter axes to be disjoint.
  if (llvm::is_contained(shardingAxes, scatterAxis)) {
    return false;
  }

  return true;
}

// This function creates a new operation state with updated shapes and
// attributes based on the provided operation, global mesh, new types, and
// tensor shardings. It handles special cases for certain operations like
// `ConstantOp`, `SliceOp`, and `GatherOp` to ensure their attributes are
// correctly updated according to the new shapes and shardings.
static FailureOr<mlir::OperationState> createNewOperationState(
    MLIRContext *context, mlir::Operation *op, mlir::sdy::MeshOp &globalMeshOp,
    llvm::ArrayRef<mlir::RankedTensorType> newTypes,
    llvm::ArrayRef<mlir::sdy::TensorShardingAttr> tensorShardings) {
  mlir::OperationState state(op->getLoc(), op->getName());
  state.types.append(newTypes.begin(), newTypes.end());
  state.operands.append(op->operand_begin(), op->operand_end());
  mlir::DictionaryAttr attrDict = op->getAttrDictionary();
  llvm::SmallVector<mlir::NamedAttribute> namedAttrs(op->getAttrs());

  // Handle special operations that need their attribute dictionary updated.
  mlir::LogicalResult updatedAttributeResult =
      llvm::TypeSwitch<mlir::Operation *, mlir::LogicalResult>(op)
          .Case<mlir::stablehlo::ConstantOp>([&](auto constantOp) {
            llvm::StringRef valueAttrName = "value";
            assert(attrDict.contains(valueAttrName) &&
                   "Constant operation does not have a value attribute. "
                   "Ill-formed operation");

            mlir::DenseElementsAttr denseElementsAttr =
                mlir::cast<mlir::DenseElementsAttr>(
                    attrDict.get(valueAttrName));

            // If the element is not a splat value (ie. the same value
            // for the entire constant) we fail as this is currently
            // not supported.
            if (!denseElementsAttr.isSplat()) {
              constantOp->emitError(
                  "Shardy automatic parallelization currently does "
                  "not support non-splat constant tensors");
              return mlir::failure();
            }
            mlir::DenseElementsAttr newAttr = mlir::DenseElementsAttr::get(
                newTypes[0],
                denseElementsAttr.getSplatValue<mlir::Attribute>());
            auto namedAttrIt = llvm::find_if(
                namedAttrs, [&](const mlir::NamedAttribute &attr) {
                  return attr.getName() == valueAttrName;
                });
            namedAttrIt->setValue(newAttr);

            return mlir::success();
          })
          .Case<mlir::stablehlo::SliceOp>([&](auto sliceOp) {
            // 1. Get the sharding for each operand dimension.
            llvm::ArrayRef<mlir::sdy::DimensionShardingAttr>
                operandDimShardings =
                    shardy_utils::getOperandShardingAttr(
                        sliceOp.getOperation()->getOpOperand(0), globalMeshOp)
                        .getDimShardings();

            // 2. Copy the current start_indices and limit_indices attributes.
            llvm::SmallVector<int64_t> startIndices(sliceOp.getStartIndices());
            llvm::SmallVector<int64_t> limitIndices(sliceOp.getLimitIndices());

            // 3. Iterate through start and limit indices and update them based
            // on the sharding annotation for that dimension.
            for (auto [i, dimShardings] :
                 llvm::enumerate(operandDimShardings)) {
              llvm::ArrayRef<mlir::sdy::DimensionShardingAttr> shardings =
                  dimShardings;

              if (shardings.size() > 1) {
                sliceOp->emitError(
                    "Slice operation has multiple shardings on a single tensor "
                    "dimension. This is not supported.");
                return mlir::failure();
              }

              FailureOr<int64_t> updatedLimitDim =
                  shardy_utils::calculateUpdatedDim(
                      globalMeshOp.getMesh(), shardings[0], limitIndices[i]);

              if (failed(updatedLimitDim)) {
                sliceOp->emitError(
                    "Could not apply propagated tensor shardings "
                    "to attribute dictionary for slice op");
                return mlir::failure();
              }

              limitIndices[i] = *updatedLimitDim;
            }

            // 4. Update start and limit indices in op named attributes.
            llvm::StringRef startIndicesAttrName = "start_indices";
            llvm::StringRef limitIndicesAttrName = "limit_indices";

            assert(attrDict.contains(startIndicesAttrName) &&
                   "Slice operation does not have start indices attribute. "
                   "Ill-formed operation");
            assert(attrDict.contains(limitIndicesAttrName) &&
                   "Slice operation does not have limit indices attribute. "
                   "Ill-formed operation");

            auto namedAttrStartIt = llvm::find_if(
                namedAttrs, [&](const mlir::NamedAttribute &attr) {
                  return attr.getName() == startIndicesAttrName;
                });
            auto namedAttrLimitIt = llvm::find_if(
                namedAttrs, [&](const mlir::NamedAttribute &attr) {
                  return attr.getName() == limitIndicesAttrName;
                });
            namedAttrStartIt->setValue(
                mlir::DenseI64ArrayAttr::get(context, startIndices));
            namedAttrLimitIt->setValue(
                mlir::DenseI64ArrayAttr::get(context, limitIndices));

            return mlir::success();
          })
          .Case<mlir::stablehlo::GatherOp>([&](auto gatherOp) {
            // 1. Get the sharding for each operand dimension.
            llvm::ArrayRef<mlir::sdy::DimensionShardingAttr>
                operandDimShardings =
                    shardy_utils::getOperandShardingAttr(
                        gatherOp.getOperation()->getOpOperand(0), globalMeshOp)
                        .getDimShardings();

            // 2. Copy the current slice_sizes attribute.
            llvm::SmallVector<int64_t> newSliceSizes(gatherOp.getSliceSizes());

            // 3. Get collapsed_slice_dims and start_index_map from the op.
            auto collapsedSliceDims =
                gatherOp.getDimensionNumbers().getCollapsedSliceDims();
            auto startIndexMap =
                gatherOp.getDimensionNumbers().getStartIndexMap();

            // 4. For each dimension, update slice size if not in
            // start_index_map.
            for (auto [index, sliceSize] : llvm::enumerate(newSliceSizes)) {
              // If this dimension is collapsed, it must be 1.
              if (llvm::is_contained(collapsedSliceDims, index)) {
                if (sliceSize != 1) {
                  gatherOp->emitError("collapsed slice dims should be 1");
                  return mlir::failure();
                }
                continue;
              }

              // If this dim is not indexed, its slice size should match the
              // local (sharded) size of the operand. We use operandDimShardings
              // to get the local shape for this dim.
              if (!llvm::is_contained(startIndexMap, index)) {
                FailureOr<int64_t> updatedSliceDim =
                    shardy_utils::calculateUpdatedDim(
                        globalMeshOp.getMesh(), operandDimShardings[index],
                        newSliceSizes[index]);
                if (failed(updatedSliceDim)) {
                  gatherOp->emitError(
                      "Could not apply propagated tensor shardings to "
                      "attribute dictionary for gather op");
                  return mlir::failure();
                }
                newSliceSizes[index] = *updatedSliceDim;
              }
            }

            // 5. Update the slice_sizes attribute in the op's attribute
            // dictionary.
            llvm::StringRef sliceSizesAttrName = "slice_sizes";
            assert(attrDict.contains(sliceSizesAttrName) &&
                   "Gather operation does not have slice sizes attribute. "
                   "Ill-formed operation");
            auto namedAttrSliceSizesIt = llvm::find_if(
                namedAttrs, [&](const mlir::NamedAttribute &attr) {
                  return attr.getName() == sliceSizesAttrName;
                });
            namedAttrSliceSizesIt->setValue(
                mlir::DenseI64ArrayAttr::get(context, newSliceSizes));

            return mlir::success();
          })
          .Case<mlir::stablehlo::ScatterOp>([&](auto scatterOp) {
            // Check if this is a safe cache update that can be handled
            if (isSafeShardedScatter(scatterOp, globalMeshOp)) {
              return mlir::success();
            }

            // If not a safe cache update, emit the error as before
            scatterOp->emitError(
                "Scatter operation is not supported in stablehlo-pipeline for "
                "meshes not 1x1: "
                "https://github.com/tenstorrent/tt-mlir/issues/3496.");
            return mlir::failure();
          })
          .Default([](mlir::Operation *op) { return mlir::success(); });

  if (failed(updatedAttributeResult)) {
    op->emitError("Could not updated attribute dictionary for operation");
    return mlir::failure();
  }

  state.attributes = namedAttrs;
  state.regions.resize(op->getNumRegions());
  return state;
}

// Update the manual axes for each computation block and update the argument
// tensor shapes according to their tensor sharding annotation.
static mlir::LogicalResult updateManualAxes(MLIRContext *context,
                                            mlir::OpBuilder &builder,
                                            mlir::sdy::MeshOp &globalMeshOp,
                                            func::FuncOp &funcOp) {
  // Set the manual axes from meshOp since this pass currently only supports 1
  // mesh.
  llvm::SmallVector<mlir::StringAttr> manualAxes;
  for (auto meshAxisAttr : globalMeshOp.getMesh().getAxes()) {
    manualAxes.push_back(
        mlir::StringAttr::get(context, meshAxisAttr.getName()));
  }
  mlir::sdy::ManualAxesAttr manualAxesAttr =
      mlir::sdy::ManualAxesAttr::get(context, manualAxes);

  // Update the manual axes in the mlir module.
  mlir::WalkResult result = funcOp.getBody().walk([&](mlir::Operation *op) {
    if (!mlir::isa<mlir::sdy::ManualComputationOp>(op)) {
      return WalkResult::advance();
    }

    mlir::sdy::ManualComputationOp manualComputationOp =
        mlir::cast<mlir::sdy::ManualComputationOp>(op);
    manualComputationOp.setManualAxesAttr(manualAxesAttr);

    // Walk through each argument in the manual computation op and update the
    // shape based on it's in_sharding attribute.
    mlir::Block &entryBlock = manualComputationOp.getRegion().front();
    llvm::ArrayRef<mlir::sdy::TensorShardingAttr> tensorShardings =
        manualComputationOp.getInShardings().getShardings();

    uint32_t initialArgSize = entryBlock.getArguments().size();
    for (uint32_t i = 0; i < initialArgSize; i++) {
      mlir::BlockArgument arg = entryBlock.getArgument(i);
      mlir::RankedTensorType oldType =
          mlir::cast<mlir::RankedTensorType>(arg.getType());
      FailureOr<mlir::RankedTensorType> newType =
          shardy_utils::populateShardedOutputType(
              globalMeshOp.getMesh(), oldType,
              tensorShardings[arg.getArgNumber()]);

      if (failed(newType)) {
        manualComputationOp.emitError("Could not apply propagated tensor "
                                      "shardings to tensor dimensions");
        return WalkResult::interrupt();
      }

      mlir::Value newArg = entryBlock.addArgument(*newType, arg.getLoc());
      arg.replaceAllUsesWith(newArg);
    }

    // Remove all unused arguments
    entryBlock.eraseArguments(0, initialArgSize);
    return WalkResult::advance();
  });

  return mlir::failure(result.wasInterrupted());
}

// Update all shapes in the module based on their sdy tensor sharding attribute.
static mlir::LogicalResult updateShapes(MLIRContext *context,
                                        mlir::OpBuilder &builder,
                                        mlir::sdy::MeshOp &globalMeshOp,
                                        func::FuncOp &funcOp) {
  mlir::WalkResult result = funcOp.getBody().walk([&](mlir::Operation *op) {
    for (auto namedAttr : op->getAttrs()) {
      if (namedAttr.getName() != mlir::sdy::TensorShardingAttr::name) {
        continue;
      }

      // Get tensor sharding annotation for this op.
      mlir::sdy::TensorShardingPerValueAttr tensorShardingPerValueAttr =
          mlir::cast<mlir::sdy::TensorShardingPerValueAttr>(
              op->getAttr(mlir::sdy::TensorShardingAttr::name));
      llvm::ArrayRef<mlir::sdy::TensorShardingAttr> tensorShardings =
          tensorShardingPerValueAttr.getShardings();
      FailureOr<llvm::SmallVector<mlir::RankedTensorType>> newTypes =
          shardy_utils::getNewResultTypes(op, globalMeshOp, tensorShardings);

      if (failed(newTypes)) {
        op->emitError("Could not apply propagated tensor shardings to "
                      "tensor dimensions");
        return WalkResult::interrupt();
      }

      // Create new operation state to update the original operation with
      // it's new computed shapes.
      FailureOr<mlir::OperationState> state =
          mlir::tt::stablehlo::createNewOperationState(
              context, op, globalMeshOp, *newTypes, tensorShardings);

      if (failed(state)) {
        op->emitError("Could not create a new operation with updated shapes");
        return WalkResult::interrupt();
      }

      mlir::Operation *newOp = mlir::Operation::create(*state);

      // If the operation has nested regions, we need to remap and copy
      // them over (eg. stablehlo.reduce).
      shardy_utils::copyNestedRegions(builder, op, newOp);

      // Update all old op results with new op results.
      for (uint32_t i = 0; i < op->getNumResults(); i++) {
        op->getResult(i).replaceAllUsesWith(newOp->getResult(i));
      }
      builder.setInsertionPoint(op);
      builder.insert(newOp);
      op->erase();
    }

    return WalkResult::advance();
  });

  return result.wasInterrupted() ? mlir::failure() : mlir::success();

  return mlir::success();
}

// Convert all sdy ccl ops into stablehlo ccl ops.
static mlir::LogicalResult
convertShardyCCLToStableHLOCCL(MLIRContext *context,
                               mlir::ModuleOp &rootModule) {
  RewritePatternSet patterns(context);
  populateShardyCCLToStableHLOCCLPatterns(context, patterns);
  FrozenRewritePatternSet patternSet(std::move(patterns));

  // Apply patterns greedily.
  if (failed(applyPatternsGreedily(rootModule, patternSet))) {
    rootModule.emitError("Could not convert shardy ccl operations into "
                         "stablehlo ccl operations");
    return mlir::failure();
  }

  return mlir::success();
}

class UpdateGlobalToLocalShapesPass
    : public impl::UpdateGlobalToLocalShapesPassBase<
          UpdateGlobalToLocalShapesPass> {
public:
  using impl::UpdateGlobalToLocalShapesPassBase<
      UpdateGlobalToLocalShapesPass>::UpdateGlobalToLocalShapesPassBase;

  void runOnOperation() final {
    mlir::ModuleOp rootModule = getOperation();
    MLIRContext *context = rootModule.getContext();
    mlir::OpBuilder builder(context);

    // Check if the graph is already solved by shardy. If so, we will remove
    // all sdy tensor shardings from the arguments and results.
    if (shardy_utils::isGraphSolved(rootModule)) {
      // Run conversion pattern to convert all sdy ccl operations into stablehlo
      // ccl operations.
      if (failed(convertShardyCCLToStableHLOCCL(context, rootModule))) {
        rootModule.emitError(
            "Could not convert shardy ccl ops into stablehlo ccl ops.\n");
        signalPassFailure();
        return;
      }

      rootModule.walk([&](func::FuncOp funcOp) {
        shardy_utils::removeSdyTensorShardings(context, funcOp);
      });

      return;
    }

    // If the module has gspmd annotations, we skip this pass.
    bool gspmdAnnotationsExist = gspmd_utils::gspmdAnnotationsExist(rootModule);
    if (gspmdAnnotationsExist) {
      rootModule.emitWarning("GSPMD style graphs are already in local device "
                             "shapes so we don't need to update them");
      return;
    }

    // Get the shardy mesh op in the root module.
    mlir::sdy::MeshOp globalMeshOp;
    llvm::SmallVector<mlir::sdy::MeshOp> parsedMeshOps =
        shardy_utils::getMeshOps(rootModule);

    if (parsedMeshOps.size() == 0) {
      rootModule.emitError(
          "Pass requires a shardy mesh op to be present in the root module");
      signalPassFailure();
      return;
    }

    if (parsedMeshOps.size() > 1) {
      rootModule.emitError("Pass currently only support a single shardy mesh "
                           "op in the module");
      signalPassFailure();
      return;
    }

    globalMeshOp = parsedMeshOps[0];

    // Update the manual axes to include the correct mesh shape dimensions in
    // the manual computation op. Also update all the argument shapes.
    rootModule.walk([&](func::FuncOp funcOp) {
      if (failed(updateManualAxes(context, builder, globalMeshOp, funcOp))) {
        rootModule.emitError(
            "Could not update manual axes for manual computation op");
        signalPassFailure();
        return;
      }
    });

    // Analyze the graph and cut all the shapes of each operation according to
    // their sdy sharding attribute.
    rootModule.walk([&](func::FuncOp funcOp) {
      if (failed(updateShapes(context, builder, globalMeshOp, funcOp))) {
        rootModule.emitError("Could not update shapes based on their tensor "
                             "sharding attributes");
        signalPassFailure();
        return;
      }
    });

    // Run conversion pattern to convert all sdy ccl operations into stablehlo
    // ccl operations
    if (failed(convertShardyCCLToStableHLOCCL(context, rootModule))) {
      rootModule.emitError(
          "Could not convert shardy ccl ops into stablehlo ccl ops");
      signalPassFailure();
      return;
    }

    // Remove all sdy tensor sharding annotations since all the analysis is
    // complete. Otherwise, verification of sdy manual computation op will fail
    // since it expects all shapes to be local device shapes with no sdy
    // shardings attached.
    rootModule.walk([&](func::FuncOp funcOp) {
      shardy_utils::removeSdyTensorShardings(context, funcOp);
    });
  }
};

} // namespace mlir::tt::stablehlo
