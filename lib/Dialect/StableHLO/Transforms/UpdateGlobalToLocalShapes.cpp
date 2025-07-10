// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "ttmlir/Dialect/StableHLO/Transforms/ShardyCCLToStableHLOCCL.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_UPDATEGLOBALTOLOCALSHAPESPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

static FailureOr<mlir::OperationState> createNewOperationState(
    MLIRContext *context, mlir::Operation *op, mlir::sdy::MeshOp &globalMeshOp,
    llvm::ArrayRef<mlir::RankedTensorType> newTypes,
    llvm::ArrayRef<mlir::sdy::TensorShardingAttr> tensorShardings) {
  mlir::OperationState state(op->getLoc(), op->getName());
  for (auto type : newTypes) {
    state.types.push_back(type);
  }
  state.operands.append(op->operand_begin(), op->operand_end());
  llvm::SmallVector<mlir::NamedAttribute> namedAttrs(op->getAttrs());

  // Handle special operations that need their attribute dictionary updated.
  mlir::LogicalResult updatedAttributeResult =
      llvm::TypeSwitch<mlir::Operation *, mlir::LogicalResult>(op)
          .Case<mlir::stablehlo::ConstantOp>([&](auto constantOp) {
            auto namedAttrIt =
                llvm::find_if(namedAttrs, [](const mlir::NamedAttribute &attr) {
                  return attr.getName() == "value";
                });

            assert(namedAttrIt != namedAttrs.end() &&
                   "Constant operation does not have a value attribute. "
                   "Ill-formed operation.\n");
            mlir::DenseElementsAttr denseElementsAttr =
                mlir::cast<mlir::DenseElementsAttr>(namedAttrIt->getValue());

            // If the element is not a splat value (ie. the same value
            // for the entire constant) we fail as this is currently
            // not supported.
            if (!denseElementsAttr.isSplat()) {
              constantOp->emitError(
                  "Shardy automatic parallelization currently does "
                  "not support non-splat constant tensors.\n");
              return mlir::failure();
            }
            mlir::DenseElementsAttr newAttr = mlir::DenseElementsAttr::get(
                newTypes[0],
                denseElementsAttr.getSplatValue<mlir::Attribute>());
            namedAttrIt->setValue(newAttr);

            return mlir::success();
          })
          .Case<mlir::stablehlo::SliceOp>([&](auto sliceOp) {
            llvm::SmallVector<int64_t> startIndices(sliceOp.getStartIndices());
            llvm::SmallVector<int64_t> limitIndices(sliceOp.getLimitIndices());

            // Iterate through start and limit indices and update them based on
            // the sharding annotation for that dimension.
            for (uint32_t i = 0; i < tensorShardings.size(); i++) {
              llvm::ArrayRef<mlir::sdy::DimensionShardingAttr> dimShardings =
                  tensorShardings[i].getDimShardings();

              for (const auto [index, dimShardingAttr] :
                   llvm::enumerate(dimShardings)) {
                FailureOr<int64_t> updatedStartDim =
                    shardy_utils::calculateUpdatedDim(globalMeshOp.getMesh(),
                                                      dimShardingAttr,
                                                      startIndices[index]);
                FailureOr<int64_t> updatedLimitDim =
                    shardy_utils::calculateUpdatedDim(globalMeshOp.getMesh(),
                                                      dimShardingAttr,
                                                      limitIndices[index]);

                if (failed(updatedStartDim) || failed(updatedLimitDim)) {
                  sliceOp->emitError(
                      "Could not apply propagated tensor shardings "
                      "to attribute dictionary for slice op.\n");
                  return mlir::failure();
                }

                startIndices[index] = *updatedStartDim;
                limitIndices[index] = *updatedLimitDim;
              }
            }

            // Update start and limit indices in op named attributes.
            auto namedAttrStartIt =
                llvm::find_if(namedAttrs, [](const mlir::NamedAttribute &attr) {
                  return attr.getName() == "start_indices";
                });
            auto namedAttrLimitIt =
                llvm::find_if(namedAttrs, [](const mlir::NamedAttribute &attr) {
                  return attr.getName() == "limit_indices";
                });

            assert(namedAttrStartIt != namedAttrs.end() &&
                   "Slice operation does not have start indices attribute. "
                   "Ill-formed operation.\n");
            assert(namedAttrStartIt != namedAttrs.end() &&
                   "Slice operation does not have limit indices attribute. "
                   "Ill-formed operation.\n");

            namedAttrStartIt->setValue(
                mlir::DenseI64ArrayAttr::get(context, startIndices));
            namedAttrLimitIt->setValue(
                mlir::DenseI64ArrayAttr::get(context, limitIndices));

            return mlir::success();
          })
          .Case<mlir::stablehlo::GatherOp>([&](auto gatherOp) {
            llvm::SmallVector<int64_t> newSliceSizes(gatherOp.getSliceSizes());

            for (uint32_t i = 0; i < tensorShardings.size(); i++) {
              llvm::ArrayRef<mlir::sdy::DimensionShardingAttr> dimShardings =
                  tensorShardings[i].getDimShardings();

              for (const auto [index, dimShardingAttr] :
                   llvm::enumerate(dimShardings)) {
                FailureOr<int64_t> updatedSliceDim =
                    shardy_utils::calculateUpdatedDim(globalMeshOp.getMesh(),
                                                      dimShardingAttr,
                                                      newSliceSizes[index]);

                if (failed(updatedSliceDim)) {
                  gatherOp->emitError(
                      "Could not apply propagated tensor shardings to "
                      "attribute dictionary for gather op.\n");
                  return mlir::failure();
                }

                newSliceSizes[index] = *updatedSliceDim;
              }
            }

            auto namedAttrSliceSizesIt =
                llvm::find_if(namedAttrs, [](const mlir::NamedAttribute &attr) {
                  return attr.getName() == "slice_sizes";
                });

            assert(namedAttrSliceSizesIt != namedAttrs.end() &&
                   "Gather operation does not have slice sizes attribute. "
                   "Ill-formed operation.\n");
            namedAttrSliceSizesIt->setValue(
                mlir::DenseI64ArrayAttr::get(context, newSliceSizes));

            return mlir::success();
          })
          .Default([](mlir::Operation *op) { return mlir::success(); });

  if (failed(updatedAttributeResult)) {
    op->emitError("Could not updated attribute dictionary for operation.\n");
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
                                      "shardings to tensor dimensions.");
        return WalkResult::interrupt();
      }

      mlir::Value newArg = entryBlock.addArgument(*newType, arg.getLoc());
      arg.replaceAllUsesWith(newArg);
    }

    // Remove all unused arguments
    entryBlock.eraseArguments(0, initialArgSize);
    return WalkResult::advance();
  });

  return result.wasInterrupted() ? mlir::failure() : mlir::success();
}

// Update all shapes in the module based on their sdy tensor sharding attribute.
static mlir::LogicalResult updateShapes(MLIRContext *context,
                                        mlir::OpBuilder &builder,
                                        mlir::sdy::MeshOp &globalMeshOp,
                                        func::FuncOp &funcOp) {
  // Run a topological sort and apply tensor sharding annotations to each op.
  llvm::SetVector<mlir::Operation *> opSet;
  funcOp.getBody().walk([&](mlir::Operation *op) { opSet.insert(op); });
  llvm::SetVector<mlir::Operation *> sortedOpSet = mlir::topologicalSort(opSet);

  for (mlir::Operation *op : sortedOpSet) {
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
                      "tensor dimensions.\n");
        return mlir::failure();
      }

      // Create new operation state to update the original operation with
      // it's new computed shapes.
      FailureOr<mlir::OperationState> state =
          mlir::tt::stablehlo::createNewOperationState(
              context, op, globalMeshOp, *newTypes, tensorShardings);

      if (failed(state)) {
        op->emitError("Could not create a new operation with updated shapes.");
        return mlir::failure();
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
  }

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
                         "stablehlo ccl operations.\n");
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

    // If the module has gspmd annotations, we skip this pass.
    bool gspmdAnnotationsExist =
        shardy_utils::gspmdAnnotationsExist(rootModule);
    if (gspmdAnnotationsExist) {
      rootModule.emitWarning("Wrapping under manual computation pass does not "
                             "support GSPMD annotated module for now.\n");
      return;
    }

    // Get the shardy mesh op in the root module.
    mlir::sdy::MeshOp globalMeshOp;
    llvm::SmallVector<mlir::sdy::MeshOp> parsedMeshOps =
        shardy_utils::getMeshOps(rootModule);

    if (parsedMeshOps.size() == 0) {
      rootModule.emitWarning(
          "Pass requires a shardy mesh op to be present in the root module.\n");
      return;
    }

    if (parsedMeshOps.size() > 1) {
      rootModule.emitError("Pass currently only support a single shardy mesh "
                           "op in the module.\n");
      signalPassFailure();
      return;
    }

    globalMeshOp = parsedMeshOps[0];

    // Update the manual axes to include the correct mesh shape dimensions in
    // the manual computation op. Also update all the argument shapes.
    rootModule.walk([&](func::FuncOp funcOp) {
      if (failed(updateManualAxes(context, builder, globalMeshOp, funcOp))) {
        rootModule.emitError(
            "Could not update manual axes for manual computation op.\n");
        signalPassFailure();
        return;
      }
    });

    // Analyze the graph and cut all the shapes of each operation according to
    // their sdy sharding attribute.
    rootModule.walk([&](func::FuncOp funcOp) {
      if (failed(updateShapes(context, builder, globalMeshOp, funcOp))) {
        rootModule.emitError("Could not update shapes based on their tensor "
                             "sharding attributes.\n");
        signalPassFailure();
        return;
      }
    });

    // Run conversion pattern to convert all sdy ccl operations into stablehlo
    // ccl operations
    if (failed(convertShardyCCLToStableHLOCCL(context, rootModule))) {
      rootModule.emitError(
          "Could not convert shardy ccl ops into stablehlo ccl ops.\n");
      signalPassFailure();
      return;
    }

    // Remove all sdy tensor sharding annotations since all the analysis is
    // complete
    rootModule.walk([&](func::FuncOp funcOp) {
      shardy_utils::removeSdyTensorShardings(context, funcOp);
    });
  }
};

} // namespace mlir::tt::stablehlo
