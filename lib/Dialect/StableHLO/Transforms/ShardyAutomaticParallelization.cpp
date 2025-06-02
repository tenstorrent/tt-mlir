// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "ttmlir/Dialect/StableHLO/Transforms/ShardyCCLToStableHLOCCL.h"
#include "ttmlir/Dialect/StableHLO/Transforms/ShardyUtils.h"
#include "ttmlir/Dialect/TT/Utils/PopulateArgumentTypes.h"
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
#define GEN_PASS_DEF_SHARDYANNOTATEARGUMENTSPASS
#define GEN_PASS_DEF_SHARDYCLEANUPMODULEPASS
#define GEN_PASS_DEF_SHARDYUPDATEAUTOMATICSHARDSHAPESPASS
#define GEN_PASS_DEF_SHARDYWRAPMANUALCOMPUTATIONPASS
#define GEN_PASS_DEF_UPDATEAUTOMATICSHARDSHAPESPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

/*
StableHLO graph can be
a. no annotation
b. tt argument annotations
c. gspmd annotations
d. sdy annotations

Algorithm:
1. Apply sharding constraints in case of any sharding mismatches. This will
insert sdy.sharding_constraint operations into the graph.
2. Check if the graph is annotated with sharding attributes.
  a. If it's annotated with gspmd, throw pass error since this pass doesn't
support that yet (need to add a conversion from gspmd into sdy).
  b. If it's annotated with sdy, we get the meshOp from the module.
  c. If it's annotated with tt argument annotations, we can use those to
determine the inputs and insert custom meshOp and sharding annotations for batch
parallelization.
  d. If it's not annotated, we insert custom meshOp and sharding
annotations for batch parallelization.
3. Run sdy sharding propogation pass.
4. Convert all sharding constraints into reshard operations.
5. Insert explicit reshards in case of sharding mismatches.
6. Wrap all operations under a sdy.manual_computationOp.
7. Convert all reshards into sdy collective operations.
8. Run topological sort on the graph and update all shapes with a new shape
based on their sharding annotation determine by the sdy sharding propagation
pass. Also convert all sdy collective operations into stablehlo collective
operations.
9. Remove all sdy annotations since analysis is complete.
10. Remove any dead operations that are no longer needed.
11. Close tensor shardings and drop replicated axes.
*/

// Check if tt argument annotations exist in the module.
static inline bool ttAnnotationsExist(mlir::ModuleOp &rootModule) {
  mlir::WalkResult result = rootModule.walk([&](func::FuncOp funcOp) {
    // Check if ttir.name exists for any of the arguments.
    for (BlockArgument arg : funcOp.getBody().front().getArguments()) {
      if (auto currentArgAttrDict = funcOp.getArgAttrDict(arg.getArgNumber())) {
        if (currentArgAttrDict.contains(mlir::tt::ArgumentTypeAttr::name)) {
          return WalkResult::interrupt();
        }
      }
    }

    return WalkResult::advance();
  });

  return result.wasInterrupted();
}

// Wrap all operations within a module under a manual computation op.
static inline mlir::LogicalResult wrapFunctionBodyInManualComputationOp(
    MLIRContext *context, mlir::OpBuilder &builder,
    mlir::sdy::MeshOp &globalMeshOp, func::FuncOp &funcOp) {
  // Get in_shardings of current function.
  llvm::SmallVector<mlir::sdy::TensorShardingAttr> inShardingAttrs =
      sdy_utils::getInShardingAttrs(context, funcOp, globalMeshOp);
  mlir::sdy::TensorShardingPerValueAttr inShardings =
      mlir::sdy::TensorShardingPerValueAttr::get(context, inShardingAttrs);

  // Get out_shardings of current function.
  llvm::SmallVector<mlir::sdy::TensorShardingAttr> outShardingAttrs =
      sdy_utils::getOutShardingAttrs(context, funcOp, globalMeshOp);
  mlir::sdy::TensorShardingPerValueAttr outShardings =
      mlir::sdy::TensorShardingPerValueAttr::get(context, outShardingAttrs);

  // Create sdy.manual_computation op
  mlir::FunctionType funcType = funcOp.getFunctionType();
  mlir::Block &entryBlock = funcOp.getBody().front();
  builder.setInsertionPointToStart(&entryBlock);
  mlir::sdy::ManualComputationOp manualComputationOp =
      builder.create<mlir::sdy::ManualComputationOp>(
          builder.getUnknownLoc(), funcType.getResults(), funcOp.getArguments(),
          inShardings, outShardings, llvm::SmallVector<mlir::StringAttr>());

  // Determine the argumentTypes and argumentLocations that need to get
  // added to the new region in manualComputationOp.
  llvm::SmallVector<mlir::Type> argumentTypes;
  llvm::SmallVector<mlir::Location> argumentLocations;

  for (auto arg : funcOp.getArguments()) {
    // All arguments must be annotated with sdy.sharding attribute at this
    // point.
    mlir::RankedTensorType oldType =
        mlir::cast<mlir::RankedTensorType>(arg.getType());
    argumentTypes.push_back(oldType);
    argumentLocations.push_back(arg.getLoc());
  }

  // Create a new block and new region in manualComputationOp since it is
  // a region based op.
  mlir::Block &sourceBlock = funcOp.getBody().front();
  mlir::Region &targetRegion = manualComputationOp->getRegion(0);
  std::unique_ptr<mlir::Block> newBlock = std::make_unique<mlir::Block>();
  targetRegion.push_back(
      newBlock.release()); // Dynamically created block memory moves
                           // ownership into region.
  mlir::Block &targetBlock = targetRegion.front();
  targetBlock.addArguments(argumentTypes, argumentLocations);

  // Migrate all the ops currently in funcOp into the manualComputationOp
  // region. This will also copy func.ReturnOp.
  mlir::Block::iterator it = ++mlir::Block::iterator(manualComputationOp);
  targetBlock.getOperations().splice(
      targetBlock.end(), sourceBlock.getOperations(), it, sourceBlock.end());

  // Create a new func.ReturnOp in the original func.funcOp that takes the
  // manualComputationOp as it's operand.
  builder.setInsertionPointAfter(manualComputationOp);
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(),
                                       manualComputationOp->getResults());

  // Update old arguments with new arguments inside of the
  // manualComputationBlock.
  for (uint32_t i = 0; i < sourceBlock.getArguments().size(); i++) {
    auto oldArg = sourceBlock.getArgument(i);
    auto newArg = targetBlock.getArgument(i);

    for (auto &targetOp : targetBlock.getOperations()) {
      for (auto &operand : targetOp.getOpOperands()) {
        if (operand.get() == oldArg) {
          operand.set(newArg);
        }
      }
    }
  }

  // Update all func.return ops in manualComputationOp with sdy.return op.
  // This is because the manualComputationOp requires a sdy.returnOp.
  for (Operation &op :
       llvm::make_early_inc_range(targetBlock.getOperations())) {
    if (auto returnOp = llvm::dyn_cast<mlir::func::ReturnOp>(op)) {
      builder.setInsertionPoint(returnOp);
      builder.create<mlir::sdy::ReturnOp>(builder.getUnknownLoc(),
                                          returnOp->getOperands());
      returnOp->erase();
    }
  }

  return mlir::success();
}

// Update the manual axes for each computation block and update the argument
// tensor shapes according to their tensor sharding annotation.
static inline mlir::LogicalResult
updateManualAxes(MLIRContext *context, mlir::OpBuilder &builder,
                 mlir::sdy::MeshOp &globalMeshOp, func::FuncOp &funcOp) {
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
    auto manualComputationOp =
        llvm::dyn_cast<mlir::sdy::ManualComputationOp>(op);
    if (!manualComputationOp) {
      return WalkResult::advance();
    }

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
          sdy_utils::populateShardedOutputType(
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
static inline mlir::LogicalResult updateShapes(MLIRContext *context,
                                               mlir::OpBuilder &builder,
                                               mlir::sdy::MeshOp &globalMeshOp,
                                               func::FuncOp &funcOp) {
  // Run a topological sort and apply tensor sharding annotations to each op.
  llvm::SetVector<mlir::Operation *> opSet;
  funcOp.getBody().walk([&](mlir::Operation *op) { opSet.insert(op); });
  llvm::SetVector<mlir::Operation *> sortedOpSet = mlir::topologicalSort(opSet);

  for (mlir::Operation *op : sortedOpSet) {
    for (auto namedAttr : op->getAttrs()) {
      if (namedAttr.getName() == mlir::sdy::TensorShardingAttr::name) {
        mlir::sdy::TensorShardingPerValueAttr tensorShardingPerValueAttr =
            mlir::dyn_cast<mlir::sdy::TensorShardingPerValueAttr>(
                op->getAttr(mlir::sdy::TensorShardingAttr::name));
        llvm::ArrayRef<mlir::sdy::TensorShardingAttr> tensorShardings =
            tensorShardingPerValueAttr.getShardings();

        // Loop through all the tensorShardings and apply them to each
        // output
        llvm::SmallVector<mlir::RankedTensorType> newTypes;

        for (uint32_t i = 0; i < tensorShardings.size(); i++) {
          mlir::sdy::TensorShardingAttr tensorShardingAttr = tensorShardings[i];
          mlir::RankedTensorType oldType =
              mlir::cast<mlir::RankedTensorType>(op->getResult(i).getType());
          FailureOr<mlir::RankedTensorType> newType =
              sdy_utils::populateShardedOutputType(globalMeshOp.getMesh(),
                                                   oldType, tensorShardingAttr);

          if (failed(newType)) {
            op->emitError("Could not apply propagated tensor shardings to "
                          "tensor dimensions.\n");
            return mlir::failure();
          }

          newTypes.push_back(*newType);
        }

        // Create new operation state to update the original operation with
        // it's new computed shapes.
        mlir::OperationState state(op->getLoc(), op->getName());
        for (auto type : newTypes) {
          state.types.push_back(type);
        }
        state.operands.append(op->operand_begin(), op->operand_end());
        llvm::SmallVector<mlir::NamedAttribute> namedAttrs(op->getAttrs());

        // stablehlo.constant needs its argument dictionary updated.
        if (auto constantOp =
                mlir::dyn_cast<mlir::stablehlo::ConstantOp>(*op)) {
          for (auto &namedAttr : namedAttrs) {
            if (namedAttr.getName() != "value") {
              continue;
            }

            mlir::DenseElementsAttr denseElementsAttr =
                mlir::dyn_cast<mlir::DenseElementsAttr>(namedAttr.getValue());

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
            namedAttr.setValue(newAttr);
          }
        }

        // stablehlo.slice needs its argument dictionary updated.
        if (auto sliceOp = mlir::dyn_cast<mlir::stablehlo::SliceOp>(*op)) {
          llvm::SmallVector<int64_t> startIndices(sliceOp.getStartIndices());
          llvm::SmallVector<int64_t> limitIndices(sliceOp.getLimitIndices());

          // Iterate through start and limit indices and update them based on
          // the sharding annotation for that dimension.
          for (uint32_t i = 0; i < tensorShardings.size(); i++) {
            llvm::ArrayRef<mlir::sdy::DimensionShardingAttr> dimShardings =
                tensorShardings[i].getDimShardings();

            for (const auto [index, dimShardingAttr] :
                 llvm::enumerate(dimShardings)) {
              FailureOr<int64_t> updatedStartShapeDim =
                  sdy_utils::calculateUpdatedShapeDim(globalMeshOp.getMesh(),
                                                      dimShardingAttr,
                                                      startIndices[index]);
              FailureOr<int64_t> updatedLimitShapeDim =
                  sdy_utils::calculateUpdatedShapeDim(globalMeshOp.getMesh(),
                                                      dimShardingAttr,
                                                      limitIndices[index]);

              if (failed(updatedStartShapeDim) ||
                  failed(updatedLimitShapeDim)) {
                sliceOp->emitError(
                    "Could not apply propagated tensor shardings "
                    "to attribute dictionary for slice op.\n");
                return mlir::failure();
              }

              startIndices[index] = *updatedStartShapeDim;
              limitIndices[index] = *updatedLimitShapeDim;
            }
          }

          // Update start and limit indices in op named attributes.
          for (auto &namedAttr : namedAttrs) {
            if (namedAttr.getName() == "start_indices") {
              namedAttr.setValue(
                  mlir::DenseI64ArrayAttr::get(context, startIndices));
            } else if (namedAttr.getName() == "limit_indices") {
              namedAttr.setValue(
                  mlir::DenseI64ArrayAttr::get(context, limitIndices));
            }
          }
        }

        state.attributes = namedAttrs;
        state.regions.resize(op->getNumRegions());
        mlir::Operation *newOp = mlir::Operation::create(state);

        // If the operation has nested regions, we need to remap and copy
        // them over (eg. stablehlo.reduce).
        for (unsigned i = 0; i < op->getNumRegions(); i++) {
          auto &oldRegion = op->getRegion(i);
          auto &newRegion = newOp->getRegion(i);

          // Figure out new operand mapping and apply to newly created op.
          mlir::IRMapping mapping;
          auto *oldBlock = &oldRegion.front();
          std::unique_ptr<mlir::Block> newBlock =
              std::make_unique<mlir::Block>();
          for (auto oldArg : oldBlock->getArguments()) {
            auto newArg =
                newBlock->addArgument(oldArg.getType(), oldArg.getLoc());
            mapping.map(oldArg, newArg);
          }

          for (auto &op : *oldBlock) {
            builder.setInsertionPointToEnd(newBlock.get());
            builder.clone(op, mapping);
          }
          newRegion.push_back(newBlock.release());
        }

        // Update all old op results with new op results.
        for (uint32_t i = 0; i < op->getNumResults(); i++) {
          op->getResult(i).replaceAllUsesWith(newOp->getResult(i));
        }
        builder.setInsertionPoint(op);
        builder.insert(newOp);
        op->erase();
      }
    }
  }

  return mlir::success();
}

// Convert all sdy ccl ops into stablehlo ccl ops.
static inline mlir::LogicalResult
convertShardyCCLToStableHLOCCL(MLIRContext *context,
                               mlir::ModuleOp &rootModule) {
  RewritePatternSet patterns(context);
  ShardyTypeConverter typeConverter(context);
  populateShardyCCLToStableHLOCCLPatterns(context, patterns, typeConverter);

  mlir::ConversionTarget target(*context);
  target.addLegalDialect<mlir::tt::ttir::TTIRDialect>();
  target.addLegalDialect<mlir::sdy::SdyDialect>();
  target.addLegalDialect<mlir::stablehlo::StablehloDialect>();
  target.addLegalDialect<mlir::func::FuncDialect>();
  target.addLegalOp<mlir::ModuleOp>();
  target.addLegalOp<mlir::func::ReturnOp>();
  target.addLegalOp<mlir::func::CallOp>();
  target.addLegalOp<mlir::arith::ConstantOp>();
  target.addIllegalOp<mlir::sdy::AllGatherOp>();
  target.addIllegalOp<mlir::sdy::ReduceScatterOp>();
  target.addIllegalOp<mlir::sdy::AllReduceOp>();

  // Apply conversion.
  if (failed(applyFullConversion(rootModule, target, std::move(patterns)))) {
    rootModule.emitError("Could not convert shardy ccl operations into "
                         "stablehlo ccl operations.\n");
    return mlir::failure();
  }

  return mlir::success();
}

// Remove all sdy tensor shardings from the module.
static inline mlir::LogicalResult
removeSdyTensorShardings(MLIRContext *context, mlir::OpBuilder &builder,
                         mlir::sdy::MeshOp &globalMeshOp,
                         func::FuncOp &funcOp) {
  // Remove sharding annotations from arguments
  for (auto arg : funcOp.getArguments()) {
    if (auto argAttrDict = funcOp.getArgAttrDict(arg.getArgNumber())) {
      funcOp.setArgAttrs(arg.getArgNumber(),
                         sdy_utils::removeDictionaryAttrSdyShardingAnnotations(
                             context, argAttrDict));
    }
  }

  // Remove sharding annotations from results
  mlir::FunctionType funcType = funcOp.getFunctionType();
  for (uint32_t i = 0; i < funcType.getNumResults(); i++) {
    if (auto resultAttrDict =
            mlir::DictionaryAttr::get(context, funcOp.getResultAttrs(i))) {
      funcOp.setResultAttrs(
          i, sdy_utils::removeDictionaryAttrSdyShardingAnnotations(
                 context, resultAttrDict));
    }
  }

  // Remove sharding annotations from operations
  funcOp.getBody().walk([&](mlir::Operation *op) {
    if (auto opAttrDict = op->getAttrDictionary()) {
      op->setAttrs(sdy_utils::removeDictionaryAttrSdyShardingAnnotations(
          context, opAttrDict));
    }
  });

  return mlir::success();
}

class ArgumentAnalysis {
public:
  ArgumentAnalysis(int64_t largestRank) : largestRank(largestRank) {}
  virtual mlir::LogicalResult processArgument(BlockArgument *arg,
                                              func::FuncOp &funcOp) = 0;
  virtual mlir::DictionaryAttr getUpdatedArgumentDictionaryAttr(
      mlir::MLIRContext *context, func::FuncOp &funcOp, BlockArgument *arg) = 0;
  virtual ~ArgumentAnalysis() = default;

public:
  int64_t largestRank;
};

// This class is used to analyze arguments if no shard hints are provided.
class AutomaticArgumentAnalysis : public ArgumentAnalysis {
public:
  // todo: (tapspatel) Need to generalize largest rank such that batch dim
  // doesn't always have to be with tensors of rank 4.
  // https://github.com/tenstorrent/tt-mlir/issues/3292
  AutomaticArgumentAnalysis() : ArgumentAnalysis(4) {}

  // Add an argument to the automaticArgumentAnalysis and update largestRank.
  mlir::LogicalResult processArgument(BlockArgument *arg,
                                      func::FuncOp &funcOp) override {
    mlir::RankedTensorType tensorType =
        mlir::cast<mlir::RankedTensorType>(arg->getType());
    this->largestRank = std::max(this->largestRank, tensorType.getRank());
    return mlir::success();
  }

  // Get the updated argument dictionary with new sdy.sharding annotations based
  // on how the argument should be sharded.
  mlir::DictionaryAttr
  getUpdatedArgumentDictionaryAttr(mlir::MLIRContext *context,
                                   func::FuncOp &funcOp,
                                   BlockArgument *arg) override {
    llvm::SmallVector<mlir::NamedAttribute> newArgAttrs;

    // Copy the current dictionary if it exists for the op.
    if (auto currentArgAttrDict = funcOp.getArgAttrDict(arg->getArgNumber())) {
      newArgAttrs =
          SmallVector<mlir::NamedAttribute>(currentArgAttrDict.getValue());
    }

    // Determine sdy.sharding annotation to add to this argument based on
    // largest rank seen.
    mlir::RankedTensorType argType =
        mlir::cast<mlir::RankedTensorType>(arg->getType());
    llvm::SmallVector<mlir::sdy::DimensionShardingAttr> dimShardings;

    // todo: (tapspatel) Currently we only support batch parallel in this pass.
    // We can extend this to other types of parallelism which will require some
    // more analysis.
    // https://github.com/tenstorrent/tt-mlir/issues/3289
    if (argType.getRank() == this->largestRank && argType.getShape()[0] != 1) {
      mlir::sdy::AxisRefAttr axisAttr =
          mlir::sdy::AxisRefAttr::get(context, "batch");
      mlir::sdy::DimensionShardingAttr dimShardingAttr =
          mlir::sdy::DimensionShardingAttr::get(context, {axisAttr},
                                                /*is_closed*/ false);
      dimShardings.push_back(dimShardingAttr);

      mlir::sdy::DimensionShardingAttr full =
          mlir::sdy::DimensionShardingAttr::get(context, {},
                                                /*is_closed*/ false);
      dimShardings.append(argType.getRank() - 1, full);
    } else {
      mlir::sdy::DimensionShardingAttr full =
          mlir::sdy::DimensionShardingAttr::get(context, {},
                                                /*is_closed*/ false);
      dimShardings.append(argType.getRank(), full);
    }

    // Add the shardy sharding attribute to the argument
    mlir::sdy::TensorShardingAttr sharding = mlir::sdy::TensorShardingAttr::get(
        context, "mesh", dimShardings, {}, {});
    newArgAttrs.emplace_back(
        mlir::StringAttr::get(context, mlir::sdy::TensorShardingAttr::name),
        sharding);
    return mlir::DictionaryAttr::get(context, newArgAttrs);
  }
};

class TTArgumentAnalysis : public ArgumentAnalysis {
public:
  TTArgumentAnalysis() : ArgumentAnalysis(0) {}

  mlir::LogicalResult processArgument(BlockArgument *arg,
                                      func::FuncOp &funcOp) override {
    auto currentArgAttrDict = funcOp.getArgAttrDict(arg->getArgNumber());

    if (!currentArgAttrDict) {
      funcOp.emitError("In function ")
          << funcOp.getName() << " argument #: " << arg->getArgNumber()
          << " does not have an argument dictionary. This is required in order "
             "to do analysis on ttir.name tensor annotations.\n";
      return mlir::failure();
    }

    if (!currentArgAttrDict.contains(mlir::tt::ArgumentTypeAttr::name)) {
      funcOp.emitError("In function ")
          << funcOp.getName() << " argument #: " << arg->getArgNumber()
          << " is not annotated with ttir.name tensor annotations.\n";
      return mlir::failure();
    }

    mlir::tt::ArgumentTypeAttr argumentTypeAttr =
        mlir::cast<mlir::tt::ArgumentTypeAttr>(
            currentArgAttrDict.get(mlir::tt::ArgumentTypeAttr::name));
    mlir::tt::ArgumentType argTypeValue = argumentTypeAttr.getValue();
    if (argTypeValue == mlir::tt::ArgumentType::Input) {
      mlir::RankedTensorType tensorType =
          mlir::cast<mlir::RankedTensorType>(arg->getType());
      this->largestRank = std::max(this->largestRank, tensorType.getRank());
    }

    return mlir::success();
  }

  mlir::DictionaryAttr
  getUpdatedArgumentDictionaryAttr(mlir::MLIRContext *context,
                                   func::FuncOp &funcOp,
                                   BlockArgument *arg) override {
    llvm::SmallVector<mlir::NamedAttribute> newArgAttrs;
    mlir::tt::ArgumentType argTypeValue = mlir::tt::ArgumentType::Default;

    // Copy the current dictionary if it exists for the op.
    if (auto currentArgAttrDict = funcOp.getArgAttrDict(arg->getArgNumber())) {
      mlir::tt::ArgumentTypeAttr argumentTypeAttr =
          mlir::cast<mlir::tt::ArgumentTypeAttr>(
              currentArgAttrDict.get(mlir::tt::ArgumentTypeAttr::name));
      argTypeValue = argumentTypeAttr.getValue();
      newArgAttrs =
          SmallVector<mlir::NamedAttribute>(currentArgAttrDict.getValue());
    }

    // Determine sdy.sharding annotation to add to this argument based on tt
    // argument type. If it's an input, we annotate it. Otherwise, we keep it
    // open for shardy propogation to fill it in if required.
    mlir::RankedTensorType argType =
        mlir::cast<mlir::RankedTensorType>(arg->getType());
    llvm::SmallVector<mlir::sdy::DimensionShardingAttr> dimShardings;

    if (argType.getRank() == this->largestRank &&
        argTypeValue == mlir::tt::ArgumentType::Input) {
      mlir::sdy::AxisRefAttr axisAttr =
          mlir::sdy::AxisRefAttr::get(context, "batch");
      mlir::sdy::DimensionShardingAttr dimShardingAttr =
          mlir::sdy::DimensionShardingAttr::get(context, {axisAttr},
                                                /*is_closed*/ false);
      dimShardings.push_back(dimShardingAttr);

      mlir::sdy::DimensionShardingAttr full =
          mlir::sdy::DimensionShardingAttr::get(context, {},
                                                /*is_closed*/ false);
      dimShardings.append(argType.getRank() - 1, full);
    } else {
      mlir::sdy::DimensionShardingAttr full =
          mlir::sdy::DimensionShardingAttr::get(context, {},
                                                /*is_closed*/ false);
      dimShardings.append(argType.getRank(), full);
    }

    // Add the shardy sharding attribute to the argument
    mlir::sdy::TensorShardingAttr sharding = mlir::sdy::TensorShardingAttr::get(
        context, "mesh", dimShardings, {}, {});
    newArgAttrs.emplace_back(
        mlir::StringAttr::get(context, mlir::sdy::TensorShardingAttr::name),
        sharding);

    return mlir::DictionaryAttr::get(context, newArgAttrs);
  }
};

class ShardyAnnotateArgumentsPass
    : public impl::ShardyAnnotateArgumentsPassBase<
          ShardyAnnotateArgumentsPass> {
public:
  using impl::ShardyAnnotateArgumentsPassBase<
      ShardyAnnotateArgumentsPass>::ShardyAnnotateArgumentsPassBase;

  void runOnOperation() final {
    mlir::ModuleOp rootModule = getOperation();
    MLIRContext *context = rootModule.getContext();
    mlir::OpBuilder builder(context);

    // Check if the graph is annotated with sharding attributes. If annotated,
    // get the meshOp. If not annotated, insert custom meshOp and sharding
    // annotations.
    mlir::sdy::MeshOp globalMeshOp;
    llvm::ArrayRef<int64_t> meshShapeRef = *meshShape;
    bool userProvidedMesh = meshShapeRef.size() != 0 ? true : false;

    // Determine whether any existing sharding annotations exist.
    bool gspmdAnnotationsExist = sdy_utils::gspmdAnnotationsExist(rootModule);
    bool sdyAnnotationsExist = sdy_utils::sdyAnnotationsExist(rootModule);
    bool ttArgAnnotationsExist =
        mlir::tt::stablehlo::ttAnnotationsExist(rootModule);

    if (!sdyAnnotationsExist && !gspmdAnnotationsExist &&
        !ttArgAnnotationsExist && !automaticArgAnalysis) {
      rootModule.emitError("Could not find sdy, gspmd, tt annotations and "
                           "automatic arg analysis is "
                           "disabled. Skipping pass.\n");
      return;
    }

    // GSPMD annotations.
    if (gspmdAnnotationsExist) {
      rootModule.emitError("Shardy automatic parallelization pass does not "
                           "support GSPMD annotated module for now.\n");
      signalPassFailure();
      return;
    }

    // SDY annotations.
    if (sdyAnnotationsExist) {
      // Get the shardy mesh op in the root module.
      llvm::SmallVector<mlir::sdy::MeshOp> parsedMeshOps =
          sdy_utils::getMeshOps(rootModule);

      if (parsedMeshOps.size() == 0) {
        rootModule.emitError(
            "Shardy annotations exist in the module but there is "
            "no sdy.meshOp. If annotating with sdy, you must "
            "provide a meshOp symbol.\n");
        signalPassFailure();
        return;
      }

      if (parsedMeshOps.size() > 1) {
        rootModule.emitError(
            "Shardy automatic parallelization pass only works on 1 "
            "meshOp for now.\n");
        signalPassFailure();
        return;
      }

      if (userProvidedMesh) {
        rootModule.emitWarning(
            "User provided mesh but mesh already exists "
            "in mlir module. Using existing mesh in module.\n");
      }

      return;
    }

    // If graph is not annotated with sdy tensor shardings, determine how the
    // arguments should be sharded and insert sdy.sharding annotations and
    // sdy.meshOp. Use tt argument annotations if they exists as a guide.
    if (automaticArgAnalysis || ttArgAnnotationsExist) {
      // Remove any sdy meshOps that exists and insert our own to run analysis
      // on.
      sdy_utils::removeMeshOps(rootModule);

      // Check if user provided a mesh.
      if (meshShapeRef.size() == 0) {
        rootModule.emitError("User did not provide a mesh.\n");
        signalPassFailure();
        return;
      }

      // Generate new meshOp based on user provided mesh.
      if (meshShapeRef.size() != 2 || meshShapeRef[0] != 1) {
        rootModule.emitError(
            "Currently, shardy automatic parallel pass only supports 2d mesh "
            "shape and mesh shape dim0 must be 1.\n");
        signalPassFailure();
        return;
      }

      std::string meshName = "mesh";
      sdy_utils::MeshMap meshMap;
      meshMap["default"] = meshShapeRef[0];
      meshMap["batch"] = meshShapeRef[1];
      mlir::sdy::MeshAttr sdyMeshAttr =
          sdy_utils::createMeshAttrFromMeshMap(context, meshMap);
      builder.setInsertionPoint(&(rootModule.getBody()->front()));
      globalMeshOp = builder.create<mlir::sdy::MeshOp>(
          builder.getUnknownLoc(), builder.getStringAttr(meshName),
          sdyMeshAttr);

      for (auto &op : rootModule.getBody()->getOperations()) {
        auto funcOp = llvm::dyn_cast<func::FuncOp>(op);
        if (!funcOp) {
          continue;
        }

        // Get appropriate analysis manager.
        std::unique_ptr<ArgumentAnalysis> analysis;
        if (ttArgAnnotationsExist) {
          analysis = std::make_unique<TTArgumentAnalysis>();
        } else if (automaticArgAnalysis) {
          analysis = std::make_unique<AutomaticArgumentAnalysis>();
        }

        Block &entryBlock = funcOp.getBody().front();

        // Process each argument and add it to the analysis manager.
        for (BlockArgument arg : entryBlock.getArguments()) {
          if (mlir::failed(analysis->processArgument(&arg, funcOp))) {
            funcOp.emitError("Failed to process arguments in function.\n");
            signalPassFailure();
            return;
          }
        }

        // Once we processed all the elements, we want to iterate through all
        // the arguments again and update it's attributes to add the shardy
        // tensor sharding attribute.
        for (BlockArgument arg : entryBlock.getArguments()) {
          funcOp.setArgAttrs(arg.getArgNumber(),
                             analysis->getUpdatedArgumentDictionaryAttr(
                                 context, funcOp, &arg));
        }
      }

      return;
    }
  }
};

class ShardyWrapManualComputationPass
    : public impl::ShardyWrapManualComputationPassBase<
          ShardyWrapManualComputationPass> {
public:
  using impl::ShardyWrapManualComputationPassBase<
      ShardyWrapManualComputationPass>::ShardyWrapManualComputationPassBase;

  void runOnOperation() final {
    mlir::ModuleOp rootModule = getOperation();
    MLIRContext *context = rootModule.getContext();
    mlir::OpBuilder builder(context);
    mlir::PassManager pm(context);

    // Get the shardy mesh op in the root module.
    mlir::sdy::MeshOp globalMeshOp;
    llvm::SmallVector<mlir::sdy::MeshOp> parsedMeshOps =
        sdy_utils::getMeshOps(rootModule);

    if (parsedMeshOps.size() == 0) {
      rootModule.emitError(
          "Pass requires a shardy mesh op to be present in the root module.\n");
      signalPassFailure();
      return;
    }

    if (parsedMeshOps.size() > 1) {
      rootModule.emitError("Pass currently only support a single shardy mesh "
                           "op in the module.\n");
      signalPassFailure();
      return;
    }

    globalMeshOp = parsedMeshOps[0];

    // Sdy does not have support for a custom call for FullToShard or
    // ShardToFull like gspmd. Therefore, wrap all operations in each function
    // under a sdy.manual_computation op. This is required to enable the
    // conversion from sdy into ttir.
    rootModule.walk([&](func::FuncOp funcOp) {
      if (failed(wrapFunctionBodyInManualComputationOp(context, builder,
                                                       globalMeshOp, funcOp))) {
        rootModule.emitError(
            "Could not wrap function body inside a manual computation op.\n");
        signalPassFailure();
        return;
      }
    });
  }
};

class UpdateAutomaticShardShapesPass
    : public impl::UpdateAutomaticShardShapesPassBase<
          UpdateAutomaticShardShapesPass> {
public:
  using impl::UpdateAutomaticShardShapesPassBase<
      UpdateAutomaticShardShapesPass>::UpdateAutomaticShardShapesPassBase;

  void runOnOperation() final {
    mlir::ModuleOp rootModule = getOperation();
    MLIRContext *context = rootModule.getContext();
    mlir::OpBuilder builder(context);

    // Get the shardy mesh op in the root module.
    mlir::sdy::MeshOp globalMeshOp;
    llvm::SmallVector<mlir::sdy::MeshOp> parsedMeshOps =
        sdy_utils::getMeshOps(rootModule);

    if (parsedMeshOps.size() == 0) {
      rootModule.emitError(
          "Pass requires a shardy mesh op to be present in the root module.\n");
      signalPassFailure();
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
      if (failed(removeSdyTensorShardings(context, builder, globalMeshOp,
                                          funcOp))) {
        rootModule.emitError("Could not remove tensor sdy shardings.\n");
        signalPassFailure();
        return;
      }
    });
  }
};

} // namespace mlir::tt::stablehlo
