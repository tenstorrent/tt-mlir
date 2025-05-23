// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRHOISTTRANSFORM
#define GEN_PASS_DEF_TTIRWORKAROUNDREENABLEDPS
#define GEN_PASS_DEF_TTIRREMOVERETURNVALUES
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Hoist CPU ops to standalone funcs pass
//===----------------------------------------------------------------------===//

// Helper function to get ranks of an op's operands
// we use this to populate attrs which we need to tensor unpacking operations
// later.
static llvm::SmallVector<int64_t, 4>
getOperandTensorRanks(mlir::Operation *op) {
  llvm::SmallVector<int64_t, 4> ranks;

  // Iterate over operands (inputs)
  for (auto operand : op->getOperands()) {
    // Check if the operand is a tensor
    if (auto tensorType =
            mlir::dyn_cast<mlir::RankedTensorType>(operand.getType())) {
      // Add the rank of the tensor (number of dimensions)
      ranks.push_back(tensorType.getRank());
    }
  }

  return ranks;
}

// Generate unique name base on operation type + argument tensors dims & types.
static llvm::SmallString<16> generateHoistedFuncName(mlir::Operation *op) {
  llvm::SmallString<16> uniqueName("hoisted_");
  uniqueName.append(op->getName().getStringRef());

  for (auto operand : op->getOperands()) {
    if (auto tensorType =
            mlir::dyn_cast<mlir::RankedTensorType>(operand.getType())) {
      uniqueName += "_";
      llvm::raw_svector_ostream os(uniqueName);
      llvm::interleave(tensorType.getShape(), os, "x");
    }
  }

  uniqueName += "_func";
  std::replace(uniqueName.begin(), uniqueName.end(), '.', '_');
  return uniqueName;
}

// Tag bufferization access options based on operand semantics.
static void tagBufferizationAccess(mlir::func::FuncOp funcOp, unsigned argIdx,
                                   mlir::Operation *origOp,
                                   mlir::OpBuilder &builder) {
  if (auto dpsOp = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(origOp)) {
    if (dpsOp.isDpsInit(&origOp->getOpOperand(argIdx))) {
      funcOp.setArgAttr(argIdx, "bufferization.access",
                        builder.getStringAttr("write"));
    } else {
      funcOp.setArgAttr(argIdx, "bufferization.access",
                        builder.getStringAttr("read"));
    }
  } else {
    funcOp.setArgAttr(argIdx, "bufferization.access",
                      builder.getStringAttr("read-write"));
  }
}

// Helper function to hoist an arbitrary op into a new function in targetModule,
// generate a matching extern prototype in the sourceModule, and replace the
// original op with a callOp to the extern function.
static Value hoistOperationToFunction(mlir::Operation *opToHoist,
                                      mlir::ModuleOp sourceModule,
                                      mlir::ModuleOp targetModule) {

  const llvm::SmallVector<int64_t, 4> ranks = getOperandTensorRanks(opToHoist);
  mlir::MLIRContext *context = sourceModule.getContext();
  mlir::OpBuilder typeBuilder(opToHoist);
  auto f32Type = mlir::Float32Type::get(context);

  // Convert operands and gather types for function signature
  llvm::SmallVector<mlir::Type> operandTypes;
  llvm::SmallVector<mlir::Value> convertedOperands;

  for (auto operand : opToHoist->getOperands()) {
    if (auto tensorType =
            mlir::dyn_cast<mlir::RankedTensorType>(operand.getType())) {
      if (!tensorType.getElementType().isF32()) {
        // Create f32 version of tensor type
        auto f32TensorType = RankedTensorType::get(
            tensorType.getShape(), f32Type, tensorType.getEncoding());
        operandTypes.push_back(f32TensorType);

        // Create converted tensor value
        auto emptyTensor = typeBuilder.create<mlir::tt::ttir::EmptyOp>(
            opToHoist->getLoc(), tensorType.getShape(), f32Type);
        auto converted = typeBuilder.create<mlir::tt::ttir::ToLayoutOp>(
            opToHoist->getLoc(), operand, emptyTensor);
        convertedOperands.push_back(converted->getResult(0));
      } else {
        operandTypes.push_back(tensorType);
        convertedOperands.push_back(operand);
      }
    } else {
      operandTypes.push_back(operand.getType());
      convertedOperands.push_back(operand);
    }
  }

  // Gather result types for function signature
  llvm::SmallVector<mlir::Type> resultTypes;
  for (auto result : opToHoist->getResultTypes()) {
    if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(result)) {
      if (!tensorType.getElementType().isF32()) {
        resultTypes.push_back(RankedTensorType::get(
            tensorType.getShape(), f32Type, tensorType.getEncoding()));
      } else {
        resultTypes.push_back(tensorType);
      }
    } else {
      resultTypes.push_back(result);
    }
  }

  // Create function types
  mlir::FunctionType localFuncType =
      mlir::FunctionType::get(context, operandTypes, resultTypes);
  mlir::FunctionType funcType =
      mlir::FunctionType::get(context, operandTypes, resultTypes);

  const llvm::SmallString<16> functionName = generateHoistedFuncName(opToHoist);
  llvm::SmallString<16> localFunctionName = functionName;
  localFunctionName.append("_decl");

  auto localFunc = llvm::dyn_cast_if_present<func::FuncOp>(
      sourceModule.lookupSymbol(localFunctionName.str()));

  // Create a new hoisted function only if an equivalent one does not exist.
  if (localFunc == nullptr) {
    // Insert the function and the terminator
    auto hoistedFunc =
        func::FuncOp::create(opToHoist->getLoc(), functionName, funcType);
    targetModule.push_back(hoistedFunc);

    // Add a basic block to the function.
    mlir::Block *block = hoistedFunc.addEntryBlock();
    mlir::OpBuilder builder(block, block->end());

    // Map operands to block arguments and clone the operation.
    llvm::SmallVector<mlir::Value> newOperands;
    for (auto operand : llvm::enumerate(opToHoist->getOperands())) {
      newOperands.push_back(block->getArgument(operand.index()));
    }

    // Add bufferization access attributes to function arguments
    for (auto arg : llvm::enumerate(hoistedFunc.getArguments())) {
      if (auto tensorType =
              mlir::dyn_cast<mlir::RankedTensorType>(arg.value().getType())) {
        tagBufferizationAccess(hoistedFunc, arg.index(), opToHoist, builder);
      }
    }

    mlir::IRMapping mapping;
    for (auto operand : llvm::zip(opToHoist->getOperands(), newOperands)) {
      mapping.map(std::get<0>(operand), std::get<1>(operand));
    }

    // Clone the operation but modify its type if needed
    auto *clonedOp = builder.clone(*opToHoist, mapping);

    // Update operand types to f32 for tensor types
    for (size_t i = 0; i < clonedOp->getNumOperands(); i++) {
      if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(
              clonedOp->getOperand(i).getType())) {
        if (!tensorType.getElementType().isF32()) {
          auto newType = RankedTensorType::get(tensorType.getShape(), f32Type,
                                               tensorType.getEncoding());
          clonedOp->getOperand(i).setType(newType);
        }
      }
    }

    // Update result types to f32 for tensor types
    for (size_t i = 0; i < clonedOp->getNumResults(); i++) {
      if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(
              clonedOp->getResult(i).getType())) {
        if (!tensorType.getElementType().isF32()) {
          auto newType = RankedTensorType::get(tensorType.getShape(), f32Type,
                                               tensorType.getEncoding());
          clonedOp->getResult(i).setType(newType);
        }
      }
    }

    // Add an attribute to the function that maps return values to output
    // arguments
    if (auto dpsOp =
            mlir::dyn_cast<mlir::DestinationStyleOpInterface>(opToHoist)) {
      // Ensure there's only a single output
      assert(dpsOp.getDpsInits().size() == 1 &&
             "Only operations with a single output are supported");

      // Get the index of the output operand
      unsigned outputIdx =
          opToHoist->getNumOperands() - dpsOp.getDpsInits().size();

      // Store this mapping as an attribute on the function
      hoistedFunc->setAttr("ttir.return_to_output_mapping",
                           builder.getI32IntegerAttr(outputIdx));
    }

    // Add a return operation to the function with the operation results
    builder.create<mlir::func::ReturnOp>(opToHoist->getLoc(),
                                         clonedOp->getResults());

    // Declare the function prototype in the source module.
    localFunc = func::FuncOp::create(opToHoist->getLoc(),
                                     localFunctionName.str(), localFuncType);
    localFunc.setPrivate();

    // Add the function to the module first
    sourceModule.push_back(localFunc);

    // Now that the function is in the module, add bufferization access
    // attributes.
    for (auto arg : llvm::enumerate(localFunc.getArguments())) {
      if (auto tensorType =
              mlir::dyn_cast<mlir::RankedTensorType>(arg.value().getType())) {
        tagBufferizationAccess(localFunc, arg.index(), opToHoist, builder);
      }
    }

    hoistedFunc->setAttr("arg_ranks", builder.getI64ArrayAttr(ranks));
  }

  // Create the call using already converted inputs
  mlir::OpBuilder opBuilder(opToHoist);
  auto callOp = opBuilder.create<mlir::func::CallOp>(
      opToHoist->getLoc(), localFunc, convertedOperands);

  // Add the hoisted_call attribute
  callOp->setAttr(HoistedCallAttr::name,
                  UnitAttr::get(opToHoist->getContext()));

  // Convert results back to original types if needed
  llvm::SmallVector<mlir::Value> finalResults;
  for (auto [result, callResult] :
       llvm::zip(opToHoist->getResults(), callOp.getResults())) {
    if (auto tensorType =
            mlir::dyn_cast<mlir::RankedTensorType>(result.getType())) {
      if (!tensorType.getElementType().isF32()) {
        auto converted = opBuilder.create<mlir::tt::ttir::EmptyOp>(
            opToHoist->getLoc(), tensorType.getShape(),
            tensorType.getElementType());
        auto toOriginal = opBuilder.create<mlir::tt::ttir::ToLayoutOp>(
            opToHoist->getLoc(), callResult, converted);
        finalResults.push_back(toOriginal->getResult(0));
      } else {
        finalResults.push_back(callResult);
      }
    } else {
      finalResults.push_back(callResult);
    }
  }

  // Replace original op with the converted results
  opToHoist->replaceAllUsesWith(finalResults);

  // Erase the original operation
  opToHoist->erase();

  return finalResults.empty() ? Value() : finalResults[0];
}

// An analysis class which currently relies on manually tagging ops with a
// `should_hoist` attribute, but in the future will also tag fall-back ops, etc.
namespace {
class TTIRHoistAnalyze {
public:
  using HoistOpSet = llvm::SmallVector<llvm::SmallSet<mlir::Operation *, 4>>;

  TTIRHoistAnalyze(mlir::ModuleOp moduleOp) {
    moduleOp.walk([&](mlir::Operation *nestedOp) {
      if (nestedOp->hasAttr("should_hoist")) {
        llvm::SmallSet<mlir::Operation *, 4> opSet;
        opSet.insert(nestedOp);
        hoistedOps.push_back(opSet);
      }
    });
  }

  HoistOpSet getResults() { return hoistedOps; }

private:
  HoistOpSet hoistedOps;
};
} // namespace

namespace {
// Transform pass to hoist specific ops (based on configured analysis pass) into
// a cpu submodule for later independent lowering.
class TTIRHoistTransform
    : public impl::TTIRHoistTransformBase<TTIRHoistTransform> {
public:
  using impl::TTIRHoistTransformBase<
      TTIRHoistTransform>::TTIRHoistTransformBase;

  void runOnOperation() final {
    mlir::ModuleOp rootModule = getOperation();

    // We must run this transform on the root ModuleOp, since we are creating
    // new Op's within the root.
    if (rootModule->getParentOp() != nullptr) {
      return;
    }

    tt::DeviceModuleOp deviceModule;
    for (Operation &op : rootModule.getBodyRegion().front()) {
      if (auto maybeDeviceModule = dyn_cast<tt::DeviceModuleOp>(op)) {
        deviceModule = maybeDeviceModule;
        break;
      }
    }
    assert(deviceModule &&
           "must run tt::WrapDeviceModulePass on IR before hoisting!");

    ModuleOp deviceInnerModule = dyn_cast_if_present<mlir::ModuleOp>(
        deviceModule.getBodyRegion().front().front());
    assert(deviceInnerModule &&
           "tt::DeviceModuleOp must have single ModuleOp child!");

    IRRewriter rewriter(&getContext());

    auto loc = rootModule->getLoc();

    TTIRHoistAnalyze analysisPass(deviceInnerModule);
    const TTIRHoistAnalyze::HoistOpSet &hoistOpSets = analysisPass.getResults();

    // We don't want to create a CPUModuleOp etc. if we aren't hoisting any ops.
    if (hoistOpSets.empty()) {
      return;
    }

    // Check if a "cpu_module" already exists.
    tt::CPUModuleOp cpuModule;
    mlir::ModuleOp cpuInnerModule;
    for (auto &op : rootModule.getBody()->getOperations()) {
      if (auto module = llvm::dyn_cast<tt::CPUModuleOp>(op)) {
        cpuModule = module;
        cpuInnerModule = dyn_cast_if_present<mlir::ModuleOp>(
            cpuModule.getBodyRegion().front().front());
        assert(cpuInnerModule && "CPUModuleOp must contain 1 ModuleOp!");
        break;
      }
    }

    // If no CPU module exists, create one.
    if (!cpuModule) {
      rewriter.setInsertionPointToEnd(rootModule.getBody());
      cpuModule = rewriter.create<tt::CPUModuleOp>(loc);
      rewriter.setInsertionPointToStart(&cpuModule.getBodyRegion().front());
      cpuInnerModule = rewriter.create<mlir::ModuleOp>(loc);
    }

    for (const auto &opSet : hoistOpSets) {
      assert(opSet.size() == 1 &&
             "currently don't support hoisting multiple instructions at once!");
      hoistOperationToFunction(*opSet.begin(), deviceInnerModule,
                               cpuInnerModule);
    }
  }
};
} // namespace

// Function to transform defining ops of return values s.t. they use original
// DPS parameter (stashed in return_to_output_mapping attr) instead of empty ops
// created during lowering.
static LogicalResult reenableDpsFromAttr(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());

  bool failed = false;
  // Find all functions with the return_to_output_mapping attribute
  moduleOp.walk([&](func::FuncOp funcOp) {
    // This transform is only meaningful on an op which has been lowered TTIR ->
    // TOSA -> Linalg, so we ignore any funcs without both these attrs.
    if (!funcOp->hasAttr("ttir.return_to_output_mapping") ||
        !funcOp->hasAttr("ttir.processed_by_tosa")) {
      return;
    }

    // Get the return_to_output_mapping attribute
    auto mappingAttr =
        funcOp->getAttrOfType<IntegerAttr>("ttir.return_to_output_mapping");
    if (!mappingAttr) {
      funcOp->emitError() << "Function has ttir.return_to_output_mapping "
                             "attribute but it's not an IntegerAttr";
      failed = true;
      return;
    }

    // Find the return operation
    func::ReturnOp returnOp;
    for (Block &block : funcOp.getBlocks()) {
      if (auto retOp = dyn_cast<func::ReturnOp>(block.getTerminator())) {
        returnOp = retOp;
        break;
      }
    }

    if (!returnOp) {
      funcOp->emitError() << "Function does not have a return operation";
      failed = true;
      return;
    }

    if (returnOp.getNumOperands() == 0) {
      funcOp->emitWarning()
          << "Function already has no return values, nothing to transform";
      funcOp->removeAttr("ttir.return_to_output_mapping");
      return;
    }

    // Get the output operand index
    unsigned outputArgIdx = mappingAttr.getInt();
    if (outputArgIdx >= funcOp.getNumArguments()) {
      funcOp->emitError() << "Output argument index " << outputArgIdx
                          << " is out of range (function has "
                          << funcOp.getNumArguments() << " arguments)";
      failed = true;
      return;
    }

    // Get the return value
    Value returnVal = returnOp.getOperands()[0];

    // Find the operation that produces the return value
    Operation *producer = returnVal.getDefiningOp();
    if (!producer) {
      funcOp->emitError() << "Return value is not produced by an operation";
      failed = true;
      return;
    }

    // Handle different types of operations
    bool transformed = false;

    // Handle tensor.expand_shape operations
    if (auto expandOp = mlir::dyn_cast<tensor::ExpandShapeOp>(producer)) {
      // Get the input to the expand_shape operation
      Value expandInput = expandOp.getSrc();
      Operation *expandInputProducer = expandInput.getDefiningOp();

      // Check if the input is produced by a linalg.reduce operation
      if (auto reduceOp =
              mlir::dyn_cast_or_null<linalg::ReduceOp>(expandInputProducer)) {
        // Find the tensor.empty operation that feeds into the linalg.reduce
        for (OpOperand &output : reduceOp->getOpOperands()) {
          // Check if this is an output operand (init tensor)
          if (reduceOp.isInitTensor(&output)) {
            Value outputBuffer = output.get();

            // Check if it's defined by a linalg.fill operation
            if (auto fillOp = mlir::dyn_cast_or_null<linalg::FillOp>(
                    outputBuffer.getDefiningOp())) {
              // Get the destination of the fill operation
              Value fillDest = fillOp.getDpsInitOperand(0)->get();

              // Check if it's defined by a tensor.empty operation
              if (auto emptyOp = mlir::dyn_cast_or_null<tensor::EmptyOp>(
                      fillDest.getDefiningOp())) {
                // Get the output tensor from the function arguments
                Value outputTensor = funcOp.getArgument(outputArgIdx);

                // We need to reshape the output tensor to match the shape
                // expected by the reduce operation This is because
                // linalg.reduce expects a tensor with the reduced dimension
                // removed
                auto outputTensorType =
                    mlir::cast<RankedTensorType>(outputTensor.getType());
                auto emptyOpType =
                    mlir::cast<RankedTensorType>(emptyOp.getType());

                if (outputTensorType.getRank() != emptyOpType.getRank()) {
                  // We need to collapse the output tensor to match the shape of
                  // the empty op
                  rewriter.setInsertionPoint(emptyOp);

                  // Create a collapse_shape operation to get the right shape
                  SmallVector<ReassociationIndices> reassociationIndices;

                  // For a reduce along dimension 0 with keep_dim=true, we need
                  // to collapse [0, 1] -> [0] This is the inverse of the
                  // expand_shape operation
                  SmallVector<int64_t> indices;
                  for (size_t i = 0;
                       i < expandOp.getReassociationIndices().size(); i++) {
                    auto reassociation = expandOp.getReassociationIndices()[i];
                    reassociationIndices.push_back(reassociation);
                  }

                  auto collapsedType =
                      RankedTensorType::get(emptyOpType.getShape(),
                                            outputTensorType.getElementType());

                  Value collapsedTensor =
                      rewriter.create<tensor::CollapseShapeOp>(
                          emptyOp.getLoc(), collapsedType, outputTensor,
                          reassociationIndices);

                  // Replace the empty op with the collapsed tensor
                  rewriter.replaceOp(emptyOp, collapsedTensor);
                  transformed = true;
                } else {
                  // If ranks match, just replace directly
                  rewriter.replaceOp(emptyOp, outputTensor);
                  transformed = true;
                }
              }
            } else if (auto emptyOp = mlir::dyn_cast_or_null<tensor::EmptyOp>(
                           outputBuffer.getDefiningOp())) {
              // Get the output tensor from the function arguments
              Value outputTensor = funcOp.getArgument(outputArgIdx);

              // We need to reshape the output tensor to match the shape
              // expected by the reduce operation
              auto outputTensorType =
                  mlir::cast<RankedTensorType>(outputTensor.getType());
              auto emptyOpType =
                  mlir::cast<RankedTensorType>(emptyOp.getType());

              if (outputTensorType.getRank() != emptyOpType.getRank()) {
                // We need to collapse the output tensor to match the shape of
                // the empty op
                rewriter.setInsertionPoint(emptyOp);

                // Create a collapse_shape operation to get the right shape
                SmallVector<ReassociationIndices> reassociationIndices;

                // For a reduce along dimension 0 with keep_dim=true, we need to
                // collapse [0, 1] -> [0] This is the inverse of the
                // expand_shape operation
                for (size_t i = 0;
                     i < expandOp.getReassociationIndices().size(); i++) {
                  auto reassociation = expandOp.getReassociationIndices()[i];
                  reassociationIndices.push_back(reassociation);
                }

                auto collapsedType = RankedTensorType::get(
                    emptyOpType.getShape(), outputTensorType.getElementType());

                Value collapsedTensor =
                    rewriter.create<tensor::CollapseShapeOp>(
                        emptyOp.getLoc(), collapsedType, outputTensor,
                        reassociationIndices);

                // Replace the empty op with the collapsed tensor
                rewriter.replaceOp(emptyOp, collapsedTensor);
                transformed = true;
              } else {
                // If ranks match, just replace directly
                rewriter.replaceOp(emptyOp, outputTensor);
                transformed = true;
              }
            }
          }
        }
      }
    }
    // Handle linalg.generic operations
    else if (auto linalgOp = mlir::dyn_cast<linalg::GenericOp>(producer)) {
      // Find all tensor.empty operations that feed into this linalg.generic
      for (OpOperand &output : linalgOp.getDpsInitsMutable()) {
        Value outputBuffer = output.get();
        if (auto emptyOp = mlir::dyn_cast_or_null<tensor::EmptyOp>(
                outputBuffer.getDefiningOp())) {
          // Get the output tensor from the function arguments
          Value outputTensor = funcOp.getArgument(outputArgIdx);
          // Replace the tensor.empty with the output tensor
          rewriter.setInsertionPoint(emptyOp);
          rewriter.replaceOp(emptyOp, outputTensor);
          transformed = true;
        }
      }
    }
    // Handle linalg.transpose operations
    else if (auto linalgOp = mlir::dyn_cast<linalg::TransposeOp>(producer)) {
      // Find all tensor.empty operations that feed into this linalg.transpose
      for (OpOperand &output : linalgOp.getDpsInitsMutable()) {
        Value outputBuffer = output.get();
        if (auto emptyOp = mlir::dyn_cast_or_null<tensor::EmptyOp>(
                outputBuffer.getDefiningOp())) {
          // Get the output tensor from the function arguments
          Value outputTensor = funcOp.getArgument(outputArgIdx);
          // Replace the tensor.empty with the output tensor
          rewriter.setInsertionPoint(emptyOp);
          rewriter.replaceOp(emptyOp, outputTensor);
          transformed = true;
        }
      }
    }
    // Handle other operations that might create temporary tensors
    else {
      // For now, we only handle linalg.generic and tensor.expand_shape
      // operations
      funcOp->emitWarning()
          << "Unhandled operation type: " << producer->getName().getStringRef()
          << ". Only linalg.generic and tensor.expand_shape operations are "
             "currently supported.";
    }

    if (!transformed) {
      funcOp->emitWarning()
          << "Could not find any tensor.empty operations to replace";
      return;
    }

    // Replace the return operation with an empty return
    rewriter.setInsertionPoint(returnOp);
    rewriter.replaceOpWithNewOp<func::ReturnOp>(returnOp);

    // Update the function type to remove the return value
    auto funcType = funcOp.getFunctionType();
    auto newFuncType =
        FunctionType::get(funcOp.getContext(), funcType.getInputs(), {});
    funcOp.setType(newFuncType);

    // Remove the mapping attribute since we've applied it
    funcOp->removeAttr("ttir.return_to_output_mapping");
  });

  return failed ? failure() : success();
}

namespace {
// Simple pass which implements a TTIR -> TOSA -> Linalg workaround to ensure
// the final linalg ops use the same DPS outputs as the original TTIR (TOSA will
// drop DPS outputs and replace with new empty ops otherwise).
class TTIRWorkaroundReenableDPS
    : public impl::TTIRWorkaroundReenableDPSBase<TTIRWorkaroundReenableDPS> {
  using impl::TTIRWorkaroundReenableDPSBase<
      TTIRWorkaroundReenableDPS>::TTIRWorkaroundReenableDPSBase;
  void runOnOperation() override {
    if (failed(reenableDpsFromAttr(getOperation()))) {
      signalPassFailure();
    }
  }
};
} // namespace

namespace {
// Pass to remove return values from functions that have been cleaned up; we do
// not want our hoisted funcs to return, since this would involve allocating new
// tensors etc.
class TTIRRemoveReturnValuesPass
    : public impl::TTIRRemoveReturnValuesBase<TTIRRemoveReturnValuesPass> {
public:
  using impl::TTIRRemoveReturnValuesBase<
      TTIRRemoveReturnValuesPass>::TTIRRemoveReturnValuesBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    // Find all functions with return values
    moduleOp.walk([&](func::FuncOp funcOp) {
      auto funcType = funcOp.getFunctionType();
      if (funcType.getResults().empty()) {
        return;
      }

      // Find the return operation.
      func::ReturnOp returnOp;
      for (Block &block : funcOp.getBlocks()) {
        if (auto retOp = dyn_cast<func::ReturnOp>(block.getTerminator())) {
          returnOp = retOp;
          break;
        }
      }

      if (!returnOp) {
        funcOp->emitError() << "Function does not have a return operation";
        signalPassFailure();
        return;
      }

      if (returnOp.getNumOperands() == 0) {
        // Function already has no return values, nothing to transform.
        return;
      }

      // Replace the return operation with an empty return.
      rewriter.setInsertionPoint(returnOp);
      rewriter.replaceOpWithNewOp<func::ReturnOp>(returnOp);

      // Update the function type to remove the return values
      auto newFuncType =
          FunctionType::get(funcOp.getContext(), funcType.getInputs(), {});
      funcOp.setType(newFuncType);
    });
  }
};
} // namespace

} // namespace mlir::tt::ttir
