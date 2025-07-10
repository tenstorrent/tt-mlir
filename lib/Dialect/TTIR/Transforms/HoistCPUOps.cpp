// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRHOISTTRANSFORM
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
static void hoistOperationToFunction(mlir::Operation *opToHoist,
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
    for (auto operand : clonedOp->getOperands()) {
      if (auto tensorType =
              mlir::dyn_cast<mlir::RankedTensorType>(operand.getType())) {
        if (!tensorType.getElementType().isF32()) {
          auto newType = RankedTensorType::get(tensorType.getShape(), f32Type,
                                               tensorType.getEncoding());
          operand.setType(newType);
        }
      }
    }

    // Update result types to f32 for tensor types
    for (auto result : clonedOp->getResults()) {
      if (auto tensorType =
              mlir::dyn_cast<mlir::RankedTensorType>(result.getType())) {
        if (!tensorType.getElementType().isF32()) {
          auto newType = RankedTensorType::get(tensorType.getShape(), f32Type,
                                               tensorType.getEncoding());
          result.setType(newType);
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
      hoistedFunc->setAttr(ttir::ReturnToOutputMappingAttr::name,
                           builder.getI64IntegerAttr(outputIdx));
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
}

// An analysis class which currently relies on manually tagging ops with a
// `should_hoist` attribute, but in the future will also tag fall-back ops, etc.
namespace {
class TTIRHoistAnalyze {
public:
  using HoistOpSet = llvm::SmallVector<llvm::SmallSet<mlir::Operation *, 4>>;

  TTIRHoistAnalyze(mlir::ModuleOp moduleOp) {
    moduleOp.walk([&](mlir::Operation *nestedOp) {
      if (nestedOp->hasAttr(ttir::ShouldHoistAttr::name)) {
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

    ttcore::DeviceModuleOp deviceModule;
    for (Operation &op : rootModule.getBodyRegion().front()) {
      if (auto maybeDeviceModule = dyn_cast<ttcore::DeviceModuleOp>(op)) {
        deviceModule = maybeDeviceModule;
        break;
      }
    }
    assert(deviceModule &&
           "must run tt::WrapDeviceModulePass on IR before hoisting!");

    ModuleOp deviceInnerModule = dyn_cast_if_present<mlir::ModuleOp>(
        deviceModule.getBodyRegion().front().front());
    assert(deviceInnerModule &&
           "ttcore::DeviceModuleOp must have single ModuleOp child!");

    IRRewriter rewriter(&getContext());

    auto loc = rootModule->getLoc();

    TTIRHoistAnalyze analysisPass(deviceInnerModule);
    const TTIRHoistAnalyze::HoistOpSet &hoistOpSets = analysisPass.getResults();

    // We don't want to create a CPUModuleOp etc. if we aren't hoisting any ops.
    if (hoistOpSets.empty()) {
      return;
    }

    // Check if a "cpu_module" already exists.
    ttcore::CPUModuleOp cpuModule;
    mlir::ModuleOp cpuInnerModule;
    for (auto &op : rootModule.getBody()->getOperations()) {
      if (auto module = llvm::dyn_cast<ttcore::CPUModuleOp>(op)) {
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
      cpuModule = rewriter.create<ttcore::CPUModuleOp>(loc);
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

} // namespace mlir::tt::ttir
