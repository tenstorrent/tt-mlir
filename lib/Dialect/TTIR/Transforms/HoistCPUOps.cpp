// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"

#include "llvm/ADT/SmallSet.h"

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
    if (auto tensorType = dyn_cast<mlir::RankedTensorType>(operand.getType())) {
      // Add the rank of the tensor (number of dimensions)
      ranks.push_back(tensorType.getRank());
    }
  }

  return ranks;
}

// Generate unique name base on operation type + argument tensors dims & types.
static llvm::SmallString<16> generateHoistedFuncName(mlir::Operation *op) {
  // Start building the unique function name
  llvm::SmallString<16> uniqueName("hoisted_");
  uniqueName.append(op->getName().getStringRef().begin(),
                    op->getName().getStringRef().end());

  // Iterate over operands to extract tensor shapes and types
  for (auto operand : op->getOperands()) {
    auto rankedTensorType = dyn_cast<mlir::RankedTensorType>(operand.getType());
    if (rankedTensorType) {
      // Append the shape (dimensions) and the element type
      llvm::SmallString<5> shapeStr("_");
      for (auto dim : rankedTensorType.getShape()) {
        shapeStr += std::to_string(dim) + "x";
      }

      // Append the element type (e.g., f32, i32) -- unforunately I don't think
      // there's a better way to get string from mlir::Type
      std::string elementTypeStr;
      llvm::raw_string_ostream stream(elementTypeStr);
      rankedTensorType.getElementType().print(stream);

      uniqueName.append(shapeStr.begin(), shapeStr.end());
      uniqueName.append(elementTypeStr.begin(), elementTypeStr.end());
    }
  }

  // Add suffix to indicate it's a function
  uniqueName += "_func";

  return uniqueName;
}

// Helper function to hoist an arbitrary op into a new function in targetModule,
// generate a matching extern prototype in the sourceModule, and replace the
// original op with a callOp to the extern function.
static void hoistOperationToFunction(mlir::Operation *opToHoist,
                                     mlir::ModuleOp sourceModule,
                                     mlir::ModuleOp targetModule) {
  const llvm::SmallVector<int64_t, 4> ranks = getOperandTensorRanks(opToHoist);

  const llvm::SmallString<16> functionName = generateHoistedFuncName(opToHoist);

  auto localFunc = llvm::dyn_cast_or_null<func::FuncOp>(
      sourceModule.lookupSymbol(functionName));

  // Create a new hoisted function only if an equivalent one does not exist.
  if (localFunc == nullptr) {

    mlir::MLIRContext *context = sourceModule.getContext();

    // Gather operand and result types.
    llvm::SmallVector<mlir::Type> operandTypes, resultTypes;
    for (auto operand : opToHoist->getOperands()) {
      operandTypes.push_back(operand.getType());
    }
    for (auto result : opToHoist->getResultTypes()) {
      resultTypes.push_back(result);
    }

    // Create the function signature.
    mlir::FunctionType funcType =
        mlir::FunctionType::get(context, operandTypes, resultTypes);

    // Create the function in the target module.
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

    mlir::IRMapping mapping;
    for (auto operand : llvm::zip(opToHoist->getOperands(), newOperands)) {
      mapping.map(std::get<0>(operand), std::get<1>(operand));
    }

    mlir::Operation *clonedOp = builder.clone(*opToHoist, mapping);

    // Add a return operation to the function.
    builder.create<mlir::func::ReturnOp>(opToHoist->getLoc(),
                                         clonedOp->getResults());

    // Declare the function prototype in the source module.
    localFunc =
        func::FuncOp::create(opToHoist->getLoc(), functionName, funcType);
    localFunc.setPrivate();
    sourceModule.push_back(localFunc);

    hoistedFunc->setAttr("arg_ranks", builder.getI64ArrayAttr(ranks));
  }

  // Replace the original operation with a call to the hoisted function.
  mlir::OpBuilder opBuilder(opToHoist);
  auto callOp = opBuilder.create<mlir::func::CallOp>(
      opToHoist->getLoc(), localFunc, opToHoist->getOperands());

  // Replace all results of the original operation with the call results.
  for (auto result : llvm::zip(opToHoist->getResults(), callOp.getResults())) {
    std::get<0>(result).replaceAllUsesWith(std::get<1>(result));
  }

  // Erase the original operation.
  opToHoist->erase();
}

// An analysis class which relies on manual input locations to mark ops to be
// hoisted, useful for testing.
class TTIRHoistAnalyzeManual {
public:
  using HoistOpSet = llvm::SmallVector<llvm::SmallSet<mlir::Operation *, 4>>;

  TTIRHoistAnalyzeManual(mlir::ModuleOp moduleOp,
                         const llvm::SmallSet<llvm::StringRef, 4> &targetLocs) {

    moduleOp.walk([&](mlir::Operation *nestedOp) {
      if (const auto nameLoc = dyn_cast<NameLoc>(nestedOp->getLoc())) {
        if (targetLocs.contains(nameLoc.getName().getValue())) {
          llvm::SmallSet<mlir::Operation *, 4> opSet;
          opSet.insert(nestedOp);
          hoistedOps.push_back(opSet);
        }
      }
    });
  }

  HoistOpSet getResults() { return hoistedOps; }

private:
  HoistOpSet hoistedOps;
};

// Transform pass to hoist specific ops (based on configured analysis pass) into
// a cpu submodule for later independent lowering.
class TTIRHoistTransform
    : public impl::TTIRHoistTransformBase<TTIRHoistTransform> {
public:
  using impl::TTIRHoistTransformBase<
      TTIRHoistTransform>::TTIRHoistTransformBase;

  void runOnOperation() final {
    mlir::ModuleOp moduleOp = getOperation();

    IRRewriter rewriter(&getContext());

    auto loc = moduleOp->getLoc();

    // TODO (#1629): Add logic for automatic vs manual flag here when we
    // have automatic analysis pass.

    // Only create cpu-module etc. if we actually have ops to hoist.
    if (!hoistLocs.hasValue()) {
      return;
    }
    const auto targetLocs = getLocations();
    if (targetLocs.empty()) {
      return;
    }

    TTIRHoistAnalyzeManual analysisPass(moduleOp, targetLocs);
    const auto &hoistOpSets = analysisPass.getResults();

    // Check if a "cpu_module" already exists.
    mlir::ModuleOp cpuModule;
    for (auto &op : moduleOp.getBody()->getOperations()) {
      if (auto module = llvm::dyn_cast<mlir::ModuleOp>(op)) {
        if (module->hasAttr("ttir.cpu_module")) {
          cpuModule = module;
          break;
        }
      }
    }

    // If no CPU module exists, create one.
    if (!cpuModule) {
      rewriter.setInsertionPointToEnd(moduleOp.getBody());
      cpuModule = rewriter.create<mlir::ModuleOp>(loc);
      cpuModule->setAttr("ttir.cpu_module", rewriter.getUnitAttr());
      cpuModule->setAttr(mlir::SymbolTable::getSymbolAttrName(),
                         rewriter.getStringAttr("cpu_module"));
      // try to make cpu module global
      mlir::SymbolTable::setSymbolVisibility(
          cpuModule, mlir::SymbolTable::Visibility::Public);
    }

    for (const auto &opSet : hoistOpSets) {
      assert(opSet.size() == 1 &&
             "currently don't support hoisting multiple instructions at once!");
      hoistOperationToFunction(*opSet.begin(), moduleOp, cpuModule);
    }
  }

protected:
  llvm::SmallSet<llvm::StringRef, 4> getLocations() {
    llvm::SmallSet<llvm::StringRef, 4> locs;
    for (const auto &loc : hoistLocs) {
      locs.insert(loc);
    }
    return locs;
  }
};

} // namespace mlir::tt::ttir
