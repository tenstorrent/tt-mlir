// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TTOps.h"
#include "ttmlir/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::transforms {

#define GEN_PASS_DEF_CONSTEVALHOISTTRANSFORM
#include "ttmlir/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Hoist const-eval subgraphs to separate funcs pass
//===----------------------------------------------------------------------===//

namespace {

static bool isBlockArgument(mlir::Value value) {
  return mlir::isa<mlir::BlockArgument>(value);
}

static llvm::SmallVector<mlir::Value, 4>
getInputs(const llvm::ArrayRef<mlir::Value> &subgraph) {
  llvm::SmallVector<mlir::Value, 4> inputs;
  for (auto value : subgraph) {
    if (isBlockArgument(value)) {
      inputs.push_back(value);
    }
  }

  return inputs;
}

static llvm::SmallVector<mlir::Operation *, 4>
getOperations(llvm::ArrayRef<mlir::Value> subgraph) {
  llvm::SmallVector<mlir::Operation *, 4> ops;
  llvm::SmallPtrSet<mlir::Operation *, 4> opSet;
  for (auto value : subgraph) {
    if (isBlockArgument(value)) {
      continue;
    }

    Operation *op = value.getDefiningOp();
    if (opSet.find(op) == opSet.end()) {
      ops.push_back(op);
      opSet.insert(op);
    }
  }

  return ops;
}

// Filter out the outputs of the graph
static llvm::SmallVector<mlir::Value, 4>
getOutputs(const llvm::ArrayRef<mlir::Value> graph) {
  llvm::SmallVector<mlir::Operation *, 4> ops = getOperations(graph);
  llvm::SmallPtrSet<mlir::Operation *, 4> opSet(ops.begin(), ops.end());
  llvm::SmallVector<mlir::Value, 4> outputs;

  for (auto value : graph) {
    if (isBlockArgument(value)) {
      continue;
    }

    bool hasExternaluser = llvm::any_of(value.getUsers(), [&](Operation *user) {
      return !opSet.contains(user);
    });

    if (hasExternaluser) {
      outputs.push_back(value);
    }
  }

  return outputs;
}
} // namespace

namespace {
// Helper class to wrap variadic list of ops and get if a given op is one of
// these types.
// OpTypeChecker - template for checking if an operation matches one of the
// specified types
template <typename... OpTypes>
class OpTypeChecker {
public:
  // Check if the given operation is of any of the specified types
  bool isOfType(mlir::Operation *op) const { return isOfTypeImpl(op); }

private:
  // Implementation method - use fold expression for C++17
  bool isOfTypeImpl(mlir::Operation *op) const {
    return (... || (op && mlir::isa<OpTypes>(op)));
  }
};

// Specialization for empty parameter pack
template <>
class OpTypeChecker<> {
public:
  bool isOfType(mlir::Operation *op) const { return false; }
};
} // namespace

namespace {
// Analyzer class to detect const-eval subgraphs in a given FuncOp.
// Template argument allows user to specify set of ops to not consider hoisting.
template <typename IgnoreChecker = OpTypeChecker<>>
class ConstEvalAnalyze {
public:
  ConstEvalAnalyze(func::FuncOp funcOp) { populateHoistOpSet(funcOp); }

  const llvm::SmallVector<Value, 4> &getAnalysisResults() const {
    return constEvalValuesVector;
  }

private:
  IgnoreChecker ignoreChecker;

private:
  void populateHoistOpSet(func::FuncOp funcOp) {
    if (funcOp.isDeclaration()) {
      return;
    }

    auto args = funcOp.getArguments();

    // Iterate through arguments and check their tt.argument_type attributes
    for (auto arg : args) {
      auto argAttrs = funcOp.getArgAttrDict(arg.getArgNumber());
      if (!argAttrs) {
        continue;
      }
      auto typeAttr = argAttrs.get("tt.argument_type");
      if (!typeAttr) {
        continue;
      }

      // Cast to ArgumentTypeAttr
      if (auto enumAttr =
              mlir::dyn_cast<mlir::tt::ArgumentTypeAttr>(typeAttr)) {
        // Get the enum value
        mlir::tt::ArgumentType attrValue = enumAttr.getValue();

        // Compare with Parameter and Constant
        if (attrValue == mlir::tt::ArgumentType::Parameter ||
            attrValue == mlir::tt::ArgumentType::Constant) {
          insert(arg);
        }
      }
    }
    // Build const-eval subgraphs based on dependency analysis.
    buildConstEvalSubgraphs(funcOp);
  }

  void buildConstEvalSubgraphs(func::FuncOp funcOp) {

    for (auto &block : funcOp.getBlocks()) {
      for (auto &opRef : block.getOperations()) {
        if (canConstEvalOp(&opRef)) {
          llvm::for_each(opRef.getResults(),
                         [&](Value result) { insert(result); });
        }
      }
    }
  }

  // Checks if an operation can be const evaluated
  // An operation can be const evaluated if all its operands are part of the
  // same subgraph
  bool canConstEvalOp(Operation *op) {
    // If op is of type to be ignored, return false
    if (ignoreChecker.isOfType(op)) {
      return false;
    }

    // If it's creation op (zeros/ones/constant) return true
    if (isCreationOp(op)) {
      return true;
    }

    // If all remaining operands are in `constEvalValuesSet`, return true
    for (auto operand : op->getOperands()) {
      if (constEvalValuesSet.find(operand) == constEvalValuesSet.end()) {
        return false;
      }
    }

    return true;
  }

  // Check if an operation is a tensor creation op (no inputs, output is a
  // tensor)
  bool isCreationOp(mlir::Operation *op) {
    assert(op != nullptr);
    // Check if the operation has no operands
    if (op->getNumOperands() != 0) {
      return false;
    }

    // Check if the operation has at least one result
    if (op->getNumResults() == 0) {
      return false;
    }

    // Check if any result is a tensor type
    for (auto result : op->getResults()) {
      if (isa<mlir::TensorType>(result.getType())) {
        return true;
      }
    }

    return false;
  }

  void insert(mlir::Value value) {
    constEvalValuesSet.insert(value);
    constEvalValuesVector.push_back(value);
  }

  llvm::SmallPtrSet<mlir::Value, 4> constEvalValuesSet;
  llvm::SmallVector<mlir::Value, 4> constEvalValuesVector;
};
} // namespace

namespace {
// Transform pass to hoist const-eval subgraphs into separate funcs, invoked
// w/ tt.load_cached ops.
template <typename IgnoreChecker = OpTypeChecker<>>
class ConstEvalHoistTransform : public impl::ConstEvalHoistTransformBase<
                                    ConstEvalHoistTransform<IgnoreChecker>> {
public:
  using impl::ConstEvalHoistTransformBase<
      ConstEvalHoistTransform<IgnoreChecker>>::ConstEvalHoistTransformBase;

  void runOnOperation() final {
    mlir::ModuleOp module = this->getOperation();
    llvm::SmallVector<func::FuncOp, 4> functionsToProcess;

    // Collect functions that need processing
    module.walk([&](func::FuncOp funcOp) { processFunction(funcOp); });
  }

private:
  // Process a single function for const-eval hoisting
  void processFunction(func::FuncOp funcOp) {
    if (funcOp->getAttr("const_eval")) {
      return;
    }

    // Run the analysis to identify const-eval subgraphs
    ConstEvalAnalyze<IgnoreChecker> analyzer(funcOp);
    auto constEvalOpSet = analyzer.getAnalysisResults();

    if (getOperations(constEvalOpSet).empty()) {
      return;
    }

    createConstEvalFunction(funcOp, constEvalOpSet);
  }

  // Create a new function for a const-eval subgraph and replace the original
  // ops with a call
  void createConstEvalFunction(func::FuncOp originalFunc,
                               const llvm::ArrayRef<mlir::Value> subgraph) {
    mlir::MLIRContext *context = &this->getContext();
    mlir::OpBuilder builder(context);

    llvm::SmallVector<mlir::Value, 4> inputs = getInputs(subgraph);
    llvm::SmallVector<mlir::Value, 4> outputs = getOutputs(subgraph);

    llvm::SmallVector<mlir::Type, 4> inputTypes;
    // Get types for function signature.
    for (auto input : inputs) {
      inputTypes.push_back(input.getType());
    }

    llvm::SmallVector<mlir::Type, 4> outputTypes;
    for (auto output : outputs) {
      outputTypes.push_back(output.getType());
    }

    // Create the new function.
    std::string newFuncName = originalFunc.getName().str() + "_const_eval";
    auto funcType = builder.getFunctionType(inputTypes, outputTypes);

    mlir::ModuleOp moduleOp =
        dyn_cast<mlir::ModuleOp>(originalFunc->getParentOp());
    builder.setInsertionPointToEnd(moduleOp.getBody());
    auto newFuncOp = builder.create<func::FuncOp>(originalFunc.getLoc(),
                                                  newFuncName, funcType);
    // Mark the new function as const-eval.
    newFuncOp->setAttr("const_eval", builder.getUnitAttr());

    // Build the body of the new function.
    auto *entryBlock = newFuncOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Create a mapping from original inputs to function arguments.
    llvm::DenseMap<mlir::Value, mlir::Value> valueMap;
    for (size_t i = 0; i < inputs.size(); ++i) {
      valueMap.insert({inputs[i], entryBlock->getArgument(i)});
    }

    llvm::SmallVector<mlir::Operation *, 4> opsToHoist =
        getOperations(subgraph);

    // Clone operations into the new function in their original order.
    for (auto *op : opsToHoist) {
      llvm::SmallVector<mlir::Value, 4> remappedOperands;
      for (auto operand : op->getOperands()) {
        if (valueMap.count(operand)) {
          remappedOperands.push_back(valueMap[operand]);
        } else {
          assert(false && "Operand not found in valueMap");
        }
      }

      auto *clonedOp = builder.clone(*op);

      // Update operands to use new func's params.
      for (size_t i = 0; i < clonedOp->getNumOperands(); ++i) {
        clonedOp->setOperand(i, remappedOperands[i]);
      }

      // Map original values to new results.
      for (size_t i = 0; i < clonedOp->getNumResults(); ++i) {
        valueMap[op->getResult(i)] = clonedOp->getResult(i);
      }
    }

    // Create return operation.
    llvm::SmallVector<mlir::Value, 4> returnValues;
    for (auto output : outputs) {
      if (valueMap.count(output)) {
        returnValues.push_back(valueMap[output]);
      } else {
        assert(false && "Output not found in valueMap");
      }
    }

    builder.create<func::ReturnOp>(originalFunc.getLoc(), returnValues);

    auto &originalEntryBlock = originalFunc.getBody().front();
    // // Manually order LoadCachedOp as first n ops in original func--we may
    // // have folded some creation ops into the subgraph, so we need to ensure
    // // these ops come before existing ops.
    auto iter = originalEntryBlock.begin();
    builder.setInsertionPoint(&*iter);
    auto calleeAttr =
        mlir::SymbolRefAttr::get(builder.getContext(), newFuncName);

    // Create the LoadCachedOp with the correct argument order
    auto callOp =
        builder.create<tt::LoadCachedOp>(originalFunc.getLoc(), // Location
                                         outputTypes,           // Result types
                                         calleeAttr, // Callee symbol reference
                                         inputs      // Input values
        );

    // Replace uses of original outputs with call results.
    for (size_t i = 0; i < outputs.size(); ++i) {
      outputs[i].replaceAllUsesWith(callOp.getResult(i));
    }

    for (auto op = opsToHoist.rbegin(); op != opsToHoist.rend(); ++op) {
      (*op)->erase();
    }
  }
};
} // namespace

template <typename... OpTypes>
std::unique_ptr<Pass> createConstEvalHoistTransformWithIgnoreTypes() {
  return std::make_unique<ConstEvalHoistTransform<OpTypeChecker<OpTypes...>>>();
}

std::unique_ptr<Pass> createConstEvalHoistTransformNoIgnoreTypes() {
  return std::make_unique<ConstEvalHoistTransform<>>();
}

} // namespace mlir::tt::transforms
