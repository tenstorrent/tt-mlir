// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include <mlir/IR/Value.h>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRHOISTTRANSFORM
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Hoist const-eval subgraphs to separate funcs pass
//===----------------------------------------------------------------------===//

struct ConstEvalSubgraph {
  // Set of parameters to the original function that this subgraph depends on.
  llvm::SmallSet<mlir::BlockArgument, 4> inputParameters;
  // Ops from the original function that this subgraph contains.
  llvm::SmallVector<mlir::Operation *, 4> ops;
  // Values produced by operations in this subgraph -- this is useful for
  // merging dependent subgraph during analysis
  llvm::SmallSet<mlir::Value, 4> values;
};

class TTIRConstEvalAnalyze {
public:
  using HoistOpSet = llvm::SmallVector<llvm::SmallSet<mlir::Operation *, 4>, 4>;

  TTIRConstEvalAnalyze(func::FuncOp funcOp) { populateHoistOpSet(funcOp); }

  HoistOpSet getResults() { return finalizeHoistSets(); }

private:
  // Analysis rules:
  // 1. Input parameters have known const-eval or non-const-eval state
  // 2. Any op which takes any non-const-eval inputs has non-const-eval outputs,
  //    otherwise it has const-eval outputs
  // 3. Post-process to handle tensor.empty() ops - only include if their users
  //    are within const-eval subgraphs
  void populateHoistOpSet(func::FuncOp funcOp) {
    if (funcOp.isDeclaration()) {
      return;
    }

    auto inputTypesAttr = funcOp->getAttrOfType<mlir::ArrayAttr>("input_types");
    if (!inputTypesAttr) {
      return;
    }

    auto args = funcOp.getArguments();
    llvm::SmallSet<mlir::BlockArgument, 4> constParams;

    // Iterate through both arguments and input types simultaneously
    for (auto [i, arg] : llvm::enumerate(args)) {
      auto typeAttr = inputTypesAttr[i].cast<mlir::StringAttr>();
      StringRef typeStr = typeAttr.getValue();

      // Add to result set if it's a param or constant
      if (typeStr == "#tt.param" || typeStr == "#tt.constant") {
        constParams.insert(arg);
      }
    }

    // Build const-eval subgraphs based on dependency analysis.
    buildConstEvalSubgraphs(funcOp, constParams);
  }

  // Main algorithm to build const-eval subgraphs
  void buildConstEvalSubgraphs(
      func::FuncOp funcOp,
      const llvm::SmallSet<mlir::BlockArgument, 4> &constParams) {
    funcOp.walk([&](mlir::Operation *op) {
      // Creation type ops are only hoisted to subgraphs if a user is found
      // which depends on them, to prevent spuriously hoisting all creation ops.
      if (isCreationOp(op)) {
        continue;
      }
      llvm::SmallSet<size_t, 2> subgraphIdxs;
      llvm::SmallSet<mlir::BlockArgument, 2> inputParams;
      llvm::SmallSet<mlir::Operation, 2> creationOps;
      bool allOperandsConstEval = true;

      for (auto operand : op->getOperands()) {
        if (auto blockArg = operand.dyn_cast<mlir::BlockArgument>()) {
          if (constParams.contains(blockArg)) {
            inputParams.insert(blockArg);
            continue;
          }
        }

        bool constEvalOperand = false;
        for (size_t i = 0; i < constEvalSubgraphs.size(); ++i) {
          const auto &subgraph = constEvalSubgraphs[i];
          if (subgraph.values.contains(operand)) {
            constEvalOperand = true;
            subgraphIdxs.insert(i);
            break;
          }
        }

        if (mlir::Operation defOp =
                operand.getDefiningOp() && isCreationOp(defOp)) {
          creationOps.insert(defOp);
        }

        // If this operand cannot be const-eval'ed, this op cannot be
        // const-eval'ed.
        if (!constEvalOperand) {
          allOperandsConstEval = false;
          break;
        }
      }

      if (!allOperandsConstEval) {
        return;
      }

      llvm::SmallVector<Operation * op, 3> opsToInsert;
      // Ensure any needed creation ops are inserted before the user op.
      for (mlir::Operation creationOp : creationOps) {
        opsToInsert.emplace_back(creationOp);
      }
      opsToInsert.emplace_back(op);

      if (subgraphIdxs.empty()) {
        // This op does not depend on any existing subgraphs, create a new
        // one.
        ConstEvalSubgraph newSubgraph;
        newSubgraph.inputParameters = inputParams;
        for (const auto opToMove : opsToInsert) {
          newSubgraph.ops.insert(opToMove);

          for (auto result : opToMove->getResults()) {
            newSubgraph.values.insert(result);
          }
        }

        constEvalSubgraphs.push_back(newSubgraph);
      } else if (subgraphIdxs.size() == 1) {
        // This op can be added to a single existing subgraph.
        size_t idx = *subgraphIdxs.begin();
        ConstEvalSubgraph &subgraph = constEvalSubgraphs[idx];
        subgraph.inputParameters.insert(inputParams.begin(), inputParams.end());
        for (const auto opToMove : opsToInsert) {
          newSubgraph.ops.insert(opToMove);

          for (auto result : opToMove->getResults()) {
            newSubgraph.values.insert(result);
          }
        }
      } else {
        // This op is connected to multiple subgraphs, must merge them all.

        // Convert set to vector for sorting
        llvm::SmallVector<size_t, 4> allIndices(subgraphIdxs.begin(),
                                                subgraphIdxs.end());

        // Sort indices to find the lowest index.
        std::sort(allIndices.begin(), allIndices.end());

        // Get the lowest index as the target.
        size_t targetIdx = allIndices[0];

        // Add op's param dependencies to target subgraph.
        constEvalSubgraphs[targetIdx].inputParameters.insert(
            inputParams.begin(), inputParams.end());

        for (auto result : op->getResults()) {
          constEvalSubgraphs[targetIdx].values.insert(result);
        }

        // Merge all other subgraphs into the target one, starting from the
        // highest index to avoid invalidating indices during removal.
        for (int i = allIndices.size() - 1; i > 0; --i) {
          mergeSubgraphs(targetIdx, allIndices[i]);
        }
        for (const auto movingOp : opsToInsert) {
          newSubgraph.ops.insert(movingOp);

          for (auto result : movingOp->getResults()) {
            newSubgraph.values.insert(result);
          }
        }
      }
    });
  }

  // Check if an operation is a tensor creation op (tensor.empty, ttir.zeros,
  // etc.)
  bool isCreationOp(mlir::Operation *op) {
    StringRef opName = op->getName().getStringRef();
    return opName == "tensor.empty" || opName == "ttir.zeros" ||
           opName == "ttir.ones" || opName == "ttir.arange";
  }

  // Merge all contents of subgraph at sourceIdx into subgraph at targetIdx +
  // erase source subgraph.
  void mergeSubgraphs(size_t targetIdx, size_t sourceIdx) {
    assert(targetIdx < constEvalSubgraphs.size() &&
           sourceIdx < constEvalSubgraphs.size());

    auto &target = constEvalSubgraphs[targetIdx];
    auto &source = constEvalSubgraphs[sourceIdx];

    target.inputParameters.insert(source.inputParameters.begin(),
                                  source.inputParameters.end());
    target.ops.insert(source.ops.begin(), source.ops.end());
    target.values.insert(source.values.begin(), source.values.end());

    constEvalSubgraphs.erase(constEvalSubgraphs.begin() + sourceIdx);
  }

  llvm::SmallVector<ConstEvalSubgraph, 4> getAnalysisResults() {
    return constEvalSubgraphs;
  }

  // Internal representation of subgraphs
  llvm::SmallVector<ConstEvalSubgraph, 4> constEvalSubgraphs;
};

// Transform pass to hoist specific ops (based on configured analysis pass) into
// a cpu submodule for later independent lowering.
class TTIRConstEvalHoistTransform
    : public impl::TTIRConstEvalHoistTransformBase<
          TTIRConstEvalHoistTransform> {
public:
  using impl::TTIRConstEvalHoistTransformBase<
      TTIRConstEvalHoistTransform>::TTIRConstEvalHoistTransformBase;

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    llvm::SmallVector<func::FuncOp, 4> functionsToProcess;

    // Collect functions that need processing
    module.walk([&](func::FuncOp funcOp) { processFunction(funcOp); });
  }

private:
  // Process a single function for const-eval hoisting
  void processFunction(func::FuncOp funcOp) {
    // Run the analysis to identify const-eval subgraphs
    TTIRConstEvalAnalyze analyzer(funcOp);
    auto constEvalOpSets = analyzer.getResults();

    if (constEvalOpSets.empty()) {
      return; // No const-eval sets found
    }

    // Create new functions for each subgraph
    for (size_t i = 0; i < constEvalOpSets.size(); ++i) {
      auto &opVec = constEvalOpSets[i];

      // Skip empty sets
      if (opVec.empty()) {
        continue;
      }

      // Create a new function for this const-eval subgraph
      createConstEvalFunction(funcOp, opVec, i);
    }
  }

  // Create a new function for a const-eval subgraph and replace the original
  // ops with a call
  void
  createConstEvalFunction(func::FuncOp originalFunc,
                          const llvm::SmallVector<mlir::Operation *, 4> &opVec,
                          size_t subgraphIdx) {
    mlir::MLIRContext *context = &getContext();
    mlir::OpBuilder builder(context);

    // Step 1: Identify inputs and outputs of the subgraph
    llvm::SmallVector<mlir::Value, 4> inputs;
    llvm::SmallVector<mlir::Type, 4> inputTypes;
    llvm::SmallVector<mlir::Value, 4> outputs;
    llvm::SmallVector<mlir::Type, 4> outputTypes;

    collectSubgraphBoundary(opVec, inputs, outputs);

    // Get types for function signature
    for (auto input : inputs) {
      inputTypes.push_back(input.getType());
    }

    for (auto output : outputs) {
      outputTypes.push_back(output.getType());
    }

    // Step 2: Create the new function
    std::string newFuncName = originalFunc.getName().str() + "_const_eval_" +
                              std::to_string(subgraphIdx);
    auto funcType = builder.getFunctionType(inputTypes, outputTypes);

    builder.setInsertionPointToEnd(
        originalFunc->getParentOp()->getRegion(0).getBlock());
    auto newFuncOp = builder.create<func::FuncOp>(originalFunc.getLoc(),
                                                  newFuncName, funcType);

    // Mark the new function as const-eval
    newFuncOp->setAttr("const_eval", builder.getUnitAttr());

    // Copy relevant attributes from the original function
    if (auto ttDevice =
            originalFunc->getAttrOfType<mlir::StringAttr>("tt.device")) {
      newFuncOp->setAttr("tt.device", ttDevice);
    }

    // Step 3: Build the body of the new function
    auto *entryBlock = newFuncOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Create a mapping from original inputs to function arguments
    llvm::DenseMap<mlir::Value, mlir::Value> valueMap;
    for (size_t i = 0; i < inputs.size(); ++i) {
      valueMap[inputs[i]] = entryBlock->getArgument(i);
    }

    // Clone operations into the new function in their original order
    for (auto *op : opVec) {
      // Skip terminator operations
      if (op->hasTrait<mlir::OpTrait::IsTerminator>()) {
        continue;
      }

      // Remap operands
      llvm::SmallVector<mlir::Value, 4> remappedOperands;
      for (auto operand : op->getOperands()) {
        if (valueMap.count(operand)) {
          remappedOperands.push_back(valueMap[operand]);
        } else {
          llvm::errs() << "Error: Operand not found in value map\n";
          return;
        }
      }

      // Clone the operation
      auto *clonedOp = builder.clone(*op);

      // Update operands
      for (size_t i = 0; i < clonedOp->getNumOperands(); ++i) {
        clonedOp->setOperand(i, remappedOperands[i]);
      }

      // Add results to the value map
      for (size_t i = 0; i < clonedOp->getNumResults(); ++i) {
        valueMap[op->getResult(i)] = clonedOp->getResult(i);
      }
    }

    // Create return operation
    llvm::SmallVector<mlir::Value, 4> returnValues;
    for (auto output : outputs) {
      if (valueMap.count(output)) {
        returnValues.push_back(valueMap[output]);
      } else {
        llvm::errs() << "Error: Output not found in value map\n";
        return;
      }
    }

    builder.create<func::ReturnOp>(originalFunc.getLoc(), returnValues);

    // Step 4: Replace the original operations with a call to the new function
    // Insert before the first operation in the subgraph
    builder.setInsertionPoint(opVec.front());
    auto callOp = builder.create<func::CallOp>(
        originalFunc.getLoc(), newFuncName, outputTypes, inputs);

    // Replace uses of original outputs with call results
    for (size_t i = 0; i < outputs.size(); ++i) {
      outputs[i].replaceAllUsesWith(callOp.getResult(i));
    }

    // Step 5: Remove the original operations (in reverse order to handle
    // dependencies)
    for (auto it = opVec.rbegin(); it != opVec.rend(); ++it) {
      (*it)->erase();
    }
  }

  // Collect inputs and outputs of the subgraph
  void
  collectSubgraphBoundary(const llvm::SmallVector<mlir::Operation *, 4> &opVec,
                          llvm::SmallVector<mlir::Value, 4> &inputs,
                          llvm::SmallVector<mlir::Value, 4> &outputs) {
    // Create a set of operations for quick lookup
    llvm::SmallSet<mlir::Operation *, 8> opSet;
    for (auto *op : opVec) {
      opSet.insert(op);
    }

    // Track values defined within the subgraph
    llvm::SmallSet<mlir::Value, 8> subgraphValues;

    // Add all results from operations in the subgraph
    for (auto *op : opVec) {
      for (auto result : op->getResults()) {
        subgraphValues.insert(result);
      }
    }

    // Collect inputs: operands used by subgraph ops but not defined within the
    // subgraph
    llvm::SmallSet<mlir::Value, 8> inputSet;
    for (auto *op : opVec) {
      for (auto operand : op->getOperands()) {
        if (!subgraphValues.contains(operand)) {
          // This is an input from outside the subgraph
          inputSet.insert(operand);
        }
      }
    }

    // Convert the set to a vector
    inputs.assign(inputSet.begin(), inputSet.end());

    // Collect outputs: values defined in the subgraph that are used outside
    llvm::SmallSet<mlir::Value, 8> outputSet;
    for (auto *op : opVec) {
      for (auto result : op->getResults()) {
        for (auto &use : result.getUses()) {
          mlir::Operation *user = use.getOwner();

          // Check if the user is outside the subgraph
          if (!opSet.contains(user)) {
            outputSet.insert(result);
            break; // Once we know it's an output, no need to check other uses
          }
        }
      }
    }

    // Convert the set to a vector
    outputs.assign(outputSet.begin(), outputSet.end());
  }
};

} // namespace mlir::tt::ttir
