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
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::transforms {

#define GEN_PASS_DEF_CONSTEVALHOISTTRANSFORM
#include "ttmlir/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Hoist const-eval subgraphs to separate funcs pass
//===----------------------------------------------------------------------===//

struct ConstEvalSubgraph {
  // Set of parameters to the original function that this subgraph depends on.
  llvm::SmallPtrSet<mlir::BlockArgument, 4> inputParameters;
  // Ops from the original function that this subgraph contains.
  llvm::SmallVector<mlir::Operation *, 4> ops;
  // Values produced by operations in this subgraph -- this is useful for
  // merging dependent subgraph during analysis
  llvm::SmallPtrSet<mlir::Value, 4> values;
};

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
template <typename IgnoreChecker = OpTypeChecker<>>
class ConstEvalAnalyze {
public:
  ConstEvalAnalyze(func::FuncOp funcOp) { populateHoistOpSet(funcOp); }

  llvm::SmallVector<ConstEvalSubgraph, 4> getAnalysisResults() {
    return constEvalSubgraphs;
  }

private:
  IgnoreChecker ignoreChecker;

private:
  void populateHoistOpSet(func::FuncOp funcOp) {
    if (funcOp.isDeclaration()) {
      return;
    }

    auto args = funcOp.getArguments();
    llvm::SmallPtrSet<mlir::BlockArgument, 4> constParams;

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
          constParams.insert(arg);
        }
      }
    }
    // Build const-eval subgraphs based on dependency analysis.
    buildConstEvalSubgraphs(funcOp, constParams);
  }

  // Main algorithm to build const-eval subgraphs:
  // 1. We have a vector of sets of ops, each representing 1 subgraph
  // 2. For every op in the original graph, select one of 3 cases
  //   a. If an op depends on any non-const ops, it cannot be const-eval'ed
  //   b. If an op depends on only const input params (to the func) or creation
  // ops, it becomes a new const-eval subgraph (along with its creation ops)
  //   c. If an op depends on output from an op in any existing subgraphs, it
  //   must be merged into these subgraphs (if there are multiple such
  //   subgraphs, they must be merged).
  // 3. Result is list of all disjoint const-eval subgraphs (as sets of ops)
  void buildConstEvalSubgraphs(
      func::FuncOp funcOp,
      const llvm::SmallPtrSet<mlir::BlockArgument, 4> &constParams) {

    // Iterate over all blocks in the function
    for (auto &block : funcOp.getBlocks()) {
      // Iterate over all operations in the block
      for (auto &opRef : block.getOperations()) {
        Operation *op = &opRef;
        // Creation type ops are only hoisted to subgraphs if a user is found
        // which depends on them, to prevent spuriously hoisting all creation
        // ops. We also completely ignore certain ops.
        if (ignoreChecker.isOfType(op) || isCreationOp(op)) {
          continue;
        }

        llvm::SmallSet<size_t, 2> subgraphIdxs;
        llvm::SmallPtrSet<mlir::BlockArgument, 2> inputParams;
        llvm::SmallPtrSet<mlir::Operation *, 2> creationOps;
        bool allOperandsConstEval = true;

        for (auto operand : op->getOperands()) {
          // Case 1: this operand is an const-eval param.
          if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
            if (constParams.contains(blockArg)) {
              inputParams.insert(blockArg);
              continue;
            }
          }

          // Case 2: this operand is result of an existing op in a const-eval
          // subgraph.
          bool constEvalOperand = false;
          for (size_t i = 0; i < constEvalSubgraphs.size(); ++i) {
            const auto &subgraph = constEvalSubgraphs[i];
            if (subgraph.values.contains(operand)) {
              constEvalOperand = true;
              subgraphIdxs.insert(i);
              break;
            }
          }

          if (!constEvalOperand) {
            // Case 3: this operand is an intermediate tensor from a neutral
            // creation op.
            if (mlir::Operation *defOp = operand.getDefiningOp();
                defOp != nullptr && !ignoreChecker.isOfType(op) &&
                isCreationOp(defOp)) {
              creationOps.insert(defOp);
              continue;
            }
            // If all 3 cases fail, this operand is not const-eval, so this
            // operation cannot be const-eval'ed.
            allOperandsConstEval = false;
            break;
          }
        }

        if (!allOperandsConstEval) {
          continue;
        }

        // Ensure any needed creation ops are inserted before the user op.
        llvm::SmallVector<mlir::Operation *, 2> opsToInsert(creationOps.begin(),
                                                            creationOps.end());
        opsToInsert.emplace_back(op);

        if (subgraphIdxs.empty()) {
          // This op does not depend on any existing subgraphs, create a new
          // one.
          ConstEvalSubgraph newSubgraph;
          newSubgraph.inputParameters.insert(inputParams.begin(),
                                             inputParams.end());
          for (auto *opToMove : opsToInsert) {
            newSubgraph.ops.emplace_back(opToMove);
            for (auto result : opToMove->getResults()) {
              newSubgraph.values.insert(result);
            }
          }

          constEvalSubgraphs.push_back(newSubgraph);
        } else if (subgraphIdxs.size() == 1) {
          // This op can be added to a single existing subgraph.
          size_t idx = *subgraphIdxs.begin();

          ConstEvalSubgraph &subgraph = constEvalSubgraphs[idx];
          subgraph.inputParameters.insert(inputParams.begin(),
                                          inputParams.end());
          for (auto *opToMove : opsToInsert) {
            subgraph.ops.emplace_back(opToMove);

            for (auto result : opToMove->getResults()) {
              subgraph.values.insert(result);
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

          ConstEvalSubgraph &targetSubgraph = constEvalSubgraphs[targetIdx];

          // Add op's param dependencies to target subgraph.
          targetSubgraph.inputParameters.insert(inputParams.begin(),
                                                inputParams.end());

          for (auto result : op->getResults()) {
            targetSubgraph.values.insert(result);
          }

          // Merge all other subgraphs into the target one, starting from the
          // highest index to avoid invalidating indices during removal.
          for (int i = allIndices.size() - 1; i > 0; --i) {
            mergeSubgraphs(targetIdx, allIndices[i]);
          }

          for (auto *const movingOp : opsToInsert) {
            targetSubgraph.ops.emplace_back(movingOp);

            for (auto result : movingOp->getResults()) {
              targetSubgraph.values.insert(result);
            }
          }
        }
      }
    }
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

  // Merge all contents of subgraph at sourceIdx into subgraph at targetIdx +
  // erase source subgraph.
  void mergeSubgraphs(size_t targetIdx, size_t sourceIdx) {
    assert(targetIdx < constEvalSubgraphs.size() &&
           sourceIdx < constEvalSubgraphs.size());

    auto &target = constEvalSubgraphs[targetIdx];
    auto &source = constEvalSubgraphs[sourceIdx];

    target.inputParameters.insert(source.inputParameters.begin(),
                                  source.inputParameters.end());
    target.ops.append(source.ops.begin(), source.ops.end());
    target.values.insert(source.values.begin(), source.values.end());

    constEvalSubgraphs.erase(constEvalSubgraphs.begin() + sourceIdx);
  }

  // Internal representation of subgraphs
  llvm::SmallVector<ConstEvalSubgraph, 4> constEvalSubgraphs;
};
ConstEvalAnalyze(func::FuncOp) -> ConstEvalAnalyze<OpTypeChecker<>>;

// Transform pass to hoist specific ops (based on configured analysis pass)
// into a cpu submodule for later independent lowering.
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
    // Run the analysis to identify const-eval subgraphs
    ConstEvalAnalyze<IgnoreChecker> analyzer(funcOp);
    auto constEvalOpSets = analyzer.getAnalysisResults();

    if (constEvalOpSets.empty()) {
      return; // No const-eval sets found
    }

    // Create new functions for each subgraph
    for (size_t i = 0; i < constEvalOpSets.size(); ++i) {
      auto &subgraph = constEvalOpSets[i];

      // Create a new function for this const-eval subgraph
      createConstEvalFunction(funcOp, subgraph, i);
    }
  }

  // Create a new function for a const-eval subgraph and replace the original
  // ops with a call
  void createConstEvalFunction(func::FuncOp originalFunc,
                               ConstEvalSubgraph &subgraph,
                               size_t subgraphIdx) {
    mlir::MLIRContext *context = &this->getContext();
    mlir::OpBuilder builder(context);

    // Identify all outputs of the subgraph.
    llvm::SmallVector<mlir::Value, 4> inputs(subgraph.inputParameters.begin(),
                                             subgraph.inputParameters.end());
    llvm::SmallVector<mlir::Type, 4> inputTypes;
    llvm::SmallVector<mlir::Value, 4> outputs;
    llvm::SmallVector<mlir::Type, 4> outputTypes;

    collectSubgraphBoundary(subgraph.ops, outputs);

    // Get types for function signature.
    for (auto input : inputs) {
      inputTypes.push_back(input.getType());
    }

    for (auto output : outputs) {
      outputTypes.push_back(output.getType());
    }

    // Create the new function.
    std::string newFuncName = originalFunc.getName().str() + "_const_eval_" +
                              std::to_string(subgraphIdx);
    auto funcType = builder.getFunctionType(inputTypes, outputTypes);

    mlir::ModuleOp moduleOp =
        dyn_cast<mlir::ModuleOp>(originalFunc->getParentOp());
    assert(moduleOp);
    builder.setInsertionPointToEnd(moduleOp.getBody());
    auto newFuncOp = builder.create<func::FuncOp>(originalFunc.getLoc(),
                                                  newFuncName, funcType);
    // Mark the new function as const-eval.
    newFuncOp->setAttr("const_eval", builder.getUnitAttr());

    // Copy relevant attributes from the original function.
    if (auto ttDevice =
            originalFunc->getAttrOfType<mlir::StringAttr>("tt.device")) {
      newFuncOp->setAttr("tt.device", ttDevice);
    }

    // Build the body of the new function.
    auto *entryBlock = newFuncOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Create a mapping from original inputs to function arguments.
    llvm::DenseMap<mlir::Value, mlir::Value> valueMap;
    for (size_t i = 0; i < inputs.size(); ++i) {
      valueMap[inputs[i]] = entryBlock->getArgument(i);
    }

    // Clone operations into the new function in their original order.
    for (auto *op : subgraph.ops) {
      if (op->hasTrait<mlir::OpTrait::IsTerminator>()) {
        continue;
      }

      llvm::SmallVector<mlir::Value, 4> remappedOperands;
      for (auto operand : op->getOperands()) {
        if (valueMap.count(operand)) {
          remappedOperands.push_back(valueMap[operand]);
        } else {
          return;
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
        return;
      }
    }

    builder.create<func::ReturnOp>(originalFunc.getLoc(), returnValues);

    auto &originalEntryBlock = originalFunc.getBody().front();
    // Manually order LoadCachedOp as first n ops in original func--we may have
    // folded some creation ops into the subgraph, so we need to ensure these
    // ops come before existing ops.
    auto iter = originalEntryBlock.begin();
    std::advance(iter, subgraphIdx);
    assert(iter != originalEntryBlock.end());
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

    // Remove the original operations (in reverse order to handle
    // dependencies).
    for (auto it = subgraph.ops.rbegin(); it != subgraph.ops.rend(); ++it) {
      (*it)->erase();
    }
  }

  // Collect inputs and outputs of the subgraph.
  void
  collectSubgraphBoundary(const llvm::SmallVector<mlir::Operation *, 4> &opVec,
                          llvm::SmallVector<mlir::Value, 4> &outputs) {
    // Create a set of operations for quick lookup.
    llvm::SmallPtrSet<mlir::Operation *, 8> opSet;
    for (auto *op : opVec) {
      opSet.insert(op);
    }

    // Collect outputs: values defined in the subgraph that are used outside.
    for (auto *op : opVec) {
      for (auto result : op->getResults()) {
        for (auto &use : result.getUses()) {
          mlir::Operation *user = use.getOwner();
          // Check if the user is outside the subgraph.
          if (!opSet.contains(user)) {
            outputs.emplace_back(result);
            break;
          }
        }
      }
    }
  }
};

template <typename... OpTypes>
std::unique_ptr<Pass> createConstEvalHoistTransformWithIgnoreTypes() {
  return std::make_unique<ConstEvalHoistTransform<OpTypeChecker<OpTypes...>>>();
}

std::unique_ptr<Pass> createConstEvalHoistTransformNoIgnoreTypes() {
  return std::make_unique<ConstEvalHoistTransform<>>();
}

} // namespace mlir::tt::transforms
