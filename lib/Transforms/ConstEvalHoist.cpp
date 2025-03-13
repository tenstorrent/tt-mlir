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
  // Analysis rules:
  // 1. Input parameters have known const-eval or non-const-eval state
  // 2. Any op which takes any non-const-eval inputs has non-const-eval outputs,
  //    otherwise it has const-eval outputs
  // 3. Creation ops (with no inputs) are a special case: we include them in a
  // subgraph iff they are used in that subgraph (cannot form their own
  // subgraphs)
  void populateHoistOpSet(func::FuncOp funcOp) {
    if (funcOp.isDeclaration()) {
      return;
    }

    auto args = funcOp.getArguments();
    llvm::SmallPtrSet<mlir::BlockArgument, 4> constParams;
    llvm::errs() << "parsing args\n";

    // Iterate through arguments and check their tt.argument_type attributes
    for (auto arg : args) {
      auto argAttrs = funcOp.getArgAttrDict(arg.getArgNumber());
      if (!argAttrs)
        continue;
      auto typeAttr = argAttrs.get("tt.argument_type");
      if (!typeAttr)
        continue;

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
    llvm::errs() << "parsed args\n";

    llvm::errs() << "constParams contains " << constParams.size()
                 << " arguments:\n";
    for (const auto &arg : constParams) {
      llvm::errs() << "  Arg #" << arg.getArgNumber()
                   << " type: " << arg.getType() << "\n";

      // If you want to print any attributes associated with the argument
      if (auto argAttrs = funcOp.getArgAttrDict(arg.getArgNumber())) {
        llvm::errs() << "    with attributes: ";
        argAttrs.print(llvm::errs());
        llvm::errs() << "\n";
      }
    }

    // Build const-eval subgraphs based on dependency analysis.
    buildConstEvalSubgraphs(funcOp, constParams);
  }

  // Main algorithm to build const-eval subgraphs
  void buildConstEvalSubgraphs(
      func::FuncOp funcOp,
      const llvm::SmallPtrSet<mlir::BlockArgument, 4> &constParams) {
    llvm::errs() << "walking ops\n";

    // Iterate over all blocks in the function
    for (auto &block : funcOp.getBlocks()) {
      // Iterate over all operations in the block
      for (auto &opRef : block.getOperations()) {
        Operation *op = &opRef;
        llvm::errs() << "processing " << op->getName() << "\n";
        // Creation type ops are only hoisted to subgraphs if a user is found
        // which depends on them, to prevent spuriously hoisting all creation
        // ops. We also completely ignore certain ops.
        if (ignoreChecker.isOfType(op) || isCreationOp(op)) {
          llvm::errs() << "failed 1st check\n";
          continue;
        }

        llvm::SmallSet<size_t, 2> subgraphIdxs;
        llvm::SmallPtrSet<mlir::BlockArgument, 2> inputParams;
        llvm::SmallPtrSet<mlir::Operation *, 2> creationOps;
        bool allOperandsConstEval = true;

        for (auto operand : op->getOperands()) {
          if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
            if (constParams.contains(blockArg)) {
              llvm::errs() << "operand was const param\n";
              inputParams.insert(blockArg);
              continue;
            }
          }

          if (mlir::Operation *defOp = operand.getDefiningOp();
              defOp != nullptr && !ignoreChecker.isOfType(op) &&
              isCreationOp(defOp)) {
            llvm::errs() << "operand was creationOp\n";
            creationOps.insert(defOp);
            continue;
          }

          bool constEvalOperand = false;
          for (size_t i = 0; i < constEvalSubgraphs.size(); ++i) {
            const auto &subgraph = constEvalSubgraphs[i];
            if (subgraph.values.contains(operand)) {
              llvm::errs() << "operand was const-eval from another subgraph\n";
              constEvalOperand = true;
              subgraphIdxs.insert(i);
              break;
            }
          }

          // If this operand cannot be const-eval'ed, this op cannot be
          // const-eval'ed.
          if (!constEvalOperand) {
            allOperandsConstEval = false;
            break;
          }
        }

        if (!allOperandsConstEval) {
          llvm::errs() << "failed 2nd check\n";
          continue;
        }

        // Ensure any needed creation ops are inserted before the user op.
        llvm::SmallVector<mlir::Operation *, 2> opsToInsert(creationOps.begin(),
                                                            creationOps.end());
        opsToInsert.emplace_back(op);

        if (subgraphIdxs.empty()) {
          // This op does not depend on any existing subgraphs, create a new
          // one.
          llvm::errs() << "creating new subgraph " << constEvalSubgraphs.size()
                       << "for op: " << op->getName() << "\n";
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
          llvm::errs() << "appending op: " << op->getName()
                       << " to existing subgraph " << idx << "\n";

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
          llvm::errs() << "op: " << op->getName()
                       << " requires merging 2+ subgraphs\n";
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

          for (const auto movingOp : opsToInsert) {
            targetSubgraph.ops.emplace_back(movingOp);

            for (auto result : movingOp->getResults()) {
              targetSubgraph.values.insert(result);
            }
          }
        }
      }
    }
    llvm::errs() << "walked ops\n";
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
    llvm::errs() << "merging subgraph " << sourceIdx << "into " << targetIdx
                 << "\n";
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
    llvm::errs() << "analyzing func\n";
    auto constEvalOpSets = analyzer.getAnalysisResults();
    llvm::errs() << "analyzed func, found " << constEvalOpSets.size()
                 << " subgraphs \n";

    if (constEvalOpSets.empty()) {
      return; // No const-eval sets found
    }

    llvm::errs() << "creating subgraphs\n";

    // Create new functions for each subgraph
    for (size_t i = 0; i < constEvalOpSets.size(); ++i) {
      auto &subgraph = constEvalOpSets[i];

      // Create a new function for this const-eval subgraph
      createConstEvalFunction(funcOp, subgraph, i);
    }
    llvm::errs() << "created subgraphs\n";
  }

  // Create a new function for a const-eval subgraph and replace the original
  // ops with a call
  void createConstEvalFunction(func::FuncOp originalFunc,
                               ConstEvalSubgraph &subgraph,
                               size_t subgraphIdx) {
    mlir::MLIRContext *context = &this->getContext();
    mlir::OpBuilder builder(context);

    // Step 1: Identify inputs and outputs of the subgraph
    llvm::SmallVector<mlir::Value, 4> inputs;
    llvm::SmallVector<mlir::Type, 4> inputTypes;
    llvm::SmallVector<mlir::Value, 4> outputs;
    llvm::SmallVector<mlir::Type, 4> outputTypes;

    llvm::errs() << "collecting subgraph boundaries\n";

    collectSubgraphBoundary(subgraph.ops, inputs, outputs);

    llvm::errs() << "collected subgraph boundaries\n";

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

    llvm::errs() << "creating new func\n";

    mlir::ModuleOp moduleOp =
        dyn_cast<mlir::ModuleOp>(originalFunc->getParentOp());
    assert(moduleOp);
    builder.setInsertionPointToEnd(moduleOp.getBody());
    auto newFuncOp = builder.create<func::FuncOp>(originalFunc.getLoc(),
                                                  newFuncName, funcType);
    llvm::errs() << "created funcOp itself\n";
    moduleOp.dump();
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

    llvm::errs() << "cloning " << subgraph.ops.size() << " ops in subgraph\n";
    // Clone operations into the new function in their original order
    for (auto *op : subgraph.ops) {
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
      llvm::errs() << "cloning: ";
      clonedOp->dump();

      // Update operands
      for (size_t i = 0; i < clonedOp->getNumOperands(); ++i) {
        clonedOp->setOperand(i, remappedOperands[i]);
      }

      // Add results to the value map
      for (size_t i = 0; i < clonedOp->getNumResults(); ++i) {
        valueMap[op->getResult(i)] = clonedOp->getResult(i);
      }
    }
    llvm::errs() << "cloned ops into funcOp\n";
    moduleOp.dump();

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

    llvm::errs() << "created return values\n";
    moduleOp.dump();

    builder.create<func::ReturnOp>(originalFunc.getLoc(), returnValues);

    // Step 4: Replace the original operations with a call to the new function
    // Always insert

    auto &originalEntryBlock = originalFunc.getBody().front();
    // Manually order LoadCachedOp as first n ops in original func.
    auto iter = originalEntryBlock.begin();
    std::advance(iter, subgraphIdx);
    assert(iter != originalEntryBlock.end());

    // Set insertion point
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
    llvm::errs() << "created tt.load_cached\n";

    // Replace uses of original outputs with call results
    for (size_t i = 0; i < outputs.size(); ++i) {
      outputs[i].replaceAllUsesWith(callOp.getResult(i));
    }

    // Step 5: Remove the original operations (in reverse order to handle
    // dependencies)
    for (auto it = subgraph.ops.rbegin(); it != subgraph.ops.rend(); ++it) {
      (*it)->erase();
    }
  }

  // Collect inputs and outputs of the subgraph
  void
  collectSubgraphBoundary(const llvm::SmallVector<mlir::Operation *, 4> &opVec,
                          llvm::SmallVector<mlir::Value, 4> &inputs,
                          llvm::SmallVector<mlir::Value, 4> &outputs) {
    // Create a set of operations for quick lookup
    llvm::SmallPtrSet<mlir::Operation *, 8> opSet;
    for (auto *op : opVec) {
      opSet.insert(op);
    }

    // Track values defined within the subgraph
    llvm::SmallPtrSet<mlir::Value, 8> subgraphValues;

    // Add all results from operations in the subgraph
    for (auto *op : opVec) {
      for (auto result : op->getResults()) {
        subgraphValues.insert(result);
      }
    }

    // Collect inputs: operands used by subgraph ops but not defined within
    // the subgraph
    llvm::SmallPtrSet<mlir::Value, 8> inputSet;
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
    llvm::SmallPtrSet<mlir::Value, 8> outputSet;
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

template <typename... OpTypes>
std::unique_ptr<Pass> createConstEvalHoistTransformWithIgnoreTypes() {
  return std::make_unique<ConstEvalHoistTransform<OpTypeChecker<OpTypes...>>>();
}

std::unique_ptr<Pass> createConstEvalHoistTransformNoIgnoreTypes() {
  return std::make_unique<ConstEvalHoistTransform<>>();
}

} // namespace mlir::tt::transforms
