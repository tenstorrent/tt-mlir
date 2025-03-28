// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TTOps.h"
#include "ttmlir/Dialect/TT/IR/TTTraits.h"
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
#include <mlir/IR/BuiltinAttributes.h>

namespace mlir::tt::transforms {

#define GEN_PASS_DEF_CONSTEVALHOISTTRANSFORM
#include "ttmlir/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Hoist const-eval subgraphs to separate funcs pass
//===----------------------------------------------------------------------===//

namespace {
struct ConstEvalSubgraph {
  // Set of parameters to the original function that this subgraph depends on.
  llvm::SmallPtrSet<mlir::BlockArgument, 4> inputParameters;
  // Ops from the original function that this subgraph contains.
  llvm::SmallVector<mlir::Operation *, 4> ops;

  void insert(SmallPtrSet<BlockArgument, 4> inputParameters,
              SmallVector<Operation *, 4> ops) {
    ConstEvalSubgraph::inputParameters.insert(inputParameters.begin(),
                                              inputParameters.end());
    ConstEvalSubgraph::ops.append(ops.begin(), ops.end());
  }
  void insert(ConstEvalSubgraph const &other) {
    insert(other.inputParameters, other.ops);
  }
};
} // namespace

namespace {
struct ConstEvalAnalysisResults {
  llvm::SmallVector<ConstEvalSubgraph, 4> subgraphs;
  llvm::SmallVector<Operation *, 1> sharedOps;
};
} // namespace

namespace {
// Analyzer class to detect const-eval subgraphs in a given FuncOp.
// Template argument allows user to specify set of ops to not consider hoisting.
class ConstEvalAnalyze {
public:
  ConstEvalAnalyze(func::FuncOp funcOp) {
    populateConstParams(funcOp);
    buildConstEvalSubgraphs(funcOp);
  }

  ConstEvalAnalysisResults getAnalysisResults() {
    llvm::SmallVector<ConstEvalSubgraph, 4> subgraphVector;
    subgraphVector.reserve(subgraphMap.size());
    for (const auto &[_, subgraph] : subgraphMap) {
      subgraphVector.emplace_back(subgraph);
    }
    return ConstEvalAnalysisResults{subgraphVector, sharedOps};
  }

private:
  void populateConstParams(func::FuncOp funcOp) {
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
          constParams.insert(arg);
        }
      }
    }
  }

  // Main algorithm to build const-eval subgraphs:
  // 1. We have a map of sets of ops, each representing 1 subgraph
  // 2. For every op in the original graph, select one of 3 cases
  //   a. If an op depends on any non-const ops, it cannot be const-eval'ed
  //   b. If an op depends on only const input params (to the func) or creation
  // ops, it becomes a new const-eval subgraph (along with its creation ops)
  //   c. If an op depends on output from an op in any existing subgraphs, it
  //   must be merged into these subgraphs (if there are multiple such
  //   subgraphs, they must be merged).
  // 3. Result is map of all disjoint const-eval subgraphs (as sets of ops)
  void buildConstEvalSubgraphs(func::FuncOp funcOp) {
    // Iterate over all blocks in the function
    for (auto &block : funcOp.getBlocks()) {
      // Iterate over all operations in the block
      for (auto &opRef : block.getOperations()) {
        processOp(&opRef);
      }
    }
  }

  void processOp(Operation *op) {
    // Creation type ops are only hoisted to subgraphs if a user is found
    // which depends on them, to prevent spuriously hoisting all creation
    // ops. We also completely ignore certain ops.
    if (isUnhoistableOp(op) || isCreationOp(op)) {
      return;
    }

    // Some specific ops need to be duplicated
    if (isSharedOp(op)) {
      sharedOps.push_back(op);
      return;
    }

    // Set of idxs of any existing const-eval subgraphs this op may depend on.
    llvm::SmallSet<size_t, 2> subgraphIdxs;
    // Set of any input params this op may depend on.
    llvm::SmallPtrSet<mlir::BlockArgument, 4> inputParams;
    // Set of creation ops this op may depend on.
    llvm::SmallPtrSet<mlir::Operation *, 2> creationOps;

    for (auto operand : op->getOperands()) {
      if (!operandIsConstEval(operand, inputParams, creationOps,
                              subgraphIdxs)) {
        return;
      }
    }

    // Ensure any needed creation ops are inserted before the user op.
    llvm::SmallVector<mlir::Operation *, 4> opsToInsert(creationOps.begin(),
                                                        creationOps.end());
    opsToInsert.push_back(op);

    size_t targetIdx{};
    if (subgraphIdxs.empty()) {
      // This op does not depend on any existing subgraphs, create a new
      // one.
      targetIdx = nextSubgraphId++;
      subgraphMap[targetIdx] = {};
    } else if (subgraphIdxs.size() == 1) {
      // This op can be added to a single existing subgraph.
      targetIdx = *subgraphIdxs.begin();
    } else {
      // This op is connected to multiple subgraphs, must merge them all.
      targetIdx = mergeSetOfSubgraphs(subgraphIdxs);
    }
    auto it = subgraphMap.find(targetIdx);
    assert(it != subgraphMap.end());
    ConstEvalSubgraph &targetSubgraph = it->second;
    targetSubgraph.insert(inputParams, opsToInsert);
    for (auto *opToMove : opsToInsert) {
      for (auto result : opToMove->getResults()) {
        valueToSubgraphMap[result] = targetIdx;
      }
    }
  }

  // Process an operand, returning true + modifying appropriate input data
  // structure if it is const-eval, and otherwise returning false without
  // modifying data structures.
  bool
  operandIsConstEval(mlir::Value operand,
                     llvm::SmallPtrSet<mlir::BlockArgument, 4> &inputParams,
                     llvm::SmallPtrSet<mlir::Operation *, 2> &creationOps,
                     llvm::SmallSet<size_t, 2> &subgraphIdxs) {
    // Case 1: this operand is an const-eval param.
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
      if (constParams.contains(blockArg)) {
        inputParams.insert(blockArg);
        return true;
      }
      // If operand is a BlockArgument but not a const one, this op cannot be
      // const-eval.
      return false;
    }

    // Case 2: this operand is result of an existing op in a const-eval
    // subgraph.
    if (auto it = valueToSubgraphMap.find(operand);
        it != valueToSubgraphMap.end()) {
      subgraphIdxs.insert(it->second);
      return true;
    }

    // Case 3: this operand is an intermediate tensor from a neutral
    // creation op.
    if (mlir::Operation *defOp = operand.getDefiningOp(); defOp != nullptr) {
      // Shared ops will always be available in all subgraphs (but not in
      // valueToSubgraphMap).
      if (isSharedOp(defOp)) {
        return true;
      }
      if (!isUnhoistableOp(defOp) && isCreationOp(defOp)) {
        creationOps.insert(defOp);
        return true;
      }
    }
    // If all 3 cases fail, this operand is not const-eval, so this
    // operation cannot be const-eval'ed.
    return false;
  }

  // Check if an operation is a tensor creation op (no inputs, output is a
  // tensor).
  bool isCreationOp(mlir::Operation *op) {
    assert(op != nullptr);
    return op->hasTrait<mlir::tt::Trait::TTCreationOpTrait>() ||
           isa<mlir::tensor::EmptyOp>(op);
  }

  // Check if an operation cannot be hoisted.
  bool isUnhoistableOp(mlir::Operation *op) {
    assert(op != nullptr);
    return op->hasTrait<mlir::tt::Trait::TTIgnoreConstEvalTrait>();
  }

  bool isSharedOp(mlir::Operation *op) {
    assert(op != nullptr);
    return op->hasTrait<mlir::tt::Trait::TTDuplicateConstEvalTrait>();
  }

  // Merge content of all subgraphs in subgraphIdxs into the subgraph at the
  // lowest index, and return index of that subgraph.
  size_t mergeSetOfSubgraphs(const llvm::SmallSet<size_t, 2> &subgraphIdxs) {

    // Convert set to vector for sorting
    llvm::SmallVector<size_t, 2> allIndices(subgraphIdxs.begin(),
                                            subgraphIdxs.end());

    // Sort indices to find the lowest index.
    std::sort(allIndices.begin(), allIndices.end());

    // Get the lowest index as the target.
    size_t targetIdx = allIndices[0];

    // Merge all other subgraphs into the target one, starting from the
    // highest index to avoid invalidating indices during removal.
    for (int i = allIndices.size() - 1; i > 0; --i) {
      mergeSubgraphs(targetIdx, allIndices[i]);
    }

    return targetIdx;
  }

  // Merge all contents of subgraph at sourceIdx into subgraph at targetIdx +
  // erase source subgraph.
  void mergeSubgraphs(size_t targetIdx, size_t sourceIdx) {
    auto targetIt = subgraphMap.find(targetIdx);
    auto sourceIt = subgraphMap.find(sourceIdx);
    assert(targetIt != subgraphMap.end() && sourceIt != subgraphMap.end());

    ConstEvalSubgraph &target = targetIt->second;
    ConstEvalSubgraph &source = sourceIt->second;

    // Update mapping for all values produced by operations in the source
    // subgraph
    for (Operation *op : subgraphMap[sourceIdx].ops) {
      for (Value result : op->getResults()) {
        valueToSubgraphMap[result] = targetIdx;
      }
    }

    target.insert(source);

    subgraphMap.erase(sourceIdx);
  }

private:
  // Internal representation of subgraphs
  // This is a map so that we can efficiently erase elements in the middle
  // without invalidating indices.
  llvm::DenseMap<size_t, ConstEvalSubgraph> subgraphMap;
  // Index counter we use to ensure each subgraph has a unique id.
  size_t nextSubgraphId = 0;

  // Map to determine which subgraph a value belongs to (if any).
  llvm::DenseMap<mlir::Value, size_t> valueToSubgraphMap;
  // Set of params to original func which can be const-eval'ed.
  llvm::SmallPtrSet<mlir::BlockArgument, 4> constParams;
  // Set of ops which every subgraph + original graph must duplicate.
  llvm::SmallVector<mlir::Operation *, 1> sharedOps;
};
} // namespace

namespace {
// Transform pass to hoist const-eval subgraphs into separate funcs, invoked
// w/ tt.load_cached ops.
class ConstEvalHoistTransform
    : public impl::ConstEvalHoistTransformBase<ConstEvalHoistTransform> {
public:
  using impl::ConstEvalHoistTransformBase<
      ConstEvalHoistTransform>::ConstEvalHoistTransformBase;

  void runOnOperation() final {
    mlir::ModuleOp module = this->getOperation();
    llvm::SmallVector<func::FuncOp, 4> functionsToProcess;

    // Collect functions that need processing
    module.walk([&](func::FuncOp funcOp) { processFunction(funcOp); });
  }

private:
  // Process a single function for const-eval hoisting
  void processFunction(func::FuncOp funcOp) {
    if (funcOp->hasAttr("const_eval")) {
      return;
    }
    // Run the analysis to identify const-eval subgraphs
    ConstEvalAnalyze analyzer(funcOp);
    ConstEvalAnalysisResults analysisResults = analyzer.getAnalysisResults();
    llvm::SmallVector<ConstEvalSubgraph, 4> subgraphs =
        analysisResults.subgraphs;
    llvm::SmallVector<Operation *, 1> sharedOps = analysisResults.sharedOps;

    if (subgraphs.empty()) {
      return;
    }

    // Create new functions for each subgraph
    for (size_t i = 0; i < subgraphs.size(); ++i) {
      auto &subgraph = subgraphs[i];

      // Create a new function for this const-eval subgraph
      createConstEvalFunction(funcOp, subgraph, sharedOps, i);
    }
  }

  // Create a new function for a const-eval subgraph and replace the original
  // ops with a call
  void createConstEvalFunction(func::FuncOp originalFunc,
                               ConstEvalSubgraph &subgraph,
                               llvm::SmallVector<Operation *, 1> sharedOps,
                               size_t subgraphIdx) {
    mlir::MLIRContext *context = &this->getContext();
    mlir::OpBuilder builder(context);

    // Identify all outputs of the subgraph.
    llvm::SmallVector<mlir::BlockArgument, 4> inputs(
        subgraph.inputParameters.begin(), subgraph.inputParameters.end());
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

    // Build the body of the new function.
    auto *newEntryBlock = newFuncOp.addEntryBlock();
    builder.setInsertionPointToStart(newEntryBlock);

    // Create a mapping from original inputs to new function arguments.
    llvm::DenseMap<mlir::Value, mlir::Value> valueMap;
    for (size_t i = 0; i < inputs.size(); ++i) {
      valueMap.insert({inputs[i], newEntryBlock->getArgument(i)});
    }

    // Clone operations into the new function.
    for (auto *op : sharedOps) {
      processOp(op, valueMap, builder);
    }
    for (auto *op : subgraph.ops) {
      processOp(op, valueMap, builder);
    }

    // Create return operation.
    llvm::SmallVector<mlir::Value, 4> returnValues;
    for (auto output : outputs) {
      auto it = valueMap.find(output);
      assert(it != valueMap.end() &&
             "Subgraph did not contain value it should output.");
      returnValues.push_back(it->second);
    }

    builder.create<func::ReturnOp>(originalFunc.getLoc(), returnValues);

    auto &originalEntryBlock = originalFunc.getBody().front();
    // Manually order LoadCachedOp as first n ops in original func--we may
    // have folded some creation ops into the subgraph, so we need to ensure
    // these ops come before existing ops.
    auto iter = originalEntryBlock.begin();
    std::advance(iter, subgraphIdx);
    assert(iter != originalEntryBlock.end());
    builder.setInsertionPoint(&*iter);
    auto calleeAttr =
        mlir::SymbolRefAttr::get(builder.getContext(), newFuncName);

    llvm::SmallVector<int32_t> inputIdxs(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      inputIdxs[i] = inputs[i].getArgNumber();
    }
    // Create the LoadCachedOp with the correct argument order
    auto callOp = builder.create<tt::LoadCachedOp>(
        originalFunc.getLoc(), // Location
        outputTypes,           // Result types
        calleeAttr,            // Callee symbol reference
        DenseI32ArrayAttr::get(builder.getContext(), inputIdxs) // Input indexes
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

  void processOp(mlir::Operation *op,
                 llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                 mlir::OpBuilder &builder) {
    // In relevant IRs, terminators are always func::ReturnOp, which can't be
    // hoisted.
    assert(!op->hasTrait<mlir::OpTrait::IsTerminator>());

    llvm::SmallVector<mlir::Value, 4> remappedOperands;
    for (auto operand : op->getOperands()) {
      auto it = valueMap.find(operand);
      assert(it != valueMap.end() && "Subgraph depends on out-of-scope value!");
      remappedOperands.push_back(it->second);
    }

    auto *clonedOp = builder.clone(*op);

    // Update operands to use new func's params.
    for (size_t i = 0; i < clonedOp->getNumOperands(); ++i) {
      clonedOp->setOperand(i, remappedOperands[i]);
    }

    // Map original values to new results.
    for (size_t i = 0; i < clonedOp->getNumResults(); ++i) {
      valueMap.insert({op->getResult(i), clonedOp->getResult(i)});
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
            outputs.push_back(result);
            break;
          }
        }
      }
    }
  }
};
} // namespace
} // namespace mlir::tt::transforms
