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
};
} // namespace

namespace {
struct ConstEvalAnalysisResults {
  llvm::SmallVector<ConstEvalSubgraph, 4> subgraphs;
  llvm::SmallVector<Operation *, 1> sharedOps;
};
} // namespace

namespace {
// Analyzer class to detect const-eval subgraphs in a given FuncOp using
// union-find.
class ConstEvalAnalyze {
public:
  ConstEvalAnalyze(func::FuncOp *funcOp) : funcOp(funcOp) {
    llvm::outs() << "analyzing: " << funcOp->getSymName() << "\n";
    populateConstParams();
    buildConstEvalSubgraphs();
  }

  ConstEvalAnalysisResults getAnalysisResults() {
    llvm::SmallVector<ConstEvalSubgraph, 4> finalSubgraphs;

    // Create a map of root ID to input parameters
    llvm::DenseMap<size_t, llvm::SmallPtrSet<mlir::BlockArgument, 4>>
        rootToParamsMap;

    // Collect all operations in the original order
    llvm::DenseMap<size_t, llvm::SmallVector<Operation *, 4>> rootToOpsMap;

    // Process all operations and collect them by root
    for (auto &block : funcOp->getBlocks()) {
      for (auto &op : block.getOperations()) {
        auto opIt = opToSubgraphMap.find(&op);
        if (opIt != opToSubgraphMap.end()) {
          size_t subgraphId = opIt->second;
          size_t root = findRoot(subgraphId);
          rootToOpsMap[root].push_back(&op);
          // Also collect params for this root
          auto paramsIt = subgraphParams.find(subgraphId);
          if (paramsIt != subgraphParams.end()) {
            rootToParamsMap[root].insert(paramsIt->second.begin(),
                                         paramsIt->second.end());
          }
        }
      }
    }

    // Convert the maps to final subgraphs
    for (auto [root, ops] : rootToOpsMap) {
      ConstEvalSubgraph subgraph;

      // Add input parameters
      auto paramsIt = rootToParamsMap.find(root);
      if (paramsIt != rootToParamsMap.end()) {
        subgraph.inputParameters.insert(paramsIt->second.begin(),
                                        paramsIt->second.end());
      }

      // Add operations
      subgraph.ops = ops;

      finalSubgraphs.push_back(subgraph);
    }

    return ConstEvalAnalysisResults{finalSubgraphs, sharedOps};
  }

private:
  void populateConstParams() {
    if (funcOp->isDeclaration()) {
      return;
    }

    auto args = funcOp->getArguments();

    // Iterate through arguments and check their tt.argument_type attributes
    for (auto arg : args) {
      auto argAttrs = funcOp->getArgAttrDict(arg.getArgNumber());
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

  // ======  Union-find operations  ====== //
  size_t findRoot(size_t id) {
    auto [it, inserted] = parent.insert({id, id});
    if (inserted) {
      // Initialize if this is first access
      rank[id] = 0;
      return id;
    }

    if (it->second != id) {
      it->second = findRoot(it->second); // Path compression
    }
    return it->second;
  }

  void unionSubgraphs(size_t x, size_t y) {
    llvm::outs() << "merging " << x << " and " << y << "\n";
    size_t rootX = findRoot(x);
    size_t rootY = findRoot(y);

    if (rootX == rootY)
      return;

    // Union by rank
    if (rank[rootX] < rank[rootY]) {
      parent[rootX] = rootY;
    } else {
      parent[rootY] = rootX;
      if (rank[rootX] == rank[rootY]) {
        rank[rootX]++;
      }
    }
  }
  // ======  ---------------  ====== //

  void buildConstEvalSubgraphs() {
    // Iterate over all blocks in the function
    for (auto &block : funcOp->getBlocks()) {
      // Iterate over all operations in the block
      for (auto &opRef : block.getOperations()) {
        processOp(&opRef);
      }
    }
  }

  void processOp(Operation *op) {
    llvm::outs() << "processing " << op->getName() << "\n";
    // Skip unhoistable and creation ops
    if (isUnhoistableOp(op) || isCreationOp(op)) {
      return;
    }

    // Handle shared ops separately
    if (isSharedOp(op)) {
      sharedOps.push_back(op);
      return;
    }

    // Track which existing subgraphs this op depends on
    llvm::SmallSet<size_t, 2> dependentSubgraphIds;
    // Track input parameters this op depends on
    llvm::SmallPtrSet<mlir::BlockArgument, 4> inputParams;
    // Track creation ops this op depends on
    llvm::SmallPtrSet<mlir::Operation *, 2> creationOps;

    // Check if all operands are const-eval
    for (auto operand : op->getOperands()) {
      if (!operandIsConstEval(operand, inputParams, creationOps,
                              dependentSubgraphIds)) {
        return; // Not a const-eval op
      }
    }

    // Determine which subgraph this op belongs to
    size_t targetSubgraphId{};

    if (dependentSubgraphIds.empty()) {
      // Create a new subgraph
      targetSubgraphId = nextSubgraphId++;

      // Initialize union-find entry
      parent[targetSubgraphId] = targetSubgraphId;
      rank[targetSubgraphId] = 0;
    } else if (dependentSubgraphIds.size() == 1) {
      // Join existing subgraph
      targetSubgraphId = *dependentSubgraphIds.begin();
    } else {
      // Need to merge multiple subgraphs
      // Convert to vector for easier handling
      llvm::SmallVector<size_t, 2> subgraphIdVector(
          dependentSubgraphIds.begin(), dependentSubgraphIds.end());
      // std::sort(subgraphIdVector.begin(), subgraphIdVector.end());

      // Choose first as target
      targetSubgraphId = subgraphIdVector[0];

      // Union all others with the target
      for (size_t i = 1; i < subgraphIdVector.size(); ++i) {
        unionSubgraphs(targetSubgraphId, subgraphIdVector[i]);
      }

      // Get the actual root after unions
      targetSubgraphId = findRoot(targetSubgraphId);
    }

    // Track this op in the subgraph
    opToSubgraphMap[op] = targetSubgraphId;
    llvm::outs() << "mapping op to subgraph: " << targetSubgraphId << "\n";

    // Add creation ops to the subgraph as well
    for (Operation *creationOp : creationOps) {
      llvm::outs() << "mapping creationOp " << creationOp->getName()
                   << " to subgraph: " << targetSubgraphId << "\n";
      opToSubgraphMap[creationOp] = targetSubgraphId;
    }

    // Store input parameters for this subgraph
    if (!inputParams.empty()) {
      llvm::SmallPtrSet<mlir::BlockArgument, 4> &paramSet =
          subgraphParams[targetSubgraphId];
      paramSet.insert(inputParams.begin(), inputParams.end());
    }

    // Update value to subgraph mapping for results
    for (auto result : op->getResults()) {
      valueToSubgraphMap[result] = targetSubgraphId;
    }
  }

  // Process an operand, returning true + modifying appropriate input data
  // structure if it is const-eval, and otherwise returning false without
  // modifying data structures.
  bool
  operandIsConstEval(mlir::Value operand,
                     llvm::SmallPtrSet<mlir::BlockArgument, 4> &inputParams,
                     llvm::SmallPtrSet<mlir::Operation *, 2> &creationOps,
                     llvm::SmallSet<size_t, 2> &dependentSubgraphIds) {

    // Case 1: this operand is a const-eval param.
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
      if (constParams.contains(blockArg)) {
        inputParams.insert(blockArg);
        return true;
      }
      // Not a const param, can't be const-eval
      return false;
    }

    // Case 2: this operand is from an op in an existing const-eval subgraph
    if (auto it = valueToSubgraphMap.find(operand);
        it != valueToSubgraphMap.end()) {
      dependentSubgraphIds.insert(it->second);
      return true;
    }

    // Case 3: operand from creation op
    if (mlir::Operation *defOp = operand.getDefiningOp(); defOp != nullptr) {
      // Shared ops always available
      if (isSharedOp(defOp)) {
        return true;
      }

      // Check if it's a valid creation op
      if (!isUnhoistableOp(defOp) && isCreationOp(defOp)) {
        creationOps.insert(defOp);
        return true;
      }
    }

    // Not const-eval
    return false;
  }

  // Helper functions
  bool isCreationOp(mlir::Operation *op) {
    assert(op != nullptr);
    return op->hasTrait<mlir::tt::Trait::TTCreationOpTrait>();
  }

  bool isUnhoistableOp(mlir::Operation *op) {
    assert(op != nullptr);
    return op->hasTrait<mlir::tt::Trait::TTIgnoreConstEvalTrait>();
  }

  bool isSharedOp(mlir::Operation *op) {
    assert(op != nullptr);
    return op->hasTrait<mlir::tt::Trait::TTDuplicateConstEvalTrait>();
  }

private:
  // ======  Union-find data structures  ======
  //
  // Map of child to parent relationships, represents basic union-find tree.
  llvm::DenseMap<size_t, size_t> parent;
  // Rank map, for union-by-rank optimization to keep tree balanced.
  llvm::DenseMap<size_t, size_t> rank;
  // Maps value to its subgraph ID, to store dependencies.
  llvm::DenseMap<mlir::Value, size_t> valueToSubgraphMap;
  // Maps operation to its subgraph ID, for final gathering.
  llvm::DenseMap<mlir::Operation *, size_t> opToSubgraphMap;
  // Maps subgraph ID to its input parameters, for final gathering.
  llvm::DenseMap<size_t, llvm::SmallPtrSet<mlir::BlockArgument, 4>>
      subgraphParams;
  // Unique id per graph.
  size_t nextSubgraphId = 0;
  //
  // ======  -------------------------  ======

  func::FuncOp *funcOp;

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
    ConstEvalAnalyze analyzer(&funcOp);
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

    llvm::outs() << subgraph.inputParameters.size() << " params mapped to "
                 << originalFunc.getSymName() << "\n";

    // Identify all outputs of the subgraph.
    llvm::SmallVector<mlir::BlockArgument, 4> inputs(
        subgraph.inputParameters.begin(), subgraph.inputParameters.end());
    // Sort by argument number to keep order consistent
    std::sort(inputs.begin(), inputs.end(),
              [](BlockArgument a, BlockArgument b) {
                return a.getArgNumber() < b.getArgNumber();
              });
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

    // Create the LoadCachedOp with the correct argument order
    auto callOp = builder.create<tt::LoadCachedOp>(
        originalFunc.getLoc(), outputTypes, calleeAttr, ValueRange(inputs));

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
    llvm::outs() << "processing operands for " << op->getName() << "\n";
    for (auto operand : op->getOperands()) {
      auto it = valueMap.find(operand);
      operand.dump();
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
