// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreTraits.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/FunctionTypes.h"
#include "ttmlir/Transforms/Passes.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <iterator>
#include <optional>
#include <string>
#include <utility>

namespace mlir::tt::transforms {

#define GEN_PASS_DEF_CONSTEVALHOISTTRANSFORM
#include "ttmlir/Transforms/Passes.h.inc"

static bool isSharedOp(mlir::Operation *op) {
  assert(op != nullptr);
  return op->hasTrait<mlir::tt::ttcore::Trait::TTCoreDuplicateConstEvalTrait>();
}

// Check if any result of the op is written in-place by any of its users.
static bool isResultWrittenInPlace(mlir::Operation *op) {
  for (auto result : op->getResults()) {
    for (auto *user : result.getUsers()) {
      auto memEffectOp = dyn_cast<mlir::MemoryEffectOpInterface>(user);
      if (!memEffectOp) {
        continue;
      }
      llvm::SmallVector<mlir::MemoryEffects::EffectInstance> effects;
      memEffectOp.getEffects(effects);
      if (llvm::any_of(effects, [&](const auto &effect) {
            return isa<mlir::MemoryEffects::Write>(effect.getEffect()) &&
                   effect.getValue() == result;
          })) {
        return true;
      }
    }
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Hoist const-eval subgraphs to separate funcs pass
//===----------------------------------------------------------------------===//

namespace {
struct DisjointSetUnion {
  void init(mlir::Value v) {
    assert(v.getDefiningOp() != nullptr &&
           "Only values from ops can be in DSU");
    parent[v] = v;
  }

  bool valueExists(mlir::Value v) { return parent.find(v) != parent.end(); }

  mlir::Value findRoot(mlir::Value v) {
    auto it = parent.find(v);
    assert(it != parent.end() && "Value not found in DSU");
    if (v == it->second) {
      return v;
    }

    return it->second = findRoot(it->second);
  }

  mlir::Value unionSets(mlir::Value x, mlir::Value y) {
    mlir::Value rootX = findRoot(x);
    mlir::Value rootY = findRoot(y);
    return parent[rootX] = rootY;
  }

  llvm::DenseMap<mlir::Value, mlir::Value> parent;
};

struct ConstEvalSubgraph {
  // Set of parameters to the original function that this subgraph depends on.
  llvm::SmallPtrSet<mlir::BlockArgument, 4> inputParameters;
  // load_cached ops that this subgraph depends on.
  llvm::SetVector<mlir::tt::ttcore::LoadCachedOp> loadCachedOps;
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
// a Disjoint Subset Union algorithm.
class ConstEvalAnalyze {
public:
  ConstEvalAnalyze(func::FuncOp funcOp)
      : funcOp(funcOp),
        constParams(mlir::tt::ttcore::getConstsAndParams(funcOp)) {
    buildConstEvalSubgraphs();
  }

  bool failed() const { return analysisFailed; }

  ConstEvalAnalysisResults getAnalysisResults() {
    llvm::DenseMap<mlir::Value, size_t> rootToId;
    llvm::DenseMap<size_t, ConstEvalSubgraph> idToSubgraph;
    size_t currentId = 0;
    for (auto &block : funcOp.getBlocks()) {
      for (auto &op : block.getOperations()) {
        // Not const-eval-able op.
        if (op.getNumResults() == 0 || !dsu.valueExists(op.getResult(0))) {
          continue;
        }

        // Get root of the subgraph this op belongs to.
        mlir::Value root = dsu.findRoot(op.getResult(0));

        // If this is the first time we see this root assign a new id.
        if (rootToId.find(root) == rootToId.end()) {
          rootToId[root] = currentId++;
        }

        // Find the subgraph corresponding to this root.
        size_t id = rootToId[root];
        auto &subgraph = idToSubgraph[id];

        if (auto loadCachedOp =
                mlir::dyn_cast<mlir::tt::ttcore::LoadCachedOp>(&op)) {
          subgraph.loadCachedOps.insert(loadCachedOp);
          for (auto input : loadCachedOp.getInputs()) {
            if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(input)) {
              assert(constParams.contains(blockArg) &&
                     "Only const/param block args should be in DSU");
              subgraph.inputParameters.insert(blockArg);
            }
          }
          continue;
        }

        // Add op to the corresponding subgraph.
        subgraph.ops.push_back(&op);

        for (auto operand : op.getOperands()) {
          if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
            assert(constParams.contains(blockArg) &&
                   "Only const/param block args should be in DSU");
            subgraph.inputParameters.insert(blockArg);
          }
        }
      }
    }

    ConstEvalAnalysisResults results;
    results.sharedOps = sharedOps;
    for (auto &[id, subgraph] : idToSubgraph) {
      results.subgraphs.push_back(subgraph);
    }

    return results;
  }

private:
  void buildConstEvalSubgraphs() {
    for (auto &block : funcOp.getBlocks()) {
      for (auto &opRef : block.getOperations()) {
        processOp(&opRef);
      }
    }
  }

  void processOp(Operation *op) {
    // Skip terminator ops.
    if (op->hasTrait<mlir::OpTrait::IsTerminator>()) {
      return;
    }

    // Handle shared ops separately as well.
    if (isSharedOp(op)) {
      sharedOps.push_back(op);
      return;
    }

    // Non-cacheable ops cannot be part of const-eval subgraphs.
    if (op->hasTrait<mlir::tt::ttcore::Trait::TTCoreNonCacheableTrait>()) {
      return;
    }

    if (isResultWrittenInPlace(op)) {
      return;
    }

    auto operandConstEval = [&](mlir::Value operand) {
      if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
        return constParams.contains(blockArg);
      }

      if (isSharedOp(operand.getDefiningOp())) {
        return true;
      }

      return dsu.valueExists(operand);
    };

    // Check if all operands can be const-eval'ed.
    if (!llvm::isa<mlir::tt::ttcore::LoadCachedOp>(op) &&
        !llvm::all_of(op->getOperands(), operandConstEval)) {
      return;
    }

    // L1-resident results must opt in via "ttnn.const_eval_allowed";
    // const-eval outputs persist, so untagged L1 reservations would silently
    // steal budget from the forward function.
    bool anyResultInL1 = false;

    for (auto result : op->getResults()) {
      if (auto tensorType =
              mlir::dyn_cast<mlir::RankedTensorType>(result.getType())) {
        if (auto layoutAttr = mlir::dyn_cast_or_null<ttnn::TTNNLayoutAttr>(
                tensorType.getEncoding())) {
          if (layoutAttr.hasL1BufferType()) {
            anyResultInL1 = true;
            break;
          }
        }
      }
    }

    if (anyResultInL1 &&
        !op->hasAttr(mlir::tt::ttnn::utils::g_ConstEvalAllowedAttrName)) {
      op->emitError("result of an op resides in L1, and the op is a const-eval "
                    "candidate, but it is not tagged with '")
          << mlir::tt::ttnn::utils::g_ConstEvalAllowedAttrName
          << "'; tag the op or change its memory space";
      analysisFailed = true;
      return;
    }

    // Initialize DSU entries for results.
    llvm::for_each(op->getResults(),
                   [&](mlir::Value result) { dsu.init(result); });

    // Union all operands and results except for block arguments and shared ops.
    llvm::SmallVector<mlir::Value> graphsToJoin;
    for (auto operand : op->getOperands()) {
      if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
        continue;
      }
      if (isSharedOp(operand.getDefiningOp())) {
        continue;
      }
      graphsToJoin.push_back(operand);
    }

    for (auto result : op->getResults()) {
      graphsToJoin.push_back(result);
    }

    // Union all graphs together.
    if (!graphsToJoin.empty()) {
      mlir::Value first = graphsToJoin[0];
      for (size_t i = 1; i < graphsToJoin.size(); ++i) {
        dsu.unionSets(first, graphsToJoin[i]);
      }
    }
  }

private:
  // Disjoint Set Union data structure.
  DisjointSetUnion dsu;

  // Current function being analyzed.
  func::FuncOp funcOp;

  // Set of params to original func which can be const-eval'ed.
  llvm::SmallPtrSet<mlir::BlockArgument, 4> constParams;

  // Set of ops which every subgraph + original graph must duplicate.
  llvm::SmallVector<mlir::Operation *, 1> sharedOps;

  bool analysisFailed = false;
};
} // namespace

namespace {
// Deduplicate operations with TTCoreDuplicateConstEvalTrait in a function.
// Assumes any op with TTCoreDuplicateConstEvalTrait is equivalent to the same
// op with the same attrs.
static void deduplicateSharedOps(func::FuncOp funcOp) {
  // Map from operation signature to first instance
  using OpKey = std::pair<StringRef, DictionaryAttr>;
  llvm::DenseMap<OpKey, Operation *> sharedOps;

  // Collect operations that need to be erased
  SmallVector<Operation *, 8> opsToErase;

  funcOp.walk([&](Operation *op) {
    if (op->hasTrait<
            mlir::tt::ttcore::Trait::TTCoreDuplicateConstEvalTrait>()) {
      // Create a key based on operation name and all attributes
      StringRef opName = op->getName().getStringRef();
      DictionaryAttr attrs = op->getAttrDictionary();

      OpKey key = std::make_pair(opName, attrs);

      // If this is the first instance with these attributes, record it
      auto [it, inserted] = sharedOps.insert({key, op});
      if (inserted) {
        // This was the first instance with these attributes, no need to
        // substite.
        return;
      }

      // This is a duplicate, replace its uses with the first instance.
      Operation *firstOp = it->second;
      // Replace all uses of this op's results with the first instance's
      // results.
      for (size_t i = 0; i < op->getNumResults(); ++i) {
        op->getResult(i).replaceAllUsesWith(firstOp->getResult(i));
      }

      // Mark for later erasure to ensure we don't invalidate the block we're
      // stepping through.
      opsToErase.push_back(op);
    }
  });

  // Now erase all duplicate operations.
  for (Operation *op : opsToErase) {
    op->erase();
  }
}
} // namespace

namespace {
// Sums per-core L1 footprint of const-eval function outputs. Returns nullopt
// when no device is registered.
static std::optional<uint64_t> computeL1ConstEvalUsage(mlir::ModuleOp module) {
  auto deviceOp = ttcore::lookupDeviceOp(module);
  if (!deviceOp) {
    return std::nullopt;
  }
  ttcore::GridAttr workerGrid = deviceOp.getDeviceAttr().getWorkerGrid();
  uint64_t numCores =
      static_cast<uint64_t>(ttmlir::utils::volume(workerGrid.getShape()));
  if (numCores == 0) {
    return std::nullopt;
  }

  uint64_t total = 0;
  module.walk([&](mlir::func::FuncOp funcOp) {
    if (!ttmlir::utils::isConstEvalFunc(funcOp)) {
      return;
    }
    auto &entryBlock = funcOp.getBody().front();
    auto returnOp =
        mlir::dyn_cast<mlir::func::ReturnOp>(entryBlock.getTerminator());
    if (!returnOp) {
      return;
    }
    for (mlir::Value operand : returnOp.getOperands()) {
      auto tensorType =
          mlir::dyn_cast<mlir::RankedTensorType>(operand.getType());
      if (!tensorType) {
        continue;
      }
      auto layout = mlir::dyn_cast_or_null<ttnn::TTNNLayoutAttr>(
          tensorType.getEncoding());
      if (!layout || !layout.hasL1BufferType()) {
        continue;
      }
      total += ttnn::utils::getPerCoreL1Usage(layout, numCores);
    }
  });
  return total;
}
} // namespace

// Infix used to construct const-eval function names of the form
// "<originalName>_const_eval_<subgraphIdx>".
static constexpr llvm::StringLiteral g_constEvalFuncNameInfix = "_const_eval_";

// Construct a const-eval function name from the original function name and
// subgraph index.
static std::string constructConstEvalFunctionName(llvm::StringRef originalName,
                                                  uint64_t subgraphIdx) {
  return (originalName + g_constEvalFuncNameInfix + std::to_string(subgraphIdx))
      .str();
}

// Parse a const-eval function name produced by this pass, returning the
// original function name and the subgraph index it was created from if `name`
// is a well-formed const-eval function name.
static std::optional<std::pair<std::string, uint64_t>>
parseConstEvalFunctionName(llvm::StringRef name) {
  // Isolate the trailing "_const_eval_<subgraphIdx>" portion.
  size_t pos = name.rfind(g_constEvalFuncNameInfix);
  if (pos == llvm::StringRef::npos) {
    return std::nullopt;
  }

  llvm::StringRef originalName = name.take_front(pos);
  llvm::StringRef indexStr =
      name.drop_front(pos + g_constEvalFuncNameInfix.size());

  uint64_t subgraphIdx;
  if (originalName.empty() || indexStr.empty() ||
      indexStr.getAsInteger(/*Radix=*/10, subgraphIdx)) {
    return std::nullopt;
  }

  return std::make_pair(originalName.str(), subgraphIdx);
}

namespace {
// Transform pass to hoist const-eval subgraphs into separate funcs, invoked
// w/ ttcore.load_cached ops.
class ConstEvalHoistTransform
    : public impl::ConstEvalHoistTransformBase<ConstEvalHoistTransform> {
public:
  using impl::ConstEvalHoistTransformBase<
      ConstEvalHoistTransform>::ConstEvalHoistTransformBase;

  void runOnOperation() final {
    mlir::ModuleOp module = this->getOperation();
    llvm::DenseMap<mlir::func::FuncOp, size_t> maxSubgraphIndexMap;

    // Determine existing const-eval function names to determine the indeces for
    // naming new const-eval functions.
    module.walk([&](func::FuncOp funcOp) {
      if (ttmlir::utils::isConstEvalFunc(funcOp)) {
        auto name = funcOp.getName();
        auto parsed = parseConstEvalFunctionName(name);
        if (!parsed) {
          // This is a const-eval function, but it doesn't follow the naming
          // convention. It doesn't affect naming of new const-eval functions,
          // so we can ignore it.
          return WalkResult::advance();
        }
        auto [orgFuncName, subgraphIdx] = parsed.value();
        func::FuncOp orgFuncOp =
            module.lookupSymbol<mlir::func::FuncOp>(orgFuncName);
        if (!orgFuncOp) {
          // No original parent found in module for this const-eval function. It
          // doesn't affect naming of new const-eval functions, so we can ignore
          // it.
          return WalkResult::advance();
        }
        auto it = maxSubgraphIndexMap.find(orgFuncOp);
        if (it == maxSubgraphIndexMap.end()) {
          maxSubgraphIndexMap[orgFuncOp] = subgraphIdx;
        } else {
          it->second = std::max(it->second, subgraphIdx);
        }
      }
      return WalkResult::advance();
    });

    // Collect functions that need processing
    bool failed = false;
    module.walk([&](func::FuncOp funcOp) {
      // If the function has existing const-eval functions, offset the new
      // subgraph indices by the max index + 1 to avoid name collisions.
      auto it = maxSubgraphIndexMap.find(funcOp);
      auto end = maxSubgraphIndexMap.end();
      size_t offset = (it == end) ? 0 : it->second + 1;

      if (!processFunction(funcOp, offset)) {
        failed = true;
      }
    });

    if (failed) {
      signalPassFailure();
      return;
    }

    if (auto usage = computeL1ConstEvalUsage(module)) {
      OpBuilder builder(&getContext());
      auto u64 = mlir::IntegerType::get(&getContext(), 64,
                                        mlir::IntegerType::Unsigned);
      constexpr uint64_t kCushion = 1024;
      uint64_t alignedUsage = llvm::alignTo(*usage + kCushion, kCushion);
      module->setAttr(ttnn::utils::g_L1ConstEvalUsageAttrName,
                      builder.getIntegerAttr(u64, alignedUsage));
    }
  }

private:
  bool processFunction(func::FuncOp funcOp, size_t nameIndexOffset) {
    if (ttmlir::utils::isConstEvalFunc(funcOp)) {
      return true;
    }
    // Skip kernel functions: load_cached only supports tensor results, but
    // kernel const-eval may return !emitc.size_t, i32, etc.
    if (ttmlir::utils::isKernelFunc(funcOp)) {
      return true;
    }
    // Run the analysis to identify const-eval subgraphs
    ConstEvalAnalyze analyzer(funcOp);
    if (analyzer.failed()) {
      return false;
    }
    ConstEvalAnalysisResults analysisResults = analyzer.getAnalysisResults();
    llvm::SmallVector<ConstEvalSubgraph, 4> subgraphs =
        std::move(analysisResults.subgraphs);
    llvm::SmallVector<Operation *, 1> sharedOps =
        std::move(analysisResults.sharedOps);

    if (subgraphs.empty()) {
      return true;
    }

    // Create new functions for each subgraph
    size_t newFuncIndex = 0;
    for (const auto &subgraph : subgraphs) {
      if (subgraph.loadCachedOps.empty()) {
        // Create a new function for this const-eval subgraph
        createConstEvalFunction(funcOp, subgraph, sharedOps,
                                newFuncIndex + nameIndexOffset);
        ++newFuncIndex;
      } else {
        addToExistingConstEvalFunction(funcOp, subgraph, sharedOps);
      }
    }
    return true;
  }

  // Create a new function for a const-eval subgraph and replace the original
  // ops with a call
  void createConstEvalFunction(
      func::FuncOp originalFunc, const ConstEvalSubgraph &subgraph,
      const llvm::SmallVector<Operation *, 1> &sharedOps, size_t nameIndex) {
    mlir::MLIRContext *context = &this->getContext();
    mlir::OpBuilder builder(context);

    // Identify all inputs and outputs of the subgraph.
    llvm::SmallVector<mlir::BlockArgument, 4> inputs;
    llvm::SmallVector<mlir::Value, 4> outputs;
    collectSubgraphBoundary(subgraph, inputs, outputs);

    // Create the new function.
    std::string newFuncName =
        constructConstEvalFunctionName(originalFunc.getName(), nameIndex);
    mlir::FunctionType funcType = createFunctionType(inputs, outputs, builder);

    mlir::ModuleOp moduleOp =
        dyn_cast<mlir::ModuleOp>(originalFunc->getParentOp());
    assert(moduleOp);
    // Create the const-eval function before the parent function
    // This ensures proper ordering in the generated EmitC code.
    builder.setInsertionPoint(originalFunc);
    auto newFuncOp = builder.create<func::FuncOp>(originalFunc.getLoc(),
                                                  newFuncName, funcType);
    // Mark the new function as const-eval and private.
    ttmlir::utils::setFunctionType(newFuncOp,
                                   ttmlir::utils::FunctionType::ConstEval);
    newFuncOp.setPrivate();

    // Retain conv2dWeight input attributes from original function.
    // This is required because TTNNLayout pass places the
    // conv2d weights in the system memory.
    for (auto [newArgIdx, input] : llvm::enumerate(inputs)) {
      transferConv2DWeightAttr(originalFunc, newFuncOp, input, newArgIdx);
    }

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
    cloneSubgraph(subgraph, outputs, originalFunc.getLoc(), valueMap, builder);

    // Replace the original ops in the parent function with a call to the new
    // function.
    replaceOpsWithCall(originalFunc, newFuncOp, subgraph, inputs, outputs,
                       builder);
  }

  void addToExistingConstEvalFunction(
      func::FuncOp originalFunc, const ConstEvalSubgraph &subgraph,
      const llvm::SmallVector<Operation *, 1> &sharedOps) {
    auto constEvalFuncLoadCachedOp = subgraph.loadCachedOps.front();
    auto constEvalFuncName = constEvalFuncLoadCachedOp.getCallee();
    auto constEvalFunc = originalFunc->getParentOfType<mlir::ModuleOp>()
                             .lookupSymbol<func::FuncOp>(constEvalFuncName);
    assert(constEvalFunc && ttmlir::utils::isConstEvalFunc(constEvalFunc) &&
           "Const-eval function not found for load_cached op");

    mlir::MLIRContext *context = &this->getContext();
    mlir::OpBuilder builder(context);

    // Identify all inputs and outputs of the subgraph.
    llvm::SmallVector<mlir::BlockArgument, 4> inputs;
    llvm::SmallVector<mlir::Value, 4> outputs;
    collectSubgraphBoundary(subgraph, inputs, outputs);

    mlir::FunctionType newFuncType =
        createFunctionType(inputs, outputs, builder);
    constEvalFunc.setType(newFuncType);

    // Update function inputs and create a mapping from original inputs to
    // function arguments.
    llvm::DenseMap<mlir::Value, mlir::Value> valueMap;
    assert(!constEvalFunc.getBody().empty() &&
           "constEvalFunc must have a body");
    Block *entryBlock = &constEvalFunc.getBody().front();
    const auto &oldInputs = constEvalFuncLoadCachedOp.getInputs();
    auto oldIt = oldInputs.begin();
    for (size_t i = 0; i != inputs.size(); ++i) {
      if (oldIt != oldInputs.end() && *oldIt == inputs[i]) {
        valueMap.insert({inputs[i], entryBlock->getArgument(i)});
        ++oldIt;
      } else {
        entryBlock->insertArgument(i, inputs[i].getType(),
                                   constEvalFunc.getLoc());
        valueMap.insert({inputs[i], entryBlock->getArgument(i)});
        // Retain conv2dWeight input attributes from original function.
        // This is required because TTNNLayout pass places the
        // conv2d weights in the system memory.
        transferConv2DWeightAttr(originalFunc, constEvalFunc,
                                 entryBlock->getArgument(i), i);
      }
    }

    // Map load_cached outputs to the const-eval function's current return
    // values.
    const auto &loadCachedResults = constEvalFuncLoadCachedOp.getResults();
    auto returnOp = cast<mlir::func::ReturnOp>(entryBlock->getTerminator());
    for (size_t i = 0; i < loadCachedResults.size(); ++i) {
      valueMap.insert({loadCachedResults[i], returnOp.getOperand(i)});
    }

    // Remove the old return operation.
    returnOp.erase();

    // Clone the subgraph ops at the end of the block; they may depend on values
    // already present in the existing const-eval function.
    builder.setInsertionPointToStart(entryBlock);
    for (auto *op : sharedOps) {
      processOp(op, valueMap, builder);
    }
    builder.setInsertionPointToEnd(entryBlock);
    // Clone the bodies of all const-eval functions called by the other
    // load_cached ops into this const-eval function.
    for (auto loadCachedOp : subgraph.loadCachedOps) {
      if (loadCachedOp == constEvalFuncLoadCachedOp) {
        continue;
      }
      auto calleeFunc =
          originalFunc->getParentOfType<mlir::ModuleOp>()
              .lookupSymbol<func::FuncOp>(loadCachedOp.getCallee());
      assert(calleeFunc && ttmlir::utils::isConstEvalFunc(calleeFunc) &&
             "Const-eval function not found for load_cached op");
      cloneConstEvalFunctionBody(loadCachedOp, calleeFunc, valueMap, builder);
    }
    cloneSubgraph(subgraph, outputs, constEvalFunc.getLoc(), valueMap, builder);
    // Some shared ops that we cloned in the const-eval function may have
    // already been there.
    deduplicateSharedOps(constEvalFunc);

    // Update the original function.
    replaceOpsWithCall(originalFunc, constEvalFunc, subgraph, inputs, outputs,
                       builder);
    // Remove the original load_cached ops.
    for (auto loadCachedOp : subgraph.loadCachedOps) {
      if (loadCachedOp != constEvalFuncLoadCachedOp) {
        func::FuncOp calleeFunc =
            originalFunc->getParentOfType<mlir::ModuleOp>()
                .lookupSymbol<func::FuncOp>(loadCachedOp.getCallee());
        assert(calleeFunc && ttmlir::utils::isConstEvalFunc(calleeFunc) &&
               "Const-eval function not found for load_cached op");
        calleeFunc.erase();
      }
      loadCachedOp.erase();
    }
  }

  mlir::FunctionType
  createFunctionType(const llvm::SmallVector<mlir::BlockArgument, 4> &inputs,
                     const llvm::SmallVector<mlir::Value, 4> &outputs,
                     mlir::OpBuilder &builder) {
    llvm::SmallVector<mlir::Type, 4> inputTypes;
    llvm::SmallVector<mlir::Type, 4> outputTypes;

    // Get types for function signature.
    for (auto input : inputs) {
      inputTypes.push_back(input.getType());
    }

    for (auto output : outputs) {
      outputTypes.push_back(output.getType());
    }

    return builder.getFunctionType(inputTypes, outputTypes);
  }

  void transferConv2DWeightAttr(mlir::func::FuncOp originalFunc,
                                mlir::func::FuncOp newFuncOp,
                                mlir::BlockArgument inputArg,
                                size_t newArgIdx) {
    // Check if input argument is also original function argument.
    auto *maybeFunctionArgument =
        std::find(originalFunc.getArguments().begin(),
                  originalFunc.getArguments().end(), inputArg);

    if (maybeFunctionArgument == originalFunc.getArguments().end()) {
      return;
    }

    auto originalArgIdx = maybeFunctionArgument->getArgNumber();

    // Check for existence of ttmlir::utils::g_conv2dWeightAttrName.
    if (auto attr = originalFunc.getArgAttrOfType<mlir::Attribute>(
            originalArgIdx, ttmlir::utils::g_conv2dWeightAttrName)) {
      newFuncOp.setArgAttr(newArgIdx, ttmlir::utils::g_conv2dWeightAttrName,
                           attr);
    }
  }

  void replaceOpsWithCall(
      mlir::func::FuncOp originalFunc, mlir::func::FuncOp constEvalFunction,
      const ConstEvalSubgraph &subgraph,
      const llvm::SmallVector<mlir::BlockArgument, 4> &inputs,
      llvm::SmallVector<mlir::Value, 4> &outputs, mlir::OpBuilder &builder) {
    auto &originalEntryBlock = originalFunc.getBody().front();
    // Manually order LoadCachedOp as first n ops in original func--we may
    // have folded some creation ops into the subgraph, so we need to ensure
    // these ops come before existing ops.
    auto iter = originalEntryBlock.begin();
    while (iter != originalEntryBlock.end() &&
           mlir::isa<mlir::tt::ttcore::LoadCachedOp>(*iter)) {
      ++iter;
    }
    assert(iter != originalEntryBlock.end());
    builder.setInsertionPoint(&*iter);
    auto calleeAttr = mlir::SymbolRefAttr::get(builder.getContext(),
                                               constEvalFunction.getName());

    // Create the LoadCachedOp with the correct argument order
    auto callOp = builder.create<ttcore::LoadCachedOp>(
        originalFunc.getLoc(), constEvalFunction.getFunctionType().getResults(),
        calleeAttr, ValueRange(inputs));

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

  void cloneSubgraph(const ConstEvalSubgraph &subgraph,
                     const llvm::SmallVector<mlir::Value, 4> &outputs,
                     mlir::Location loc,
                     llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                     mlir::OpBuilder &builder) {
    // Clone operations into the new function.
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

    builder.create<func::ReturnOp>(loc, returnValues);
  }

  // Clone the body of a const-eval function called by a load_cached op into the
  // current insertion point, and update the value map.
  void
  cloneConstEvalFunctionBody(mlir::tt::ttcore::LoadCachedOp loadCachedOp,
                             func::FuncOp calleeFunc,
                             llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                             mlir::OpBuilder &builder) {
    Block &calleeEntryBlock = calleeFunc.getBody().front();

    // Map the callee's block arguments to the values passed by the load_cached
    // op.
    llvm::DenseMap<mlir::Value, mlir::Value> localValueMap;
    auto inputs = loadCachedOp.getInputs();
    for (auto [arg, input] :
         llvm::zip_equal(calleeEntryBlock.getArguments(), inputs)) {
      auto it = valueMap.find(input);
      assert(it != valueMap.end() &&
             "load_cached input not present in value map");
      localValueMap.insert({arg, it->second});
    }

    // Clone the ops of the callee's body.
    for (auto &op : calleeEntryBlock.without_terminator()) {
      processOp(&op, localValueMap, builder);
    }

    // Map the load_cached results to the cloned return values.
    auto returnOp =
        cast<mlir::func::ReturnOp>(calleeEntryBlock.getTerminator());
    for (auto [result, returnValue] :
         llvm::zip_equal(loadCachedOp.getResults(), returnOp.getOperands())) {
      auto it = localValueMap.find(returnValue);
      assert(it != localValueMap.end() &&
             "load_cached return value not present in value map");
      valueMap.insert({result, it->second});
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
  collectSubgraphBoundary(const ConstEvalSubgraph &subgraph,
                          llvm::SmallVector<mlir::BlockArgument, 4> &inputs,
                          llvm::SmallVector<mlir::Value, 4> &outputs) {

    std::copy(subgraph.inputParameters.begin(), subgraph.inputParameters.end(),
              std::back_inserter(inputs));

    // Sort by argument number to keep order consistent.
    std::sort(inputs.begin(), inputs.end(),
              [](BlockArgument a, BlockArgument b) {
                return a.getArgNumber() < b.getArgNumber();
              });
    // Remove duplicates.
    auto *newEnd = std::unique(inputs.begin(), inputs.end());
    inputs.erase(newEnd, inputs.end());

    // Create a set of operations for quick lookup.
    llvm::SmallPtrSet<mlir::Operation *, 8> opSet;
    for (auto *op : subgraph.ops) {
      opSet.insert(op);
    }

    // Collect outputs: values defined in the subgraph that are used outside.
    for (auto *op : subgraph.ops) {
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

    // If adding to existing const-eval function, also consider outputs of
    // load_cached ops that are going to be removed, as outputs of the
    // subgraph.
    for (auto loadCachedOp : subgraph.loadCachedOps) {
      for (auto result : loadCachedOp.getResults()) {
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
