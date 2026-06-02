// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreTraits.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/FunctionTypes.h"
#include "ttmlir/Transforms/Passes.h"
#include "ttmlir/Utils.h"

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
#define GEN_PASS_DEF_UNDOCONSTEVALTRANSFORM
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

        // Add op to the corresponding subgraph.
        size_t id = rootToId[root];
        auto &subgraph = idToSubgraph[id];
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
    if (!llvm::all_of(op->getOperands(), operandConstEval)) {
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

// Common implementation shared between passes
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

// Helper to inline a const-eval function.
static void inlineConstEvalFunction(mlir::func::FuncOp funcOp,
                                    mlir::tt::ttcore::LoadCachedOp callOp,
                                    OpBuilder &builder) {
  builder.setInsertionPoint(callOp);

  // Use IRMapping to handle the mapping from original values to cloned values
  mlir::IRMapping valueMapper;

  // Map function arguments to call operands
  for (size_t i = 0; i < funcOp.getNumArguments(); ++i) {
    valueMapper.map(funcOp.getArgument(i), callOp.getOperand(i));
  }

  // Clone operations from const-eval function
  llvm::SmallVector<mlir::Operation *, 16> inlinedOps;
  auto &funcBody = funcOp.getBody().front();
  for (auto &op : funcBody) {
    // Skip the terminator operations
    if (op.hasTrait<mlir::OpTrait::IsTerminator>()) {
      continue;
    }

    // Clone the operation and update operands using the mapper
    inlinedOps.push_back(builder.clone(op, valueMapper));
  }

  // Get the return operation and map its values to the cloned values
  auto returnOp = cast<mlir::func::ReturnOp>(funcBody.back());
  for (size_t i = 0; i < returnOp.getNumOperands(); ++i) {
    auto mappedVal = valueMapper.lookup(returnOp.getOperand(i));
    callOp.getResult(i).replaceAllUsesWith(mappedVal);
  }

  // Erase the call operation
  callOp.erase();

  // Sink ops to just before their earliest user to minimize intermediate
  // tensor liveness on device. Process bottom-to-top so that consumers
  // are already in their final position when their producers are sunk.
  // Use the last moved op as an upper bound to preserve relative order.
  mlir::Operation *lastMovedOp = nullptr;
  for (int i = inlinedOps.size() - 1; i >= 0; --i) {
    mlir::Operation *op = inlinedOps[i];
    mlir::Operation *earliestUser = nullptr;
    for (auto result : op->getResults()) {
      for (auto *user : result.getUsers()) {
        if (!earliestUser || user->isBeforeInBlock(earliestUser)) {
          earliestUser = user;
        }
      }
    }
    if (earliestUser) {
      mlir::Operation *insertPt = earliestUser;
      if (lastMovedOp && lastMovedOp->isBeforeInBlock(insertPt)) {
        insertPt = lastMovedOp;
      }
      op->moveBefore(insertPt);
      lastMovedOp = op;
    }
  }
}

static void undoConstEvalImpl(mlir::ModuleOp module,
                              mlir::MLIRContext *context) {
  OpBuilder builder(context);

  // Find all const-eval functions and their callers
  llvm::DenseMap<mlir::func::FuncOp, mlir::tt::ttcore::LoadCachedOp> funcToCall;
  llvm::SmallVector<mlir::func::FuncOp, 4> constEvalFuncs;
  llvm::SmallVector<mlir::func::FuncOp, 4> parentFuncs;

  // Find all const-eval functions
  module.walk([&](mlir::func::FuncOp funcOp) {
    if (ttmlir::utils::isConstEvalFunc(funcOp)) {
      constEvalFuncs.push_back(funcOp);
    }
  });

  // Find all calls to const-eval functions
  module.walk([&](mlir::tt::ttcore::LoadCachedOp loadOp) {
    mlir::StringRef calleeName = loadOp.getCallee();
    auto funcOp = module.lookupSymbol<mlir::func::FuncOp>(calleeName);
    assert(funcOp && ttmlir::utils::isConstEvalFunc(funcOp));
    auto [_, inserted] = funcToCall.insert({funcOp, loadOp});
    assert(inserted && "Found const-eval func used more than once!");
  });

  // Inline each const-eval function
  for (auto funcOp : constEvalFuncs) {
    auto callIt = funcToCall.find(funcOp);
    if (callIt == funcToCall.end()) {
      // No call found; this can happen if the const-evaled value
      // is optimized-away.
      continue;
    }
    mlir::tt::ttcore::LoadCachedOp &callOp = callIt->second;
    // Get the parent function of this call
    mlir::func::FuncOp parentFunc =
        callOp->getParentOfType<mlir::func::FuncOp>();
    if (parentFunc) {
      parentFuncs.emplace_back(parentFunc);
    }

    inlineConstEvalFunction(funcOp, callOp, builder);
  }

  // Deduplicate shared ops in each function where we performed inlining
  for (auto funcOp : parentFuncs) {
    deduplicateSharedOps(funcOp);
  }

  // Delete inlined functions
  for (auto funcOp : constEvalFuncs) {
    funcOp.erase();
  }
}
} // namespace

namespace {
// Standalone pass to undo const-eval transformations
class UndoConstEvalTransform
    : public impl::UndoConstEvalTransformBase<UndoConstEvalTransform> {
public:
  using impl::UndoConstEvalTransformBase<
      UndoConstEvalTransform>::UndoConstEvalTransformBase;

  void runOnOperation() final {
    mlir::ModuleOp module = this->getOperation();
    undoConstEvalImpl(module, &getContext());
  }
};
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

//===----------------------------------------------------------------------===//
// Force const-eval inputs to system memory
//===----------------------------------------------------------------------===//
//
// After hoisting, forward-function arguments whose sole consumer is a
// load_cached op never need to live on device: the const-eval function can
// transfer them to device internally. Keeping them in system memory avoids a
// host->device copy in the forward function. This logic is TTNN-specific and
// no-ops on IR without TTNN layouts (e.g. TTIR-level invocations).
//
namespace {

// Create a system memory layout attribute for the given tensor type.
static ttnn::TTNNLayoutAttr
createSystemMemoryLayoutAttr(RankedTensorType type) {
  return ttnn::TTNNLayoutAttr::Builder(type)
      .setBufferType(ttnn::BufferType::SystemMemory)
      .setLayout(ttnn::Layout::RowMajor)
      .build();
}

// Convert a tensor type to its system memory equivalent.
static RankedTensorType toSystemMemoryType(RankedTensorType ty) {
  ttnn::TTNNLayoutAttr newLayout = createSystemMemoryLayoutAttr(ty);
  return RankedTensorType::get(ty.getShape(), ty.getElementType(), newLayout);
}

// Whether the given tensor type already has a TTNN system memory layout.
static bool isSystemMemory(RankedTensorType ty) {
  auto layout = mlir::dyn_cast_or_null<ttnn::TTNNLayoutAttr>(ty.getEncoding());
  return layout && layout.isSystemBufferType();
}

// Whether the const-eval argument should be transferred back to device memory.
//
// The argument should NOT be transferred to device memory only if its sole user
// is a ttnn.to_layout op which already transfers it to system memory - this
// happens if the const-eval function is CPU-hoisted.
static bool shouldTransferArgumentToDevice(BlockArgument blockArgument) {
  if (blockArgument.getNumUses() != 1) {
    return true;
  }

  auto toLayoutOp =
      mlir::dyn_cast<ttnn::ToLayoutOp>(*blockArgument.getUsers().begin());

  if (!toLayoutOp) {
    return true;
  }

  return mlir::cast<ttnn::TTNNMemoryConfigOpInterface>(
             toLayoutOp.getOperation())
             .getMemoryConfigAttr()
             .getBufferType()
             .getValue() != ttnn::BufferType::SystemMemory;
}

// Convert the arguments of the forward function which are const-eval inputs to
// system memory. Returns the list of converted arguments.
//
// An argument is considered a const-eval input if it is consumed only by
// load_cached ops.
static SmallVector<BlockArgument> moveConstEvalArgsToHost(func::FuncOp funcOp) {
  SmallVector<Type> argumentTypes;
  SmallVector<BlockArgument> convertedArguments;

  for (auto blockArgument : funcOp.getRegion().getArguments()) {
    auto tensorType = mlir::dyn_cast<RankedTensorType>(blockArgument.getType());

    // Skip args without a TTNN layout (e.g. TTIR-level IR); nothing to convert.
    if (!tensorType || !mlir::isa_and_nonnull<ttnn::TTNNLayoutAttr>(
                           tensorType.getEncoding())) {
      argumentTypes.push_back(blockArgument.getType());
      continue;
    }

    // Already in system memory.
    if (isSystemMemory(tensorType)) {
      argumentTypes.push_back(tensorType);
      continue;
    }

    // No uses, nothing to gain.
    if (blockArgument.getNumUses() == 0) {
      argumentTypes.push_back(tensorType);
      continue;
    }

    // Only convert if every user is a load_cached op.
    if (llvm::any_of(blockArgument.getUsers(), [](Operation *userOp) {
          return !mlir::isa<ttcore::LoadCachedOp>(userOp);
        })) {
      argumentTypes.push_back(tensorType);
      continue;
    }

    auto systemMemoryTensorType = toSystemMemoryType(tensorType);
    blockArgument.setType(systemMemoryTensorType);
    argumentTypes.push_back(systemMemoryTensorType);
    convertedArguments.push_back(blockArgument);
  }

  if (!convertedArguments.empty()) {
    auto newFunctionType =
        mlir::FunctionType::get(funcOp->getContext(), argumentTypes,
                                funcOp.getFunctionType().getResults());
    funcOp.setFunctionType(newFunctionType);
  }

  return convertedArguments;
}

// Update the const-eval function so that the given argument index is in system
// memory, inserting a to_layout op to restore the original device layout when
// the argument is still consumed on device.
static void convertArgumentOfConstEvalFunc(func::FuncOp constEvalFuncOp,
                                           size_t argumentIndex,
                                           RankedTensorType systemMemoryType) {
  SmallVector<Type> constEvalArgumentTypes(
      constEvalFuncOp.getFunctionType().getInputs());

  // Already converted.
  if (constEvalArgumentTypes[argumentIndex] == systemMemoryType) {
    return;
  }

  auto blockArgument = constEvalFuncOp.getArgument(argumentIndex);

  if (shouldTransferArgumentToDevice(blockArgument)) {
    mlir::OpBuilder builder(constEvalFuncOp.getRegion());

    // Insert after any existing get_device op.
    constEvalFuncOp.walk([&builder](ttnn::GetDeviceOp getDeviceOp) {
      builder.setInsertionPointAfter(getDeviceOp);
    });

    auto deviceTensorType =
        mlir::cast<RankedTensorType>(blockArgument.getType());
    auto deviceTensorLayout =
        mlir::cast<ttnn::TTNNLayoutAttr>(deviceTensorType.getEncoding());

    auto originalDataTypeAttr = ttcore::DataTypeAttr::get(
        constEvalFuncOp.getContext(), deviceTensorLayout.getDataType());

    auto toLayoutOp = builder.create<ttnn::ToLayoutOp>(
        blockArgument.getLoc(), deviceTensorType, blockArgument,
        deviceTensorLayout.getLayout(), originalDataTypeAttr);

    blockArgument.replaceAllUsesExcept(toLayoutOp.getResult(), toLayoutOp);
  }

  blockArgument.setType(systemMemoryType);

  constEvalArgumentTypes[argumentIndex] = systemMemoryType;
  auto newConstEvalFunctionType = mlir::FunctionType::get(
      constEvalFuncOp->getContext(), constEvalArgumentTypes,
      constEvalFuncOp.getFunctionType().getResults());
  constEvalFuncOp.setFunctionType(newConstEvalFunctionType);
}

// Drive the system-memory conversion across all forward functions in the
// module. Idempotent: already-converted arguments are skipped, so it is safe to
// run on every hoist invocation.
static void forceConstEvalInputsToSystemMemory(mlir::ModuleOp moduleOp) {
  moduleOp->walk([&](func::FuncOp funcOp) {
    if (!ttmlir::utils::isForwardDeviceFunc(funcOp)) {
      return;
    }

    SmallVector<BlockArgument> convertedArguments =
        moveConstEvalArgsToHost(funcOp);

    for (auto convertedArgument : convertedArguments) {
      llvm::SmallVector<ttcore::LoadCachedOp> loadCachedOps;
      for (auto *userOp : convertedArgument.getUsers()) {
        loadCachedOps.push_back(mlir::cast<ttcore::LoadCachedOp>(userOp));
      }

      for (auto loadCachedOp : loadCachedOps) {
        auto constEvalFuncOp =
            mlir::cast<func::FuncOp>(mlir::SymbolTable::lookupNearestSymbolFrom(
                loadCachedOp, loadCachedOp.getCalleeAttr()));

        // Argument index in the load_cached op matches the index in the
        // const-eval function we need to update.
        size_t argumentIndex = std::distance(
            loadCachedOp.getOperands().begin(),
            llvm::find(loadCachedOp.getOperands(), convertedArgument));

        assert(argumentIndex < constEvalFuncOp.getNumArguments() &&
               "Argument index out of bounds while updating LoadCachedOp.");

        convertArgumentOfConstEvalFunc(
            constEvalFuncOp, argumentIndex,
            mlir::cast<RankedTensorType>(convertedArgument.getType()));
      }
    }
  });
}
} // namespace

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
    llvm::SmallVector<func::FuncOp, 4> functionsToProcess;

    bool hasExistingConstEvalFuncs = false;
    module.walk([&](func::FuncOp funcOp) {
      if (ttmlir::utils::isConstEvalFunc(funcOp)) {
        hasExistingConstEvalFuncs = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    // If we found existing const-eval functions, undo them first
    if (hasExistingConstEvalFuncs) {
      undoConstEvalImpl(module, &getContext());
    }

    // Collect functions that need processing
    bool failed = false;
    module.walk([&](func::FuncOp funcOp) {
      if (!processFunction(funcOp)) {
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

    // Bring the memory space of const-eval-only inputs up to date with the
    // current const-eval state. No-op on TTIR-level IR (no TTNN layouts).
    forceConstEvalInputsToSystemMemory(module);
  }

private:
  bool processFunction(func::FuncOp funcOp) {
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
    for (size_t i = 0; i < subgraphs.size(); ++i) {
      auto &subgraph = subgraphs[i];

      // Create a new function for this const-eval subgraph
      createConstEvalFunction(funcOp, subgraph, sharedOps, i);
    }
    return true;
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
    // Create the const-eval function before the parent function
    // This ensures proper ordering in the generated EmitC code.
    builder.setInsertionPoint(originalFunc);
    auto newFuncOp = builder.create<func::FuncOp>(originalFunc.getLoc(),
                                                  newFuncName, funcType);
    // Mark the new function as const-eval and private.
    ttmlir::utils::setFunctionType(newFuncOp,
                                   ttmlir::utils::FunctionType::ConstEval);
    newFuncOp.setPrivate();

    // Retain connv2dWeight input attributes from original function.
    // This is required because TTNNLayout pass places the
    // conv2d weights in the system memory.
    for (auto [newArgIdx, input] : llvm::enumerate(inputs)) {
      // Check if input argument is also original function argument.
      auto *maybeFunctionArgument =
          std::find(originalFunc.getArguments().begin(),
                    originalFunc.getArguments().end(), input);

      if (maybeFunctionArgument == originalFunc.getArguments().end()) {
        continue;
      }

      auto originalArgIdx = maybeFunctionArgument->getArgNumber();

      // Check for existence of ttmlir::utils::g_conv2dWeightAttrName.
      if (auto attr = originalFunc.getArgAttrOfType<mlir::Attribute>(
              originalArgIdx, ttmlir::utils::g_conv2dWeightAttrName)) {
        newFuncOp.setArgAttr(newArgIdx, ttmlir::utils::g_conv2dWeightAttrName,
                             attr);
      }
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
    auto callOp = builder.create<ttcore::LoadCachedOp>(
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
