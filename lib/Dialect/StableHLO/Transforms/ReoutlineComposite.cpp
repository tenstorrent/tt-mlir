// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardingUtils.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"
#include "llvm/Support/Error.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_REOUTLINECOMPOSITEPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

using GroupKey = mlir::StringAttr;

// Collect ops per group string inside a func.
static void collectGroups(
    mlir::func::FuncOp func,
    llvm::DenseMap<GroupKey, llvm::SmallVector<mlir::Operation *>> &groups) {
  func.walk([&](mlir::Operation *op) {
    if (auto attr =
            op->getAttrOfType<mlir::StringAttr>(sharding_utils::kGroupAttr)) {
      groups[attr].push_back(op);
    }
  });
}

static bool isOpBefore(mlir::Operation *opA, mlir::Operation *opB) {
  if (opA->getBlock() != opB->getBlock()) {
    return false;
  }
  return opA->isBeforeInBlock(opB);
}

// Sort ops by dominance order within a single block (we require one block
// later).
static void sortByBlockOrder(llvm::SmallVector<mlir::Operation *> &ops) {
  llvm::sort(ops, isOpBefore);
}

// Compute captured inputs (defined outside, used by the group) and escaping
// outputs (defined in the group, used outside). Also verify the group is
// inside a single block and forms a contiguous range (movable as a chunk).
static bool analyzeBoundary(const llvm::SmallVector<mlir::Operation *> &ops,
                            llvm::SmallVector<mlir::Value> &captures,
                            llvm::SmallVector<mlir::Value> &escapes,
                            mlir::Operation *&firstOp,
                            mlir::Operation *&lastOp) {
  // Sort and find contiguous range [first, last].
  llvm::SmallVector<mlir::Operation *> sorted = ops;
  sortByBlockOrder(sorted);
  firstOp = sorted.front();
  lastOp = sorted.back();

  // Build a set for quick membership tests.
  llvm::SmallDenseSet<mlir::Operation *> inGroup;
  inGroup.reserve(sorted.size());
  for (auto *op : sorted) {
    inGroup.insert(op);
  }

  for (mlir::Operation &it :
       llvm::make_range(mlir::Block::iterator(firstOp),
                        std::next(mlir::Block::iterator(lastOp)))) {
    // All ops between first and last must be in the group.
    // If new ops were inserted in between, we cannot reoutline.
    if (!inGroup.contains(&it)) {
      return false;
    }
  }

  // Captures: operands whose defining op is not in the group (or is a block
  // arg).
  llvm::SmallDenseSet<mlir::Value> captureSet;
  for (auto *op : sorted) {
    for (mlir::Value v : op->getOperands()) {
      if (auto *defOp = v.getDefiningOp()) {
        if (!inGroup.contains(defOp)) {
          captureSet.insert(v);
        }
      } else {
        // Block argument (outside def).
        captureSet.insert(v);
      }
    }
  }

  // Escapes: results of group ops with any use outside the group.
  llvm::SmallDenseSet<mlir::Value> escapeSet;
  for (auto *op : sorted) {
    for (mlir::Value result : op->getResults()) {
      for (mlir::OpOperand &use : result.getUses()) {
        if (!inGroup.contains(use.getOwner())) {
          escapeSet.insert(result);
          break;
        }
      }
    }
  }
  // Materialize vectors in deterministic order (by appearance in the block).
  auto orderByBlockPos = [](mlir::Value opA, mlir::Value opB) {
    mlir::Operation *defA = opA.getDefiningOp();
    mlir::Operation *defB = opB.getDefiningOp();
    if (!defA && !defB) {
      // both block args
      return opA.getImpl() < opB.getImpl();
    }
    if (!defA) {
      // block arg first
      return true;
    }
    if (!defB) {
      return false;
    }
    return isOpBefore(defA, defB);
  };

  captures.assign(captureSet.begin(), captureSet.end());
  escapes.assign(escapeSet.begin(), escapeSet.end());
  llvm::sort(captures, orderByBlockPos);
  llvm::sort(escapes, orderByBlockPos);

  return true;
}

// Create a new private function and clone the group into it.
// - Captures become function arguments (in declared order).
// - Escapes become function results (in declared order).
// Returns the new callee symbol and also fills 'mapping' from old->new values.
static mlir::func::FuncOp outlineToFunc(
    mlir::ModuleOp module, mlir::StringRef baseName,
    mlir::ArrayRef<mlir::Value> captures, mlir::ArrayRef<mlir::Value> escapes,
    mlir::ArrayRef<mlir::Operation *> opsToClone, mlir::IRMapping &mapping) {
  mlir::OpBuilder builder(module.getContext());

  // Build function type: (captures) -> (escapes types)
  llvm::SmallVector<mlir::Type> argumentTypes, resultTypes;
  argumentTypes.reserve(captures.size());
  resultTypes.reserve(escapes.size());
  for (mlir::Value v : captures) {
    argumentTypes.push_back(v.getType());
  }
  for (mlir::Value v : escapes) {
    resultTypes.push_back(v.getType());
  }

  std::string fnName;
  {
    llvm::raw_string_ostream os(fnName);
    os << "outlined_" << baseName;
  }

  auto fnType = builder.getFunctionType(argumentTypes, resultTypes);
  auto func = mlir::func::FuncOp::create(module.getLoc(), fnName, fnType);
  func.setPrivate();
  module.push_back(func);

  // Create entry block and map captures to block arguments.
  mlir::Block *entry = func.addEntryBlock();
  for (auto it : llvm::enumerate(captures)) {
    mapping.map(it.value(),
                entry->getArgument(static_cast<unsigned>(it.index())));
  }

  // Clone ops in order into the new function.
  mlir::OpBuilder internalBuilder(entry, entry->end());
  for (mlir::Operation *op : opsToClone) {
    mlir::Operation *cloned = internalBuilder.clone(*op, mapping);
    // Strip the reoutline attrs inside the callee.
    cloned->removeAttr(sharding_utils::kGroupAttr);
    cloned->removeAttr(sharding_utils::kSeedAttr);
  }

  // Emit return with remapped escape values.
  llvm::SmallVector<mlir::Value> retVals;
  retVals.reserve(escapes.size());
  for (mlir::Value esc : escapes) {
    mlir::Value escVal = mapping.lookupOrNull(esc);
    retVals.push_back(escVal);
  }
  internalBuilder.create<mlir::func::ReturnOp>(func.getLoc(), retVals);

  return func;
}

static mlir::Operation *
computeSafeInsertAfter(llvm::ArrayRef<mlir::Value> captures,
                       mlir::Operation *fallbackBefore) {
  // find the latest defining op among captures in the same block.
  mlir::Block *block = fallbackBefore->getBlock();
  mlir::Operation *latestDef = nullptr;

  for (mlir::Value v : captures) {
    if (auto *def = v.getDefiningOp()) {
      if (def->getBlock() != block) {
        // different block → ignore; fallback handles it
        continue;
      }
      if (!latestDef || isOpBefore(latestDef, def)) {
        latestDef = def;
      }
    }
  }

  // If no defining ops in this block, insert before fallback.
  if (!latestDef) {
    return fallbackBefore;
  }

  // Otherwise, insert after the latest def.
  return latestDef;
}

// Replace the original ops range with a stablehlo.composite, wiring
// captures/results. Erase the old ops.
void replaceWithComposite(mlir::func::FuncOp parentFunc,
                          mlir::Operation *insertBefore,
                          mlir::Operation *firstOp, mlir::Operation *lastOp,
                          mlir::func::FuncOp callee,
                          mlir::ArrayRef<mlir::Value> captures,
                          mlir::ArrayRef<mlir::Value> escapes,
                          llvm::ArrayRef<mlir::Operation *> opsToErase,
                          mlir::StringAttr groupKey) {
  mlir::Operation *after = computeSafeInsertAfter(captures, firstOp);
  mlir::OpBuilder builder(after);
  if (after == firstOp) {
    // No capture defined after firstOp
    builder.setInsertionPoint(firstOp);
  } else {
    // latest capture def is after firstOp
    builder.setInsertionPointAfter(after);
  }
  mlir::Location loc = callee.getLoc();
  mlir::MLIRContext *ctx = builder.getContext();

  // Result types = escape value types.
  llvm::SmallVector<mlir::Type> resultTypes;
  resultTypes.reserve(escapes.size());
  for (mlir::Value v : escapes) {
    resultTypes.push_back(v.getType());
  }

  // Operands = captures.
  llvm::SmallVector<mlir::Value> operands(captures.begin(), captures.end());

  // Attributes:
  // - decomposition: SymbolRef to callee
  // - composite_attributes: an (optionally empty) dict
  // - call_target_name (or your fork's key for the printed string after
  // 'composite')
  mlir::SymbolRefAttr decomp =
      mlir::SymbolRefAttr::get(ctx, callee.getSymName());
  mlir::DictionaryAttr compAttrs = mlir::DictionaryAttr::get(ctx);
  mlir::StringAttr targetName = builder.getStringAttr(callee.getSymName());
  for (mlir::Operation *op : opsToErase) {
    if (op->hasAttr(sharding_utils::kSeedAttr)) {
      if (auto origName = op->getAttrOfType<mlir::StringAttr>(
              sharding_utils::kOrigNameAttr)) {
        targetName = origName;
      }
      if (auto compAttr = op->getAttrOfType<mlir::DictionaryAttr>(
              sharding_utils::kCompAttrsAttr)) {
        compAttrs = compAttr;
      }
    }
  }

  // Some forks name this attribute differently ("call_target_name" vs "name").
  // Match your textual IR; in your dump it prints as: composite
  // "tenstorrent.gelu_tanh" ...
  constexpr llvm::StringLiteral kDecompKey("decomposition");
  constexpr llvm::StringLiteral kCompAttrsKey("composite_attributes");
  constexpr llvm::StringLiteral kNameKey("name");

  mlir::OperationState state(loc,
                             mlir::stablehlo::CompositeOp::getOperationName());
  state.addOperands(operands);
  state.addTypes(resultTypes);
  state.addAttribute(kDecompKey, decomp);
  state.addAttribute(kCompAttrsKey, compAttrs);
  state.addAttribute(kNameKey, targetName);

  mlir::Operation *newOp = mlir::Operation::create(state);
  builder.insert(newOp);
  auto comp = llvm::cast<mlir::stablehlo::CompositeOp>(newOp);

  // Replace external uses of escape values with composite results (1:1).
  for (auto it : llvm::enumerate(escapes)) {
    mlir::Value oldV = it.value();
    mlir::Value newV = comp.getResult(static_cast<unsigned>(it.index()));
    oldV.replaceAllUsesWith(newV);
  }

  // Erase grouped ops in reverse block order (users first, defs later)
  llvm::SmallVector<mlir::Operation *> toErase(opsToErase.begin(),
                                               opsToErase.end());
  sortByBlockOrder(toErase);

  for (mlir::Operation *op : llvm::reverse(toErase)) {
    if (op == firstOp) {
      continue;
    }
    if (groupKey) {
      auto attr =
          op->getAttrOfType<mlir::StringAttr>(sharding_utils::kGroupAttr);
      if (!attr || attr != groupKey) {
        continue;
      }
    }
    op->erase();
  }
}

// ReoutlineCompositePass: Rebuilds stablehlo.composite from grouped ops after
// flatten → sharding → re-outline. For each function, it gathers groups,
// checks they form a single-block, “closed-enough” contiguous range, outlines
// them into a private callee ((captures)->(escapes)), and replaces the range
// with a composite op (restoring name/composite_attributes).
class ReoutlineCompositePass
    : public impl::ReoutlineCompositePassBase<ReoutlineCompositePass> {
  using impl::ReoutlineCompositePassBase<
      ReoutlineCompositePass>::ReoutlineCompositePassBase;

public:
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::SymbolTable symTable(module);

    // For each function, gather groups and try to re-outline.
    for (mlir::func::FuncOp func : module.getOps<mlir::func::FuncOp>()) {
      llvm::DenseMap<GroupKey, llvm::SmallVector<mlir::Operation *>> groups;
      // Collect ops per group. Group key is defined by
      // sharding_utils::kGroupAttr.
      collectGroups(func, groups);
      if (groups.empty()) {
        continue;
      }

      // Process each group independently.
      for (auto &kv : groups) {
        const GroupKey &group = kv.first;
        auto &ops = kv.second;

        // Analyze boundary and contiguity.
        mlir::Operation *firstOp = nullptr;
        mlir::Operation *lastOp = nullptr;
        llvm::SmallVector<mlir::Value> captures, escapes;

        if (!analyzeBoundary(ops, captures, escapes, firstOp, lastOp)) {
          // Not a closed-enough group
          continue;
        }

        // Choose insertion point right before 'firstOp'.
        mlir::StringAttr base = group;

        // Outline to a new callee function.
        mlir::IRMapping mapping;
        mlir::func::FuncOp callee = outlineToFunc(
            module, base.getValue(), captures, escapes, ops, mapping);

        // Replace range with a call and erase the grouped ops.
        replaceWithComposite(func, firstOp, firstOp, lastOp, callee, captures,
                             escapes, ops, group);
      }
    }
  }
};

} // namespace mlir::tt::stablehlo
