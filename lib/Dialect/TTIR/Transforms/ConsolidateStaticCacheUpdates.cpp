// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRCONSOLIDATESTATICCACHEUPDATES
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

struct WritebackInfo {
  BlockArgument blockArg;
  Value delta;
};

// Return the scalar integer value of a constant delta operand.
// Accepts ttir.full (FillValue attr) and ttir.constant (dense splat).
static std::optional<int64_t> scalarIntKey(Value v) {
  if (auto fullOp = v.getDefiningOp<FullOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(fullOp.getFillValue())) {
      return ia.getInt();
    }
    return std::nullopt;
  }
  if (auto constOp = v.getDefiningOp<ConstantOp>()) {
    auto dense = dyn_cast<DenseIntOrFPElementsAttr>(constOp.getValue());
    if (dense && dense.isSplat() && dense.getElementType().isInteger()) {
      return dense.getSplatValue<APInt>().getSExtValue();
    }
    return std::nullopt;
  }
  return std::nullopt;
}

// Return the update-index operand of a cache op that carries one, or null if
// the op doesn't expose one. Only update-style cache ops have a position
// operand that traces back to a cumulative_length arg.
static Value getCacheUpdateIndex(CacheOpInterface cacheOp) {
  if (auto op = llvm::dyn_cast<UpdateCacheOp>(cacheOp.getOperation())) {
    return op.getUpdateIndex();
  }
  if (auto op = llvm::dyn_cast<PagedUpdateCacheOp>(cacheOp.getOperation())) {
    return op.getUpdateIndex();
  }
  return {};
}

// Walk back from a cache op's update_index operand through TM-like ops to
// collect any entry-block arguments that feed into it. The cumulative_length
// arg sits behind some combination of mesh_shard / broadcast / repeat /
// reshape / typecast and an add with an arange-derived offset.
static void collectArgsFeedingValue(Value start, func::FuncOp funcOp,
                                    llvm::DenseSet<BlockArgument> &out) {
  Block &entry = funcOp.getBody().front();
  SmallVector<Value> worklist;
  llvm::DenseSet<Value> visited;
  worklist.push_back(start);

  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!visited.insert(v).second) {
      continue;
    }

    if (auto blockArg = llvm::dyn_cast<BlockArgument>(v)) {
      if (blockArg.getOwner() == &entry) {
        out.insert(blockArg);
      }
      continue;
    }

    Operation *defOp = v.getDefiningOp();
    if (!defOp) {
      continue;
    }

    // Look through TM-like single-operand ops.
    if (isa<MeshShardOp, BroadcastOp, RepeatOp, ReshapeOp, TypecastOp>(defOp)) {
      worklist.push_back(defOp->getOperand(0));
      continue;
    }

    // For add, the cumulative_length sits on one side; the other side is
    // typically an arange or constant offset. Push both and let recursion
    // terminate on non-arg leaves.
    if (auto addOp = llvm::dyn_cast<AddOp>(defOp)) {
      worklist.push_back(addOp.getLhs());
      worklist.push_back(addOp.getRhs());
      continue;
    }

    // Anything else (arange, full, constant, arbitrary compute) terminates
    // this branch.
  }
}

// Collect entry-block args that provably feed an `update_index` of some
// update-style cache op. These are the args we'll treat as cumulative_length
// for unification.
static llvm::DenseSet<BlockArgument>
collectCumulativeLengthArgs(func::FuncOp funcOp) {
  llvm::DenseSet<BlockArgument> result;
  funcOp.walk([&](CacheOpInterface cacheOp) {
    Value updateIndex = getCacheUpdateIndex(cacheOp);
    if (!updateIndex) {
      return;
    }
    collectArgsFeedingValue(updateIndex, funcOp, result);
  });
  return result;
}

} // namespace

class TTIRConsolidateStaticCacheUpdates
    : public impl::TTIRConsolidateStaticCacheUpdatesBase<
          TTIRConsolidateStaticCacheUpdates> {
public:
  using impl::TTIRConsolidateStaticCacheUpdatesBase<
      TTIRConsolidateStaticCacheUpdates>::TTIRConsolidateStaticCacheUpdatesBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    moduleOp.walk([&](func::FuncOp funcOp) {
      if (funcOp.getBody().empty()) {
        return;
      }

      Block &lastBlock = funcOp.getBody().back();
      auto returnOp = llvm::dyn_cast<func::ReturnOp>(lastBlock.getTerminator());
      if (!returnOp) {
        return;
      }

      // Identify candidate cumulative_length args by tracing back from
      // update-style cache ops in this function. An arg is only eligible for
      // unification if it provably feeds a cache update_index — this proves
      // the lockstep invariant and prevents misfiring on unrelated programs
      // that happen to share the [add(blockArg, const_delta)] shape.
      llvm::DenseSet<BlockArgument> cumulativeLengthArgs =
          collectCumulativeLengthArgs(funcOp);
      if (cumulativeLengthArgs.empty()) {
        return;
      }

      // Collect write-back candidates.
      // Pattern (with optional mesh_shard wrapper):
      //   [mesh_shard(] add(blockArg-or-mesh_shard(blockArg),
      //                     constant_delta) [)]  →  return operand
      SmallVector<WritebackInfo> candidates;

      for (auto en : llvm::enumerate(returnOp.getOperands())) {
        Value retVal = en.value();
        // Optionally look through an outer mesh_shard.
        Value addResult = retVal;
        if (auto outerShard = retVal.getDefiningOp<MeshShardOp>()) {
          if (!retVal.hasOneUse()) {
            continue;
          }
          addResult = outerShard.getInput();
        }

        auto addOp = addResult.getDefiningOp<AddOp>();
        if (!addOp || !addResult.hasOneUse()) {
          continue;
        }

        auto tryMatch = [&](Value maybeArgSide, Value maybeDelta) -> bool {
          BlockArgument blockArg;
          if (auto innerShard = maybeArgSide.getDefiningOp<MeshShardOp>()) {
            blockArg = llvm::dyn_cast<BlockArgument>(innerShard.getInput());
          } else {
            blockArg = llvm::dyn_cast<BlockArgument>(maybeArgSide);
          }
          if (!blockArg || blockArg.getOwner()->getParentOp() != funcOp) {
            return false;
          }
          // Require the arg to be in the set of args provably feeding a cache
          // update_index. This is what proves the lockstep invariant holds
          // for this argument; without it, two independent counters with the
          // same delta could be miscompiled into one.
          if (!cumulativeLengthArgs.contains(blockArg)) {
            return false;
          }
          if (!scalarIntKey(maybeDelta).has_value()) {
            return false;
          }
          // Guard against broadcasting: block arg type must equal add result
          // type so that replacing arg uses preserves the result shape.
          if (blockArg.getType() != addResult.getType()) {
            return false;
          }
          candidates.push_back({blockArg, maybeDelta});
          return true;
        };

        if (!tryMatch(addOp.getLhs(), addOp.getRhs())) {
          tryMatch(addOp.getRhs(), addOp.getLhs());
        }
      }

      // Group candidates by (delta scalar key, blockArg element type).
      // Two write-backs belong to the same group when they apply the same
      // constant increment to arguments of the same type, so that keeping one
      // result and broadcasting it to all return slots is value-equivalent.
      // (For per-layer StaticCache tensors, all layer cumulative_lengths have
      // equal values at function entry, making this substitution correct.)
      using GroupKey = std::pair<int64_t, mlir::Type>;
      SmallVector<std::pair<GroupKey, SmallVector<WritebackInfo>>> groups;

      for (auto &wb : candidates) {
        auto keyOpt = scalarIntKey(wb.delta);
        if (!keyOpt.has_value()) {
          continue;
        }
        GroupKey key{*keyOpt, wb.blockArg.getType()};

        bool found = false;
        for (auto &[k, vec] : groups) {
          if (k == key) {
            vec.push_back(wb);
            found = true;
            break;
          }
        }
        if (!found) {
          groups.push_back({key, {wb}});
        }
      }

      // Replace all uses of eliminated block args with the canonical block arg.
      // All per-layer cumulative_length args are value-equivalent at function
      // entry (lockstep decode invariant), so routing every use to a single arg
      // is value-preserving. The CSE pass that follows will then deduplicate
      // the now-identical add/repeat/delta ops, collapsing N per-layer ops
      // to 1.
      //
      // We drop the eliminated args from the function signature so downstream
      // passes don't carry the dead arg through as an identity passthrough in
      // @main, which costs runtime perf for every layer. We intentionally do
      // NOT drop the matching return slots: replaceAllUsesWith above already
      // retargeted the return op's operands to the canonical SSA value, so the
      // return is still valid and the function's result type list is unchanged.
      // Keeping the return slots avoids forcing the caller (e.g. tt-xla) to
      // also drop matching output buffers, which would require non-trivial
      // host-side plumbing for relatively small device->host savings.
      llvm::BitVector argsToErase(funcOp.getNumArguments());

      // Track the canonical arg chosen per type so the second phase
      // (unification of args without a matching write-back) can route to the
      // same canonical and avoid splitting groups.
      llvm::DenseMap<mlir::Type, BlockArgument> canonicalByType;

      for (auto &[key, group] : groups) {
        if (group.size() <= 1) {
          continue;
        }

        BlockArgument canonicalArg = group.back().blockArg;
        canonicalByType[key.second] = canonicalArg;
        for (size_t i = 0; i + 1 < group.size(); ++i) {
          const WritebackInfo &eliminated = group[i];
          BlockArgument eliminatedArg = eliminated.blockArg;
          if (eliminatedArg == canonicalArg) {
            continue;
          }
          eliminatedArg.replaceAllUsesWith(canonicalArg);
          argsToErase.set(eliminatedArg.getArgNumber());
        }
      }

      // Phase 2: unify any remaining cumulative_length args that don't have a
      // matching write-back add/return pattern (e.g. production decode-only
      // graphs where the per-layer position is consumed by `ttir.update_cache`
      // but never written back to a return slot). Membership in
      // `cumulativeLengthArgs` is the safety proof: each such arg was traced
      // back from a cache-update position, so all of them carry the same
      // lockstep-invariant value at function entry and can share a single SSA
      // value.
      //
      // Grouping key is the block-arg type, matching Phase 1. If Phase 1
      // already picked a canonical for a given type, we route Phase 2's
      // eliminations to that same canonical so a single SSA value covers the
      // whole group.
      llvm::DenseMap<mlir::Type, SmallVector<BlockArgument>> remainingByType;
      for (BlockArgument arg : cumulativeLengthArgs) {
        if (argsToErase.test(arg.getArgNumber())) {
          continue;
        }
        remainingByType[arg.getType()].push_back(arg);
      }

      for (auto &[type, args] : remainingByType) {
        if (args.size() <= 1 && !canonicalByType.count(type)) {
          continue;
        }
        BlockArgument canonicalArg;
        auto it = canonicalByType.find(type);
        if (it != canonicalByType.end()) {
          canonicalArg = it->second;
        } else {
          // Pick the last arg (matches Phase 1's group.back() convention).
          canonicalArg = args.back();
          canonicalByType[type] = canonicalArg;
        }
        for (BlockArgument arg : args) {
          if (arg == canonicalArg) {
            continue;
          }
          arg.replaceAllUsesWith(canonicalArg);
          argsToErase.set(arg.getArgNumber());
        }
      }

      if (argsToErase.none()) {
        return;
      }

      // Update the function signature: drop the eliminated arguments only.
      // eraseArguments rewrites the entry-block arg list and the function
      // type's input list. The result list is intentionally left untouched.
      if (failed(funcOp.eraseArguments(argsToErase))) {
        funcOp.emitError(
            "ConsolidateStaticCacheUpdates: failed to erase orphaned arguments");
        signalPassFailure();
        return;
      }
    });
  }
};

} // namespace mlir::tt::ttir
