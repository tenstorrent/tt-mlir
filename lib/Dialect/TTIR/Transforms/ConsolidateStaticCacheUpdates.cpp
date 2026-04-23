// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRCONSOLIDATESTATICCACHEUPDATES
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

struct WritebackInfo {
  unsigned retIdx;
  BlockArgument blockArg;
  AddOp addOp;
  Value delta;
  Value retSlotVal; // What is currently in the return slot (may be outer shard)
};

// Return the scalar integer value of a constant delta operand.
// Accepts ttir.full (FillValue attr) and ttir.constant (dense splat).
static std::optional<int64_t> scalarIntKey(Value v) {
  if (auto fullOp = v.getDefiningOp<FullOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(fullOp.getFillValue()))
      return ia.getInt();
    return std::nullopt;
  }
  if (auto constOp = v.getDefiningOp<ConstantOp>()) {
    auto dense = dyn_cast<DenseIntOrFPElementsAttr>(constOp.getValue());
    if (dense && dense.isSplat() && dense.getElementType().isInteger())
      return dense.getSplatValue<APInt>().getSExtValue();
    return std::nullopt;
  }
  return std::nullopt;
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
      if (funcOp.getBody().empty())
        return;

      Block &lastBlock = funcOp.getBody().back();
      auto returnOp =
          llvm::dyn_cast<func::ReturnOp>(lastBlock.getTerminator());
      if (!returnOp)
        return;

      // Collect write-back candidates.
      // Pattern (with optional mesh_shard wrapper):
      //   retSlotVal = [mesh_shard(] add(blockArg-or-mesh_shard(blockArg),
      //                                   constant_delta) [)]
      SmallVector<WritebackInfo> candidates;

      for (auto [idx, retVal] : llvm::enumerate(returnOp.getOperands())) {
        unsigned retIdx = static_cast<unsigned>(idx);

        // Optionally look through an outer mesh_shard.
        Value addResult = retVal;
        if (auto outerShard = retVal.getDefiningOp<MeshShardOp>()) {
          if (!retVal.hasOneUse())
            continue;
          addResult = outerShard.getInput();
        }

        auto addOp = addResult.getDefiningOp<AddOp>();
        if (!addOp || !addResult.hasOneUse())
          continue;

        // Copy the structured binding index to avoid C++17 lambda capture
        // limitation for structured bindings (C++20 extension).
        unsigned capturedRetIdx = retIdx;
        Value capturedRetVal = retVal;

        auto tryMatch = [&](Value maybeArgSide, Value maybeDelta) -> bool {
          // maybeArgSide is either a BlockArgument directly, or
          // mesh_shard(BlockArgument).
          BlockArgument blockArg;
          if (auto innerShard = maybeArgSide.getDefiningOp<MeshShardOp>()) {
            blockArg = llvm::dyn_cast<BlockArgument>(innerShard.getInput());
          } else {
            blockArg = llvm::dyn_cast<BlockArgument>(maybeArgSide);
          }
          if (!blockArg || blockArg.getOwner()->getParentOp() != funcOp)
            return false;

          // Require a recognizable scalar-integer constant delta.
          if (!scalarIntKey(maybeDelta).has_value())
            return false;

          // The blockArg type must match the return slot type so that the
          // consolidated result can substitute into any slot of the group.
          if (blockArg.getType() != capturedRetVal.getType())
            return false;

          candidates.push_back(
              {capturedRetIdx, blockArg, addOp, maybeDelta, capturedRetVal});
          return true;
        };

        if (!tryMatch(addOp.getLhs(), addOp.getRhs()))
          tryMatch(addOp.getRhs(), addOp.getLhs());
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
        if (!keyOpt.has_value())
          continue;
        GroupKey key{*keyOpt, wb.blockArg.getType()};

        bool found = false;
        for (auto &[k, vec] : groups) {
          if (k == key) {
            vec.push_back(wb);
            found = true;
            break;
          }
        }
        if (!found)
          groups.push_back({key, {wb}});
      }

      // Consolidate each group with more than one write-back.
      // Keep the last write-back's return-slot value as the canonical result
      // and redirect all earlier slots to it, erasing the now-dead ops.
      for (auto &[key, group] : groups) {
        if (group.size() <= 1)
          continue;

        Value keptResult = group.back().retSlotVal;

        for (size_t i = 0; i + 1 < group.size(); ++i) {
          WritebackInfo &wb = group[i];
          returnOp.setOperand(wb.retIdx, keptResult);

          // Erase in dependency order: outer shard first, then the add.
          // The inner mesh_shard (if any) may still have other uses; leave it
          // for dead-code elimination.
          if (wb.retSlotVal != wb.addOp.getResult() &&
              wb.retSlotVal.use_empty())
            wb.retSlotVal.getDefiningOp()->erase();

          if (wb.addOp.getResult().use_empty())
            wb.addOp.erase();
        }
      }
    });
  }
};

} // namespace mlir::tt::ttir
