// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/DataMovementRules.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNFORCEFINALDEALLOCS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

// Maps a control flow region's block argument to the operand it is bound to.
// Overloaded per op rather than taking an `Operation *`, so that there is no
// unhandled op to guard against: a `ttnn.while` region observes
// `inits ++ captures`, a `ttnn.case` branch observes just the captures, and
// nothing else has a binding to describe.
Value getBoundOperand(WhileOp op, unsigned argNumber) {
  unsigned numInits = op.getInits().size();
  if (argNumber < numInits) {
    return op.getInits()[argNumber];
  }
  return op.getCaptures()[argNumber - numInits];
}

Value getBoundOperand(CaseOp op, unsigned argNumber) {
  return op.getCaptures()[argNumber];
}

// The block argument a region hands straight back out as `resultNumber`, or a
// null value if the region computes that result itself.
BlockArgument getForwardedArgument(Region &region, unsigned resultNumber) {
  auto yieldOp =
      mlir::dyn_cast_if_present<YieldOp>(region.front().getTerminator());
  if (!yieldOp || resultNumber >= yieldOp.getNumOperands()) {
    return BlockArgument();
  }
  return mlir::dyn_cast<BlockArgument>(yieldOp.getOperand(resultNumber));
}

// The operands a control flow op may hand back as `resultNumber` without
// touching them, i.e. the buffers that result may turn out to alias.
//
// A `ttnn.while` result is only reported when the binding holds for every
// iteration: a capture is bound to the same value throughout, and a carried
// value forwarded into its own slot is never overwritten with anything else.
// A carried value forwarded into a *different* slot depends on the iteration
// count, so nothing is reported for it.
llvm::SmallVector<Value> getForwardedOperands(Operation *op,
                                              unsigned resultNumber) {
  llvm::SmallVector<Value> operands;

  if (auto whileOp = mlir::dyn_cast<WhileOp>(op)) {
    BlockArgument arg = getForwardedArgument(whileOp.getBody(), resultNumber);
    if (arg && (arg.getArgNumber() >= whileOp.getInits().size() ||
                arg.getArgNumber() == resultNumber)) {
      operands.push_back(getBoundOperand(whileOp, arg.getArgNumber()));
    }
    return operands;
  }

  if (auto caseOp = mlir::dyn_cast<CaseOp>(op)) {
    // Exactly one branch runs, so the result may alias whatever any of them
    // forwards.
    for (Region &branch : caseOp.getBranches()) {
      if (BlockArgument arg = getForwardedArgument(branch, resultNumber)) {
        Value operand = getBoundOperand(caseOp, arg.getArgNumber());
        if (!llvm::is_contained(operands, operand)) {
          operands.push_back(operand);
        }
      }
    }
  }

  return operands;
}

// Returns the value whose device buffer `value` aliases, or a null Value if it
// names a buffer of its own (i.e. is not acting as a view).
//
// Two things alias today: a view-eligible reshape aliases its input, and a
// control flow op result aliases the operand a region forwards straight out of
// it - the runtime publishes a second handle on that same buffer. This helper
// is the single place that knowledge lives so that the rest of the pass is
// decoupled from it. A future ViewOpInterface can replace the body without
// touching the pass logic.
//
// A `ttnn.case` result whose branches forward *different* operands has no
// single source and so none is reported here; `collectDoNotForceRoots` keeps
// those buffers out of the forcing decision instead.
Value getViewSource(Value value) {
  Operation *op = value.getDefiningOp();
  if (!op) {
    return Value();
  }
  if (canReshapeBeView(op)) {
    return op->getOperand(0);
  }
  if (mlir::isa<WhileOp, CaseOp>(op)) {
    llvm::SmallVector<Value> forwarded =
        getForwardedOperands(op, mlir::cast<OpResult>(value).getResultNumber());
    if (forwarded.size() == 1) {
      return forwarded.front();
    }
  }
  return Value();
}

// Returns the activation of the given conv op if the conv deallocates it
// itself, or a null value otherwise.
template <typename ConvOpTy>
Value getConvDeallocatedActivation(ConvOpTy conv) {
  auto config = conv.getConv2dConfigAttr();
  Value input = conv.getInput();
  auto inputType = mlir::cast<RankedTensorType>(input.getType());

  // The conv deallocates its activation when the flag is set and the
  // activation is in L1 memory.
  bool deallocatesActivation =
      config && config.getDeallocateActivation() &&
      config.getDeallocateActivation().getValue() &&
      utils::getBufferTypeFromTensor(inputType) == BufferType::L1;

  return deallocatesActivation ? input : Value();
}

// Walks up the chain of view ops until reaching the value that produces the
// underlying buffer, caching the result for every value on the path.
Value canonicalRoot(Value value, llvm::DenseMap<Value, Value> &valueToRoot) {
  llvm::SmallVector<Value, 4> path;
  Value current = value;
  while (true) {
    auto it = valueToRoot.find(current);
    if (it != valueToRoot.end()) {
      current = it->second;
      break;
    }
    Value source = getViewSource(current);
    if (!source) {
      break;
    }
    path.push_back(current);
    current = source;
  }
  Value root = current;
  for (Value aliased : path) {
    valueToRoot[aliased] = root;
  }
  valueToRoot[root] = root;
  return root;
}

// Groups roots that may name the same buffer without provably doing so, which
// today means a control flow result whose branches forward different operands:
// it aliases one of them, decided at runtime.
//
// These cannot be merged into a single root the way a view can. A root stands
// for one buffer, and the members of a group are still distinct buffers - only
// one of them turns out to be shared - so their deallocations are not
// interchangeable and none of them is redundant.
class MayAliasGroups {
public:
  Value find(Value root) const {
    auto it = parent.find(root);
    while (it != parent.end() && it->second != root) {
      root = it->second;
      it = parent.find(root);
    }
    return root;
  }

  void join(Value lhs, Value rhs) {
    Value lhsGroup = find(lhs);
    Value rhsGroup = find(rhs);
    parent[lhsGroup] = lhsGroup;
    if (lhsGroup != rhsGroup) {
      parent[rhsGroup] = lhsGroup;
    }
  }

  bool isGrouped(Value root) const { return parent.contains(root); }

private:
  llvm::DenseMap<Value, Value> parent;
};

MayAliasGroups buildMayAliasGroups(func::FuncOp funcOp,
                                   llvm::DenseMap<Value, Value> &valueToRoot) {
  MayAliasGroups groups;
  funcOp.walk([&](Operation *op) {
    if (!mlir::isa<WhileOp, CaseOp>(op)) {
      return;
    }
    for (OpResult result : op->getResults()) {
      llvm::SmallVector<Value> forwarded =
          getForwardedOperands(op, result.getResultNumber());
      if (forwarded.size() < 2) {
        continue;
      }
      Value resultRoot = canonicalRoot(result, valueToRoot);
      for (Value operand : forwarded) {
        groups.join(resultRoot, canonicalRoot(operand, valueToRoot));
      }
    }
  });
  return groups;
}

} // namespace

// A `ttnn.deallocate` with the force flag set to false frees the buffer only
// when its input variable is the last one referencing that buffer. This
// becomes a problem when several handles alias one buffer: e.g. a
// view-eligible reshape op returns a tensor that points to its input's device
// buffer, and a control flow op returns the operand a region forwarded straight
// out of it. Deallocate ops are inserted in the IR per SSA value by the
// `TTNNDeallocate` pass. However, in the mentioned cases, they act as no-ops,
// so the buffer is never freed. This can result in L1 allocation failure.
//
// Getting the aliasing wrong the other way round is worse than a leak: forcing
// a deallocation of a buffer another live handle still names frees it out from
// under that handle.
//
// Aliasing comes in two strengths. A view *must* alias its source, so all the
// handles share one root and every deallocation but the last is a no-op that
// can be removed. A control flow result whose branches forward different
// operands only *may* alias each of them - which one is decided at runtime - so
// those handles are grouped rather than merged: they are still distinct
// buffers, no deallocation among them is redundant, and only the bottom-most is
// forced, to free the one that is shared and whose refcount therefore never
// drops to zero on its own.
//
// For each underlying buffer, this pass walks that buffer's deallocate ops from
// bottom to top and sets the force flag to true on the last one in program
// order (the true final use of that buffer), so the buffer is properly freed.
// For buffers that are freed elsewhere, no deallocation is forced and all of
// their (no-op) deallocations are removed. Those are buffers that escape the
// block computing them (returned from the function and freed by the caller, or
// yielded out of a region), buffers a region only borrows through its block
// arguments, and conv activations the conv op force-deallocates itself.
//
// Bottom-to-top only means program order because every deallocation of a
// block-owned buffer sits in that same block: a value defined in a region is
// invisible outside it, and an op in a region cannot alias a value from the
// enclosing scope while all region-carrying ops here are IsolatedFromAbove. A
// region op that is not isolated from above needs this revisited.
class TTNNForceFinalDeallocs
    : public impl::TTNNForceFinalDeallocsBase<TTNNForceFinalDeallocs> {
public:
  using impl::TTNNForceFinalDeallocsBase<
      TTNNForceFinalDeallocs>::TTNNForceFinalDeallocsBase;

  void runOnOperation() final {
    getOperation()->walk([&](func::FuncOp funcOp) {
      if (funcOp.isDeclaration()) {
        return;
      }
      assert(funcOp.getBody().hasOneBlock() &&
             "found func that didn't have one block!");
      processFunc(funcOp);
    });
  }

private:
  // Collects the roots that must never be force-freed: buffers that escape the
  // block that computes them, buffers a region is only borrowing, and conv
  // activations that the conv op deallocates itself.
  llvm::DenseSet<Value>
  collectDoNotForceRoots(func::FuncOp funcOp,
                         llvm::DenseMap<Value, Value> &valueToRoot) {
    llvm::DenseSet<Value> doNotForceRoots;
    funcOp.walk([&](Operation *op) {
      // A terminator hands its operands to whoever its block returns to, so
      // this block is not the one that frees them: the caller for func.return,
      // the enclosing scope or the next iteration for a region terminator such
      // as ttnn.yield.
      if (op->hasTrait<OpTrait::IsTerminator>()) {
        for (Value operand : op->getOperands()) {
          doNotForceRoots.insert(canonicalRoot(operand, valueToRoot));
        }
        return WalkResult::advance();
      }

      // A region's block arguments name buffers the region did not allocate:
      // they belong to the enclosing scope, or - across a loop back edge - to
      // the previous iteration. Functions are deliberately excluded: calling
      // one transfers ownership of its arguments, entering a region does not.
      if (!mlir::isa<func::FuncOp>(op)) {
        for (Region &region : op->getRegions()) {
          for (Block &block : region) {
            for (BlockArgument arg : block.getArguments()) {
              doNotForceRoots.insert(canonicalRoot(arg, valueToRoot));
            }
          }
        }
      }

      Value convActivation =
          llvm::TypeSwitch<Operation *, Value>(op)
              .Case<Conv2dOp, ConvTranspose2dOp>([](auto convOp) {
                return getConvDeallocatedActivation(convOp);
              })
              .Default(Value());
      if (convActivation) {
        doNotForceRoots.insert(canonicalRoot(convActivation, valueToRoot));
      }
      return WalkResult::advance();
    });
    return doNotForceRoots;
  }

  // Forces the last deallocation of each buffer that has more than one
  // (aliasing) deallocations.
  void processFunc(func::FuncOp funcOp) {
    // Resolves each value to the root value identifying its underlying
    // buffer.
    llvm::DenseMap<Value, Value> valueToRoot;

    // Built before anything else resolves a root, so that every later lookup
    // sees the same grouping.
    MayAliasGroups mayAliasGroups = buildMayAliasGroups(funcOp, valueToRoot);
    auto groupOf = [&](Value value) {
      return mayAliasGroups.find(canonicalRoot(value, valueToRoot));
    };

    // Buffers that are used outside the function (returned variables) or
    // deallocated by a conv op (conv op L1 activations) cannot be force-freed.
    llvm::DenseSet<Value> doNotForceGroups;
    for (Value root : collectDoNotForceRoots(funcOp, valueToRoot)) {
      doNotForceGroups.insert(mayAliasGroups.find(root));
    }

    // Count deallocations per buffer so we only touch buffers that
    // actually have multiple (aliasing) deallocations.
    llvm::SmallVector<DeallocateOp> deallocs;
    llvm::DenseMap<Value, unsigned> deallocCountByGroup;
    funcOp.walk([&](DeallocateOp deallocOp) {
      deallocs.push_back(deallocOp);
      deallocCountByGroup[groupOf(deallocOp.getInput())]++;
    });

    // Walk deallocations bottom-to-top and decide, per buffer, which single
    // deallocate (if any) should free it. All other deallocations of that
    // buffer are no-ops and are removed.
    llvm::DenseSet<Value> forcedGroups;
    llvm::SmallVector<DeallocateOp> redundantDeallocs;
    for (auto deallocOp : llvm::reverse(deallocs)) {
      Value group = groupOf(deallocOp.getInput());

      // A may-alias group holds distinct buffers, only one of which turns out
      // to be shared, so none of its deallocations is redundant: each frees
      // whichever buffer it alone owns. Only the bottom-most needs forcing, to
      // free the one that is shared and whose refcount therefore never drops
      // to zero on its own.
      bool mayAlias = mayAliasGroups.isGrouped(group);

      // The buffer is freed elsewhere: escapes the function (freed by the
      // caller) or is a conv activation the conv force-deallocates itself.
      if (doNotForceGroups.contains(group)) {
        if (!mayAlias) {
          redundantDeallocs.push_back(deallocOp);
        }
        continue;
      }

      // A single deallocate already frees the buffer (its input variable is the
      // sole reference), so leave it as is. A may-alias group is the exception:
      // a lone deallocation there may be the shared buffer's, whose refcount is
      // still above zero, so it has to be forced.
      if (!mayAlias && deallocCountByGroup.lookup(group) < 2) {
        continue;
      }

      // Multiple aliasing deallocations: the first one seen is the last in
      // program order, so force it. The rest are no-ops and are removed.
      if (forcedGroups.insert(group).second) {
        deallocOp.setForce(true);
      } else if (!mayAlias) {
        redundantDeallocs.push_back(deallocOp);
      }
    }

    for (DeallocateOp deallocOp : redundantDeallocs) {
      deallocOp->erase();
    }
  }
};

} // namespace mlir::tt::ttnn
