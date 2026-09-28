// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/DataMovementRules.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNFORCEFINALDEALLOCS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

// Maps a `ttnn.while` region's block argument to the operand it is bound to.
// The region observes `inits ++ captures`.
Value getBoundOperand(WhileOp op, unsigned argNumber) {
  unsigned numInits = op.getInits().size();
  if (argNumber < numInits) {
    return op.getInits()[argNumber];
  }
  return op.getCaptures()[argNumber - numInits];
}

// Maps a `ttnn.case` branch's block argument to the operand it is bound to.
// A branch observes just the captures.
Value getBoundOperand(CaseOp op, unsigned argNumber) {
  return op.getCaptures()[argNumber];
}

using Origins = llvm::SmallSetVector<Value, 4>;

Origins getOrigins(OpResult result);

// Collects what `value` may be, seen from its block: a block argument it
// forwards (directly, through views or nested control flow), or the value that
// owns its buffer.
void collectOrigins(Value value, Origins &origins) {
  Operation *op = value.getDefiningOp();
  while (op && canReshapeBeView(op)) {
    value = op->getOperand(0);
    op = value.getDefiningOp();
  }
  if (!mlir::isa_and_present<WhileOp, CaseOp>(op)) {
    origins.insert(value);
    return;
  }
  // A nested op's operands live in this block, so follow them further.
  for (Value origin : getOrigins(mlir::cast<OpResult>(value))) {
    if (llvm::is_contained(op->getOperands(), origin)) {
      collectOrigins(origin, origins);
    } else {
      origins.insert(origin);
    }
  }
}

// Adds the origins of `yielded`, seen from outside `op`, mapping `block`'s
// arguments to their bound operands. Returns the mapped argument numbers.
template <typename OpTy>
llvm::SmallVector<unsigned> addYieldOrigins(OpTy op, Block &block,
                                            Value yielded, Origins &origins) {
  Origins yieldOrigins;
  collectOrigins(yielded, yieldOrigins);
  llvm::SmallVector<unsigned> argNumbers;
  for (Value origin : yieldOrigins) {
    auto arg = mlir::dyn_cast<BlockArgument>(origin);
    if (arg && arg.getOwner() == &block) {
      argNumbers.push_back(arg.getArgNumber());
      origin = getBoundOperand(op, arg.getArgNumber());
    }
    origins.insert(origin);
  }
  return argNumbers;
}

// What a control flow result may be: operands the op forwards (inits or
// captures), or buffers its regions allocate. A case result may be what any
// branch yields. A while result is its init if the loop may run zero times, and
// otherwise what the body yields into its slot; a carried value yielded there
// leads to its own slot, so e.g. after a swap result 0 may be init 1.
Origins getOrigins(OpResult result) {
  Origins origins;
  unsigned resultNumber = result.getResultNumber();

  if (auto caseOp = mlir::dyn_cast<CaseOp>(result.getOwner())) {
    for (unsigned i = 0; i < caseOp.getBranches().size(); ++i) {
      addYieldOrigins(caseOp, caseOp.getBranchBlock(i),
                      caseOp.getBranchYield(i).getOperand(resultNumber),
                      origins);
    }
    return origins;
  }

  auto whileOp = mlir::cast<WhileOp>(result.getOwner());
  if (whileOp.getTripCount().value_or(0) == 0) {
    // If it is possible that while runs zero times, its results are possibly
    // just a view on inits.
    origins.insert(whileOp.getInits()[resultNumber]);
  }
  unsigned numInits = whileOp.getInits().size();

  llvm::SmallVector<unsigned> stack = {resultNumber};
  llvm::SmallDenseSet<unsigned> visited = {resultNumber};
  while (!stack.empty()) {
    Value yielded = whileOp.getBodyYield().getOperand(stack.pop_back_val());
    for (unsigned argNumber :
         addYieldOrigins(whileOp, whileOp.getBodyBlock(), yielded, origins)) {
      if (argNumber < numInits && !visited.contains(argNumber)) {
        visited.insert(argNumber);
        stack.push_back(argNumber);
      }
    }
  }
  return origins;
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

// Groups handles that may name the same buffer. A must-alias group (a view and
// its source, or a result every path forwards) is one buffer. A may-alias group
// (a result that depends on the path) holds distinct buffers, one of which is
// shared at runtime.
class AliasGroups {
public:
  Value find(Value value) const {
    auto it = parent.find(value);
    while (it != parent.end()) {
      value = it->second;
      it = parent.find(value);
    }
    return value;
  }

  void join(Value lhs, Value rhs, bool mayAlias) {
    Value lhsGroup = find(lhs);
    Value rhsGroup = find(rhs);
    if (lhsGroup != rhsGroup) {
      parent[rhsGroup] = lhsGroup;
      mayAlias |= mayAliasGroups.erase(rhsGroup);
    }
    if (mayAlias) {
      mayAliasGroups.insert(lhsGroup);
    }
  }

  bool isMayAlias(Value group) const { return mayAliasGroups.contains(group); }

private:
  llvm::DenseMap<Value, Value> parent;
  llvm::DenseSet<Value> mayAliasGroups;
};

// The one place that knows what aliases: view-eligible reshapes, and control
// flow results that forward an operand. A future ViewOpInterface can replace
// the reshape check.
AliasGroups buildAliasGroups(func::FuncOp funcOp) {
  AliasGroups groups;
  funcOp.walk([&](Operation *op) {
    if (canReshapeBeView(op)) {
      groups.join(op->getResult(0), op->getOperand(0), /*mayAlias=*/false);
      return;
    }
    if (!mlir::isa<WhileOp, CaseOp>(op)) {
      return;
    }

    llvm::SmallVector<Origins> origins =
        llvm::map_to_vector(op->getResults(), getOrigins);
    auto isOperand = [&](Value value) {
      return llvm::is_contained(op->getOperands(), value);
    };
    for (OpResult result : op->getResults()) {
      const Origins &resultOrigins = origins[result.getResultNumber()];
      bool mustAlias =
          resultOrigins.size() == 1 && isOperand(resultOrigins.front());
      for (Value origin : resultOrigins) {
        if (isOperand(origin)) {
          groups.join(result, origin, /*mayAlias=*/!mustAlias);
          continue;
        }
        // A region-allocated buffer, shared with any other result that may be
        // it (e.g. a value yielded twice).
        for (OpResult other : op->getResults()) {
          if (other != result &&
              origins[other.getResultNumber()].contains(origin)) {
            groups.join(result, other, /*mayAlias=*/true);
          }
        }
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
// For each group of aliasing handles (see AliasGroups), this pass forces the
// last deallocation in program order. The other deallocations are removed in a
// must-alias group and kept in a may-alias one, where each frees the buffer it
// alone owns. Nothing is forced for buffers freed elsewhere: those that escape
// their block (returned or yielded), region block arguments, buffers that
// outlive the call (constant and parameter arguments, const-eval results), and
// conv activations the conv frees itself.
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
  // Collects the groups that must never be force-freed: buffers that escape the
  // block that computes them, buffers a region is only borrowing, buffers that
  // outlive the call, and conv activations that the conv op deallocates itself.
  llvm::DenseSet<Value> collectDoNotForceGroups(func::FuncOp funcOp,
                                                const AliasGroups &groups) {
    llvm::DenseSet<Value> doNotForceGroups;

    // Weights and const-eval results are reused by the next call.
    // TTNNAdjustDeallocs removed their own deallocations, not those of views or
    // control flow results that alias them.
    for (BlockArgument arg : ttcore::getConstsAndParams(funcOp)) {
      doNotForceGroups.insert(groups.find(arg));
    }

    funcOp.walk([&](Operation *op) {
      if (mlir::isa<ttcore::LoadCachedOp>(op)) {
        for (Value result : op->getResults()) {
          doNotForceGroups.insert(groups.find(result));
        }
        return WalkResult::advance();
      }

      // A terminator hands its operands to whoever its block returns to, so
      // this block is not the one that frees them: the caller for func.return,
      // the enclosing scope or the next iteration for a region terminator such
      // as ttnn.yield.
      if (op->hasTrait<OpTrait::IsTerminator>()) {
        for (Value operand : op->getOperands()) {
          doNotForceGroups.insert(groups.find(operand));
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
              doNotForceGroups.insert(groups.find(arg));
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
        doNotForceGroups.insert(groups.find(convActivation));
      }
      return WalkResult::advance();
    });
    return doNotForceGroups;
  }

  // Forces the last deallocation of each buffer that has more than one
  // (aliasing) deallocations.
  void processFunc(func::FuncOp funcOp) {
    AliasGroups groups = buildAliasGroups(funcOp);
    llvm::DenseSet<Value> doNotForceGroups =
        collectDoNotForceGroups(funcOp, groups);

    // Count deallocations per group so we only touch buffers that actually
    // have multiple (aliasing) deallocations.
    llvm::SmallVector<DeallocateOp> deallocs;
    llvm::DenseMap<Value, unsigned> deallocCountByGroup;
    funcOp.walk([&](DeallocateOp deallocOp) {
      deallocs.push_back(deallocOp);
      deallocCountByGroup[groups.find(deallocOp.getInput())]++;
    });

    // Walk deallocations bottom-to-top and decide, per group, which single
    // deallocate (if any) should be forced.
    llvm::DenseSet<Value> forcedGroups;
    llvm::SmallVector<DeallocateOp> redundantDeallocs;
    for (auto deallocOp : llvm::reverse(deallocs)) {
      Value group = groups.find(deallocOp.getInput());
      bool mayAlias = groups.isMayAlias(group);

      // The buffer is freed elsewhere: escapes the function (freed by the
      // caller), outlives the call, or is a conv activation the conv
      // force-deallocates itself.
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

      // The first one seen is the last in program order, so force it. The rest
      // of a must-alias group are no-ops and are removed.
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
