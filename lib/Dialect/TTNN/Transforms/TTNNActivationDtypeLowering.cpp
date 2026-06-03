// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNACTIVATIONDTYPELOWERING
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

// View-like ops: data values pass through unchanged; dtype is preserved.
static bool isViewLikeOp(Operation *op) {
  return mlir::isa<ReshapeOp, SliceStaticOp, ToMemoryConfigOp, ToLayoutOp>(op);
}

// CCL ops: shape changes, dtype is preserved.
static bool isCCLOp(Operation *op) {
  return mlir::isa<AllGatherOp, ReduceScatterOp, AllReduceOp>(op);
}

// Dtype carried by a value's TTNN layout encoding, if it has one.
static std::optional<ttcore::DataType> getValueDtype(Value v) {
  auto rt = mlir::dyn_cast<RankedTensorType>(v.getType());
  if (!rt) {
    return std::nullopt;
  }
  auto layout = mlir::dyn_cast_if_present<TTNNLayoutAttr>(rt.getEncoding());
  if (!layout) {
    return std::nullopt;
  }
  return layout.getDataType();
}

// The float dtypes this pass is allowed to lower from. Activations entering a
// CCL chain are bf16 (or f32); anything else is left alone.
static bool isLowerableFloat(std::optional<ttcore::DataType> dtype) {
  return dtype && (*dtype == ttcore::DataType::BFloat16 ||
                   *dtype == ttcore::DataType::Float32);
}

// Rewrite an op's single result type so its encoding carries `targetDtype`.
// No-op if the op already produces `targetDtype` or has no TTNN layout
// encoding. Returns true when a rewrite occurred.
//
// TTNN reads an op's output dtype from its result layout encoding, so
// rewriting the encoding is all that's needed to set the output dtype.
static bool rewriteResultDtype(Operation *op, ttcore::DataType targetDtype) {
  if (op->getNumResults() != 1) {
    return false;
  }
  Value result = op->getResult(0);
  auto rt = mlir::dyn_cast<RankedTensorType>(result.getType());
  if (!rt || !isLowerableFloat(getValueDtype(result))) {
    return false;
  }
  if (getValueDtype(result) == targetDtype) {
    return false;
  }
  auto newType = ttnn::utils::RankedTensorTypeFactory::create(rt, targetDtype);
  result.setType(newType);
  return true;
}

// Walks the def-use chain downstream from `start`, treating ops for which
// `isFlowThrough(op)` returns true as transparent (their results are also
// followed). Collects the visited flow-through ops in `flowThroughOps` and
// the first non-flow-through users in `exits` (paired with the operand index
// at which `start`'s descendant feeds them).
template <typename FlowThroughFn>
static void
collectChain(Value start, FlowThroughFn isFlowThrough,
             llvm::SmallVectorImpl<Operation *> &flowThroughOps,
             llvm::SmallVectorImpl<std::pair<Operation *, unsigned>> &exits) {
  llvm::SmallPtrSet<Operation *, 16> seenFlow;
  llvm::SmallVector<Value, 8> worklist;
  worklist.push_back(start);
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    for (OpOperand &use : v.getUses()) {
      Operation *user = use.getOwner();
      if (isFlowThrough(user)) {
        if (seenFlow.insert(user).second) {
          flowThroughOps.push_back(user);
          for (Value result : user->getResults()) {
            worklist.push_back(result);
          }
        }
      } else {
        exits.emplace_back(user, use.getOperandNumber());
      }
    }
  }
}

// Validate-only core of both matchers. A matmul matches when:
//   - its result reaches one or more exits through a chain of `isFlow` ops,
//   - at least one of those flow-through ops is a CCL (otherwise there is no
//     ethernet traffic to save and nothing to lower),
//   - every exit satisfies `isValidExit(exitOp, operandIdx)`, and
//   - the matmul itself currently produces a lowerable float (bf16 / f32).
//
// On success the visited flow-through ops and exits are returned via the
// out-params. Mutates nothing, so it doubles as a read-only predicate.
template <typename FlowFn, typename ExitFn>
static bool matchMatmulChain(
    MatmulOp matmul, FlowFn isFlow, ExitFn isValidExit,
    llvm::SmallVectorImpl<Operation *> &flowThroughOps,
    llvm::SmallVectorImpl<std::pair<Operation *, unsigned>> &exits) {
  collectChain(matmul.getResult(), isFlow, flowThroughOps, exits);

  if (exits.empty() || !llvm::any_of(flowThroughOps, isCCLOp)) {
    return false;
  }
  for (auto [exitOp, operandIdx] : exits) {
    if (!isValidExit(exitOp, operandIdx)) {
      return false;
    }
  }
  return isLowerableFloat(getValueDtype(matmul.getResult()));
}

// Lowering driver: match, then (only on a full match) lower the matmul output
// to bfp_bf8 and propagate bfp_bf8 through every flow-through op. When
// `rewriteExits` is set, each exit's result encoding is also rewritten to
// bfp_bf8 (used to hand bfp_bf8 to a downstream consumer; see the MLP matcher).
template <typename FlowFn, typename ExitFn>
static bool tryLowerMatmulChain(MatmulOp matmul, FlowFn isFlow,
                                ExitFn isValidExit, bool rewriteExits) {
  llvm::SmallVector<Operation *> flowThroughOps;
  llvm::SmallVector<std::pair<Operation *, unsigned>> exits;
  if (!matchMatmulChain(matmul, isFlow, isValidExit, flowThroughOps, exits)) {
    return false;
  }

  rewriteResultDtype(matmul, ttcore::DataType::BFP_BFloat8);
  for (Operation *flow : flowThroughOps) {
    rewriteResultDtype(flow, ttcore::DataType::BFP_BFloat8);
  }
  if (rewriteExits) {
    for (const auto &exit : exits) {
      rewriteResultDtype(exit.first, ttcore::DataType::BFP_BFloat8);
    }
  }
  return true;
}

// Projection-residual chain shape:
//   matmul -> [view / CCL]* -> residual add, where the add's *other* operand is
// the bf16/f32 residual stream. The residual add is where the block returns to
// bf16: TTNN combines the bfp_bf8 CCL output with the bf16 residual operand to
// produce a bf16 result, so no explicit cast is needed at the add.
//
// Shared by the projection matcher (which lowers it) and the MLP matcher (which
// uses it read-only to confirm the gate's FF2 matmul ends at such an add).

static bool isProjChainFlow(Operation *op) {
  return isViewLikeOp(op) || isCCLOp(op);
}

static bool isProjChainExit(Operation *exitOp, unsigned operandIdx) {
  auto add = mlir::dyn_cast<AddOp>(exitOp);
  if (!add) {
    return false;
  }
  return isLowerableFloat(getValueDtype(add->getOperand(1 - operandIdx)));
}

// Read-only: does `matmul` root a projection-residual chain?
static bool isProjResidualMatmul(MatmulOp matmul) {
  llvm::SmallVector<Operation *> flowThroughOps;
  llvm::SmallVector<std::pair<Operation *, unsigned>> exits;
  return matchMatmulChain(matmul, isProjChainFlow, isProjChainExit,
                          flowThroughOps, exits);
}

// Projection matmul (attention output / MLP down) -> [view / CCL]* ->
// residual add (see "Projection-residual chain shape" above). Lowers the
// producer matmul output to bfp_bf8 and propagates it through every
// flow-through op; the residual add's encoding is left as bf16.
//
// Also used for the FF2 -> add (residual) tail of the MLP pattern.
static bool tryProjResidualAddMatcher(MatmulOp matmul) {
  return tryLowerMatmulChain(matmul, isProjChainFlow, isProjChainExit,
                             /*rewriteExits=*/false);
}

// MLP up / gate matmuls -> [view / CCL]* -> silu (FF1) or multiply (FF3).
//
// FF1 path:  matmul -> CCL -> silu -> multiply.
// FF3 path:  matmul -> CCL -> multiply.
// Both branches converge at the gate multiply whose two inputs are the two
// CCL-rooted chains; the multiply feeds the FF2 (down) matmul, whose residual
// add then restores bf16 (handled by the projection-residual matcher on FF2's
// own invocation).
//
// Action: lower the producer matmul output to bfp_bf8, propagate bfp_bf8
// through view / CCL / silu on the producer's branch, and rewrite the gate
// multiply's result encoding to bfp_bf8 so the downstream FF2 matmul receives
// bfp_bf8 input.
//
// The exit check enforces the real MLP shape: the single-use requirement
// rejects gate multiplies that fan out to consumers we should not silently
// downcast, and the FF2/proj-residual check guarantees the bf16 restore point
// downstream actually exists.
static bool tryMLPUpGateMatcher(MatmulOp matmul) {
  auto isFlow = [](Operation *op) {
    return isViewLikeOp(op) || isCCLOp(op) || mlir::isa<SiluOp>(op);
  };
  auto isValidExit = [](Operation *exitOp, unsigned /*operandIdx*/) {
    auto mul = mlir::dyn_cast<MultiplyOp>(exitOp);
    if (!mul || !mul.getResult().hasOneUse()) {
      return false;
    }
    auto ff2 = mlir::dyn_cast<MatmulOp>(*mul.getResult().getUsers().begin());
    return ff2 && isProjResidualMatmul(ff2);
  };
  return tryLowerMatmulChain(matmul, isFlow, isValidExit,
                             /*rewriteExits=*/true);
}

class TTNNActivationDtypeLoweringPass
    : public impl::TTNNActivationDtypeLoweringBase<
          TTNNActivationDtypeLoweringPass> {
public:
  using impl::TTNNActivationDtypeLoweringBase<
      TTNNActivationDtypeLoweringPass>::TTNNActivationDtypeLoweringBase;

  void runOnOperation() final {
    if (!enable) {
      return;
    }

    ModuleOp moduleOp = getOperation();

    // Collect matmuls in a stable order before mutating; rewrites change the
    // IR and we don't want to revisit a transformed matmul.
    llvm::SmallVector<MatmulOp> matmuls;
    moduleOp.walk([&](MatmulOp op) { matmuls.push_back(op); });

    for (MatmulOp matmul : matmuls) {
      // Try matchers in order. Each matcher is independent — the first one
      // that succeeds claims the matmul. Strict matching means the matchers
      // do not overlap.
      if (tryProjResidualAddMatcher(matmul)) {
        continue;
      }
      if (tryMLPUpGateMatcher(matmul)) {
        continue;
      }
    }
  }
};

} // namespace

} // namespace mlir::tt::ttnn
