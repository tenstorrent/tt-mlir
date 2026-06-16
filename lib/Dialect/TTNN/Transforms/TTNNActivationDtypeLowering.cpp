// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNCCLACTIVATIONDTYPELOWERING
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

// View-like ops: data values pass through unchanged and the dtype is preserved.
//
// ToLayoutOp is intentionally NOT here: at this stage of the pipeline it can
// carry a dtype change, so it is not safe to treat as dtype-preserving. The
// patterns this pass currently handles (projection-residual, MLP up/gate) do
// not contain any ToLayoutOp on their chains. When the LM-head / QKV->RoPE
// patterns are added (they do have a ToLayoutOp), it should be admitted here
// only when its input and output dtypes match.
static bool isViewLikeOp(Operation *op) {
  return mlir::isa<ReshapeOp, SliceStaticOp, ToMemoryConfigOp>(op);
}

// CCL ops: shape changes, dtype is preserved.
static bool isCCLOp(Operation *op) {
  return mlir::isa<AllGatherOp, ReduceScatterOp, AllReduceOp>(op);
}

// Dtype carried by a value's TTNN layout encoding. Every tensor at this stage
// of the pipeline carries a TTNNLayoutAttr, so the dtype is always present;
// callers only pass TTNN op results / operands.
static ttcore::DataType getValueDtype(Value v) {
  auto rt = mlir::cast<RankedTensorType>(v.getType());
  auto layout = mlir::cast<TTNNLayoutAttr>(rt.getEncoding());
  return layout.getDataType();
}

// The float dtypes this pass is allowed to lower from. Activations entering a
// CCL chain are bf16 (or f32); anything else is left alone.
static bool isLowerableFloat(ttcore::DataType dtype) {
  return dtype == ttcore::DataType::BFloat16 ||
         dtype == ttcore::DataType::Float32;
}

// Rewrite an op's single result type so its encoding carries `targetDtype`.
// No-op if the op doesn't have a single ranked-tensor result, already produces
// `targetDtype`, or does not currently produce a lowerable float. Returns true
// on rewrite.
//
// TTNN reads an op's output dtype from its result layout encoding, so rewriting
// the encoding is all that's needed to set the output dtype.
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

// Down-cast `matmul`'s output and every flow-through op on its chain to
// bfp_bf8.
static void downcastChain(MatmulOp matmul,
                          llvm::ArrayRef<Operation *> flowOps) {
  rewriteResultDtype(matmul, ttcore::DataType::BFP_BFloat8);
  for (Operation *op : flowOps) {
    rewriteResultDtype(op, ttcore::DataType::BFP_BFloat8);
  }
}

// Projection-residual shape (read-only check):
//   matmul -> [view / CCL]* -> residual add
// walked as a single-use linear chain. Returns the flow-through ops on the
// chain when:
//   - the matmul currently produces a lowerable float,
//   - the chain is a single-use line of view / CCL ops with at least one CCL,
//   - it terminates at an AddOp whose *other* operand is a bf16/f32 residual.
// Otherwise returns nullopt.
//
// The residual add is where the block returns to bf16: TTNN combines the
// bfp_bf8 CCL output with the bf16 residual operand to produce a bf16 result,
// so no explicit cast is needed at the add. Used by the projection matcher (to
// lower it) and by the MLP matcher (to confirm the gate's FF2 matmul -- a
// *different* matmul -- ends at such an add).
static std::optional<llvm::SmallVector<Operation *>>
matmulReachesResidualAdd(MatmulOp matmul) {
  if (!isLowerableFloat(getValueDtype(matmul.getResult()))) {
    return std::nullopt;
  }

  llvm::SmallVector<Operation *> flowOps;
  bool sawCCL = false;
  Value cur = matmul.getResult();
  while (true) {
    // A clean chain is single-use; a branch means this isn't the shape we
    // model, so don't match (under-trigger by design).
    if (!cur.hasOneUse()) {
      return std::nullopt;
    }
    OpOperand &use = *cur.getUses().begin();
    Operation *user = use.getOwner();

    if (isViewLikeOp(user) || isCCLOp(user)) {
      if (user->getNumResults() != 1) {
        return std::nullopt;
      }
      sawCCL |= isCCLOp(user);
      flowOps.push_back(user);
      cur = user->getResult(0);
      continue;
    }

    // Exit: must be a residual add whose other operand is a bf16/f32 residual.
    auto add = mlir::dyn_cast<AddOp>(user);
    if (!sawCCL || !add) {
      return std::nullopt;
    }
    unsigned otherIdx = 1 - use.getOperandNumber();
    if (!isLowerableFloat(getValueDtype(add->getOperand(otherIdx)))) {
      return std::nullopt;
    }
    return flowOps;
  }
}

// Projection matmul (attention output / MLP down) -> [view / CCL]* ->
// residual add. Lowers the producer matmul output to bfp_bf8 and propagates it
// through every flow-through op; the residual add's encoding is left as bf16.
//
// Also matches the FF2 -> add (residual) tail of the MLP pattern.
static bool tryProjResidualAddMatcher(MatmulOp matmul) {
  auto flowOps = matmulReachesResidualAdd(matmul);
  if (!flowOps) {
    return false;
  }
  downcastChain(matmul, *flowOps);
  return true;
}

// MLP up / gate matmuls -> [view / CCL / silu]* -> gate multiply.
//
//   FF1 (up)   matmul -> CCL -> silu --\
//                                       multiply (gate) -> FF2 (down) matmul
//   FF3 (gate) matmul -> CCL ----------/
//
// Lowers the producer matmul output to bfp_bf8, propagates it through the
// producer's view / CCL / silu chain, and rewrites the gate multiply's result
// to bfp_bf8 so the downstream FF2 matmul receives bfp_bf8 input. bf16 is
// restored later, at the FF2 residual add, when FF2 is matched by the
// projection matcher on its own visit.
//
// The gate multiply is matched strictly: it must have a single use, that use
// must be the FF2 matmul, and FF2 must itself reach a residual add (the bf16
// restore point). This rejects multiplies that fan out or whose output is not
// a real MLP down-projection.
static bool tryMLPUpGateMatcher(MatmulOp matmul) {
  if (!isLowerableFloat(getValueDtype(matmul.getResult()))) {
    return false;
  }

  // Walk the single-use chain from the matmul to its first non-flow user.
  llvm::SmallVector<Operation *> flowOps;
  bool sawCCL = false;
  Value cur = matmul.getResult();
  Operation *exit = nullptr;
  while (true) {
    if (!cur.hasOneUse()) {
      return false;
    }
    OpOperand &use = *cur.getUses().begin();
    Operation *user = use.getOwner();

    if (isViewLikeOp(user) || isCCLOp(user) || mlir::isa<SiluOp>(user)) {
      if (user->getNumResults() != 1) {
        return false;
      }
      sawCCL |= isCCLOp(user);
      flowOps.push_back(user);
      cur = user->getResult(0);
      continue;
    }
    exit = user;
    break;
  }
  if (!sawCCL) {
    return false;
  }

  // Exit must be the gate multiply feeding the FF2 matmul, and FF2 must reach a
  // residual add downstream (where bf16 is restored).
  auto gate = mlir::dyn_cast<MultiplyOp>(exit);
  if (!gate || !gate.getResult().hasOneUse()) {
    return false;
  }
  auto ff2 = mlir::dyn_cast<MatmulOp>(*gate.getResult().getUsers().begin());
  if (!ff2 || !matmulReachesResidualAdd(ff2)) {
    return false;
  }

  downcastChain(matmul, flowOps);
  // Down-cast the gate multiply output so FF2 receives bfp_bf8 input. This only
  // changes what the multiply hands to its consumer (the FF2 matmul, which
  // accepts any input dtype); the other gate branch is lowered by this same
  // matcher on its own producer matmul.
  rewriteResultDtype(gate, ttcore::DataType::BFP_BFloat8);
  return true;
}

class TTNNCCLActivationDtypeLoweringPass
    : public impl::TTNNCCLActivationDtypeLoweringBase<
          TTNNCCLActivationDtypeLoweringPass> {
public:
  using impl::TTNNCCLActivationDtypeLoweringBase<
      TTNNCCLActivationDtypeLoweringPass>::TTNNCCLActivationDtypeLoweringBase;

  void runOnOperation() final {
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
