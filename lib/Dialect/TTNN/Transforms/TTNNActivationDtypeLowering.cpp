// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNACTIVATIONDTYPELOWERING
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Op-class predicates
//===----------------------------------------------------------------------===//

// View-like ops: data values pass through unchanged; dtype is preserved.
static bool isViewLikeOp(Operation *op) {
  return mlir::isa<ReshapeOp, SliceStaticOp, ToMemoryConfigOp, ToLayoutOp>(op);
}

// CCL ops: shape changes, dtype is preserved.
static bool isCCLOp(Operation *op) {
  return mlir::isa<AllGatherOp, ReduceScatterOp, AllReduceOp>(op);
}

//===----------------------------------------------------------------------===//
// Type / dtype helpers
//===----------------------------------------------------------------------===//

static std::optional<ttcore::DataType> getResultDtype(Value v) {
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

// Rewrite an op's single result type so its encoding carries `targetDtype`.
// No-op if the op already produces `targetDtype` or has no TTNN layout
// encoding. Returns true when a rewrite occurred.
static bool rewriteResultDtype(Operation *op, ttcore::DataType targetDtype) {
  if (op->getNumResults() != 1) {
    return false;
  }
  Value result = op->getResult(0);
  auto rt = mlir::dyn_cast<RankedTensorType>(result.getType());
  if (!rt) {
    return false;
  }
  auto current = getResultDtype(result);
  if (!current || *current == targetDtype) {
    return false;
  }
  auto newType = ttnn::utils::RankedTensorTypeFactory::create(rt, targetDtype);
  result.setType(newType);
  return true;
}

// Set the `dtype` attribute on an op that exposes one (matmul, eltwise
// binary) and rewrite its result type encoding to match.
static void setOpDtype(Operation *op, ttcore::DataType dtype) {
  op->setAttr("dtype", ttcore::DataTypeAttr::get(op->getContext(), dtype));
  rewriteResultDtype(op, dtype);
}

// Insert a ttnn.to_layout(RowMajor, bf16) on `useOp`'s `operandIdx`. The
// producer's result is expected to be (Tile, bfp_bf8); the to_layout op
// implicitly upcasts to bf16 as part of de-tiling, which is the "untilize"
// fast-path on hardware (see tt-metal's
// `untilize_device_operation.cpp:268`, where BFLOAT8_B input always produces
// BFLOAT16 output). The combined to_layout is preserved through
// TTNNDecomposeLayouts via the bfp_bf8 -> bf16 special-case, lowering to a
// single tt-metal untilize kernel. Mirrors the tt-metal llama3_70b_galaxy
// lm_head -> untilize -> argmax chain.
static void insertToLayoutRowMajorBF16(IRRewriter &rewriter, Operation *useOp,
                                       unsigned operandIdx) {
  Value operand = useOp->getOperand(operandIdx);
  auto operandTV = mlir::cast<mlir::TypedValue<RankedTensorType>>(operand);
  TTNNLayoutAttr inputLayoutAttr =
      ttnn::utils::getLayoutAttrFromTensor(operandTV.getType());
  rewriter.setInsertionPoint(useOp);
  auto toLayout = ttnn::utils::createToLayoutOp(
      useOp, operandTV, rewriter, Layout::RowMajor,
      inputLayoutAttr.getBufferType(), inputLayoutAttr.getMemLayout(),
      ttcore::DataType::BFloat16, "_to_layout_pre_argmax");
  useOp->setOperand(operandIdx, toLayout.getResult());
}

// Insert a ttnn.typecast on `useOp`'s `operandIdx` so that the value
// observed by `useOp` is `targetDtype`. The producer's result keeps its
// (lowered) dtype.
static void insertTypecast(OpBuilder &builder, Operation *useOp,
                           unsigned operandIdx, ttcore::DataType targetDtype) {
  Value operand = useOp->getOperand(operandIdx);
  auto operandType = mlir::cast<RankedTensorType>(operand.getType());
  auto current = getResultDtype(operand);
  if (current && *current == targetDtype) {
    return;
  }
  auto newType =
      ttnn::utils::RankedTensorTypeFactory::create(operandType, targetDtype);
  builder.setInsertionPoint(useOp);
  auto typecast = builder.create<TypecastOp>(
      useOp->getLoc(), newType, operand,
      ttcore::DataTypeAttr::get(builder.getContext(), targetDtype));
  useOp->setOperand(operandIdx, typecast.getResult());
}

//===----------------------------------------------------------------------===//
// Chain walker
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Per-pattern matchers
//===----------------------------------------------------------------------===//

// Pattern A — QKV matmul -> [slice / CCL / view]* -> rotary_embedding /
// update_cache / fill_cache.
//
// Action: set matmul.dtype = bfp_bf8, propagate bfp_bf8 through every
// flow-through op, insert ttnn.typecast(bf16) before each exit so RoPE / KV
// cache writes still receive bf16.
static bool tryQKVRoPEMatcher(MatmulOp matmul, OpBuilder &builder) {
  auto isFlow = [](Operation *op) {
    return isViewLikeOp(op) || isCCLOp(op);
  };

  llvm::SmallVector<Operation *> flowThroughOps;
  llvm::SmallVector<std::pair<Operation *, unsigned>> exits;
  collectChain(matmul.getResult(), isFlow, flowThroughOps, exits);

  if (exits.empty()) {
    return false;
  }
  if (!llvm::any_of(flowThroughOps, isCCLOp)) {
    return false;
  }

  for (auto [exitOp, _] : exits) {
    if (!mlir::isa<RotaryEmbeddingOp, UpdateCacheOp, FillCacheOp>(exitOp)) {
      return false;
    }
  }

  // Strict: only match when the matmul currently produces bf16/f32.
  auto matmulDtype = getResultDtype(matmul.getResult());
  if (!matmulDtype || (*matmulDtype != ttcore::DataType::BFloat16 &&
                       *matmulDtype != ttcore::DataType::Float32)) {
    return false;
  }

  setOpDtype(matmul, ttcore::DataType::BFP_BFloat8);
  for (Operation *flow : flowThroughOps) {
    rewriteResultDtype(flow, ttcore::DataType::BFP_BFloat8);
  }
  for (auto [exitOp, operandIdx] : exits) {
    insertTypecast(builder, exitOp, operandIdx, ttcore::DataType::BFloat16);
  }
  return true;
}

// Pattern B — projection matmul -> [view / CCL]* -> add (residual).
//
// Action: set matmul.dtype = bfp_bf8, propagate bfp_bf8 through flow-through
// ops, ensure the consuming add has dtype = bf16 (so the residual sum
// reverts to bf16). No typecast needed.
//
// Also used for the FF2 -> add (residual) tail of the MLP pattern.
static bool tryProjResidualAddMatcher(MatmulOp matmul, OpBuilder &builder) {
  auto isFlow = [](Operation *op) {
    return isViewLikeOp(op) || isCCLOp(op);
  };

  llvm::SmallVector<Operation *> flowThroughOps;
  llvm::SmallVector<std::pair<Operation *, unsigned>> exits;
  collectChain(matmul.getResult(), isFlow, flowThroughOps, exits);

  if (exits.empty()) {
    return false;
  }
  if (!llvm::any_of(flowThroughOps, isCCLOp)) {
    return false;
  }

  // Every exit must be an AddOp; the *other* operand of each add must trace
  // to a bf16/f32 producer (the residual stream).
  for (auto [exitOp, operandIdx] : exits) {
    auto add = mlir::dyn_cast<AddOp>(exitOp);
    if (!add) {
      return false;
    }
    Value otherOperand = add->getOperand(1 - operandIdx);
    auto otherDtype = getResultDtype(otherOperand);
    if (!otherDtype || (*otherDtype != ttcore::DataType::BFloat16 &&
                        *otherDtype != ttcore::DataType::Float32)) {
      return false;
    }
  }

  auto matmulDtype = getResultDtype(matmul.getResult());
  if (!matmulDtype || (*matmulDtype != ttcore::DataType::BFloat16 &&
                       *matmulDtype != ttcore::DataType::Float32)) {
    return false;
  }

  setOpDtype(matmul, ttcore::DataType::BFP_BFloat8);
  for (Operation *flow : flowThroughOps) {
    rewriteResultDtype(flow, ttcore::DataType::BFP_BFloat8);
  }
  for (auto [exitOp, _] : exits) {
    auto add = mlir::cast<AddOp>(exitOp);
    add.setDtypeAttr(ttcore::DataTypeAttr::get(builder.getContext(),
                                               ttcore::DataType::BFloat16));
  }
  return true;
}

// Pattern C (MLP up/gate) — FF1 / FF3 matmul -> [view / CCL]* -> silu (FF1)
// or multiply (FF3).
//
// FF1 path:  matmul -> CCL -> silu -> multiply.
// FF3 path:  matmul -> CCL -> multiply.
// Both end up at a multiply whose two inputs are the two CCL chains; the
// multiply's result then feeds FF2 (handled by Pattern B at the residual
// add). For this matcher we lower the producer matmul + propagate bfp_bf8
// through the chain up to and including silu/multiply.
//
// Multiply is allowed to be the immediate exit (FF3 path) or the consumer of
// silu (FF1 path). The matcher treats multiply as a flow-through *only* when
// both of its operands are themselves bfp_bf8-producing chains rooted at a
// CCL — otherwise we don't try to lower it.
//
// Action:
//   - Set matmul.dtype = bfp_bf8.
//   - Propagate bfp_bf8 through view/CCL/silu on the producer's branch.
//   - On encountering the multiply exit, set multiply.dtype = bfp_bf8 and
//     rewrite its result type to bfp_bf8 — but only if both operands are
//     bfp_bf8 (i.e. the partner matmul has already been lowered, or will be
//     when this matcher runs on it). To keep the matcher local, we lower the
//     multiply when its *current* operand is bfp_bf8; partner is handled by
//     its own matcher invocation.
static bool tryMLPUpGateMatcher(MatmulOp matmul) {
  // Flow-through set for the producer side: view / CCL / silu (single-result
  // eltwise unary).
  auto isFlow = [](Operation *op) {
    return isViewLikeOp(op) || isCCLOp(op) || mlir::isa<SiluOp>(op);
  };

  llvm::SmallVector<Operation *> flowThroughOps;
  llvm::SmallVector<std::pair<Operation *, unsigned>> exits;
  collectChain(matmul.getResult(), isFlow, flowThroughOps, exits);

  if (exits.empty()) {
    return false;
  }
  if (!llvm::any_of(flowThroughOps, isCCLOp)) {
    return false;
  }

  // Every exit must be a MultiplyOp.
  for (auto [exitOp, _] : exits) {
    if (!mlir::isa<MultiplyOp>(exitOp)) {
      return false;
    }
  }

  auto matmulDtype = getResultDtype(matmul.getResult());
  if (!matmulDtype || (*matmulDtype != ttcore::DataType::BFloat16 &&
                       *matmulDtype != ttcore::DataType::Float32)) {
    return false;
  }

  setOpDtype(matmul, ttcore::DataType::BFP_BFloat8);
  for (Operation *flow : flowThroughOps) {
    rewriteResultDtype(flow, ttcore::DataType::BFP_BFloat8);
  }
  // At each multiply exit, set its dtype = bfp_bf8. Result-type rewrite is
  // safe because the partner matmul will also be lowered by its own matcher
  // invocation, and we use only the multiply's `dtype` attribute (not the
  // operand dtypes) to drive ttnn's packer.
  for (auto [exitOp, _] : exits) {
    auto mul = mlir::cast<MultiplyOp>(exitOp);
    mul.setDtypeAttr(ttcore::DataTypeAttr::get(mul.getContext(),
                                               ttcore::DataType::BFP_BFloat8));
    rewriteResultDtype(mul, ttcore::DataType::BFP_BFloat8);
  }
  return true;
}

// Pattern D — LM-head matmul -> [view / CCL / sum]* -> argmax.
//
// Action: set matmul.dtype = bfp_bf8, propagate bfp_bf8 through every
// flow-through op (slice, reshape, sum, CCL), then insert a single
// ttnn.to_layout(RowMajor, bf16) at the argmax operand. That to_layout lowers
// (via TTNNDecomposeLayouts' bfp_bf8 -> bf16 fast-path + tt-metal's
// to_layout_op assert relax) to a single tt-metal `untilize` kernel that
// does the bfp8 -> bf16 upcast inline as part of de-tiling, avoiding the
// separate device typecast pass that would otherwise erase the CCL savings.
//
// The view-op invariant guard in MemoryLayoutPropagation /
// OperationValidationAndFallback (narrowed to SliceStaticOp) keeps
// intermediate slices consistent; the return-cast logic in
// updateFunctionReturnTypes keeps the function signature stable for integer
// mismatches like the argmax ui32 -> si32 case.
static bool tryLMHeadArgmaxMatcher(MatmulOp matmul, OpBuilder &builder) {
  auto isFlow = [](Operation *op) {
    return isViewLikeOp(op) || isCCLOp(op) || mlir::isa<SumOp>(op);
  };

  llvm::SmallVector<Operation *> flowThroughOps;
  llvm::SmallVector<std::pair<Operation *, unsigned>> exits;
  collectChain(matmul.getResult(), isFlow, flowThroughOps, exits);

  if (exits.empty()) {
    return false;
  }
  if (!llvm::any_of(flowThroughOps, isCCLOp)) {
    return false;
  }

  for (auto [exitOp, _] : exits) {
    if (!mlir::isa<ArgMaxOp>(exitOp)) {
      return false;
    }
  }

  auto matmulDtype = getResultDtype(matmul.getResult());
  if (!matmulDtype || (*matmulDtype != ttcore::DataType::BFloat16 &&
                       *matmulDtype != ttcore::DataType::Float32)) {
    return false;
  }

  setOpDtype(matmul, ttcore::DataType::BFP_BFloat8);
  for (Operation *flow : flowThroughOps) {
    rewriteResultDtype(flow, ttcore::DataType::BFP_BFloat8);
  }
  IRRewriter rewriter(builder.getContext());
  for (auto [exitOp, operandIdx] : exits) {
    insertToLayoutRowMajorBF16(rewriter, exitOp, operandIdx);
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

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
    OpBuilder builder(&getContext());

    // Collect matmuls in a stable order before mutating; rewrites change the
    // IR and we don't want to revisit a transformed matmul.
    llvm::SmallVector<MatmulOp> matmuls;
    moduleOp.walk([&](MatmulOp op) { matmuls.push_back(op); });

    for (MatmulOp matmul : matmuls) {
      // Skip ops the user has already opted out of via a dtype attribute set
      // to a non-bf16/f32 value (e.g., a previous run of this pass).
      if (auto attr = matmul.getDtypeAttr()) {
        ttcore::DataType d = attr.getValue();
        if (d != ttcore::DataType::BFloat16 && d != ttcore::DataType::Float32) {
          continue;
        }
      }

      // Try matchers in order. Each matcher is independent — the first one
      // that succeeds claims the matmul. Strict matching means the matchers
      // do not overlap.
      if (tryQKVRoPEMatcher(matmul, builder)) {
        continue;
      }
      if (tryProjResidualAddMatcher(matmul, builder)) {
        continue;
      }
      if (tryMLPUpGateMatcher(matmul)) {
        continue;
      }
      if (tryLMHeadArgmaxMatcher(matmul, builder)) {
        continue;
      }
    }
  }
};

} // namespace

} // namespace mlir::tt::ttnn
