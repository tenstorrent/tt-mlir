// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Fusing/AllGatherMatmulFusingPattern.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/STLExtras.h"

#include <atomic>
#include <type_traits>

namespace mlir::tt::ttir::fusing {

namespace {

// Name of the composite this fusion emits. Must match the key registered in
// TTNNResolveComposites' composite registry.
constexpr llvm::StringLiteral kCompositeName =
    "all_gather_minimal_matmul_async";

std::string getUniqueDecompName() {
  static std::atomic<uint64_t> counter{0};
  return "all_gather_minimal_matmul_async_decomp_" +
         std::to_string(counter.fetch_add(1));
}

// Coefficient for the addcmul epilogue's scalar slot, which computes
//   result = addcmul_input1 + scalar * proj * addcmul_input2
// The gated residual we fuse carries no coefficient, so this is the identity
// 1.0 (see the addcmul pattern for the full derivation).
constexpr double kAddcmulScalar = 1.0;

// True if this matmul/linear result flows into a gated-residual epilogue
// (a multiply then an add), which the addcmul pattern folds in whole.
template <typename MatmulLikeOp>
bool feedsGatedResidualEpilogue(MatmulLikeOp matmulOp) {
  if (!matmulOp.getResult().hasOneUse()) {
    return false;
  }
  auto mulOp =
      mlir::dyn_cast<MultiplyOp>(*matmulOp.getResult().getUsers().begin());
  if (!mulOp || !mulOp.getResult().hasOneUse()) {
    return false;
  }
  return mlir::isa<AddOp>(*mulOp.getResult().getUsers().begin());
}

func::FuncOp buildDecompositionFunc(OpBuilder &builder, Location loc,
                                    ArrayRef<Value> captures,
                                    ArrayRef<Operation *> ops,
                                    Type resultType) {
  auto argTypes =
      llvm::map_to_vector(captures, [](Value v) { return v.getType(); });
  auto funcOp =
      func::FuncOp::create(loc, getUniqueDecompName(),
                           builder.getFunctionType(argTypes, {resultType}));
  funcOp.setVisibility(SymbolTable::Visibility::Private);
  funcOp->setAttr(utils::kCompositeDecompositionAttr,
                  UnitAttr::get(builder.getContext()));

  Block *block = funcOp.addEntryBlock();
  OpBuilder fb(block, block->end());

  // Map each captured value to its matching block argument so cloned ops
  // reference the function's arguments; clone() rewires chained results.
  IRMapping mapping;
  for (auto [capture, arg] : llvm::zip(captures, block->getArguments())) {
    mapping.map(capture, arg);
  }

  Operation *last = nullptr;
  for (Operation *op : ops) {
    last = fb.clone(*op, mapping);
  }

  fb.create<func::ReturnOp>(loc, last->getResults());
  return funcOp;
}

} // namespace

// Match, with no gated-residual epilogue, and fold into the composite:
//
//   proj = matmul(all_gather(input), weight) + bias
//
// where `+ bias` applies only to the linear variant (matmul has no bias).
// If a `residual + gate * proj` epilogue follows, defer to
// AllGatherMatmulAddcmulFusing so the whole thing folds at once.
template <typename MatmulLikeOp>
mlir::LogicalResult AllGatherMatmulFusing<MatmulLikeOp>::matchAndRewrite(
    MatmulLikeOp matmulOp, mlir::PatternRewriter &rewriter) const {
  // Don't re-fuse the primitive ops we cloned into a decomposition body.
  if (utils::isInsideCompositeDecomposition(matmulOp)) {
    return mlir::failure();
  }

  AllGatherOp allGatherOp =
      matmulOp.getA().template getDefiningOp<AllGatherOp>();
  if (!allGatherOp || !allGatherOp.getResult().hasOneUse()) {
    return mlir::failure();
  }

  if (matmulOp.getTransposeA() || matmulOp.getTransposeB()) {
    return mlir::failure();
  }

  // Let the addcmul pattern fold the whole gated-residual epilogue instead.
  if (feedsGatedResidualEpilogue(matmulOp)) {
    return mlir::failure();
  }

  Value bias;
  if constexpr (std::is_same_v<MatmulLikeOp, LinearOp>) {
    bias = matmulOp.getBias();
  }

  auto projType = mlir::cast<RankedTensorType>(matmulOp.getResult().getType());

  // Captures feed the composite/decomposition in order: input, weight, [bias].
  SmallVector<Value> captures{allGatherOp.getInput(), matmulOp.getB()};
  if (bias) {
    captures.push_back(bias);
  }
  SmallVector<Operation *> ops{allGatherOp, matmulOp};

  Operation *anchor = matmulOp.getOperation();
  ModuleOp moduleOp = anchor->getParentOfType<ModuleOp>();
  OpBuilder moduleBuilder(moduleOp.getContext());
  moduleBuilder.setInsertionPointToEnd(moduleOp.getBody());
  func::FuncOp decompFunc = buildDecompositionFunc(
      moduleBuilder, matmulOp.getLoc(), captures, ops, projType);
  moduleBuilder.insert(decompFunc);

  // Collective parameters and operand-presence flags travel on the composite so
  // TTNNResolveComposites can rebuild the typed op without re-inspecting the
  // IR.
  mlir::MLIRContext *ctx = rewriter.getContext();
  SmallVector<NamedAttribute> attrs;
  attrs.emplace_back(
      StringAttr::get(ctx, "all_gather_dim"),
      rewriter.getSI32IntegerAttr(allGatherOp.getAllGatherDim()));
  attrs.emplace_back(StringAttr::get(ctx, "cluster_axis"),
                     rewriter.getUI32IntegerAttr(allGatherOp.getClusterAxis()));
  attrs.emplace_back(StringAttr::get(ctx, "has_bias"),
                     rewriter.getBoolAttr(bias != nullptr));
  attrs.emplace_back(StringAttr::get(ctx, "has_addcmul"),
                     rewriter.getBoolAttr(false));

  rewriter.replaceOpWithNewOp<ttcore::CompositeOp>(
      anchor, TypeRange{projType}, captures,
      rewriter.getStringAttr(kCompositeName),
      FlatSymbolRefAttr::get(ctx, decompFunc.getName()),
      DictionaryAttr::get(ctx, attrs));
  return mlir::success();
}

template <typename MatmulLikeOp>
mlir::LogicalResult AllGatherMatmulAddcmulFusing<MatmulLikeOp>::matchAndRewrite(
    AddOp addOp, mlir::PatternRewriter &rewriter) const {
  // Don't re-fuse the primitive ops we cloned into a decomposition body.
  if (utils::isInsideCompositeDecomposition(addOp)) {
    return mlir::failure();
  }

  // Match the DiT gated residual:
  //
  //   result = residual + gate * proj,
  //   where proj = matmul(all_gather(input), weight) + bias
  //   (`+ bias` applies only to the linear variant; matmul has no bias)
  //
  // walking backwards from the anchor `add`:  add -> multiply -> matmul ->
  // all_gather. Both the add and the multiply are commutative, so we try each
  // operand order.
  //
  // This maps to tt-metal's `addcmul` epilogue, whose fixed formula is
  //
  //   result = addcmul_input1 + scalar * proj * addcmul_input2
  //
  // The gated residual carries no coefficient in front of the product, so
  // `scalar` is the multiplicative identity (kAddcmulScalar == 1.0):
  //
  //   residual + 1.0 * proj * gate  ==  residual + gate * proj
  //
  // matching tt-metal, where every DiT call site passes 1.0; the slot stays
  // configurable only because the underlying addcmul kernel is general.

  // add: one operand is the `gate * proj` multiply, the other is the residual.
  MultiplyOp gateMulOp = addOp.getLhs().getDefiningOp<MultiplyOp>();
  mlir::Value residual = addOp.getRhs();
  if (!gateMulOp) {
    gateMulOp = addOp.getRhs().getDefiningOp<MultiplyOp>();
    residual = addOp.getLhs();
  }
  if (!gateMulOp || !gateMulOp.getResult().hasOneUse()) {
    return mlir::failure();
  }

  // multiply: one operand is the projection, the other is the gate.
  MatmulLikeOp projOp = gateMulOp.getLhs().getDefiningOp<MatmulLikeOp>();
  mlir::Value gate = gateMulOp.getRhs();
  if (!projOp) {
    projOp = gateMulOp.getRhs().getDefiningOp<MatmulLikeOp>();
    gate = gateMulOp.getLhs();
  }
  if (!projOp || !projOp.getResult().hasOneUse()) {
    return mlir::failure();
  }

  AllGatherOp allGatherOp = projOp.getA().template getDefiningOp<AllGatherOp>();
  if (!allGatherOp || !allGatherOp.getResult().hasOneUse()) {
    return mlir::failure();
  }

  if (projOp.getTransposeA() || projOp.getTransposeB()) {
    return mlir::failure();
  }

  Value bias;
  if constexpr (std::is_same_v<MatmulLikeOp, LinearOp>) {
    bias = projOp.getBias();
  }

  auto resultType = mlir::cast<RankedTensorType>(addOp.getResult().getType());

  // Captures feed the composite/decomposition in order:
  //   input, weight, [bias], residual, gate.
  SmallVector<Value> captures{allGatherOp.getInput(), projOp.getB()};
  if (bias) {
    captures.push_back(bias);
  }
  captures.push_back(residual);
  captures.push_back(gate);
  SmallVector<Operation *> ops{allGatherOp, projOp, gateMulOp, addOp};

  ModuleOp moduleOp = addOp->getParentOfType<ModuleOp>();
  OpBuilder moduleBuilder(moduleOp.getContext());
  moduleBuilder.setInsertionPointToEnd(moduleOp.getBody());
  func::FuncOp decompFunc = buildDecompositionFunc(
      moduleBuilder, addOp.getLoc(), captures, ops, resultType);
  moduleBuilder.insert(decompFunc);

  // Collective parameters and operand-presence flags travel on the composite so
  // TTNNResolveComposites can rebuild the typed op without re-inspecting the
  // IR.
  mlir::MLIRContext *ctx = rewriter.getContext();
  SmallVector<NamedAttribute> attrs;
  attrs.emplace_back(
      StringAttr::get(ctx, "all_gather_dim"),
      rewriter.getSI32IntegerAttr(allGatherOp.getAllGatherDim()));
  attrs.emplace_back(StringAttr::get(ctx, "cluster_axis"),
                     rewriter.getUI32IntegerAttr(allGatherOp.getClusterAxis()));
  attrs.emplace_back(StringAttr::get(ctx, "has_bias"),
                     rewriter.getBoolAttr(bias != nullptr));
  attrs.emplace_back(StringAttr::get(ctx, "has_addcmul"),
                     rewriter.getBoolAttr(true));
  // out = residual + kAddcmulScalar * proj * gate  (scalar == 1.0; see
  // kAddcmulScalar for why).
  attrs.emplace_back(StringAttr::get(ctx, "scalar"),
                     rewriter.getF32FloatAttr(kAddcmulScalar));

  rewriter.replaceOpWithNewOp<ttcore::CompositeOp>(
      addOp, TypeRange{resultType}, captures,
      rewriter.getStringAttr(kCompositeName),
      FlatSymbolRefAttr::get(ctx, decompFunc.getName()),
      DictionaryAttr::get(ctx, attrs));
  return mlir::success();
}

// Explicit template instantiations.
template class AllGatherMatmulFusing<MatmulOp>;
template class AllGatherMatmulFusing<LinearOp>;
template class AllGatherMatmulAddcmulFusing<MatmulOp>;
template class AllGatherMatmulAddcmulFusing<LinearOp>;

} // namespace mlir::tt::ttir::fusing
