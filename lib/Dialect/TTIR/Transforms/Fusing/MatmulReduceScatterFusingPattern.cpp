// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Fusing/MatmulReduceScatterFusingPattern.h"

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
    "minimal_matmul_strided_reduce_scatter_async";

std::string getUniqueDecompName() {
  static std::atomic<uint64_t> counter{0};
  return "minimal_matmul_strided_reduce_scatter_async_decomp_" +
         std::to_string(counter.fetch_add(1));
}

// Coefficient for the addcmul epilogue's scalar slot, which computes
//   result = addcmul_input1 + scalar * proj * addcmul_input2
// The gated residual we fuse carries no coefficient, so this is the identity
// 1.0 (see the addcmul pattern for the full derivation).
constexpr double kAddcmulScalar = 1.0;

// True if this reduce_scatter result flows into a gated-residual epilogue
// (a multiply then an add), which the addcmul pattern folds in whole.
bool feedsGatedResidualEpilogue(ReduceScatterOp reduceScatterOp) {
  if (!reduceScatterOp.getResult().hasOneUse()) {
    return false;
  }
  auto mulOp = mlir::dyn_cast<MultiplyOp>(
      *reduceScatterOp.getResult().getUsers().begin());
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
//   proj = matmul(input, weight) + bias
//   out  = reduce_scatter(proj)
//
// where `+ bias` applies only to the linear variant (matmul has no bias).
// If a `residual + gate * out` epilogue follows, defer to
// MatmulReduceScatterAddcmulFusing so the whole thing folds at once.
template <typename MatmulLikeOp>
mlir::LogicalResult MatmulReduceScatterFusing<MatmulLikeOp>::matchAndRewrite(
    ReduceScatterOp reduceScatterOp, mlir::PatternRewriter &rewriter) const {
  // Don't re-fuse the primitive ops we cloned into a decomposition body.
  if (utils::isInsideCompositeDecomposition(reduceScatterOp)) {
    return mlir::failure();
  }

  MatmulLikeOp matmulOp =
      reduceScatterOp.getInput().template getDefiningOp<MatmulLikeOp>();
  if (!matmulOp || !matmulOp.getResult().hasOneUse()) {
    return mlir::failure();
  }

  if (matmulOp.getTransposeA() || matmulOp.getTransposeB()) {
    return mlir::failure();
  }

  // Let the addcmul pattern fold the whole gated-residual epilogue instead.
  if (feedsGatedResidualEpilogue(reduceScatterOp)) {
    return mlir::failure();
  }

  Value bias;
  if constexpr (std::is_same_v<MatmulLikeOp, LinearOp>) {
    bias = matmulOp.getBias();
  }

  auto resultType =
      mlir::cast<RankedTensorType>(reduceScatterOp.getResult().getType());

  // Captures feed the composite/decomposition in order: input, weight, [bias].
  SmallVector<Value> captures{matmulOp.getA(), matmulOp.getB()};
  if (bias) {
    captures.push_back(bias);
  }
  SmallVector<Operation *> ops{matmulOp, reduceScatterOp};

  Operation *anchor = reduceScatterOp.getOperation();
  ModuleOp moduleOp = anchor->getParentOfType<ModuleOp>();
  OpBuilder moduleBuilder(moduleOp.getContext());
  moduleBuilder.setInsertionPointToEnd(moduleOp.getBody());
  func::FuncOp decompFunc = buildDecompositionFunc(
      moduleBuilder, reduceScatterOp.getLoc(), captures, ops, resultType);
  moduleBuilder.insert(decompFunc);

  // Collective parameters and operand-presence flags travel on the composite so
  // TTNNResolveComposites can rebuild the typed op without re-inspecting the
  // IR.
  mlir::MLIRContext *ctx = rewriter.getContext();
  SmallVector<NamedAttribute> attrs;
  attrs.emplace_back(
      StringAttr::get(ctx, "scatter_dim"),
      rewriter.getSI32IntegerAttr(reduceScatterOp.getScatterDim()));
  attrs.emplace_back(
      StringAttr::get(ctx, "cluster_axis"),
      rewriter.getUI32IntegerAttr(reduceScatterOp.getClusterAxis()));
  attrs.emplace_back(StringAttr::get(ctx, "has_bias"),
                     rewriter.getBoolAttr(bias != nullptr));
  attrs.emplace_back(StringAttr::get(ctx, "has_addcmul"),
                     rewriter.getBoolAttr(false));

  rewriter.replaceOpWithNewOp<ttcore::CompositeOp>(
      anchor, TypeRange{resultType}, captures,
      rewriter.getStringAttr(kCompositeName),
      FlatSymbolRefAttr::get(ctx, decompFunc.getName()),
      DictionaryAttr::get(ctx, attrs));
  return mlir::success();
}

template <typename MatmulLikeOp>
mlir::LogicalResult
MatmulReduceScatterAddcmulFusing<MatmulLikeOp>::matchAndRewrite(
    AddOp addOp, mlir::PatternRewriter &rewriter) const {
  // Don't re-fuse the primitive ops we cloned into a decomposition body.
  if (utils::isInsideCompositeDecomposition(addOp)) {
    return mlir::failure();
  }

  // Match the DiT gated residual:
  //
  //   result = residual + gate * proj,
  //   where proj = reduce_scatter(matmul(input, weight) + bias)
  //   (`+ bias` applies only to the linear variant; matmul has no bias)
  //
  // walking backwards from the anchor `add`:  add -> multiply -> reduce_scatter
  // -> matmul. Both the add and the multiply are commutative, so we try each
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

  // multiply: one operand is the reduce_scatter projection, the other the gate.
  ReduceScatterOp reduceScatterOp =
      gateMulOp.getLhs().getDefiningOp<ReduceScatterOp>();
  mlir::Value gate = gateMulOp.getRhs();
  if (!reduceScatterOp) {
    reduceScatterOp = gateMulOp.getRhs().getDefiningOp<ReduceScatterOp>();
    gate = gateMulOp.getLhs();
  }
  if (!reduceScatterOp || !reduceScatterOp.getResult().hasOneUse()) {
    return mlir::failure();
  }

  MatmulLikeOp projOp =
      reduceScatterOp.getInput().template getDefiningOp<MatmulLikeOp>();
  if (!projOp || !projOp.getResult().hasOneUse()) {
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
  SmallVector<Value> captures{projOp.getA(), projOp.getB()};
  if (bias) {
    captures.push_back(bias);
  }
  captures.push_back(residual);
  captures.push_back(gate);
  SmallVector<Operation *> ops{projOp, reduceScatterOp, gateMulOp, addOp};

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
      StringAttr::get(ctx, "scatter_dim"),
      rewriter.getSI32IntegerAttr(reduceScatterOp.getScatterDim()));
  attrs.emplace_back(
      StringAttr::get(ctx, "cluster_axis"),
      rewriter.getUI32IntegerAttr(reduceScatterOp.getClusterAxis()));
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
template class MatmulReduceScatterFusing<MatmulOp>;
template class MatmulReduceScatterFusing<LinearOp>;
template class MatmulReduceScatterAddcmulFusing<MatmulOp>;
template class MatmulReduceScatterAddcmulFusing<LinearOp>;

} // namespace mlir::tt::ttir::fusing
