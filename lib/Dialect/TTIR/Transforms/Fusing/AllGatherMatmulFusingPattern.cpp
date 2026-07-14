// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Fusing/AllGatherMatmulFusingPattern.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

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

// Build the private decomposition function referenced by the composite. Its
// arguments, in order, mirror the composite's inputs:
//   x, weight, [bias], [residual, gate]
// The body is the primitive form of the fused op. Marked with
// kCompositeDecompositionAttr so fusing patterns never recurse into it.
func::FuncOp buildDecompositionFunc(OpBuilder &builder, Location loc,
                                    RankedTensorType gatheredType,
                                    RankedTensorType projType,
                                    RankedTensorType resultType,
                                    int32_t allGatherDim, uint32_t clusterAxis,
                                    Value x, Value weight, Value bias,
                                    Value residual, Value gate) {
  SmallVector<Type> argTypes{x.getType(), weight.getType()};
  if (bias) {
    argTypes.push_back(bias.getType());
  }
  if (residual) {
    argTypes.push_back(residual.getType());
    argTypes.push_back(gate.getType());
  }

  auto funcType = builder.getFunctionType(argTypes, {resultType});
  auto funcOp = func::FuncOp::create(loc, getUniqueDecompName(), funcType);
  funcOp.setVisibility(SymbolTable::Visibility::Private);
  funcOp->setAttr(utils::kCompositeDecompositionAttr,
                  UnitAttr::get(builder.getContext()));

  Block *block = funcOp.addEntryBlock();
  OpBuilder fb(builder.getContext());
  fb.setInsertionPointToStart(block);

  unsigned idx = 0;
  Value xArg = block->getArgument(idx++);
  Value weightArg = block->getArgument(idx++);
  Value biasArg = bias ? block->getArgument(idx++) : Value();
  Value residualArg = residual ? block->getArgument(idx++) : Value();
  Value gateArg = residual ? block->getArgument(idx++) : Value();

  Value gathered =
      fb.create<AllGatherOp>(loc, gatheredType, xArg, allGatherDim, clusterAxis)
          .getResult();

  Value proj;
  if (biasArg) {
    proj = fb.create<LinearOp>(loc, projType, gathered, weightArg, biasArg,
                               /*transpose_a=*/false, /*transpose_b=*/false)
               .getResult();
  } else {
    proj = fb.create<MatmulOp>(loc, projType, gathered, weightArg).getResult();
  }

  Value out = proj;
  if (residualArg) {
    // out = residual + gate * proj
    Value scaled =
        fb.create<MultiplyOp>(loc, projType, gateArg, proj).getResult();
    out = fb.create<AddOp>(loc, resultType, residualArg, scaled).getResult();
  }

  fb.create<func::ReturnOp>(loc, ValueRange{out});
  return funcOp;
}

// Emit the decomposition function into the module and replace `anchor` with a
// ttcore.composite referencing it. `bias`/`residual`/`gate` are null for the
// plain matmul fusion and populated for the gated-residual variant.
void replaceWithComposite(mlir::PatternRewriter &rewriter, Operation *anchor,
                          RankedTensorType resultType,
                          RankedTensorType gatheredType,
                          RankedTensorType projType, int32_t allGatherDim,
                          uint32_t clusterAxis, Value x, Value weight,
                          Value bias, Value residual, Value gate) {
  auto moduleOp = anchor->getParentOfType<ModuleOp>();

  OpBuilder moduleBuilder(moduleOp.getContext());
  moduleBuilder.setInsertionPointToEnd(moduleOp.getBody());
  func::FuncOp decompFunc = buildDecompositionFunc(
      moduleBuilder, anchor->getLoc(), gatheredType, projType, resultType,
      allGatherDim, clusterAxis, x, weight, bias, residual, gate);
  moduleBuilder.insert(decompFunc);

  SmallVector<Value> inputs{x, weight};
  if (bias) {
    inputs.push_back(bias);
  }
  if (residual) {
    inputs.push_back(residual);
    inputs.push_back(gate);
  }

  // Collective parameters and operand-presence flags travel on the composite so
  // TTNNResolveComposites can rebuild the typed op without re-inspecting the
  // IR.
  mlir::MLIRContext *ctx = rewriter.getContext();
  SmallVector<NamedAttribute> attrs;
  attrs.emplace_back(StringAttr::get(ctx, "all_gather_dim"),
                     rewriter.getSI32IntegerAttr(allGatherDim));
  attrs.emplace_back(StringAttr::get(ctx, "cluster_axis"),
                     rewriter.getUI32IntegerAttr(clusterAxis));
  attrs.emplace_back(StringAttr::get(ctx, "has_bias"),
                     rewriter.getBoolAttr(bias != nullptr));
  attrs.emplace_back(StringAttr::get(ctx, "has_addcmul"),
                     rewriter.getBoolAttr(residual != nullptr));
  if (residual) {
    attrs.emplace_back(StringAttr::get(ctx, "scalar"),
                       rewriter.getF32FloatAttr(1.0));
  }

  rewriter.replaceOpWithNewOp<ttcore::CompositeOp>(
      anchor, TypeRange{resultType}, inputs,
      rewriter.getStringAttr(kCompositeName),
      FlatSymbolRefAttr::get(ctx, decompFunc.getName()),
      DictionaryAttr::get(ctx, attrs));
}

} // namespace

template <typename MatmulLikeOp>
mlir::LogicalResult AllGatherMatmulFusing<MatmulLikeOp>::matchAndRewrite(
    MatmulLikeOp matmulOp, mlir::PatternRewriter &rewriter) const {
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
  replaceWithComposite(
      rewriter, matmulOp, /*resultType=*/projType,
      mlir::cast<RankedTensorType>(allGatherOp.getResult().getType()), projType,
      allGatherOp.getAllGatherDim(), allGatherOp.getClusterAxis(),
      allGatherOp.getInput(), matmulOp.getB(), bias, /*residual=*/Value(),
      /*gate=*/Value());
  return mlir::success();
}

template <typename MatmulLikeOp>
mlir::LogicalResult AllGatherMatmulAddcmulFusing<MatmulLikeOp>::matchAndRewrite(
    AddOp addOp, mlir::PatternRewriter &rewriter) const {
  if (utils::isInsideCompositeDecomposition(addOp)) {
    return mlir::failure();
  }

  // add is commutative: one operand is the (gate * proj) multiply, the other is
  // the residual.
  MultiplyOp mulOp = addOp.getLhs().getDefiningOp<MultiplyOp>();
  mlir::Value residual = addOp.getRhs();
  if (!mulOp) {
    mulOp = addOp.getRhs().getDefiningOp<MultiplyOp>();
    residual = addOp.getLhs();
  }
  if (!mulOp || !mulOp.getResult().hasOneUse()) {
    return mlir::failure();
  }

  // multiply is commutative: one operand is the projection, the other gate.
  MatmulLikeOp projOp = mulOp.getLhs().getDefiningOp<MatmulLikeOp>();
  mlir::Value gate = mulOp.getRhs();
  if (!projOp) {
    projOp = mulOp.getRhs().getDefiningOp<MatmulLikeOp>();
    gate = mulOp.getLhs();
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

  replaceWithComposite(
      rewriter, addOp,
      mlir::cast<RankedTensorType>(addOp.getResult().getType()),
      mlir::cast<RankedTensorType>(allGatherOp.getResult().getType()),
      mlir::cast<RankedTensorType>(projOp.getResult().getType()),
      allGatherOp.getAllGatherDim(), allGatherOp.getClusterAxis(),
      allGatherOp.getInput(), projOp.getB(), bias, residual, gate);
  return mlir::success();
}

// Explicit template instantiations.
template class AllGatherMatmulFusing<MatmulOp>;
template class AllGatherMatmulFusing<LinearOp>;
template class AllGatherMatmulAddcmulFusing<MatmulOp>;
template class AllGatherMatmulAddcmulFusing<LinearOp>;

} // namespace mlir::tt::ttir::fusing
