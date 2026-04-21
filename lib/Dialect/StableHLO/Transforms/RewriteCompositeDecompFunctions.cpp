// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "stablehlo/dialect/StablehloOps.h"
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "ttmlir/Dialect/StableHLO/Utils/StableHLOUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_REWRITECOMPOSITEDECOMPFUNCTIONSPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

namespace {

// Signature of a per-composite rewriter. Receives:
//   - the decomposition func::FuncOp whose body it must rewrite, and
//   - the `composite_attributes` DictionaryAttr from the composite op that
//     referenced this decomposition (never null; an empty dictionary is
//     passed if the composite had no `composite_attributes`).
// Must preserve the function's name, input types (entry-block arg types),
// and result types.
using DecompRewriter = void (*)(mlir::func::FuncOp, mlir::DictionaryAttr);

// ---------------------------------------------------------------------------
// Per-composite rewriter functions.
// Add a new function here for each composite whose decomposition body you
// want to rewrite, then register it in getCompositeRewriters() below.
// ---------------------------------------------------------------------------

// Rewriter for stablehlo.composite "tenstorrent.gather".
static void
rewriteTenstorrentGatherDecomp(mlir::func::FuncOp func,
                               mlir::DictionaryAttr compositeAttrs) {
  mlir::MLIRContext *ctx = func.getContext();
  mlir::Block &entry = func.getBody().front();

  // Erase everything currently in the entry block (terminator-last). Block
  // arguments are preserved: they carry the function's input types.
  while (!entry.empty()) {
    entry.back().erase();
  }

  mlir::OpBuilder builder(ctx);
  builder.setInsertionPointToStart(&entry);

  mlir::Location loc = func.getLoc();
  mlir::TypeRange resultTypes = func.getFunctionType().getResults();
  mlir::ValueRange args = entry.getArguments();
  (void)args;           // Available to the user's code below.
  (void)compositeAttrs; // Available to the user's code below.

  llvm::SmallVector<mlir::Value> returnValues;
  returnValues.reserve(resultTypes.size());

  // ============================================================
  // vvvvv USER: ADD YOUR CUSTOM GATHER OPS HERE vvvvv
  //
  //  - Build replacement ops with `builder` (inserts at top of entry).
  //  - Inputs:  `args` (ValueRange of block arguments; types = func inputs).
  //  - Attrs:   `compositeAttrs` (DictionaryAttr from the composite op's
  //             `composite_attributes`; empty dict if the composite had
  //             none). Example lookups:
  //                int64_t dim = 0;
  //                if (auto a = compositeAttrs.getAs<mlir::IntegerAttr>("dim"))
  //                  dim = a.getInt();
  //                bool sparseGrad = false;
  //                if (auto a =
  //                compositeAttrs.getAs<mlir::BoolAttr>("sparse_grad"))
  //                  sparseGrad = a.getValue();
  //  - Outputs: push your final Values into `returnValues` in order,
  //             matching `resultTypes`.
  //  - Delete the default zero-constant loop below once you are emitting
  //    real ops.
  //
  // Everything OUTSIDE this banner (including the final func.return below)
  // is fixed plumbing: do not modify.
  // ============================================================
  for (mlir::Type resultType : resultTypes) {
    auto rankedTy = mlir::cast<mlir::RankedTensorType>(resultType);
    mlir::Type elemTy = rankedTy.getElementType();
    mlir::Attribute zeroElt;
    if (auto floatTy = mlir::dyn_cast<mlir::FloatType>(elemTy)) {
      zeroElt = mlir::FloatAttr::get(
          floatTy, llvm::APFloat::getZero(floatTy.getFloatSemantics()));
    } else if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(elemTy)) {
      zeroElt = mlir::IntegerAttr::get(intTy, 0);
    } else {
      llvm_unreachable("unsupported element type for placeholder body");
    }
    auto splat = mlir::DenseElementsAttr::get(rankedTy, zeroElt);
    auto cst =
        builder.create<mlir::stablehlo::ConstantOp>(loc, rankedTy, splat);
    returnValues.push_back(cst.getResult());
  }
  // ^^^^^ USER: END CUSTOM GATHER OPS ^^^^^
  // ============================================================

  builder.create<mlir::func::ReturnOp>(loc, returnValues);
}

// ---------------------------------------------------------------------------
// Registry: composite op name -> rewriter for its decomposition function.
// To handle a new composite, write a rewriter above and add an entry here.
// ---------------------------------------------------------------------------
static const llvm::StringMap<DecompRewriter> &getCompositeRewriters() {
  static const llvm::StringMap<DecompRewriter> map = [] {
    llvm::StringMap<DecompRewriter> m;
    m["tenstorrent.gather"] = rewriteTenstorrentGatherDecomp;
    return m;
  }();
  return map;
}

} // namespace

struct RewriteCompositeDecompFunctionsPass
    : public impl::RewriteCompositeDecompFunctionsPassBase<
          RewriteCompositeDecompFunctionsPass> {
public:
  using impl::RewriteCompositeDecompFunctionsPassBase<
      RewriteCompositeDecompFunctionsPass>::
      RewriteCompositeDecompFunctionsPassBase;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::SymbolTable symTable(module);
    const llvm::StringMap<DecompRewriter> &rewriters = getCompositeRewriters();

    // Per unique decomposition func, record the rewriter to run and the
    // composite_attributes to pass to it. Multiple composite ops may
    // reference the same function; we rewrite each function exactly once.
    // If two different composites that share a decomposition target carry
    // different `composite_attributes`, the first one encountered wins
    // (documented caveat: decomposition functions are expected to be
    // specialized per attribute configuration; downstream ops always read
    // `composite_attributes` off the composite itself, not the body).
    struct Task {
      DecompRewriter rewriter;
      mlir::DictionaryAttr compositeAttrs;
    };
    llvm::DenseMap<mlir::func::FuncOp, Task> toRewrite;

    mlir::DictionaryAttr emptyDict =
        mlir::DictionaryAttr::get(module.getContext());

    module.walk([&](mlir::stablehlo::CompositeOp comp) {
      auto it = rewriters.find(comp.getName());
      if (it == rewriters.end()) {
        return;
      }
      auto decompAttr = comp->getAttrOfType<mlir::SymbolRefAttr>(
          utils::kCompDecompositionKey);
      if (!decompAttr) {
        comp.emitOpError() << "missing required SymbolRefAttr '"
                           << utils::kCompDecompositionKey << "'";
        signalPassFailure();
        return;
      }
      mlir::StringAttr leaf = decompAttr.getLeafReference();
      auto callee = symTable.lookup<mlir::func::FuncOp>(leaf);
      if (!callee) {
        comp.emitOpError() << "failed to resolve decomposition '" << leaf
                           << "'";
        signalPassFailure();
        return;
      }
      mlir::DictionaryAttr compAttrs = comp.getCompositeAttributes();
      if (!compAttrs) {
        compAttrs = emptyDict;
      }
      toRewrite.try_emplace(callee, Task{it->second, compAttrs});
    });

    for (auto &kv : toRewrite) {
      kv.second.rewriter(kv.first, kv.second.compositeAttrs);
    }
  }
};

} // namespace mlir::tt::stablehlo
