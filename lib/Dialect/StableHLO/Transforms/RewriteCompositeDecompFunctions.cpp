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
//     referenced this decomposition
// Must preserve the function's name, input types (entry-block arg types),
// and result types.
using DecompRewriter = void (*)(mlir::func::FuncOp, mlir::DictionaryAttr);

// ---------------------------------------------------------------------------
// Per-composite rewriter functions.
// Add a new function here for each composite whose decomposition body you
// want to rewrite, then register it in getCompositeRewriters() below.
// ---------------------------------------------------------------------------

// Rewriter for stablehlo.composite "tenstorrent.gather_dim".
//
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

  llvm::SmallVector<mlir::Value> returnValues;
  returnValues.reserve(resultTypes.size());

  // Collect inputs from the decomp function and composite_attributes
  mlir::Value input = args[0];
  mlir::Value index = args[1];
  auto inputTy = mlir::cast<mlir::RankedTensorType>(input.getType());
  auto indexTy = mlir::cast<mlir::RankedTensorType>(index.getType());
  int64_t rank = inputTy.getRank();

  int64_t dim = 0;
  if (auto dimAttr = compositeAttrs.getAs<mlir::IntegerAttr>("dim")) {
    dim = dimAttr.getInt();
  }
  if (dim < 0) {
    dim += rank;
  }

  // torch.gather only requires index.size(d) <= input.size(d) for d != dim,
  // but stablehlo.gather's batching-dim constraint requires equal sizes on
  // operand and start_indices for each batching axis. Slice input down on
  // any non-dim axis where it is larger than the index — those trailing
  // positions on input are never read by torch.gather, so this is a
  // semantic no-op that makes the batching-dim gather legal.
  llvm::ArrayRef<int64_t> inputShape = inputTy.getShape();
  llvm::ArrayRef<int64_t> indexShape = indexTy.getShape();
  llvm::SmallVector<int64_t> slicedInputShape(inputShape);
  bool needsSlice = false;
  for (int64_t i = 0; i < rank; ++i) {
    if (i != dim && indexShape[i] < inputShape[i]) {
      slicedInputShape[i] = indexShape[i];
      needsSlice = true;
    }
  }
  if (needsSlice) {
    llvm::SmallVector<int64_t> sliceStart(rank, 0);
    llvm::SmallVector<int64_t> sliceStrides(rank, 1);
    auto slicedInputTy =
        mlir::RankedTensorType::get(slicedInputShape, inputTy.getElementType());
    input = builder.create<mlir::stablehlo::SliceOp>(
        loc, slicedInputTy, input, builder.getDenseI64ArrayAttr(sliceStart),
        builder.getDenseI64ArrayAttr(slicedInputShape),
        builder.getDenseI64ArrayAttr(sliceStrides));
  }

  // Reshape the index tensor to append a trailing size-1 dim so we have an
  // explicit index_vector_dim for stablehlo.gather.
  llvm::SmallVector<int64_t> reshapedIndexShape(indexTy.getShape());
  reshapedIndexShape.push_back(1);
  auto reshapedIndexTy =
      mlir::RankedTensorType::get(reshapedIndexShape, indexTy.getElementType());
  mlir::Value reshapedIndex =
      builder.create<mlir::stablehlo::ReshapeOp>(loc, reshapedIndexTy, index);

  // Batching dims: every operand/start_indices axis except `dim`.
  llvm::SmallVector<int64_t> batchingDims;
  batchingDims.reserve(rank - 1);
  for (int64_t i = 0; i < rank; ++i) {
    if (i != dim) {
      batchingDims.push_back(i);
    }
  }
  llvm::SmallVector<int64_t> sliceSizes(rank, 1);

  auto dimNumbers = mlir::stablehlo::GatherDimensionNumbersAttr::get(
      ctx,
      /*offsetDims=*/{},
      /*collapsedSliceDims=*/{dim},
      /*operandBatchingDims=*/batchingDims,
      /*startIndicesBatchingDims=*/batchingDims,
      /*startIndexMap=*/{dim},
      /*indexVectorDim=*/rank);

  auto gather = builder.create<mlir::stablehlo::GatherOp>(
      loc, resultTypes[0], input, reshapedIndex, dimNumbers, sliceSizes);
  returnValues.push_back(gather.getResult());

  builder.create<mlir::func::ReturnOp>(loc, returnValues);
}

// ---------------------------------------------------------------------------
// Registry: maps the composite op name to the rewriter for its decomposition
// function.
// ---------------------------------------------------------------------------
static const llvm::StringMap<DecompRewriter> &getCompositeRewriters() {
  static const llvm::StringMap<DecompRewriter> map = [] {
    llvm::StringMap<DecompRewriter> m;
    m["tenstorrent.gather_dim"] = rewriteTenstorrentGatherDecomp;
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

    // Per unique decomposition func, record the rewriter to run, the
    // composite_attributes to pass to it, and the composite op that
    // originally populated the entry (for diagnostics). Multiple composite
    // ops may reference the same function; we rewrite each function exactly
    // once. If two composites share a decomposition target but carry
    // different `composite_attributes`, the pass errors out — decomposition
    // functions must be specialized per attribute configuration.
    struct Task {
      DecompRewriter rewriter;
      mlir::DictionaryAttr compositeAttrs;
      mlir::stablehlo::CompositeOp firstComposite;
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
      auto [entryIt, inserted] =
          toRewrite.try_emplace(callee, Task{it->second, compAttrs, comp});
      if (!inserted && entryIt->second.compositeAttrs != compAttrs) {
        auto diag = comp.emitOpError()
                    << "composite op '" << comp.getName()
                    << "' shares decomposition target '" << leaf
                    << "' with a prior composite but carries different "
                       "composite_attributes (this: "
                    << compAttrs
                    << ", prior: " << entryIt->second.compositeAttrs
                    << "); decomposition functions must be specialized per "
                       "attribute configuration";
        diag.attachNote(entryIt->second.firstComposite.getLoc())
            << "prior composite referencing the same decomposition";
        signalPassFailure();
        return;
      }
    });

    for (auto &kv : toRewrite) {
      kv.second.rewriter(kv.first, kv.second.compositeAttrs);
    }
  }
};

} // namespace mlir::tt::stablehlo
