// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/OpValidator.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"

#include "ttmlir/Asserts.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNRESOLVECOMPOSITES
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

using CompositeValidatorFn =
    std::function<OpValidationResult(CompositeOp, OpBuilder &)>;
using CompositeBuilderFn =
    std::function<Operation *(CompositeOp, OpBuilder &)>;

struct CompositeEntry {
  CompositeValidatorFn validate;
  CompositeBuilderFn build;
};

static llvm::StringMap<CompositeEntry> &getCompositeRegistry() {
  static llvm::StringMap<CompositeEntry> registry;
  return registry;
}

static void registerBuiltinComposites() {
  auto &registry = getCompositeRegistry();
  if (!registry.empty()) {
    return;
  }

  registry["topk_router_gpt"] = CompositeEntry{
      // Validate
      [](CompositeOp compositeOp, OpBuilder &builder) -> OpValidationResult {
        TT_assert(compositeOp.getInputs().size() == 3u);

        auto optAttrs = compositeOp.getCompositeAttributes();
        TT_assert(optAttrs);
        DictionaryAttr attrs = *optAttrs;

        auto kAttr = attrs.getAs<mlir::IntegerAttr>("k");
        auto numExpertsAttr = attrs.getAs<mlir::IntegerAttr>("num_experts");
        TT_assert(kAttr);
        TT_assert(numExpertsAttr);

        SmallVector<Type> resultTypes(compositeOp.getResultTypes());
        IsolatedIRValidationWrapper validator(compositeOp.getContext());
        return validator.validateOp<TopKRouterGptOp>(
            compositeOp.getOperation(), compositeOp.getLoc(), resultTypes,
            compositeOp.getInputs()[0], compositeOp.getInputs()[1],
            compositeOp.getInputs()[2],
            builder.getI32IntegerAttr(kAttr.getInt()),
            builder.getI32IntegerAttr(numExpertsAttr.getInt()));
      },
      // Build
      [](CompositeOp compositeOp, OpBuilder &builder) -> Operation * {
        DictionaryAttr attrs = *compositeOp.getCompositeAttributes();
        auto kAttr = attrs.getAs<mlir::IntegerAttr>("k");
        auto numExpertsAttr = attrs.getAs<mlir::IntegerAttr>("num_experts");

        return builder.create<TopKRouterGptOp>(
            compositeOp.getLoc(), compositeOp.getResultTypes(),
            compositeOp.getInputs()[0], compositeOp.getInputs()[1],
            compositeOp.getInputs()[2],
            builder.getI32IntegerAttr(kAttr.getInt()),
            builder.getI32IntegerAttr(numExpertsAttr.getInt()));
      }};
}

// Inline the decomposition function body at the composite ops location,
// replacing the composites results with the inlined operations results.
static LogicalResult inlineDecomposition(CompositeOp compositeOp,
                                         ModuleOp moduleOp) {
  auto decompName = compositeOp.getDecomposition();
  auto *symbolOp = SymbolTable::lookupSymbolIn(moduleOp, decompName);
  auto funcOp = dyn_cast_or_null<func::FuncOp>(symbolOp);
  if (!funcOp) {
    return compositeOp.emitOpError("decomposition function '")
           << decompName << "' not found";
  }

  OpBuilder builder(compositeOp);
  IRMapping mapping;

  for (auto [arg, input] :
       llvm::zip(funcOp.getArguments(), compositeOp.getInputs())) {
    mapping.map(arg, input);
  }

  Block &funcBody = funcOp.getBody().front();
  for (Operation &op : funcBody.without_terminator()) {
    builder.clone(op, mapping);
  }

  auto returnOp = dyn_cast<func::ReturnOp>(funcBody.getTerminator());
  if (!returnOp) {
    return compositeOp.emitOpError("decomposition function '")
           << decompName << "' must have func.return terminator";
  }

  for (auto [result, returnVal] :
       llvm::zip(compositeOp.getResults(), returnOp.getOperands())) {
    result.replaceAllUsesWith(mapping.lookup(returnVal));
  }

  compositeOp.erase();
  return success();
}

// Validate and create the typed op. Uses IsolatedIRValidationWrapper to
// check if the typed op is valid (passes workarounds + OpModel constraints)
// in an isolated module before creating it in the real IR.
// Returns nullptr if the composite name is not in the registry or validation
// fails — the caller should fall back to inlining the decomposition.
static Operation *tryCreateTypedOp(CompositeOp compositeOp,
                                   OpBuilder &builder) {
  auto &registry = getCompositeRegistry();
  auto it = registry.find(compositeOp.getCompositeName());
  if (it == registry.end()) {
    return nullptr;
  }

  auto &entry = it->second;
  auto validationResult = entry.validate(compositeOp, builder);
  if (!validationResult.isSuccess()) {
    return nullptr;
  }

  return entry.build(compositeOp, builder);
}

class TTNNResolveComposites
    : public impl::TTNNResolveCompositesBase<TTNNResolveComposites> {
public:
  using impl::TTNNResolveCompositesBase<
      TTNNResolveComposites>::TTNNResolveCompositesBase;

  void runOnOperation() final {
    registerBuiltinComposites();

    ModuleOp moduleOp = getOperation();
    llvm::DenseSet<func::FuncOp> decompositionFuncsToDelete;
    bool passFailed = false;

    moduleOp.walk([&](CompositeOp compositeOp) {
      if (passFailed) {
        return;
      }

      auto decompName = compositeOp.getDecomposition();
      auto *symbolOp = SymbolTable::lookupSymbolIn(moduleOp, decompName);
      auto decompFunc = dyn_cast<func::FuncOp>(symbolOp);

      OpBuilder builder(compositeOp);
      Operation *typedOp = tryCreateTypedOp(compositeOp, builder);

      if (typedOp) {
        for (auto [result, typedResult] :
             llvm::zip(compositeOp.getResults(), typedOp->getResults())) {
          result.replaceAllUsesWith(typedResult);
        }
        compositeOp.erase();
      } else {
        if (mlir::failed(inlineDecomposition(compositeOp, moduleOp))) {
          passFailed = true;
          return;
        }
      }

      if (decompFunc) {
        decompositionFuncsToDelete.insert(decompFunc);
      }
    });

    if (passFailed) {
      signalPassFailure();
      return;
    }

    // Clean up decomposition functions that are no longer referenced.
    for (func::FuncOp func : decompositionFuncsToDelete) {
      if (func && SymbolTable::symbolKnownUseEmpty(func, moduleOp)) {
        func.erase();
      }
    }
  }
};

} // namespace
} // namespace mlir::tt::ttnn
