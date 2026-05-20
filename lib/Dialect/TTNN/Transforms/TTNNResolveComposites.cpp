// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNRESOLVECOMPOSITES
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

using CompositeBuilderFn =
    std::function<Operation *(CompositeOp compositeOp, OpBuilder &builder)>;

// Registry mapping composite names to functions that build the corresponding
// typed op. Each builder creates the typed op using the composite's operands
// and attributes, returning the new operation (or nullptr on failure).
static llvm::StringMap<CompositeBuilderFn> &getCompositeRegistry() {
  static llvm::StringMap<CompositeBuilderFn> registry;
  return registry;
}

static void registerBuiltinComposites() {
  auto &registry = getCompositeRegistry();
  if (!registry.empty()) {
    return;
  }

  registry["topk_router_gpt"] = [](CompositeOp compositeOp,
                                   OpBuilder &builder) -> Operation * {
    if (compositeOp.getInputs().size() != 3) {
      return nullptr;
    }

    auto optAttrs = compositeOp.getCompositeAttributes();
    if (!optAttrs) {
      return nullptr;
    }
    DictionaryAttr attrs = *optAttrs;

    auto kAttr = attrs.getAs<mlir::IntegerAttr>("k");
    auto numExpertsAttr = attrs.getAs<mlir::IntegerAttr>("num_experts");
    if (!kAttr || !numExpertsAttr) {
      return nullptr;
    }

    return builder.create<TopKRouterGptOp>(
        compositeOp.getLoc(), compositeOp.getResultTypes(),
        compositeOp.getInputs()[0], compositeOp.getInputs()[1],
        compositeOp.getInputs()[2], builder.getI32IntegerAttr(kAttr.getInt()),
        builder.getI32IntegerAttr(numExpertsAttr.getInt()));
  };
}

// Inline the decomposition function body at the composite op's location,
// replacing the composite's results with the inlined operations' results.
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

// Try to create the typed op and validate it. Returns the created op on
// success, or nullptr if verification/validation fails.
// Requires OpModel — without it, always returns nullptr (forcing
// decomposition).
static Operation *tryCreateTypedOp(CompositeOp compositeOp,
                                   OpBuilder &builder) {
#ifndef TTMLIR_ENABLE_OPMODEL
  (void)compositeOp;
  (void)builder;
  return nullptr;
#else
  auto &registry = getCompositeRegistry();
  auto it = registry.find(compositeOp.getCompositeName());
  if (it == registry.end()) {
    return nullptr;
  }

  // Suppress diagnostics since verify() failure is an expected path.
  ScopedDiagnosticHandler diagHandler(compositeOp.getContext(),
                                      [](Diagnostic &) { return success(); });

  Operation *typedOp = it->second(compositeOp, builder);
  if (!typedOp) {
    return nullptr;
  }

  // Run static verification.
  if (failed(mlir::verify(typedOp, /*verifyRecursively=*/false))) {
    typedOp->erase();
    return nullptr;
  }

  // Run OpModel constraint validation. If we cannot validate (missing layouts,
  // non-tensor results), fall back to decomposition rather than emitting an
  // unvalidated typed op.
  std::vector<TTNNLayoutAttr> inputLayouts =
      utils::extractInputLayouts(typedOp);
  if (inputLayouts.empty()) {
    typedOp->erase();
    return nullptr;
  }

  auto resultType = dyn_cast<RankedTensorType>(typedOp->getResult(0).getType());
  if (!resultType) {
    typedOp->erase();
    return nullptr;
  }

  auto layoutAttr = dyn_cast_or_null<TTNNLayoutAttr>(resultType.getEncoding());
  if (!layoutAttr) {
    typedOp->erase();
    return nullptr;
  }

  OpConfig config(layoutAttr);
  auto validationResult = op_constraint_validation::validateOperation(
      typedOp, inputLayouts, config);
  if (!validationResult.isSuccess()) {
    typedOp->erase();
    return nullptr;
  }

  return typedOp;
#endif
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
    SmallVector<CompositeOp> compositeOps;

    moduleOp.walk([&](CompositeOp op) { compositeOps.push_back(op); });

    for (CompositeOp compositeOp : compositeOps) {
      // Track the decomposition function for potential cleanup.
      auto decompName = compositeOp.getDecomposition();
      auto *symbolOp = SymbolTable::lookupSymbolIn(moduleOp, decompName);
      auto decompFunc = dyn_cast<func::FuncOp>(symbolOp);

      OpBuilder builder(compositeOp);
      Operation *typedOp = tryCreateTypedOp(compositeOp, builder);

      if (typedOp) {
        // Typed op is valid — replace composite with it.
        for (auto [result, typedResult] :
             llvm::zip(compositeOp.getResults(), typedOp->getResults())) {
          result.replaceAllUsesWith(typedResult);
        }
        compositeOp.erase();
      } else {
        // Fallback: inline decomposition body.
        if (failed(inlineDecomposition(compositeOp, moduleOp))) {
          signalPassFailure();
          return;
        }
      }

      if (decompFunc) {
        decompositionFuncsToDelete.insert(decompFunc);
      }
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
