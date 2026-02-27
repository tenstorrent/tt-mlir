// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "stablehlo/dialect/StablehloOps.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"
#include "ttmlir/Dialect/StableHLO/Utils/StableHLOUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_FLATTENCOMPOSITEPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

// Inline a single stablehlo.composite op. Returns success if it was flattened.
static mlir::LogicalResult
flattenOneComposite(mlir::stablehlo::CompositeOp comp,
                    mlir::SymbolTable &symTable, mlir::OpBuilder &builder) {
  mlir::Operation *op = comp.getOperation();
  mlir::MLIRContext *context = builder.getContext();

  // 0) Read metadata from the original composite.
  mlir::StringAttr origName = mlir::StringAttr::get(context, comp.getName());
  // Preserve original composite attributes.
  mlir::DictionaryAttr origCompAttrs = comp.getCompositeAttributes();
  if (!origCompAttrs) {
    // Create an empty attribute if none exists.
    origCompAttrs = mlir::DictionaryAttr::get(context);
  }

  // 1) Resolve callee from 'decomposition' (SymbolRefAttr).
  // stablehlo.composite "name" %arg {decomposition = @foo}
  auto decompAttr =
      op->getAttrOfType<mlir::SymbolRefAttr>(utils::kCompDecompositionKey);
  if (!decompAttr) {
    comp.emitOpError()
        << "missing required SymbolRefAttr attribute '"
        << utils::kCompDecompositionKey
        << "' (stablehlo.composite requires a 'decomposition' target).";
    return mlir::failure();
  }

  // SymbolRefAttr may be nested; use the leaf for func name.
  mlir::StringAttr leaf = decompAttr.getLeafReference();
  mlir::func::FuncOp callee = symTable.lookup<mlir::func::FuncOp>(leaf);

  if (!callee) {
    comp.emitOpError() << "failed to resolve callee function '" << leaf
                       << "' referenced by attribute '"
                       << utils::kCompDecompositionKey << "' (full ref: '"
                       << decompAttr
                       << "'). Please ensure the symbol is defined and visible "
                          "to the symbol table.";
    return mlir::failure();
  }
  if (callee.getBody().empty()) {
    comp.emitOpError()
        << "callee function '" << leaf
        << "' has an empty body; cannot inline/flatten composite '" << origName
        << "'. The decomposition target must define a non-empty body.";
    return mlir::failure();
  }

  // We will inline before the composite op.
  builder.setInsertionPoint(comp);

  // 2) Build a group id for all cloned ops.
  std::string groupName;
  {
    llvm::raw_string_ostream os(groupName);
    os << "composite_" << callee.getSymName();
  }
  mlir::StringAttr groupAttr = builder.getStringAttr(groupName);
  mlir::UnitAttr seedAttr = builder.getUnitAttr();

  // 3) Prepare mapping from callee arguments to composite operands.
  mlir::IRMapping mapping;
  auto &calleeEntry = callee.getBody().front();
  if (static_cast<int64_t>(calleeEntry.getNumArguments()) !=
      comp.getNumOperands()) {
    comp.emitOpError() << "number of operands (" << comp.getNumOperands()
                       << ") does not match number of callee arguments ("
                       << calleeEntry.getNumArguments() << ") in function '"
                       << leaf << "' during inlining.";
    return mlir::failure();
  }
  for (int64_t i = 0; i < static_cast<int64_t>(comp->getNumOperands()); ++i) {
    mlir::BlockArgument arg = calleeEntry.getArgument(i);
    mapping.map(arg, comp->getOperand(i));
  }

  // 4) Clone callee body ops (excluding the terminator) at the call site.
  // Track the first cloned op to attach the 'seed' attribute.
  // 'seed' includes the info composite attributes and original name.
  bool seeded = false;
  llvm::SmallVector<mlir::Operation *> clonedOps;

  // Insert cloned ops right before the composite op.
  builder.setInsertionPoint(comp);

  for (mlir::Operation &inner : calleeEntry.without_terminator()) {
    mlir::Operation *cloned = builder.clone(inner, mapping);
    // Tag the cloned operation with the group marker.
    cloned->setAttr(utils::kReoutlineGroupAttr, groupAttr);

    if (!seeded) {
      cloned->setAttr(utils::kReoutlineSeedAttr, seedAttr);
      // "tenstorrent.gelu_tanh"
      cloned->setAttr(utils::kReoutlineOrigNameAttr, origName);
      // { approximate = "tanh" }
      cloned->setAttr(utils::kReoutlineCompAttrsAttr, origCompAttrs);
      seeded = true;
    }
    clonedOps.push_back(cloned);
  }

  // 4b) Annotate cloned ops with original composite operand indices.
  llvm::DenseMap<mlir::Value, int64_t> captureToArgIndex;
  for (int64_t i = 0; i < static_cast<int64_t>(comp->getNumOperands()); ++i) {
    captureToArgIndex[comp->getOperand(i)] = i;
  }

  for (mlir::Operation *cloned : clonedOps) {
    bool hasCapture = false;
    llvm::SmallVector<int64_t> argIndices;
    argIndices.reserve(cloned->getNumOperands());
    for (mlir::Value operand : cloned->getOperands()) {
      auto it = captureToArgIndex.find(operand);
      if (it != captureToArgIndex.end()) {
        argIndices.push_back(it->second);
        hasCapture = true;
      } else {
        argIndices.push_back(-1);
      }
    }
    if (hasCapture) {
      cloned->setAttr(utils::kReoutlineArgOperandIndicesAttr,
                      builder.getDenseI64ArrayAttr(argIndices));
    }
  }

  // 5) Handle callee terminator: replace composite results with mapped return
  // values. We expect a func.return with N operands, matching the composite's
  // results.
  auto ret = llvm::dyn_cast<mlir::func::ReturnOp>(
      callee.getBody().front().getTerminator());
  if (!ret) {
    comp.emitOpError() << "expected callee function '" << leaf
                       << "' to end with a 'return' operation.";
    return mlir::failure();
  }

  if (static_cast<int64_t>(ret.getNumOperands()) != comp.getNumResults()) {
    comp.emitOpError() << "number of return operands (" << ret.getNumOperands()
                       << ") does not match number of composite results ("
                       << comp.getNumResults() << ") in function '" << leaf
                       << "' during inlining.";
    return mlir::failure();
  }
  for (int64_t i = 0; i < static_cast<int64_t>(comp->getNumResults()); ++i) {
    mlir::Value mapped = mapping.lookupOrNull(ret.getOperand(i));
    if (!mapped) {
      comp.emitOpError() << "failed to map return operand #" << i << " of '"
                         << leaf << "' during inlining.";
      return mlir::failure();
    }
    comp.getResult(i).replaceAllUsesWith(mapped);
  }

  // 6) Erase the original composite op.
  comp.erase();

  return mlir::success();
}

static void eraseDeadPrivateCallees(mlir::ModuleOp module) {
  for (auto func :
       llvm::make_early_inc_range(module.getOps<mlir::func::FuncOp>())) {
    if (!func.isPrivate()) {
      continue;
    }
    if (mlir::SymbolTable::symbolKnownUseEmpty(func, module)) {
      func.erase();
    }
  }
}

// The pass that flattens all stablehlo.composite ops in the module.
struct FlattenCompositePass
    : public impl::FlattenCompositePassBase<FlattenCompositePass> {
  using impl::FlattenCompositePassBase<
      FlattenCompositePass>::FlattenCompositePassBase;

public:
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    // If the module is not sharded, we skip this pass.
    if (!shardy_utils::isShardedModule(module)) {
      return;
    }

    mlir::SymbolTable symTable(module);
    mlir::MLIRContext *context = module.getContext();
    mlir::OpBuilder builder(context);

    for (mlir::func::FuncOp funcOp : module.getOps<mlir::func::FuncOp>()) {
      for (auto compositeOp : llvm::make_early_inc_range(
               funcOp.getOps<mlir::stablehlo::CompositeOp>())) {
        if (mlir::failed(flattenOneComposite(compositeOp, symTable, builder))) {
          compositeOp.emitOpError() << "failed to inline/flatten composite '"
                                    << compositeOp.getName() << "'";
          signalPassFailure();
          return;
        }
      }
    }

    // Finally, clean up any dead private callees.
    eraseDeadPrivateCallees(module);
  }
};

} // namespace mlir::tt::stablehlo
