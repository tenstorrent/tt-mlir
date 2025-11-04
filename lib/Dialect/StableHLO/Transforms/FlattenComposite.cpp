// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "stablehlo/dialect/StablehloOps.h"
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardingUtils.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/Error.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_FLATTENCOMPOSITEPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

/// Inline a single stablehlo.composite op. Returns success if it was flattened.
static mlir::LogicalResult
flattenOneComposite(mlir::stablehlo::CompositeOp comp,
                    mlir::SymbolTable &symTable) {
  mlir::Operation *op = comp.getOperation();
  mlir::OpBuilder builder(op);

  // 0) Read metadata from the original composite.
  mlir::StringAttr origName = builder.getStringAttr(comp.getName());
  // Preserve original composite attributes.
  mlir::DictionaryAttr origCompAttrs = comp.getCompositeAttributes();
  if (!origCompAttrs) {
    // Create an empty attribute if none exists.
    origCompAttrs = mlir::DictionaryAttr::get(builder.getContext());
  }

  // 1) Resolve callee from 'decomposition' (SymbolRefAttr).
  // stablehlo.composite "name" %arg {decomposition = @foo}
  auto decompAttr = op->getAttrOfType<mlir::SymbolRefAttr>("decomposition");
  if (!decompAttr) {
    // Some forks expose an accessor; use it if available:
    // decompAttr = comp.getDecompositionAttr();
    return mlir::failure();
  }

  // SymbolRefAttr may be nested; use the leaf for func name.
  mlir::StringAttr leaf = decompAttr.getLeafReference();
  mlir::func::FuncOp callee = symTable.lookup<mlir::func::FuncOp>(leaf);
  if (!callee || callee.getBody().empty()) {
    return mlir::failure();
  }

  // We will inline *before* the composite op.
  builder.setInsertionPoint(comp);
  // mlir::OpBuilder builder(comp);

  // 2) Build a unique group id for all cloned ops.
  std::string groupName;
  {
    llvm::raw_string_ostream os(groupName);
    os << "composite_" << callee.getSymName();
  }
  mlir::StringAttr groupAttr = builder.getStringAttr(groupName);
  mlir::UnitAttr seedAttr = builder.getUnitAttr();

  // 3) Prepare mapping from callee arguments to composite operands.
  mlir::IRMapping mapping;
  {
    auto calleeEntry = &callee.getBody().front();
    for (int64_t i = 0; i < static_cast<int64_t>(comp->getNumOperands()); ++i) {
      mlir::BlockArgument arg = calleeEntry->getArgument(i);
      mapping.map(arg, comp->getOperand(i));
    }
  }

  // 4) Clone callee body ops (excluding the terminator) at the call site.
  //    Track the first cloned op to attach the 'seed' attribute.
  //    'seed' includes the info composite attributes and original name.
  bool seeded = false;
  llvm::SmallVector<mlir::Operation *> clonedOps;

  {
    auto &calleeEntry = callee.getBody().front();
    // Insert cloned ops right before the composite op.
    builder.setInsertionPoint(comp);

    for (mlir::Operation &inner : calleeEntry.without_terminator()) {
      mlir::Operation *cloned = builder.clone(inner, mapping);
      // Tag the cloned operation with the group marker.
      cloned->setAttr(sharding_utils::kGroupAttr, groupAttr);

      if (!seeded) {
        cloned->setAttr(sharding_utils::kSeedAttr, seedAttr);
        // "tenstorrent.gelu_tanh"
        cloned->setAttr(sharding_utils::kOrigNameAttr, origName);
        // { approximate = "tanh" }
        cloned->setAttr(sharding_utils::kCompAttrsAttr, origCompAttrs);
        seeded = true;
      }
      clonedOps.push_back(cloned);
    }
  }

  // 5) Handle callee terminator: replace composite results with mapped return
  // values. We expect a func.return with N operands, matching the composite's
  // results.
  auto ret = llvm::dyn_cast<mlir::func::ReturnOp>(
      callee.getBody().front().getTerminator());
  if (!ret) {
    return mlir::failure();
  }

  for (int64_t i = 0; i < static_cast<int64_t>(comp->getNumResults()); ++i) {
    mlir::Value mapped = mapping.lookupOrNull(ret.getOperand(i));
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

    for (mlir::func::FuncOp funcOp : module.getOps<mlir::func::FuncOp>()) {
      for (auto compositeOp : llvm::make_early_inc_range(
               funcOp.getOps<mlir::stablehlo::CompositeOp>())) {
        if (mlir::failed(flattenOneComposite(compositeOp, symTable))) {
          signalPassFailure();
          return;
        }
      }
    }

    eraseDeadPrivateCallees(module);
  }
};

} // namespace mlir::tt::stablehlo
