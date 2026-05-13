// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitPy/TTNNToEmitPy.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h"
#include "ttmlir/FunctionTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::tt {

#define GEN_PASS_DEF_EMITPYADDIMPORTS
#include "ttmlir/Conversion/Passes.h.inc"

namespace {

// Add a plain `import <moduleName>` to the start of the given block.
//
static void addModuleImport(OpBuilder &builder, Block &block, Location loc,
                            StringRef moduleName) {
  builder.setInsertionPointToStart(&block);
  builder.create<emitpy::ImportOp>(loc, moduleName, /*module_alias=*/nullptr,
                                   /*members_to_import=*/nullptr,
                                   /*member_aliases=*/nullptr,
                                   /*import_all=*/nullptr);
}

// For each ImportedDeclaration func op in the block, create a
// `from <source_file> import <name1>, <name2>` import, grouped by source file.
//
static LogicalResult addImportedDeclarationImports(OpBuilder &builder,
                                                   Block &block) {
  llvm::MapVector<StringRef, SmallVector<StringRef>> grouped;

  for (auto funcOp : block.getOps<func::FuncOp>()) {
    if (!ttmlir::utils::isImportedDeclarationFunc(funcOp)) {
      continue;
    }
    auto sourceFile = ttmlir::utils::getImportedFrom(funcOp);
    if (!sourceFile) {
      return funcOp.emitOpError("ImportedDeclaration missing tt.imported_from");
    }
    grouped[*sourceFile].push_back(funcOp.getSymName());
  }

  builder.setInsertionPointToStart(&block);
  for (auto &[sourceFile, names] : grouped) {
    SmallVector<StringRef> noAliases(names.size(), "");
    builder.create<emitpy::ImportOp>(
        block.front().getLoc(), sourceFile, /*module_alias=*/nullptr,
        builder.getStrArrayAttr(names), builder.getStrArrayAttr(noAliases),
        /*import_all=*/nullptr);
  }

  return success();
}

// Add all imports to a block: module-level imports and imported-declaration
// imports (from <file> import <func>).
//
static LogicalResult addImportsToBlock(OpBuilder &builder, Block &block,
                                       Location loc) {
  if (failed(addImportedDeclarationImports(builder, block))) {
    return failure();
  }

  bool hasCPUHoistedCode =
      llvm::any_of(block.getOps<func::FuncOp>(), [](func::FuncOp funcOp) {
        return ttmlir::utils::isForwardCPUFunc(funcOp);
      });

  if (hasCPUHoistedCode) {
    addModuleImport(builder, block, loc, "torch");
    addModuleImport(builder, block, loc, "ttir_cpu");
  }

  addModuleImport(builder, block, loc, "utils");
  addModuleImport(builder, block, loc, "ttnn");

  return success();
}

class EmitPyAddImportsPass
    : public impl::EmitPyAddImportsBase<EmitPyAddImportsPass> {
public:
  using impl::EmitPyAddImportsBase<EmitPyAddImportsPass>::EmitPyAddImportsBase;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    OpBuilder builder(&getContext());

    // Collect scopes to add imports into:
    // each FileOp body if file splitting was performed,
    // otherwise, the module body itself.
    //
    auto fileOps = module.getOps<emitpy::FileOp>();

    SmallVector<Block *> scopes;
    if (fileOps.empty()) {
      scopes.push_back(module.getBody());
    } else {
      for (auto fileOp : fileOps) {
        scopes.push_back(&fileOp.getBodyRegion().front());
      }
    }

    // Add imports to each scope.
    //
    for (Block *scope : scopes) {
      if (failed(addImportsToBlock(builder, *scope, module->getLoc()))) {
        return signalPassFailure();
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createEmitPyAddImportsPass() {
  return std::make_unique<EmitPyAddImportsPass>();
}

} // namespace mlir::tt
