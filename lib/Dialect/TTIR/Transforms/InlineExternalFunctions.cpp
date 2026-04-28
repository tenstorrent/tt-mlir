// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_TTIRINLINEEXTERNALFUNCTIONS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

// Merges every symbol-defining op from `externalModule` into `destModule`.
//
// For each symbol in the external module whose name already exists in the
// destination module, a unique name is generated (by appending "_0", "_1", …).
// All uses of the original name *inside* the external module are updated via
// SymbolTable::replaceAllSymbolUses before the op is moved, so that
// cross-symbol references within the incoming module remain consistent.
//
// Returns a map { originalName → finalName } for every symbol that was
// renamed.  Symbols whose name did not change are absent from the map.
// Returns failure if renaming or verification fails.
static FailureOr<llvm::StringMap<std::string>>
mergeExternalModule(ModuleOp destModule,
                    mlir::OwningOpRef<ModuleOp> &externalModule) {
  MLIRContext *ctx = destModule.getContext();

  // Build the set of symbol names already present in the destination module.
  llvm::StringSet<> usedNames;
  for (auto &op : destModule.getBody()->getOperations()) {
    if (auto symAttr =
            op.getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())) {
      usedNames.insert(symAttr.getValue());
    }
  }

  // Compute renames: walk every symbol in the external module and, if its name
  // collides with usedNames, pick a fresh name.  usedNames is updated as we
  // go so that symbols within the external module also don't collide with each
  // other after renaming.
  llvm::StringMap<std::string> renameMap;
  for (auto &op : externalModule->getBody()->getOperations()) {
    auto symAttr =
        op.getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
    if (!symAttr) {
      continue;
    }
    StringRef origName = symAttr.getValue();
    if (!usedNames.contains(origName)) {
      usedNames.insert(origName);
      continue;
    }
    // Choose a unique name by appending "_N".
    std::string newName;
    for (unsigned i = 0;; ++i) {
      newName = (origName + "_" + Twine(i)).str();
      if (!usedNames.contains(newName)) {
        break;
      }
    }
    renameMap[origName] = newName;
    usedNames.insert(newName);
  }

  // Apply each rename inside the external module:
  //   1. Replace all *uses* of the old name (call sites, symbol references).
  //   2. Rename the defining op itself.
  // Step 1 must precede step 2 because SymbolTable::replaceAllSymbolUses
  // searches by name attribute, which would not find the op after renaming.
  for (auto &[origName, newName] : renameMap) {
    StringAttr origAttr = StringAttr::get(ctx, origName);
    StringAttr newAttr = StringAttr::get(ctx, newName);

    if (failed(SymbolTable::replaceAllSymbolUses(origAttr, newAttr,
                                                 externalModule.get()))) {
      externalModule->emitError()
          << "failed to replace uses of symbol '" << origName << "' with '"
          << newName << "' inside external module";
      return failure();
    }

    // Rename the definition.  replaceAllSymbolUses only touches *uses*, so
    // the sym_name attribute on the defining op is still the original name.
    Operation *symOp =
        SymbolTable::lookupSymbolIn(externalModule.get(), origAttr);
    assert(symOp && "defining op must still be findable by original name");
    symOp->setAttr(SymbolTable::getSymbolAttrName(), newAttr);
  }

  // Move all ops from the external module body to the destination module.
  // ModuleOp uses NoTerminator, so there is no implicit terminator to skip.
  SmallVector<Operation *> toMove;
  for (auto &op : externalModule->getBody()->getOperations()) {
    toMove.push_back(&op);
  }
  for (auto *op : toMove) {
    op->remove();
    destModule.getBody()->push_back(op);
  }

  return renameMap;
}

struct TTIRInlineExternalFunctionsPass
    : public impl::TTIRInlineExternalFunctionsBase<
          TTIRInlineExternalFunctionsPass> {

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    // Cache: path → renameMap produced when that external module was merged.
    // Each external module is parsed and merged at most once regardless of how
    // many ttir.invoke_external ops reference it.
    llvm::StringMap<llvm::StringMap<std::string>> mergedPaths;

    // Collect ops before walking to avoid mutation-during-walk issues.
    SmallVector<ttir::InvokeExternalOp> invokeOps;
    moduleOp.walk([&](ttir::InvokeExternalOp op) { invokeOps.push_back(op); });

    for (auto invokeOp : invokeOps) {
      StringRef path = invokeOp.getPath();
      StringRef entry = invokeOp.getEntry();

      if (!mergedPaths.count(path)) {
        mlir::ParserConfig config(moduleOp.getContext());
        mlir::OwningOpRef<mlir::ModuleOp> externalModule =
            mlir::parseSourceFile<mlir::ModuleOp>(path, config);
        if (!externalModule) {
          invokeOp.emitOpError()
              << "failed to parse external MLIR file '" << path << "'";
          return signalPassFailure();
        }

        if (failed(mlir::verify(*externalModule))) {
          invokeOp.emitOpError()
              << "external MLIR file '" << path << "' failed verification";
          return signalPassFailure();
        }

        auto renameMapOrErr = mergeExternalModule(moduleOp, externalModule);
        if (failed(renameMapOrErr)) {
          invokeOp.emitOpError()
              << "failed to merge external module from '" << path << "'";
          return signalPassFailure();
        }

        mergedPaths.try_emplace(path, std::move(*renameMapOrErr));
      }

      // Resolve the final name of the entry symbol, accounting for any rename.
      const auto &renameMap = mergedPaths[path];
      std::string finalEntry = entry.str();
      if (auto it = renameMap.find(entry); it != renameMap.end()) {
        finalEntry = it->second;
      }

      // Replace ttir.invoke_external with func.call.
      OpBuilder builder(invokeOp);
      auto callOp = builder.create<func::CallOp>(invokeOp.getLoc(), finalEntry,
                                                 invokeOp.getResultTypes(),
                                                 invokeOp.getArguments());

      invokeOp.replaceAllUsesWith(callOp.getResults());
      invokeOp.erase();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttir
