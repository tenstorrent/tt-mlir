// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitC/TTNNToEmitC.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/FunctionTypes.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::tt {

#define GEN_PASS_DEF_EMITCLINKMODULES
#include "ttmlir/Conversion/Passes.h.inc"

namespace {

// Helper to generate C type string from LLVM type.
std::string getLLVMTypeAsCString(mlir::Type type) {
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
    unsigned width = intType.getWidth();
    if (width == 1) {
      return "bool";
    }
    if (width == 8) {
      return "int8_t";
    }
    if (width == 16) {
      return "int16_t";
    }
    if (width == 32) {
      return "int32_t";
    }
    if (width == 64) {
      return "int64_t";
    }
    return "int" + std::to_string(width) + "_t";
  }
  if (mlir::isa<mlir::Float16Type>(type)) {
    return "float16_t";
  }
  if (mlir::isa<mlir::Float32Type>(type)) {
    return "float";
  }
  if (mlir::isa<mlir::Float64Type>(type)) {
    return "double";
  }
  if (auto ptrType = mlir::dyn_cast<mlir::LLVM::LLVMPointerType>(type)) {
    return "void*";
  }
  // Default fallback
  return "void*";
}

// Helper to generate extern "C" declaration string for an LLVM function.
std::string generateExternCDeclaration(mlir::LLVM::LLVMFuncOp funcOp) {
  std::string decl = "extern \"C\" ";

  // Return type
  auto funcType = funcOp.getFunctionType();
  mlir::Type returnType = funcType.getReturnType();
  if (mlir::isa<mlir::LLVM::LLVMVoidType>(returnType)) {
    decl += "void";
  } else {
    decl += getLLVMTypeAsCString(returnType);
  }

  decl += " " + funcOp.getSymName().str() + "(";

  // Parameters
  auto paramTypes = funcType.getParams();
  for (size_t i = 0; i < paramTypes.size(); ++i) {
    if (i > 0) {
      decl += ", ";
    }
    decl += getLLVMTypeAsCString(paramTypes[i]);
  }

  if (paramTypes.empty()) {
    decl += "void";
  }

  decl += ");";
  return decl;
}

// Pass which links the CPU and Device modules for EmitC dylib compilation.
// The CPU module contains LLVM IR that will be compiled to a dynamic library.
// This pass generates extern "C" declarations in the root module and moves
// the Device module contents to the root, while preserving the CPU module
// for later dylib compilation by Alchemist.
//
class EmitCLinkModulesPass
    : public impl::EmitCLinkModulesBase<EmitCLinkModulesPass> {
public:
  using impl::EmitCLinkModulesBase<EmitCLinkModulesPass>::EmitCLinkModulesBase;

  void runOnOperation() override {
    mlir::ModuleOp rootModule = getOperation();

    // Ensure we only run this on top-level ModuleOp.
    if (rootModule->getParentOp() != nullptr) {
      rootModule.emitError("EmitCLinkModules pass must run on root module!");
      return signalPassFailure();
    }

    // Find DeviceModuleOp (optional - may have been unwrapped in legacy path).
    auto deviceModuleOps = rootModule.getOps<ttcore::DeviceModuleOp>();
    if (deviceModuleOps.empty()) {
      // No DeviceModuleOp means legacy path was used (already unwrapped).
      // Nothing to do - the EmitC ops are already in the root module.
      return;
    }

    ttcore::DeviceModuleOp deviceModuleOp = *deviceModuleOps.begin();
    auto deviceModule =
        mlir::cast<mlir::ModuleOp>(deviceModuleOp.getBody()->front());

    // Transfer attributes from device module to the root module.
    for (const auto &attr : deviceModule->getAttrs()) {
      if (!rootModule->hasAttr(attr.getName())) {
        rootModule->setAttr(attr.getName(), attr.getValue());
      }
    }

    // Find CPUModuleOp (optional).
    auto cpuModuleOps = rootModule.getOps<ttcore::CPUModuleOp>();
    ttcore::CPUModuleOp cpuModuleOp =
        cpuModuleOps.empty() ? nullptr : *cpuModuleOps.begin();

    auto &rootBody = rootModule.getBodyRegion().front();
    auto &deviceBody = deviceModule.getBodyRegion().front();

    OpBuilder builder(&getContext());
    builder.setInsertionPointToStart(&rootBody);

    // If we have a CPU module with LLVM functions, generate extern "C"
    // declarations for each function that will be in the dylib.
    if (cpuModuleOp) {
      auto cpuModule =
          mlir::cast<mlir::ModuleOp>(cpuModuleOp.getBody()->front());

      // Collect LLVM functions and generate extern declarations.
      llvm::SmallVector<std::string> externDecls;
      for (auto llvmFunc : cpuModule.getOps<mlir::LLVM::LLVMFuncOp>()) {
        // Skip external declarations.
        if (llvmFunc.isExternal()) {
          continue;
        }
        externDecls.push_back(generateExternCDeclaration(llvmFunc));
      }

      // Insert all extern declarations at the beginning of the root module.
      if (!externDecls.empty()) {
        std::string allDecls;
        for (const auto &decl : externDecls) {
          allDecls += decl + "\n";
        }
        builder.create<mlir::emitc::VerbatimOp>(rootModule.getLoc(), allDecls);
      }
    }

    // Move all operations from the Device module to root, including the
    // CPU-hoisted function declarations. The declarations serve as targets
    // for func.call operations and will be resolved at link time with the
    // dylib functions.
    //
    // Insert device content BEFORE the CPU module (if it exists) to maintain
    // expected order: extern decls -> device functions -> cpu_module.
    if (cpuModuleOp) {
      rootBody.getOperations().splice(cpuModuleOp->getIterator(),
                                      deviceBody.getOperations());
    } else {
      rootBody.getOperations().splice(rootBody.end(),
                                      deviceBody.getOperations());
    }

    // Erase the Device module wrapper.
    deviceModuleOp->erase();

    // Note: CPUModuleOp is preserved with LLVM IR for Alchemist to compile
    // to a dynamic library.
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createEmitCLinkModulesPass() {
  return std::make_unique<EmitCLinkModulesPass>();
}

} // namespace mlir::tt
