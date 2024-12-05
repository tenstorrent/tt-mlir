// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Target/LLVM/LLVMToDynamicLib.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

// #include "lld/Common/Driver.h"
// #include "lld/Common/ErrorHandler.h"
// #include "lld/Common/Memory.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"

// a lot of this code is adapted from IREE's LLVMCPU target code
namespace mlir::tt::llvm_to_cpu {

// Function to convert MLIR ModuleOp to LLVM Module
std::unique_ptr<llvm::Module>
convertToLLVMModule(mlir::ModuleOp mlirModule, llvm::LLVMContext &llvmContext) {
  // Ensure the MLIR module is in the LLVM dialect
  if (!mlirModule.getOperation()
           ->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>()) {
    llvm::errs() << "ModuleOp is not properly isolated\n";
    return nullptr;
  }

  // Use MLIR's translation utility
  auto llvmModule = mlir::translateModuleToLLVMIR(mlirModule, llvmContext,
                                                  "test-llvm-custom-name");
  if (!llvmModule) {
    llvm::errs() << "Failed to convert MLIR ModuleOp to LLVM IR\n";
    return nullptr;
  }

  return llvmModule;
}

std::unique_ptr<llvm::TargetMachine>
createTargetMachine(const std::string &targetTriple) {
  std::string errorMessage;
  auto llvmTarget =
      llvm::TargetRegistry::lookupTarget(targetTriple, errorMessage);
  if (!llvmTarget) {
    llvm::errs() << "target lookup failed for " << targetTriple
                 << " w msg: " << errorMessage << "\n";
    return nullptr;
  }

  llvm::TargetOptions options;

  std::unique_ptr<llvm::TargetMachine> machine(llvmTarget->createTargetMachine(
      targetTriple, "generic" /* cpu e.g k8 */,
      "" /* cpu features e.g avx512f */, options, llvm::Reloc::Model::PIC_));
  return machine;
}

llvm::LogicalResult compileToObject(llvm::Module &module,
                                    llvm::LLVMContext &context,
                                    const std::string &outputFilename) {
  // Initialize LLVM targets
  LLVMInitializeX86Target();
  LLVMInitializeX86TargetMC();
  LLVMInitializeX86TargetInfo();
  LLVMInitializeX86AsmPrinter();
  LLVMInitializeX86AsmParser();

  llvm::errs()
      << "(debug) Registered targets after explicit x86 registration :\n";
  for (const auto &Target : llvm::TargetRegistry::targets()) {
    llvm::errs() << "  " << Target.getName() << "\n";
  }

  // Set target triple if not already set
  if (module.getTargetTriple().empty()) {
    std::string defaultTriple = llvm::sys::getDefaultTargetTriple();
    llvm::outs() << "Setting default target triple: " << defaultTriple << "\n";
    module.setTargetTriple(defaultTriple);
  }

  // auto relocModel = llvm::Optional<llvm::Reloc::Model>();

  // Look up the target
  llvm::outs() << "Target triple for this module:" << module.getTargetTriple()
               << "\n";
  llvm::SmallVector<std::string, 0> attrs; // Empty feature list

  // llvm::TargetMachine *targetMachine = llvm::EngineBuilder().selectTarget(
  // llvm::Triple(module.getTargetTriple()), "x86-64", "generic", attrs);
  auto targetMachine = createTargetMachine(module.getTargetTriple());
  if (!targetMachine) {
    llvm::errs() << "Failed to create TargetMachine for triple: "
                 << module.getTargetTriple() << "\n";
    return llvm::failure();
  }

  // Set data layout
  module.setDataLayout(targetMachine->createDataLayout());

  // debug info:
  llvm::outs() << "TargetMachine Info: \n";
  llvm::outs() << "Triple: " << targetMachine->getTargetTriple().str() << "\n";
  llvm::outs() << "DataLayout: "
               << module.getDataLayout().getStringRepresentation() << "\n";

  // Create an output file stream to write the object file
  std::error_code EC;
  llvm::raw_fd_ostream out(outputFilename, EC, llvm::sys::fs::OF_None);
  if (EC) {
    llvm::errs() << "Error opening output file: " << EC.message() << "\n";
    return llvm::failure();
  }

  // Emit object code to the file
  llvm::legacy::PassManager passManager;
  passManager.add(
      new llvm::TargetLibraryInfoWrapperPass(targetMachine->getTargetTriple()));
  if (targetMachine->addPassesToEmitFile(passManager, out, nullptr,
                                         llvm::CodeGenFileType::ObjectFile)) {
    llvm::errs() << "Target machine cannot emit object file\n";
    return llvm::failure();
  }

  passManager.run(module);

  return llvm::success();
}

llvm::LogicalResult runLinkCommand(std::string commandLine) {
  llvm::dbgs() << "Running linker command:\n" << commandLine << "\n";
  const auto exitCode = system(commandLine.c_str());
  if (exitCode == 0) {
    return llvm::success();
  }
  llvm::errs() << "Linking failed; escaped command line returned exit code "
               << exitCode << ":\n\n"
               << commandLine << "\n\n";
  return llvm::failure();
}

// TODO: decide if we have any use cases where we actually need multiple .o
// files, or we should just pass in a single 1
llvm::LogicalResult linkDynamicLibrary(const std::string &libraryName,
                                       ArrayRef<std::string> objectFileNames,
                                       bool removeDebugSymbols) {
  SmallVector<std::string, 8> flags = {
      "ld.lld-17",
      "-o " + libraryName,
  };

  // Avoids including any libc/startup files that initialize the CRT as
  // we don't use any of that. Our shared libraries must be freestanding.
  flags.push_back("-nostdlib"); // -nodefaultlibs + -nostartfiles

  // Statically link all dependencies so we don't have any runtime deps.
  // We cannot have any imports in the module we produce.
  flags.push_back("-static");

  // Generate a dynamic library (ELF type: ET_DYN), otherwise dlopen()
  // won't succeed on it. This is not incompatible with -static. The GNU
  // man page for ld, `man ld`, says the following:
  //
  //   -static
  //       Do not link against shared libraries. [...] This option can be
  //       used with -shared. Doing so means that a shared library is
  //       being created but that all of the library's external references
  //       must be resolved by pulling in entries from static libraries.
  //
  // While that much is said in the GNU ld man page, the reality is that
  // out of ld.bfd, ld.gold and ld.lld, only ld.lld actually implements
  // that. Meanwhile, ld.bfd interprets -static -shared as just -static,
  // and ld.gold rejects -static -shared outright as "incompatible".
  flags.push_back("-shared");

  // Strip debug information (only, no relocations) when not requested.
  if (removeDebugSymbols) {
    flags.push_back("--strip-debug");
  }

  // Link all input objects. Note that we are not linking whole-archive as
  // we want to allow dropping of unused codegen outputs.
  for (auto &objectFile : objectFileNames) {
    flags.push_back(objectFile);
  }

  auto commandLine = llvm::join(flags, " ");
  if (llvm::failed(runLinkCommand(commandLine)))
    return llvm::failure();
  return llvm::success();
}

llvm::LogicalResult verifyAllLLVM(mlir::ModuleOp module) {
  auto llvmDialect = module.getContext()->getOrLoadDialect<LLVM::LLVMDialect>();

  bool isAllLLVM = true;

  module.walk([&](Operation *op) {
    // Allow the module operation itself to pass (builtin.module)
    if (llvm::isa<mlir::ModuleOp>(op)) {
      return; // Skip the check for the module operation
    }
    // check other operations to make sure they're llvm
    if (op->getDialect() != llvmDialect) {
      isAllLLVM = false;
      llvm::errs() << "Non-LLVM operation found: " << op->getName()
                   << " at location " << op->getLoc() << "\n";
    }
  });

  if (isAllLLVM) {
    llvm::outs() << "All operations belong to the LLVM dialect.\n";
    return llvm::success();
  } else {
    llvm::errs() << "Module contains non-LLVM dialect operations.\n";
    return llvm::failure();
  }
}

llvm::LogicalResult
compileAndLinkToSharedLibrary(llvm::Module &module, llvm::LLVMContext &context,
                              const std::string &outputPath) {
  std::string tmpObjFileName = "/home/vwells/sources/tt-mlir/temp.o";
  // Compile to object code
  if (llvm::failed(compileToObject(module, context, tmpObjFileName))) {
    llvm::errs() << "Failed to compile to object code\n";
    return llvm::failure();
  }

  if (llvm::failed(
          linkDynamicLibrary("/home/vwells/sources/tt-mlir/generated.so",
                             {tmpObjFileName}, false))) {
    llvm::errs() << "Failed to link object code to dynamic library\n";
    return llvm::failure();
  }

  return llvm::success();
}

llvm::LogicalResult
translateLLVMToDyLib(Operation *op, llvm::raw_ostream &,
                     std::unordered_map<std::string, GoldenTensor>) {

  if (!llvm::isa<mlir::ModuleOp>(op)) {
    llvm::errs() << "The operation is not a ModuleOp, cannot perform this "
                    "translation on anything but entire modules\n";
    return llvm::failure();
  }
  mlir::ModuleOp moduleOp = llvm::dyn_cast<mlir::ModuleOp>(op);

  if (llvm::failed(verifyAllLLVM(moduleOp))) {
    return llvm::failure();
  }
  llvm::LLVMContext llvmContext;
  auto llvmModule = convertToLLVMModule(moduleOp, llvmContext);
  if (llvm::failed(compileAndLinkToSharedLibrary(*llvmModule.get(), llvmContext,
                                                 "temp.so"))) {
    return llvm::failure();
  }
  return llvm::success();
}
} // namespace mlir::tt::llvm_to_cpu
