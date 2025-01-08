// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Target/LLVM/LLVMToDynamicLib.h"
#include "ttmlir/Dialect/TT/IR/TTOps.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"

#include <fstream>

namespace mlir::tt::llvm_to_cpu {

// flag to toggle whether we delete temp files after we're done with them
static llvm::cl::opt<bool>
    cleanupTempFiles("cleanup-dylib-temp-files",
                     llvm::cl::desc("Delete temporary files after translation"),
                     llvm::cl::init(true));

// helper to create randomized tempDir to store our temp files
llvm::SmallString<128> createTempDir() {
  llvm::SmallString<128> tempDir;
  if (llvm::sys::fs::createUniqueDirectory("ttmlir_tmp", tempDir)) {
    llvm::errs() << "Error: Could not create temporary directory.\n";
    exit(1);
  }
  return tempDir;
}

// helper to create specific temp file inside a dir
llvm::SmallString<128> createTempFile(llvm::StringRef tempDir,
                                      llvm::StringRef prefix,
                                      llvm::StringRef extension) {
  llvm::SmallString<128> tempFile;
  llvm::sys::path::append(tempFile, tempDir, prefix + "-%%%%%%" + extension);
  if (llvm::sys::fs::createUniqueFile(tempFile, tempFile)) {
    llvm::errs() << "Error: Could not create temporary file.\n";
    exit(1);
  }
  return tempFile;
}

// Function to convert MLIR ModuleOp to LLVM Module
std::unique_ptr<llvm::Module> convertToLLVMModule(CPUModuleOp cpuModule, 
                                                 llvm::LLVMContext &llvmContext) {
  mlir::registerLLVMDialectTranslation(*cpuModule.getContext());
  // Create a new MLIR module
  mlir::OpBuilder builder(cpuModule.getContext());
  auto mlirModule = builder.create<mlir::ModuleOp>(cpuModule.getLoc());
  
  // Clone the functions from CPUModule into the new ModuleOp
  builder.setInsertionPointToStart(mlirModule.getBody());
  for (auto &op : cpuModule.getBody().front().getOperations()) {
    if (isa<mlir::LLVM::LLVMFuncOp>(op)) {
      builder.clone(op);
    }
  }

  llvm::outs() << "mlir module: \n\n";
  mlirModule.dump();
  llvm::outs() << "\n\n";

  // Use the existing translation infrastructure
  auto llvmModule = mlir::translateModuleToLLVMIR(mlirModule, llvmContext,
                                                 "llvm-dylib-module");
  if (!llvmModule) {
    llvm::errs() << "Failed to convert MLIR ModuleOp to LLVM IR\n";
    return nullptr;
  }

  return llvmModule;
}

// helper to get an llvm::TargetMachine with proper default options
std::unique_ptr<llvm::TargetMachine>
createTargetMachine(llvm::StringRef targetTriple) {
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
                                    llvm::StringRef outputFilename) {

  //  Initialize LLVM targets
  // TODO (#1631): eventually, we should get this working on other archs, but
  // adding new archs requires corresponding cmake changes
  LLVMInitializeX86Target();
  LLVMInitializeX86TargetMC();
  LLVMInitializeX86TargetInfo();
  LLVMInitializeX86AsmPrinter();
  LLVMInitializeX86AsmParser();

  // Set target triple if not already set
  if (module.getTargetTriple().empty()) {
    auto defaultTriple = llvm::sys::getDefaultTargetTriple();
    module.setTargetTriple(defaultTriple);
  }

  auto targetMachine = createTargetMachine(module.getTargetTriple());
  if (!targetMachine) {
    llvm::errs() << "Failed to create TargetMachine for triple: "
                 << module.getTargetTriple() << "\n";
    return llvm::failure();
  }

  // Set data layout
  module.setDataLayout(targetMachine->createDataLayout());

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

// simple wrapper for running link command + error handling
llvm::LogicalResult runLinkCommand(llvm::StringRef commandLine) {
  llvm::dbgs() << "Running linker command:\n" << commandLine << "\n";
  const auto exitCode = system(commandLine.data());
  if (exitCode == 0) {
    return llvm::success();
  }
  llvm::errs() << "Linking failed; escaped command line returned exit code "
               << exitCode << ":\n\n"
               << commandLine << "\n\n";
  return llvm::failure();
}

// wrapper function to invoke linker w/ correct options on set of .o files
llvm::LogicalResult
linkDynamicLibrary(llvm::StringRef libraryName,
                   ArrayRef<llvm::StringRef> objectFileNames) {
  SmallVector<llvm::SmallString<13>, 8> flags = {
      llvm::SmallString<13>("ld.lld-17"), llvm::SmallString<13>("-o"),
      libraryName};

  // no stdlib dependency makes things easier for us
  flags.emplace_back("-nostdlib");

  // want to create a standalone dylib w/o dependencies on other dylibs
  // apparently, only lld supports this combo
  flags.emplace_back("-static");
  flags.emplace_back("-shared");

  // In our case, we probably don't gain much useful info from debug symbols
  // anyway
  flags.emplace_back("--strip-debug");

  // Link all input .o into 1 output .so
  for (const auto &objectFile : objectFileNames) {
    flags.emplace_back(objectFile);
  }

  auto commandLine = llvm::join(flags, " ");
  if (llvm::failed(runLinkCommand(commandLine)))
    return llvm::failure();
  return llvm::success();
}

// checker to make sure we don't attempt translation unless entire module is
// properly converted to LLVM Dialect
llvm::LogicalResult verifyAllLLVM(tt::CPUModuleOp module) {
  auto llvmDialect = module.getContext()->getOrLoadDialect<LLVM::LLVMDialect>();

  bool isAllLLVM = true;

  module.walk([&](Operation *op) {
    // check other operations to make sure they're llvm
    if (op->getDialect() != llvmDialect && !(llvm::isa<tt::CPUModuleOp>(op) || llvm::isa<tt::CPUModuleTerminatorOp>(op)) ) {
      isAllLLVM = false;
      llvm::errs() << "Non-LLVM operation found: " << op->getName()
                   << " at location " << op->getLoc() << "\n";
    }
  });

  if (isAllLLVM) {
    return llvm::success();
  } else {
    llvm::errs() << "Module contains non-LLVM dialect operations.\n";
    return llvm::failure();
  }
}

// Wrapper func to create objects, link them into dylib, and return dylib as
// binary buffer is successful
std::optional<llvm::SmallVector<char, 2048>>
compileAndLinkToSharedLibrary(llvm::Module &module,
                              llvm::LLVMContext &context) {
  const auto tmpDirName = createTempDir();
  const auto tmpObjFileName =
      createTempFile(tmpDirName, module.getName(), ".o");
  // Compile to object code
  if (llvm::failed(compileToObject(module, context, tmpObjFileName))) {
    llvm::errs() << "Failed to compile to object code\n";
    return std::nullopt;
  }

  auto dylibName = createTempFile(tmpDirName, module.getName(), ".so");
  // Link to dynamic library
  if (llvm::failed(linkDynamicLibrary(dylibName, {tmpObjFileName}))) {
    llvm::errs() << "Failed to link object code to dynamic library\n";
    return std::nullopt;
  }

  std::ifstream file(dylibName.c_str(), std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    llvm::errs() << "Could not open file: " << dylibName << "\n";
    return std::nullopt;
  }

  // Get the size of the file.
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  // Read the file into a vector.
  llvm::SmallVector<char, 2048> buffer(size);
  if (!file.read(buffer.data(), size)) {
    llvm::errs() << "Failed to read file: " << dylibName << "\n";
    return std::nullopt;
  }

  if (cleanupTempFiles) {
    llvm::sys::fs::remove_directories(tmpDirName);
  } else {
    llvm::outs() << "wrote temp files to: " << tmpDirName << "\n";
  }

  return buffer;
}

llvm::LogicalResult translateLLVMToDyLib(Operation *op, llvm::raw_ostream &os) {

  if (!llvm::isa<tt::CPUModuleOp>(op)) {
    llvm::errs() << "The operation is not a ModuleOp, cannot perform this "
                    "translation on anything but entire modules\n";
    return llvm::failure();
  }
  auto moduleOp = llvm::dyn_cast<tt::CPUModuleOp>(op);
  llvm::outs() << "CPU Module:\n\n";
  moduleOp->dump();
  llvm::outs() << "\n\n";
  if (llvm::failed(verifyAllLLVM(moduleOp))) {
    return llvm::failure();
  }
  llvm::LLVMContext llvmContext;
  auto llvmModule = convertToLLVMModule(moduleOp, llvmContext);
  if (!llvmModule)
  {
    return llvm::failure();
  }
  const auto maybeDylibBinary =
      compileAndLinkToSharedLibrary(*llvmModule.get(), llvmContext);
  if (!maybeDylibBinary.has_value()) {
    return llvm::failure();
  }
  os.write(maybeDylibBinary.value().data(), maybeDylibBinary.value().size());
  return llvm::success();
}
} // namespace mlir::tt::llvm_to_cpu
