// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"

#include "lld/Common/Driver.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/raw_ostream.h"

LogicalResult verifyAllLLVM(const mlir::ModuleOp &module) {
  auto llvmDialect = module.getContext()->getOrLoadDialect<LLVM::LLVMDialect>();

  bool isAllLLVM = true;

  module.walk([&](Operation *op) {
    if (op->getDialect() != llvmDialect) {
      isAllLLVM = false;
      llvm::errs() << "Non-LLVM operation found: " << op->getName()
                   << " at location " << op->getLoc() << "\n";
    }
  });

  if (isAllLLVM) {
    llvm::outs() << "All operations belong to the LLVM dialect.\n";
    return success();
  } else {
    llvm::errs() << "Module contains non-LLVM dialect operations.\n";
    return failure();
  }
}

std::unique_ptr<llvm::MemoryBuffer>
compileToObject(llvm::Module &module, llvm::LLVMContext &context) {
  // Set up target triple
  auto targetTriple = llvm::sys::getDefaultTargetTriple();
  module.setTargetTriple(targetTriple);

  // Look up the target
  std::string errorStr;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(targetTriple, errorStr);
  if (!target) {
    llvm::errs() << "Error finding target: " << errorStr << "\n";
    return nullptr;
  }

  // Create target machine
  llvm::TargetOptions opt;
  auto relocModel = llvm::Optional<llvm::Reloc::Model>();
  auto targetMachine =
      target->createTargetMachine(targetTriple, "generic", "", opt, relocModel);

  // Set data layout
  module.setDataLayout(targetMachine->createDataLayout());

  // Emit object code to an in-memory buffer
  llvm::SmallVector<char, 0> objectBuffer;
  llvm::raw_svector_ostream stream(objectBuffer);

  llvm::legacy::PassManager passManager;
  if (targetMachine->addPassesToEmitFile(passManager, stream, nullptr,
                                         llvm::CGFT_ObjectFile)) {
    llvm::errs() << "Target machine cannot emit object file\n";
    return nullptr;
  }

  passManager.run(module);

  return llvm::MemoryBuffer::getMemBufferCopy(objectBuffer.data(),
                                              objectBuffer.size());
}

LogicalResult linkWithLLD(const llvm::MemoryBuffer &objectBuffer,
                          const std::string &outputPath) {
  // Configure arguments for LLD
  std::vector<const char *> args = {
      "ld.lld",                 // Name of the linker
      "-shared",                // Create a shared library
      "-o", outputPath.c_str(), // Output file
      "/dev/stdin"              // Input from stdin (we'll redirect the buffer)
  };

  // Temporary file redirection for LLD's input
  std::string tempObjectPath;
  llvm::sys::fs::createTemporaryFile("temp_object", "o", tempObjectPath);

  // Write the object buffer to a temporary file
  std::error_code ec;
  llvm::raw_fd_ostream tempObjectStream(tempObjectPath, ec);
  if (ec) {
    llvm::errs() << "Error writing object to temporary file: " << ec.message()
                 << "\n";
    return failure();
  }
  tempObjectStream << objectBuffer.getBuffer();
  tempObjectStream.close();

  // Run LLD programmatically
  bool success =
      lld::elf::link(args, llvm::outs(), llvm::errs(), /*exitEarly=*/false);

  // Clean up temporary file
  llvm::sys::fs::remove(tempObjectPath);

  return success;
}

LogicalResult compileAndLinkToSharedLibrary(llvm::Module &module,
                                            llvm::LLVMContext &context,
                                            const std::string &outputPath) {
  // Compile to object code
  auto objectBuffer = compileToObject(module, context);
  if (!objectBuffer) {
    llvm::errs() << "Failed to compile to object code\n";
    return failure();
  }

  // Link the object code into a shared library using LLD
  if (!linkWithLLD(*objectBuffer, outputPath)) {
    llvm::errs()
        << "Failed to link object code into shared library using LLD\n";
    return failure();
  }

  llvm::outs() << "Shared library created at: " << outputPath << "\n";
  return success();
}

LogicalResult translateLLVMToDyLib(Operation *op, llvm::raw_ostream &os) {

  if (!verifyAllLLVM(op)) {
    return failure();
  }
  if (!compileAndLinkToSharedLibrary(op, op.getContext(), "temp.so")) {
    return failure();
  }
  return success();
}
