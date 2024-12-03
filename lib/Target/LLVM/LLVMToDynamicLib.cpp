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

llvm::LogicalResult compileToObject(llvm::Module &module,
                                    llvm::LLVMContext &context,
                                    const std::string &outputFilename) {
  // Initialize LLVM targets
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

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

  llvm::TargetMachine *targetMachine = llvm::EngineBuilder().selectTarget(
      llvm::Triple(module.getTargetTriple()), "x86-64", "generic", attrs);
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
  passManager.add(
      llvm::createBasicAliasAnalysisPass()); // Add alias analysis pass
  passManager.add(llvm::createTargetTransformInfoWrapperPass(
      targetMachine->getTargetTransformInfo()));
  passManager.add(
      llvm::createTargetLoweringPass(targetMachine->getTargetLowering()));
  passManager.add(llvm::createMachineVerifierPass()); // Optional
  if (targetMachine->addPassesToEmitFile(passManager, out, nullptr,
                                         llvm::CodeGenFileType::ObjectFile)) {
    llvm::errs() << "Target machine cannot emit object file\n";
    return llvm::failure();
  }

  passManager.run(module);

  return llvm::success();
}

// std::unique_ptr<llvm::MemoryBuffer>
// compileToObject(llvm::Module &module, llvm::LLVMContext &context) {
//   // Set up target triple
//   auto targetTriple = llvm::sys::getDefaultTargetTriple();
//   module.setTargetTriple(targetTriple);

//   // Look up the target
//   std::string errorStr;
//   const llvm::Target *target =
//       llvm::TargetRegistry::lookupTarget(targetTriple, errorStr);
//   if (!target) {
//     llvm::errs() << "Error finding target: " << errorStr << "\n";
//     return nullptr;
//   }

//   // Create target machine
//   llvm::TargetOptions opt;
//   auto relocModel = llvm::Optional<llvm::Reloc::Model>();
//   auto targetMachine =
//       target->createTargetMachine(targetTriple, "generic", "", opt,
//       relocModel);

//   // Set data layout
//   module.setDataLayout(targetMachine->createDataLayout());

//   // Emit object code to an in-memory buffer
//   llvm::SmallVector<char, 0> objectBuffer;
//   llvm::raw_svector_ostream stream(objectBuffer);

//   llvm::legacy::PassManager passManager;
//   if (targetMachine->addPassesToEmitFile(passManager, stream, nullptr,
//                                          llvm::CGFT_ObjectFile)) {
//     llvm::errs() << "Target machine cannot emit object file\n";
//     return nullptr;
//   }

//   passManager.run(module);

//   return llvm::MemoryBuffer::getMemBufferCopy(objectBuffer.data(),
//                                               objectBuffer.size());
// }

// llvm::LogicalResult linkWithLLD(const llvm::MemoryBuffer &objectBuffer,
//                           const std::string &outputPath) {
//   // Configure arguments for LLD
//   std::vector<const char *> args = {
//       "ld.lld",                 // Name of the linker
//       "-shared",                // Create a shared library
//       "-o", outputPath.c_str(), // Output file
//       "/dev/stdin"              // Input from stdin (we'll redirect the
//       buffer)
//   };

//   // Temporary file redirection for LLD's input
//   std::string tempObjectPath;
//   llvm::sys::fs::createTemporaryFile("temp_object", "o", tempObjectPath);

//   // Write the object buffer to a temporary file
//   std::error_code ec;
//   llvm::raw_fd_ostream tempObjectStream(tempObjectPath, ec);
//   if (ec) {
//     llvm::errs() << "Error writing object to temporary file: " <<
//     ec.message()
//                  << "\n";
//     return llvm::failure();
//   }
//   tempObjectStream << objectBuffer.getBuffer();
//   tempObjectStream.close();

//   // Run LLD programmatically
//   bool success =
//       lld::elf::link(args, llvm::outs(), llvm::errs(), /*exitEarly=*/false);

//   // Clean up temporary file
//   llvm::sys::fs::remove(tempObjectPath);

//   return success;
// }

// llvm::LogicalResult serializeExecutable(const SerializationOptions &options,
//                                   IREE::HAL::ExecutableVariantOp variantOp,
//                                   OpBuilder &executableBuilder) override {
//   // Perform the translation in a separate context to avoid any
//   // multi-threading issues.
//   llvm::LLVMContext context;
//   auto maybeTarget = getVariantTarget(variantOp);
//   if (!maybeTarget)
//     return llvm::failure();
//   const LLVMTarget &target = *maybeTarget;
//   LLVM_DEBUG(dbgs() << "LLVM-CPU SerializeExecutable:\n"
//                     << "-----------------------------\n";
//              target.print(dbgs()));

//   // For debugging effective options in live builds, uncomment the following.
//   // dbgs() << "LLVM-CPU ";
//   // target.print(dbgs());

//   // We name our files after the executable name so that they are easy to
//   // track both during compilation (logs/artifacts/etc), as outputs (final
//   // intermediate code/binary files), and at runtime (loaded
//   // libraries/symbols/etc).
//   auto libraryName =
//       variantOp->getParentOfType<IREE::HAL::ExecutableOp>().getName().str();

//   // Validate flags for output mode.
//   if (target.getLinkEmbedded() && target.linkStatic) {
//     return variantOp.emitError()
//            << "cannot embed ELF and produce static library simultaneously";
//   }

//   // Try to create the LLVM target machine interface for the variant target.
//   auto targetMachine = createTargetMachine(target);
//   if (!targetMachine) {
//     return mlir::emitError(variantOp.getLoc())
//            << "failed to create target machine for target triple '"
//            << target.getTriple() << "'";
//   }

//   // Specialize the module to the target triple.
//   // The executable will have been cloned into other ExecutableVariantOps for
//   // other triples so it's fine to mutate in-place.
//   const llvm::Triple &targetTriple = targetMachine->getTargetTriple();
//   variantOp.getInnerModule()->setAttr(
//       LLVM::LLVMDialect::getTargetTripleAttrName(),
//       executableBuilder.getStringAttr(targetTriple.str()));

//   // At this moment we are leaving MLIR LLVM dialect land translating module
//   // into target independent LLVMIR.
//   auto llvmModule = mlir::translateModuleToLLVMIR(variantOp.getInnerModule(),
//                                                   context, libraryName);
//   if (!llvmModule) {
//     return variantOp.emitError() << "failed to translate the MLIR LLVM "
//                                     "dialect to the native llvm::Module";
//   }

//   // Configure the functions in the module. This may override defaults set
//   // during the MLIR->LLVM conversion.
//   for (auto &func : *llvmModule) {
//     // Enable frame pointers to ensure that stack unwinding works, e.g. in
//     // Tracy. In principle this could also be achieved by enabling unwind
//     // tables, but we tried that and that didn't work in Tracy (which uses
//     // libbacktrace), while enabling frame pointers worked.
//     // https://github.com/iree-org/iree/issues/3957
//     func.addFnAttr("frame-pointer", "all");

//     // -ffreestanding-like behavior.
//     func.addFnAttr("no-builtins");

//     // Our dispatches are all hot - that's kind of the point.
//     // This may favor more aggressive optimizations.
//     func.addFnAttr("hot");
//   }

//   // Build the IREE HAL executable library metadata. The runtime uses this to
//   // find the entry point functions and their information.
//   LibraryBuilder::Mode libraryBuilderMode =
//       target.debugSymbols ? LibraryBuilder::Mode::INCLUDE_REFLECTION_ATTRS
//                           : LibraryBuilder::Mode::NONE;
//   LibraryBuilder libraryBuilder(llvmModule.get(), libraryBuilderMode,
//                                 LibraryBuilder::Version::LATEST);

//   switch (target.sanitizerKind) {
//   case SanitizerKind::kNone: {
//     libraryBuilder.setSanitizerKind(LibraryBuilder::SanitizerKind::NONE);
//     break;
//   }
//   case SanitizerKind::kAddress: {
//     libraryBuilder.setSanitizerKind(LibraryBuilder::SanitizerKind::ADDRESS);
//     for (auto &function : llvmModule->getFunctionList()) {
//       function.addFnAttr(llvm::Attribute::SanitizeAddress);
//     }
//   } break;
//   case SanitizerKind::kThread: {
//     libraryBuilder.setSanitizerKind(LibraryBuilder::SanitizerKind::THREAD);
//     for (auto &function : llvmModule->getFunctionList()) {
//       function.addFnAttr(llvm::Attribute::SanitizeThread);
//     }
//   } break;
//   }

//   // Declare dynamically imported functions.
//   auto importsAttrName =
//       StringAttr::get(variantOp.getContext(), "hal.executable.imports");
//   if (auto importsAttr =
//   variantOp->getAttrOfType<ArrayAttr>(importsAttrName)) {
//     for (auto importAttr : importsAttr.getAsValueRange<ArrayAttr>()) {
//       auto nameAttr = llvm::cast<StringAttr>(importAttr[0]);
//       auto weakAttr = llvm::cast<BoolAttr>(importAttr[1]);
//       libraryBuilder.addImport(nameAttr.getValue(), weakAttr.getValue());
//     }
//     variantOp->removeAttr(importsAttrName);
//   }

//   // Declare exported entry points.
//   auto align16 = llvm::Attribute::getWithAlignment(context, llvm::Align(16));
//   for (auto exportOp : variantOp.getBlock().getOps<ExecutableExportOp>()) {
//     // Find the matching function in the LLVM module.
//     auto *llvmFunc = llvmModule->getFunction(exportOp.getName());
//     if (!llvmFunc)
//       continue;
//     llvmFunc->setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);
//     llvmFunc->setDSOLocal(true);

//     // Tag the function parameters in case they got removed during
//     conversion.
//     // (%arg0: environment, %arg1: dispatch_state, %arg2: workgroup_state)
//     for (unsigned i = 0; i <= 2; ++i) {
//       llvmFunc->addParamAttr(i, llvm::Attribute::NonNull);
//       llvmFunc->addParamAttr(i, llvm::Attribute::NoAlias);
//       llvmFunc->addParamAttr(i, align16);
//     }

//     LibraryBuilder::DispatchAttrs dispatchAttrs = {0};

//     // Entry points may optionally specify that they require workgroup local
//     // memory. We fetch that value here and plumb it through so the runtime
//     // knows how much memory to reserve and pass in.
//     dispatchAttrs.localMemorySize = exportOp.getWorkgroupLocalMemory()
//                                         .value_or(APInt(64, 0))
//                                         .getSExtValue();

//     // Specify the constant and binding information used to validate
//     // dispatches.
//     if (auto layoutAttr = exportOp.getLayout()) {
//       dispatchAttrs.constantCount = layoutAttr.getConstants();
//       dispatchAttrs.bindingCount = layoutAttr.getBindings().size();
//     }

//     LibraryBuilder::SourceLocation sourceLocation;
//     if (options.debugLevel >= 1) {
//       if (auto loc = findFirstFileLoc(exportOp.getLoc())) {
//         sourceLocation = {"", loc->getFilename().str(), loc->getLine()};
//       }
//     }
//     SmallVector<LibraryBuilder::SourceLocation> stageLocations;
//     if (options.debugLevel >= 3) {
//       if (auto locsAttr = exportOp.getSourceLocsAttr()) {
//         for (auto locAttr : locsAttr.getValue()) {
//           if (auto loc =
//                   findFirstFileLoc(cast<LocationAttr>(locAttr.getValue()))) {
//             stageLocations.push_back({
//                 locAttr.getName().str(),
//                 loc->getFilename().str(),
//                 loc->getLine(),
//             });
//           }
//         }
//       }
//     }
//     libraryBuilder.addExport(exportOp.getName(), std::move(sourceLocation),
//                              std::move(stageLocations), /*tag=*/"",
//                              dispatchAttrs, llvmFunc);
//   }

//   // Embed source files (if present).
//   if (auto sourcesAttr = variantOp.getSourcesAttr()) {
//     for (auto sourceAttr : sourcesAttr.getValue()) {
//       if (auto resourceAttr = dyn_cast_if_present<DenseResourceElementsAttr>(
//               sourceAttr.getValue())) {
//         auto handle = resourceAttr.getRawHandle();
//         SmallVector<char> rawData;
//         llvm::append_range(rawData, handle.getBlob()->getData());
//         libraryBuilder.addSourceFile(sourceAttr.getName(),
//         std::move(rawData));
//       }
//     }
//   }

//   auto queryFunctionName = std::string(kQueryFunctionName);
//   if (target.linkStatic) {
//     // Static library query functions must be unique to support multiple
//     // libraries in the same namespace.
//     queryFunctionName = libraryName + "_library_query";
//   }
//   auto *queryLibraryFunc = libraryBuilder.build(queryFunctionName);

//   // The query function must be exported for dynamic libraries.
//   queryLibraryFunc->setDSOLocal(false);
//   queryLibraryFunc->setVisibility(
//       llvm::GlobalValue::VisibilityTypes::DefaultVisibility);
//   queryLibraryFunc->setLinkage(
//       llvm::GlobalValue::LinkageTypes::ExternalLinkage);
//   queryLibraryFunc->setDLLStorageClass(
//       llvm::GlobalValue::DLLStorageClassTypes::DLLExportStorageClass);

//   // If linking dynamically, find a suitable linker tool and configure the
//   // module with any options that tool requires.
//   std::unique_ptr<LinkerTool> linkerTool;
//   if (!target.linkStatic) {
//     // Grab a linker tool based on the options (and target environment).
//     // This uses the defaultOptions_ in order to get paths and such, which
//     // are environmental, but replace the target with the actual one.
//     LLVMTargetOptions options = defaultOptions_;
//     options.target = target;
//     linkerTool = LinkerTool::getForTarget(targetTriple, options);
//     if (!linkerTool) {
//       return mlir::emitError(variantOp.getLoc())
//              << "failed to find a target linker for the given target triple
//              '"
//              << targetTriple.str() << "'";
//     }

//     // Configure the module with any code generation options required later
//     by
//     // linking (such as initializer functions).
//     if (failed(linkerTool->configureModule(llvmModule.get(),
//                                            {queryLibraryFunc}))) {
//       return variantOp.emitError()
//              << "failed to configure LLVM module for target linker";
//     }
//   }

//   // Specialize the module to our target machine.
//   llvmModule->setDataLayout(targetMachine->createDataLayout());
//   llvmModule->setTargetTriple(targetMachine->getTargetTriple().str());

//   // Dump just the codegen bitcode before linking and optimization.
//   if (!options.dumpIntermediatesPath.empty()) {
//     dumpLLVMModuleToPath(options.dumpIntermediatesPath, options.dumpBaseName,
//                          variantOp.getName(), ".codegen", *llvmModule);
//   }

//   // Statically link libraries into our module prior to LLVM optimizations.
//   // This approximates LTO.
//   llvm::Linker moduleLinker(*llvmModule);

//   // Link any bitcode files specified on the command line.
//   if (failed(linkCmdlineBitcodeFiles(variantOp.getLoc(), moduleLinker,
//                                      llvm::Linker::OverrideFromSrc,
//                                      *targetMachine, context))) {
//     return llvm::failure();
//   }

//   // Link any bitcode objects specified in executable.object attributes and
//   // specialize them for the current config.
//   if (failed(linkBitcodeObjects(variantOp.getLoc(), moduleLinker,
//                                 llvm::Linker::LinkOnlyNeeded, *targetMachine,
//                                 variantOp.getObjectsAttr(), context))) {
//     return llvm::failure();
//   }

//   // Link our libdevice after all codegen and user objects as they may
//   // reference it. Some of the functions in here are only known used after
//   // we perform LLVM ISel and need to be pulled in whether they are used or
//   // not.
//   if (failed(linkBitcodeModule(
//           variantOp.getLoc(), moduleLinker, llvm::Linker::OverrideFromSrc,
//           *targetMachine, "libdevice",
//           loadDeviceBitcode(targetMachine.get(), context),
//           [&](llvm::Module &module) {
//             specializeDeviceModule(variantOp, module, *targetMachine);
//           }))) {
//     return mlir::emitError(variantOp.getLoc())
//            << "failed linking in builtin library for target triple '"
//            << targetTriple.str() << "'";
//   }

//   if (target.linkUkernelBitcode) {
//     // Link in ukernel bitcode.
//     if (hasUkernel(variantOp.getTarget())) {
//       llvm::Expected<std::unique_ptr<llvm::Module>> bitcode =
//           loadUKernelBitcode(targetMachine.get(), context);
//       if (!bitcode) {
//         return mlir::emitError(variantOp.getLoc())
//                << "failed to load ukernel bitcode: "
//                << llvm::toString(bitcode.takeError());
//       }

//       if (bitcode.get()) {
//         llvm::StringRef bitcodeName = bitcode.get()->getName();
//         if (failed(linkBitcodeModule(
//                 variantOp.getLoc(), moduleLinker,
//                 llvm::Linker::LinkOnlyNeeded, *targetMachine, bitcodeName,
//                 std::move(bitcode), {}))) {
//           return mlir::emitError(variantOp.getLoc())
//                  << "failed linking in architecture-specific ukernel bitcode
//                  "
//                     "for target triple '"
//                  << targetTriple.str() << "'";
//         }
//       }
//     }
//   }

//   // Strip any compiler identifiers that may have snuck in. We let the linker
//   // tag the module.
//   auto *llvmIdent = llvmModule->getNamedMetadata("llvm.ident");
//   if (llvmIdent)
//     llvmIdent->clearOperands();

//   // Dump all linked bitcode prior to optimization.
//   if (!options.dumpIntermediatesPath.empty()) {
//     dumpLLVMModuleToPath(options.dumpIntermediatesPath, options.dumpBaseName,
//                          variantOp.getName(), ".linked", *llvmModule);
//   }

//   // LLVM opt passes that perform code generation
//   optimizations/transformation
//   // similar to what a frontend would do.
//   if (failed(runLLVMIRPasses(target, targetMachine.get(), llvmModule.get())))
//   {
//     return variantOp.emitError()
//            << "failed to run LLVM-IR opt passes for IREE::HAL::ExecutableOp "
//               "targeting '"
//            << targetTriple.str() << "'";
//   }

//   // Fixup visibility from any symbols we may link in - we want to hide all
//   // but the query entry point.
//   // Note: can't move this before runLLVMIRPasses at the moment, as further
//   // symbol references may still be created past this point, namely to math
//   // functions, e.g. `llvm.frem` lowering to a call to `fmodf`.
//   SetVector<llvm::Function *> preservedFuncs;
//   preservedFuncs.insert(queryLibraryFunc);
//   fixupVisibility(*llvmModule, preservedFuncs);

//   // Dump bitcode post-linking and optimization.
//   if (!options.dumpIntermediatesPath.empty()) {
//     dumpLLVMModuleToPath(options.dumpIntermediatesPath, options.dumpBaseName,
//                          variantOp.getName(), ".optimized", *llvmModule);
//   }

//   SmallVector<Artifact> objectFiles;

//   // Emit the base object file containing the bulk of our code.
//   // This must come first such that we have the proper library linking order.
//   {
//     // NOTE: today we just use a single object file, however if we wanted to
//     // scale code generation and linking we'd want to generate one per
//     // function (or something like that). A single object file is also
//     // instrumental to static library generation (which only supports one
//     // object file per library).
//     std::string objectData;
//     if (failed(runEmitObjFilePasses(targetMachine.get(), llvmModule.get(),
//                                     llvm::CodeGenFileType::ObjectFile,
//                                     &objectData))) {
//       return variantOp.emitError()
//              << "failed to compile LLVM-IR module to an object file";
//     }
//     if (!options.dumpIntermediatesPath.empty()) {
//       dumpDataToPath(options.dumpIntermediatesPath, options.dumpBaseName,
//                      variantOp.getName(), ".o", objectData);
//     }
//     auto objectFile = Artifact::createTemporary(libraryName, "o");
//     auto &os = objectFile.outputFile->os();
//     os << objectData;
//     os.flush();
//     os.close();
//     objectFiles.push_back(std::move(objectFile));
//   }

//   // Dump assembly listing after optimization, which is just a textual
//   // representation of the object file we generate below.
//   if (!options.dumpIntermediatesPath.empty()) {
//     std::string asmData;
//     if (failed(runEmitObjFilePasses(targetMachine.get(), llvmModule.get(),
//                                     llvm::CodeGenFileType::AssemblyFile,
//                                     &asmData))) {
//       return variantOp.emitError()
//              << "failed to compile LLVM-IR module to an assembly file";
//     }
//     dumpDataToPath(options.dumpIntermediatesPath, options.dumpBaseName,
//                    variantOp.getName(), ".s", asmData);
//   }
// }

// llvm::LogicalResult runLinkCommand(std::string commandLine, llvm::StringRef
// env) {
//   LLVM_DEBUG(llvm::dbgs() << "Running linker command:\n"
//                           << env << " " << commandLine << "\n");
//   if (!env.empty()) {

//     commandLine = (env + " " + commandLine).str();
//   } else {
//     commandLine = escapeCommandLineComponent(commandLine);
//   }
//   int exitCode = system(commandLine.c_str());
//   if (exitCode == 0)
//     return llvm::success();
//   llvm::errs() << "Linking failed; escaped command line returned exit code "
//                << exitCode << ":\n\n"
//                << commandLine << "\n\n";
//   return llvm::failure();
// }

// std::optional<Artifacts>
// linkDynamicLibrary(llvm::StringRef libraryName,
//                    ArrayRef<Artifact> objectFiles) override {
//   Artifacts artifacts;

//   // Create the shared object name; if we only have a single input object we
//   // can just reuse that.
//   if (objectFiles.size() == 1) {
//     artifacts.libraryFile =
//         Artifact::createVariant(objectFiles.front().path, "so");
//   } else {
//     artifacts.libraryFile = Artifact::createTemporary(libraryName, "so");
//   }
//   artifacts.libraryFile.close();

//   llvm::SmallVector<std::string, 8> linkCmdAndArgs = {
//       "ld.lld-17",
//       "-o " + artifacts.libraryFile.path,
//   };

//   // Avoids including any libc/startup files that initialize the CRT as
//   // we don't use any of that. Our shared libraries must be freestanding.
//   linkCmdAndArgs.push_back("-nostdlib"); // -nodefaultlibs + -nostartfiles

//   // Statically link all dependencies so we don't have any runtime deps.
//   // We cannot have any imports in the module we produce.
//   linkCmdAndArgs.push_back("-static");

//   // Generate a dynamic library (ELF type: ET_DYN), otherwise dlopen()
//   // won't succeed on it. This is not incompatible with -static. The GNU
//   // man page for ld, `man ld`, says the following:
//   //
//   //   -static
//   //       Do not link against shared libraries. [...] This option can be
//   //       used with -shared. Doing so means that a shared library is
//   //       being created but that all of the library's external references
//   //       must be resolved by pulling in entries from static libraries.
//   //
//   // While that much is said in the GNU ld man page, the reality is that
//   // out of ld.bfd, ld.gold and ld.lld, only ld.lld actually implements
//   // that. Meanwhile, ld.bfd interprets -static -shared as just -static,
//   // and ld.gold rejects -static -shared outright as "incompatible".
//   linkCmdAndArgs.push_back("-shared");

//   // Strip debug information (only, no relocations) when not requested.
//   if (!targetOptions.target.debugSymbols) {
//     flags.push_back("--strip-debug");
//   }

//   // Link all input objects. Note that we are not linking whole-archive as
//   // we want to allow dropping of unused codegen outputs.
//   for (auto &objectFile : objectFiles) {
//     flags.push_back(objectFile.path);
//   }

//   auto commandLine = llvm::join(linkCmdAndArgs, " ");
//   if (failed(runLinkCommand(commandLine)))
//     return std::nullopt;
//   return artifacts;
// }

// LogicalResult serializeDynamicLibraryExecutable(
//     const SerializationOptions &options, const LLVMTarget &target,
//     IREE::HAL::ExecutableVariantOp variantOp, OpBuilder &executableBuilder,
//     const std::string &libraryName, const llvm::Triple &targetTriple,
//     const SmallVector<Artifact> &objectFiles, LinkerTool *linkerTool) {
//   // Link the generated object files into a dylib.
//   auto linkArtifactsOr =
//       linkerTool->linkDynamicLibrary(libraryName, objectFiles);
//   if (!linkArtifactsOr.has_value()) {
//     return mlir::emitError(variantOp.getLoc())
//            << "failed to link executable and generate target dylib (check "
//               "above for more specific error messages)";
//   }
//   auto &linkArtifacts = linkArtifactsOr.value();
//   if (defaultOptions_.keepLinkerArtifacts) {
//     mlir::emitRemark(variantOp.getLoc())
//         << "linker artifacts for " << variantOp.getName() << " preserved:\n"
//         << "    " << linkArtifacts.libraryFile.path;
//     linkArtifacts.keepAllFiles();
//     for (auto &objectFile : objectFiles) {
//       objectFile.keep();
//     }
//   }

//   if (target.getLinkEmbedded()) {
//     // Load the linked ELF file and pack into an attr.
//     auto elfFile = linkArtifacts.libraryFile.read();
//     if (!elfFile.has_value()) {
//       return variantOp.emitError() << "failed to read back dylib temp file at
//       "
//                                    << linkArtifacts.libraryFile.path;
//     }
//     if (!options.dumpBinariesPath.empty()) {
//       dumpDataToPath<int8_t>(options.dumpBinariesPath, options.dumpBaseName,
//                              variantOp.getName(), ".so", *elfFile);
//     }
//     auto bufferAttr = DenseIntElementsAttr::get(
//         VectorType::get({static_cast<int64_t>(elfFile->size())},
//                         IntegerType::get(executableBuilder.getContext(), 8)),
//         std::move(elfFile.value()));

//     // Add the binary to the parent hal.executable.
//     auto binaryOp = executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
//         variantOp.getLoc(), variantOp.getSymName(),
//         variantOp.getTarget().getFormat(), bufferAttr);
//     binaryOp.setMimeTypeAttr(
//         executableBuilder.getStringAttr("application/x-elf"));
//   } else {
//     const char *mimeType = nullptr;
//     const char *extension = "";
//     switch (targetTriple.getObjectFormat()) {
//     case llvm::Triple::ObjectFormatType::COFF:
//       mimeType = "application/x-msdownload";
//       extension = ".dll";
//       break;
//     case llvm::Triple::ObjectFormatType::ELF:
//       mimeType = "application/x-elf";
//       extension = ".so";
//       break;
//     case llvm::Triple::ObjectFormatType::MachO:
//       mimeType = "application/x-dylib";
//       extension = ".dylib";
//       break;
//     case llvm::Triple::ObjectFormatType::Wasm:
//       mimeType = "application/wasm";
//       extension = ".wasm";
//       break;
//     default:
//       mimeType = "application/octet-stream";
//       break;
//     }

//     // Load the linked system library and optionally tag on the debug
//     // database. This debug database sits at the tail of the file and is
//     // ignored by system loaders and tools but still accessible to the
//     runtime
//     // loader. Not all platforms have separate debug databases and need this.
//     auto libraryFileOr = linkArtifacts.libraryFile.read();
//     if (!libraryFileOr.has_value()) {
//       return variantOp.emitError() << "failed to read back dylib temp file at
//       "
//                                    << linkArtifacts.libraryFile.path;
//     }
//     auto libraryFile = std::move(libraryFileOr).value();
//     if (target.debugSymbols && linkArtifacts.debugFile.outputFile) {
//       if (failed(appendDebugDatabase(libraryFile, linkArtifacts.debugFile)))
//       {
//         return variantOp.emitError()
//                << "failed to append debug database to dylib file";
//       }
//     }
//     if (!options.dumpBinariesPath.empty()) {
//       dumpDataToPath<int8_t>(options.dumpBinariesPath, options.dumpBaseName,
//                              variantOp.getName(), extension, libraryFile);
//     }
//     auto bufferAttr = DenseIntElementsAttr::get(
//         VectorType::get({static_cast<int64_t>(libraryFile.size())},
//                         IntegerType::get(executableBuilder.getContext(), 8)),
//         std::move(libraryFile));

//     // Add the binary to the parent hal.executable.
//     auto binaryOp = executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
//         variantOp.getLoc(), variantOp.getSymName(),
//         variantOp.getTarget().getFormat(), bufferAttr);
//     binaryOp.setMimeTypeAttr(executableBuilder.getStringAttr(mimeType));
//   }

//   return llvm::success();
// }

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
  // Compile to object code
  if (llvm::failed(compileToObject(module, context,
                                   "/home/vwells/sources/tt-mlir/temp.o"))) {
    llvm::errs() << "Failed to compile to object code\n";
    return llvm::failure();
  }

  // Link the object code into a shared library using LLD
  // if (!linkWithLLD(*objectBuffer, outputPath)) {
  //   llvm::errs()
  //       << "Failed to link object code into shared library using LLD\n";
  //   return llvm::failure();
  // }

  // llvm::outs() << "Shared library created at: " << outputPath << "\n";
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
