// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Target/CUDA/CudaToFlatbuffer.h"
#include "ttmlir/Transforms/Passes.h"

#include "ttmlir/Target/CUDA/program_generated.h"

#include "ttmlir/Target/Utils/FlatbufferObjectCache.h"
#include "ttmlir/Target/Utils/GPUKernelProgram.h"
#include "ttmlir/Version.h"

#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

static llvm::cl::opt<std::string>
    cudaMcpu("cuda-mcpu",
             llvm::cl::desc("CUDA compute capability (default: sm_80)"),
             llvm::cl::init("sm_80"));

namespace mlir::tt::cuda {

static std::string getCudaMcpu() {
  static llvm::cl::opt<std::string> cudaMcpu(
      "cuda-mcpu", llvm::cl::desc("CUDA compute capability (default: sm_80)"),
      llvm::cl::init("sm_80"));
  return cudaMcpu;
}

std::string translateToPTX(Operation *op, const std::string &mcpu) {

  ModuleOp module = cast<ModuleOp>(op);

  PassManager pm(module.getContext());

  GpuToLLVMConversionPassOptions gputollvmOptions;
  gputollvmOptions.hostBarePtrCallConv = true;
  gputollvmOptions.kernelBarePtrCallConv = true;
  pm.addPass(createGpuToLLVMConversionPass(gputollvmOptions));

  // Resolves any remaining type conversion issues by reconciling unrealized
  // cast operations.
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  pm.addPass(transforms::createExtractGPUModules());

  if (failed(pm.run(module))) {
    llvm::errs() << "Failed to run GPU to LLVM conversion passes\n";
    module.erase();
    return "";
  }

  // Translate MLIR to LLVM IR.
  llvm::LLVMContext llvmContext;

  auto llvmModule = translateModuleToLLVMIR(module, llvmContext, "gpu-module");
  if (!llvmModule) {
    llvm::errs() << "Failed to translate MLIR to LLVM IR\n";
    module.erase();
    return "";
  }

  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXAsmPrinter();

  llvm::Triple targetTriple("nvptx64-nvidia-cuda");
  llvmModule->setTargetTriple(targetTriple);

  std::string error;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(targetTriple, error);
  if (!target) {
    llvm::errs() << "Failed to lookup target: " << error << "\n";
    return "";
  }

  llvm::TargetOptions options;
  std::unique_ptr<llvm::TargetMachine> targetMachine(
      target->createTargetMachine(targetTriple, mcpu, "", options,
                                  llvm::Reloc::Static, llvm::CodeModel::Small,
                                  llvm::CodeGenOptLevel::Default));
  if (!targetMachine) {
    llvm::errs() << "Failed to create TargetMachine for triple: "
                 << targetTriple.str() << "\n";
    return "";
  }

  llvmModule->setDataLayout(targetMachine->createDataLayout());

  llvm::SmallVector<char, 2048> ptxBuffer;
  llvm::raw_svector_ostream ptxStream(ptxBuffer);

  llvm::legacy::PassManager passManager;

  if (targetMachine->addPassesToEmitFile(passManager, ptxStream, nullptr,
                                         llvm::CodeGenFileType::AssemblyFile)) {
    llvm::errs() << "Target machine cannot emit PTX assembly\n";
    return "";
  }

  passManager.run(*llvmModule);

  return std::string(ptxBuffer.begin(), ptxBuffer.end());
}

std::shared_ptr<void> cudaToFlatbuffer(Operation *op) {
  ModuleOp rootModule = dyn_cast<ModuleOp>(op);
  assert(rootModule && "Expected ModuleOp as top level operation");

  ::flatbuffers::FlatBufferBuilder fbb;
  FlatbufferObjectCache cache(&fbb);
  ::ttmlir::Version ttmlirVersion = ::ttmlir::getVersion();
  ::tt::target::Version binaryVersion(ttmlirVersion.major, ttmlirVersion.minor,
                                      ttmlirVersion.patch);

  ::flatbuffers::Offset<cuda::Program> program;
  std::vector<::flatbuffers::Offset<::cuda::Kernel>> kernels;
  std::vector<::flatbuffers::Offset<::cuda::MemRefDesc>> memRefDescs;
  llvm::StringMap<MemRefDesc> memRefDescMap;
  llvm::StringMap<cuda::Kernel> kernelMap;

  std::string returnVariableName = "";
  for (auto funcOp : rootModule.getOps<mlir::func::FuncOp>()) {
    auto *returnOp = funcOp.getBody().front().getTerminator();
    auto returnOpCast = dyn_cast<mlir::func::ReturnOp>(returnOp);
    if (returnOpCast && returnOpCast.getNumOperands() > 0) {
      Value returnValue = returnOpCast.getOperand(0);
      std::string returnStr;
      llvm::raw_string_ostream returnStream(returnStr);
      AsmState asmState(rootModule);
      returnValue.printAsOperand(returnStream, asmState);
      returnVariableName = returnStream.str();
      break;
    }
  }

  for (auto gpuModule : rootModule.getOps<mlir::gpu::GPUModuleOp>()) {
    for (auto func : gpuModule.template getOps<mlir::LLVM::LLVMFuncOp>()) {
      std::string kernelName =
          gpuModule.getName().str() + "_" + func.getName().str();

      MLIRContext *ctx = gpuModule.getContext();
      OpBuilder builder(ctx);
      auto tempModule = builder.create<ModuleOp>(gpuModule.getLoc());
      builder.setInsertionPointToStart(tempModule.getBody());
      builder.clone(*gpuModule.getOperation());
      std::string ptxCode = translateToPTX(tempModule, getCudaMcpu());
      tempModule.erase();
      cuda::Kernel kernel;
      kernel.name = kernelName;
      kernel.ptx = ptxCode;
      kernelMap.insert({kernelName, kernel});
    }
  }

  rootModule.walk([&](mlir::gpu::LaunchFuncOp launchFuncOp) {
    std::string kernelName = launchFuncOp.getKernelModuleName().str() + "_" +
                             launchFuncOp.getKernelName().str();
    assert(kernelMap.count(kernelName) > 0 && "Kernel not found");
    auto kernel = kernelMap.lookup(kernelName);

    Attribute valueAttr =
        dyn_cast<arith::ConstantOp>(launchFuncOp.getGridSizeX().getDefiningOp())
            .getValue();
    auto intAttr = dyn_cast<IntegerAttr>(valueAttr);
    assert(intAttr);
    kernel.gridSizeX = intAttr.getInt();

    valueAttr =
        dyn_cast<arith::ConstantOp>(launchFuncOp.getGridSizeY().getDefiningOp())
            .getValue();
    intAttr = dyn_cast<IntegerAttr>(valueAttr);
    assert(intAttr);
    kernel.gridSizeY = intAttr.getInt();

    valueAttr =
        dyn_cast<arith::ConstantOp>(launchFuncOp.getGridSizeZ().getDefiningOp())
            .getValue();
    intAttr = dyn_cast<IntegerAttr>(valueAttr);
    assert(intAttr);
    kernel.gridSizeZ = intAttr.getInt();

    valueAttr = dyn_cast<arith::ConstantOp>(
                    launchFuncOp.getBlockSizeX().getDefiningOp())
                    .getValue();
    intAttr = dyn_cast<IntegerAttr>(valueAttr);
    assert(intAttr);
    kernel.blockSizeX = intAttr.getInt();

    valueAttr = dyn_cast<arith::ConstantOp>(
                    launchFuncOp.getBlockSizeY().getDefiningOp())
                    .getValue();
    intAttr = dyn_cast<IntegerAttr>(valueAttr);
    assert(intAttr);
    kernel.blockSizeY = intAttr.getInt();

    valueAttr = dyn_cast<arith::ConstantOp>(
                    launchFuncOp.getBlockSizeZ().getDefiningOp())
                    .getValue();
    intAttr = dyn_cast<IntegerAttr>(valueAttr);
    assert(intAttr);
    kernel.blockSizeZ = intAttr.getInt();

    std::vector<std::string> inputNames;
    auto kernelOperands = launchFuncOp.getKernelOperands();
    for (auto [idx, operand] : llvm::enumerate(kernelOperands)) {
      std::string operandName = "unnamed";
      std::string operandStr;
      llvm::raw_string_ostream operandStream(operandStr);

      Operation *definingOp = operand.getDefiningOp();
      Operation *lastOp = launchFuncOp;

      // Host code does a lot of reinterpreting casts, so we need to find the
      // original operand. This will get constants, allocations and program
      // arguments.
      while (definingOp && definingOp->getNumOperands() > 0) {
        lastOp = definingOp;
        definingOp = definingOp->getOperand(0).getDefiningOp();
      }
      AsmState asmState(lastOp->getParentOfType<ModuleOp>());

      Value operandValue = lastOp->getOpOperand(0).get();
      if (lastOp == launchFuncOp) {
        operandValue = operand;
      }

      operandValue.printAsOperand(operandStream, asmState);

      operandName = operandStream.str();

      TypedAttr constantValue = nullptr;
      auto constantOp =
          (definingOp) ? dyn_cast<arith::ConstantOp>(definingOp) : nullptr;
      if (constantOp) {
        constantValue = constantOp.getValue();
      }

      if (!memRefDescMap.count(operandName)) {
        memRefDescMap.insert(
            {operandName,
             MemRefDesc{operandName, operandValue.getType(), constantValue}});
      }

      inputNames.push_back(operandName);
    }
    kernel.inputNames = inputNames;
    auto nameOffset = fbb.CreateString(kernel.name);
    auto ptxOffset = fbb.CreateString(kernel.ptx);
    std::vector<::flatbuffers::Offset<::flatbuffers::String>> inputNameOffsets;
    for (const std::string &inputName : kernel.inputNames) {
      inputNameOffsets.push_back(fbb.CreateString(inputName));
    }
    auto inputNamesOffset = fbb.CreateVector(inputNameOffsets);

    auto kernelOffset = ::cuda::CreateKernel(
        fbb, nameOffset, ptxOffset, static_cast<uint64_t>(kernel.gridSizeX),
        static_cast<uint64_t>(kernel.gridSizeY),
        static_cast<uint64_t>(kernel.gridSizeZ),
        static_cast<uint64_t>(kernel.blockSizeX),
        static_cast<uint64_t>(kernel.blockSizeY),
        static_cast<uint64_t>(kernel.blockSizeZ), inputNamesOffset);
    kernels.push_back(kernelOffset);
  });

  for (const auto &pair : memRefDescMap) {

    const std::string memRefName = pair.first().str();
    const cuda::MemRefDesc &memRefDesc = pair.second;
    auto nameOffset = fbb.CreateString(memRefName);
    std::string typeStr;
    llvm::raw_string_ostream typeStream(typeStr);
    typeStream << memRefDesc.type;
    if (typeStr.find("memref") != std::string::npos) {
      typeStr = typeStr.substr(typeStr.find("memref") + 7);
      typeStr = typeStr.substr(0, typeStr.find(">"));
    }
    if (typeStr.find("index") != std::string::npos) {
      typeStr.replace(typeStr.find("index"), 5, "i64");
    }
    auto typeOffset = fbb.CreateString(typeStr);

    std::string valueStr;
    llvm::raw_string_ostream valueStream(valueStr);
    memRefDesc.value.print(valueStream);
    if (valueStr.find("NULL") != std::string::npos) {
      valueStr = "";
    }
    if (valueStr.find(":") != std::string::npos) {
      valueStr = valueStr.substr(0, valueStr.find(":") - 1);
    }
    auto valueOffset = fbb.CreateString(valueStr);
    auto memrefOffset =
        ::cuda::CreateMemRefDesc(fbb, nameOffset, typeOffset, valueOffset);
    memRefDescs.push_back(memrefOffset);
  }

  auto kernelsVector = fbb.CreateVector(kernels);
  auto memrefsVector = fbb.CreateVector(memRefDescs);
  auto returnVariableOffset = fbb.CreateString(returnVariableName);

  auto finalProgram = ::cuda::CreateProgram(fbb, kernelsVector, memrefsVector,
                                            returnVariableOffset);
  fbb.FinishSizePrefixed(finalProgram);

  uint8_t *buf = fbb.GetBufferPointer();
  std::size_t size = fbb.GetSize();

  std::shared_ptr<void> bufferPtr =
      std::shared_ptr<void>(std::malloc(size), std::free);
  std::memcpy(bufferPtr.get(), buf, size);
  return bufferPtr;
}

LogicalResult translateCudaToFlatbuffer(Operation *op, llvm::raw_ostream &os) {
  std::shared_ptr<void> data = cudaToFlatbuffer(op);
  std::size_t size = ::flatbuffers::GetSizePrefixedBufferLength(
      static_cast<const uint8_t *>(data.get()));
  os.write(reinterpret_cast<const char *>(data.get()), size);
  return success();
}

} // namespace mlir::tt::cuda
