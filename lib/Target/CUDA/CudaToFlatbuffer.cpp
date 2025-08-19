// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Target/CUDA/CudaToFlatbuffer.h"

#include "ttmlir/Target/CUDA/program_generated.h"
#include "ttmlir/Target/Utils/GPUKernelProgram.h"
#include "ttmlir/Transforms/Passes.h"

#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
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
#include <memory>
#include <vector>

namespace mlir::tt::cuda {

// Helper function to extract CUDA chip information from module attributes.
static std::string getCudaChipFromModule(Operation *moduleOp) {
  if (auto chipAttr = moduleOp->getAttrOfType<StringAttr>("cuda.chip")) {
    return chipAttr.getValue().str();
  }
  return "sm_80";
}

llvm::Expected<std::string> translateToPTX(Operation *op,
                                           const std::string &mcpu) {

  ModuleOp moduleOp = cast<ModuleOp>(op);

  PassManager pm(moduleOp.getContext());

  GpuToLLVMConversionPassOptions gputollvmOptions;
  gputollvmOptions.hostBarePtrCallConv = true;
  gputollvmOptions.kernelBarePtrCallConv = true;
  pm.addPass(createGpuToLLVMConversionPass(gputollvmOptions));

  // Resolves any remaining type conversion issues by reconciling unrealized
  // cast operations.
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  pm.addPass(transforms::createExtractGPUModules());

  if (failed(pm.run(moduleOp))) {
    moduleOp.erase();
    return llvm::createStringError(
        "Failed to run GPU to LLVM conversion passes");
  }

  // Translate MLIR to LLVM IR.
  llvm::LLVMContext llvmContext;

  auto llvmModule =
      translateModuleToLLVMIR(moduleOp, llvmContext, "gpu-module");
  if (!llvmModule) {
    moduleOp.erase();
    return llvm::createStringError("Failed to translate MLIR to LLVM IR");
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
    return llvm::createStringError("Failed to lookup target: " + error);
  }

  llvm::TargetOptions options;
  std::unique_ptr<llvm::TargetMachine> targetMachine(
      target->createTargetMachine(targetTriple, mcpu, "", options,
                                  llvm::Reloc::Static, llvm::CodeModel::Small,
                                  llvm::CodeGenOptLevel::Default));
  if (!targetMachine) {
    return llvm::createStringError(
        "Failed to create TargetMachine for triple: " + targetTriple.str());
  }

  llvmModule->setDataLayout(targetMachine->createDataLayout());

  llvm::SmallVector<char, 2048> ptxBuffer;
  llvm::raw_svector_ostream ptxStream(ptxBuffer);

  llvm::legacy::PassManager passManager;

  if (targetMachine->addPassesToEmitFile(passManager, ptxStream, nullptr,
                                         llvm::CodeGenFileType::AssemblyFile)) {
    return llvm::createStringError("Target machine cannot emit PTX assembly");
  }

  passManager.run(*llvmModule);

  return std::string(ptxBuffer.begin(), ptxBuffer.end());
}

static ::cuda::DataType mapMLIRTypeToCudaDataType(Type elementType) {
  if (elementType.isF64()) {
    return ::cuda::DataType::Float64;
  }
  if (elementType.isF32()) {
    return ::cuda::DataType::Float32;
  }
  if (elementType.isF16()) {
    return ::cuda::DataType::Float16;
  }
  if (elementType.isBF16()) {
    return ::cuda::DataType::BFloat16;
  }
  if (elementType.isInteger(64)) {
    if (elementType.isUnsignedInteger()) {
      return ::cuda::DataType::UInt64;
    }
    return ::cuda::DataType::Int64;
  }
  if (elementType.isInteger(32)) {
    if (elementType.isUnsignedInteger()) {
      return ::cuda::DataType::UInt32;
    }
    return ::cuda::DataType::Int32;
  }
  if (elementType.isInteger(16)) {
    if (elementType.isUnsignedInteger()) {
      return ::cuda::DataType::UInt16;
    }
    return ::cuda::DataType::Int16;
  }
  if (elementType.isIndex()) {
    return ::cuda::DataType::Int64;
  }
  return ::cuda::DataType::Float32;
}

static std::vector<uint8_t> serializeTypedAttrToBytes(TypedAttr attr) {
  std::vector<uint8_t> bytes;

  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(attr)) {
    APInt value = intAttr.getValue();
    unsigned bitWidth = value.getBitWidth();

    if (bitWidth <= 64) {
      uint64_t intValue = value.getZExtValue();
      size_t byteCount = (bitWidth + 7) / 8;

      for (size_t i = 0; i < byteCount; ++i) {
        bytes.push_back(static_cast<uint8_t>((intValue >> (i * 8)) & 0xFF));
      }
    }
  } else if (auto floatAttr = llvm::dyn_cast<FloatAttr>(attr)) {
    APFloat value = floatAttr.getValue();

    if (&value.getSemantics() == &APFloat::IEEEsingle()) {
      uint32_t bits = value.bitcastToAPInt().getZExtValue();
      for (size_t i = 0; i < 4; ++i) {
        bytes.push_back(static_cast<uint8_t>((bits >> (i * 8)) & 0xFF));
      }
    } else if (&value.getSemantics() == &APFloat::IEEEdouble()) {
      uint64_t bits = value.bitcastToAPInt().getZExtValue();
      for (size_t i = 0; i < 8; ++i) {
        bytes.push_back(static_cast<uint8_t>((bits >> (i * 8)) & 0xFF));
      }
    }
  } else if (auto denseAttr = llvm::dyn_cast<DenseElementsAttr>(attr)) {
    auto rawData = denseAttr.getRawData();
    bytes.assign(rawData.begin(), rawData.end());
  }
  return bytes;
}

static int64_t extractIntegerFromConstantOp(Value value) {
  Attribute valueAttr =
      cast<arith::ConstantOp>(value.getDefiningOp()).getValue();
  auto intAttr = cast<IntegerAttr>(valueAttr);
  return intAttr.getInt();
}

static llvm::Expected<std::string> getReturnVariableName(ModuleOp rootModule) {
  // The name of return variable is needed for runtime.
  // The expected structure is:
  //
  // func.func @function_name(args) -> (memref<...>) {
  //   %return_value = ...
  //   gpu.launch_func @kernel_name::@function_name blocks in (X, Y, Z)
  //     threads in (X, Y, Z)
  //     args(%return_value : memref<...>)
  //   ...
  //   return %return_value : memref<...>
  // }
  //
  // gpu.module @kernel_name {
  //   llvm.func @function_name() {
  //     ...
  //     }
  // }
  //
  // The return variable name is %return_value.
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
      return returnVariableName;
    }
  }
  return llvm::createStringError("No return variable found");
}

static std::pair<std::vector<uint64_t>, ::cuda::DataType>
extractShapeAndDataType(Type mlirType) {
  std::vector<uint64_t> shape;
  ::cuda::DataType dataType = ::cuda::DataType::Float32;

  if (auto memrefType = llvm::dyn_cast<MemRefType>(mlirType)) {
    auto shapeRef = memrefType.getShape();
    for (int64_t dim : shapeRef) {
      shape.push_back(static_cast<uint64_t>(dim));
    }
    Type elementType = memrefType.getElementType();
    dataType = mapMLIRTypeToCudaDataType(elementType);
  }

  return {shape, dataType};
}

static llvm::Expected<llvm::StringMap<cuda::Kernel>>
extractKernels(ModuleOp rootModule, std::string chip) {
  llvm::StringMap<cuda::Kernel> kernelMap;
  for (auto gpuModule : rootModule.getOps<mlir::gpu::GPUModuleOp>()) {
    for (auto func : gpuModule.template getOps<mlir::LLVM::LLVMFuncOp>()) {
      std::string kernelName =
          gpuModule.getName().str() + "_" + func.getName().str();

      MLIRContext *ctx = gpuModule.getContext();
      OpBuilder builder(ctx);
      auto tempModule = builder.create<ModuleOp>(gpuModule.getLoc());
      builder.setInsertionPointToStart(tempModule.getBody());
      builder.clone(*gpuModule.getOperation());
      auto ptxCode = translateToPTX(tempModule, chip);
      if (auto err = ptxCode.takeError()) {
        return llvm::createStringError("Failed to translate GPU module to PTX");
      }
      tempModule.erase();
      cuda::Kernel kernel;
      kernel.name = kernelName;
      kernel.ptx = ptxCode.get();
      kernelMap.insert({kernelName, kernel});
    }
  }
  return kernelMap;
}

static void processMemRefDescMap(
    const llvm::StringMap<MemRefDesc> &memRefDescMap,
    ::flatbuffers::FlatBufferBuilder &fbb, bool isConst,
    std::vector<::flatbuffers::Offset<::cuda::MemRefDesc>> &memRefDescs,
    std::vector<::flatbuffers::Offset<::cuda::Constant>> &constants) {

  for (const auto &pair : memRefDescMap) {
    const std::string name = pair.first().str();
    const cuda::MemRefDesc &memRefDesc = pair.second;
    auto nameOffset = fbb.CreateString(name);

    auto [shape, dataType] = extractShapeAndDataType(memRefDesc.type);

    auto shapeVector = fbb.CreateVector(shape);
    auto typeOffset = ::cuda::CreateType(fbb, shapeVector, dataType);

    if (isConst) {
      auto valueBytes = serializeTypedAttrToBytes(memRefDesc.value);
      auto valueVector = fbb.CreateVector(valueBytes);
      auto constantOffset =
          ::cuda::CreateConstant(fbb, nameOffset, typeOffset, valueVector);
      constants.push_back(constantOffset);
    } else {
      auto memrefOffset = ::cuda::CreateMemRefDesc(fbb, nameOffset, typeOffset);
      memRefDescs.push_back(memrefOffset);
    }
  }
}

std::shared_ptr<void> cudaToFlatbuffer(Operation *op) {
  ModuleOp rootModule = dyn_cast<ModuleOp>(op);
  assert(rootModule && "Expected ModuleOp as top level operation");

  ::flatbuffers::FlatBufferBuilder fbb;

  ::flatbuffers::Offset<cuda::Program> program;
  std::vector<::flatbuffers::Offset<::cuda::Kernel>> kernels;
  std::vector<::flatbuffers::Offset<::cuda::MemRefDesc>> memRefDescs;
  llvm::StringMap<MemRefDesc> memRefDescMap;
  std::vector<::flatbuffers::Offset<::cuda::Constant>> constants;
  llvm::StringMap<MemRefDesc> constantMemRefDescMap;
  llvm::StringMap<cuda::Kernel> kernelMap;

  auto getReturnVariableNameResult = getReturnVariableName(rootModule);
  if (auto err = getReturnVariableNameResult.takeError()) {
    llvm::errs() << "Failed to get return variable name: " << err << "\n";
    return nullptr;
  }
  std::string returnVariableName = getReturnVariableNameResult.get();

  std::string chip = getCudaChipFromModule(rootModule);
  auto kernelMapResult = extractKernels(rootModule, chip);
  if (auto err = kernelMapResult.takeError()) {
    llvm::errs() << "Failed to extract kernel names: " << err << "\n";
    return nullptr;
  }
  kernelMap = kernelMapResult.get();

  rootModule.walk([&](mlir::gpu::LaunchFuncOp launchFuncOp) {
    std::string kernelName = launchFuncOp.getKernelModuleName().str() + "_" +
                             launchFuncOp.getKernelName().str();
    assert(kernelMap.contains(kernelName) && "Kernel not found");
    auto kernel = kernelMap.lookup(kernelName);

    kernel.gridSizeX =
        extractIntegerFromConstantOp(launchFuncOp.getGridSizeX());
    kernel.gridSizeY =
        extractIntegerFromConstantOp(launchFuncOp.getGridSizeY());
    kernel.gridSizeZ =
        extractIntegerFromConstantOp(launchFuncOp.getGridSizeZ());
    kernel.blockSizeX =
        extractIntegerFromConstantOp(launchFuncOp.getBlockSizeX());
    kernel.blockSizeY =
        extractIntegerFromConstantOp(launchFuncOp.getBlockSizeY());
    kernel.blockSizeZ =
        extractIntegerFromConstantOp(launchFuncOp.getBlockSizeZ());

    std::vector<std::string> inputNames;
    auto kernelOperands = launchFuncOp.getKernelOperands();
    for (auto operand : kernelOperands) {
      std::string operandName = "unnamed";
      std::string operandStr;
      llvm::raw_string_ostream operandStream(operandStr);

      Operation *definingOp = operand.getDefiningOp();
      Operation *lastOp = launchFuncOp;

      // Host code does a lot of reinterpreting casts, so we need to find the
      // original operand. This will get constants, allocations and program
      // arguments.
      // Dimensions of operands are not needed as PTX code will iterate over
      // the stored data according to the shape.
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

      inputNames.push_back(operandName);

      TypedAttr constantValue = nullptr;
      auto constantOp =
          (definingOp) ? dyn_cast<arith::ConstantOp>(definingOp) : nullptr;
      if (constantOp) {
        constantValue = constantOp.getValue();
      }

      if (constantValue && !constantMemRefDescMap.count(operandName)) {
        constantMemRefDescMap.insert(
            {operandName,
             MemRefDesc{operandName, operandValue.getType(), constantValue}});
      }

      if (!constantValue && !memRefDescMap.count(operandName)) {
        memRefDescMap.insert(
            {operandName,
             MemRefDesc{operandName, operandValue.getType(), constantValue}});
      }
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
        fbb, nameOffset, ptxOffset, kernel.gridSizeX, kernel.gridSizeY,
        kernel.gridSizeZ, kernel.blockSizeX, kernel.blockSizeY,
        kernel.blockSizeZ, inputNamesOffset);
    kernels.push_back(kernelOffset);
  });

  processMemRefDescMap(memRefDescMap, fbb, false, memRefDescs, constants);
  processMemRefDescMap(constantMemRefDescMap, fbb, true, memRefDescs,
                       constants);

  auto kernelsVector = fbb.CreateVector(kernels);
  auto memrefsVector = fbb.CreateVector(memRefDescs);
  auto constantsVector = fbb.CreateVector(constants);
  auto returnVariableOffset = fbb.CreateString(returnVariableName);

  auto finalProgram = ::cuda::CreateProgram(
      fbb, kernelsVector, memrefsVector, constantsVector, returnVariableOffset);
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
