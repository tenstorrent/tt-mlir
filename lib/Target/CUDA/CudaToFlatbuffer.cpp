// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Target/CUDA/CudaToFlatbuffer.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/CUDA/program_generated.h"
#pragma clang diagnostic pop
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
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
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
  auto chipAttr = moduleOp->getAttrOfType<StringAttr>("cuda.chip");
  assert(chipAttr && "CUDA chip attribute not found");
  return chipAttr.getValue().str();
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
    return llvm::createStringError(
        "Failed to run GPU to LLVM conversion passes");
  }

  // Translate MLIR to LLVM IR.
  llvm::LLVMContext llvmContext;

  auto llvmModule =
      translateModuleToLLVMIR(moduleOp, llvmContext, "gpu-module");
  if (!llvmModule) {
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
    Type elementType = floatAttr.getType();

    if (elementType.isF32()) {
      uint32_t bits = value.bitcastToAPInt().getZExtValue();
      for (size_t i = 0; i < 4; ++i) {
        bytes.push_back(static_cast<uint8_t>((bits >> (i * 8)) & 0xFF));
      }
    } else if (elementType.isF64()) {
      uint64_t bits = value.bitcastToAPInt().getZExtValue();
      for (size_t i = 0; i < 8; ++i) {
        bytes.push_back(static_cast<uint8_t>((bits >> (i * 8)) & 0xFF));
      }
    } else if (elementType.isF16() || elementType.isBF16()) {
      uint16_t bits =
          static_cast<uint16_t>(value.bitcastToAPInt().getZExtValue());
      for (size_t i = 0; i < 2; ++i) {
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

// Helper function to walk backwards through operations to find the original
// value by traversing through reinterpret casts and other transformations.
static Value walkBackToOriginalValue(Value initialValue,
                                     Operation *fallbackOp) {
  Operation *definingOp = initialValue.getDefiningOp();
  Operation *lastOp = fallbackOp;

  while (definingOp && definingOp->getNumOperands() > 0) {
    lastOp = definingOp;
    definingOp = definingOp->getOperand(0).getDefiningOp();
  }
  Value finalValue = lastOp->getOpOperand(0).get();
  if (lastOp == fallbackOp) {
    finalValue = initialValue;
  }

  return finalValue;
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

      // Apply the same pattern as operand extraction to find the original
      // return variable by walking backwards through any reinterpret casts
      // or other transformations.
      Value finalReturnValue =
          walkBackToOriginalValue(returnValue, returnOpCast);

      std::string returnStr;
      llvm::raw_string_ostream returnStream(returnStr);
      AsmState asmState(rootModule);
      finalReturnValue.printAsOperand(returnStream, asmState);
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
    auto first = memRefDesc.first;
    auto last = memRefDesc.last;

    if (isConst) {
      auto valueBytes = serializeTypedAttrToBytes(memRefDesc.value);
      auto valueVector = fbb.CreateVector(valueBytes);
      auto constantOffset = ::cuda::CreateConstant(fbb, nameOffset, typeOffset,
                                                   valueVector, first, last);
      constants.push_back(constantOffset);
    } else {
      auto memrefOffset =
          ::cuda::CreateMemRefDesc(fbb, nameOffset, typeOffset, first, last);
      memRefDescs.push_back(memrefOffset);
    }
  }
}

flatbuffers::Offset<::cuda::CopyFunction>
processCopyOp(mlir::memref::CopyOp copyOp, flatbuffers::FlatBufferBuilder &fbb,
              mlir::ModuleOp rootModule,
              llvm::StringMap<MemRefDesc> &memRefDescMap, uint64_t opIndex) {
  auto targetType = cast<MemRefType>(copyOp.getTarget().getType());
  auto layout = dyn_cast<StridedLayoutAttr>(targetType.getLayout());

  int64_t offset = 0;

  if (layout) {
    offset = layout.getOffset();
  }
  Value source = walkBackToOriginalValue(copyOp.getSource(), copyOp);
  Value target = walkBackToOriginalValue(copyOp.getTarget(), copyOp);

  mlir::AsmState asmState(rootModule);

  // Get source name
  std::string sourceName;
  llvm::raw_string_ostream sourceStream(sourceName);
  source.printAsOperand(sourceStream, asmState);

  // Get target name
  std::string targetName;
  llvm::raw_string_ostream targetStream(targetName);
  target.printAsOperand(targetStream, asmState);

  if (!memRefDescMap.contains(sourceName)) {
    memRefDescMap.insert({sourceName, MemRefDesc{sourceName, source.getType(),
                                                 nullptr, opIndex, opIndex}});
  }
  if (memRefDescMap.contains(sourceName)) {
    auto memRefDesc = memRefDescMap.lookup(sourceName);
    memRefDesc.last = opIndex;
    memRefDescMap[sourceName] = memRefDesc;
  }

  if (!memRefDescMap.contains(targetName)) {
    memRefDescMap.insert({targetName, MemRefDesc{targetName, target.getType(),
                                                 nullptr, opIndex, opIndex}});
  }
  if (memRefDescMap.contains(targetName)) {
    auto memRefDesc = memRefDescMap.lookup(targetName);
    memRefDesc.last = opIndex;
    memRefDescMap[targetName] = memRefDesc;
  }

  auto sourceNameOffset = fbb.CreateString(sourceName);
  auto targetNameOffset = fbb.CreateString(targetName);

  auto copyFunctionOffset = ::cuda::CreateCopyFunction(
      fbb, sourceNameOffset, targetNameOffset, offset);

  return copyFunctionOffset;
}

flatbuffers::Offset<::cuda::Kernel>
processLaunchFuncOp(mlir::gpu::LaunchFuncOp launchFuncOp,
                    const llvm::StringMap<cuda::Kernel> &kernelMap,
                    flatbuffers::FlatBufferBuilder &fbb,
                    llvm::StringMap<MemRefDesc> &memRefDescMap,
                    llvm::StringMap<MemRefDesc> &constantMemRefDescMap,
                    uint64_t opIndex) {

  std::string kernelName = launchFuncOp.getKernelModuleName().str() + "_" +
                           launchFuncOp.getKernelName().str();
  assert(kernelMap.contains(kernelName) && "Kernel not found");
  auto kernel = kernelMap.lookup(kernelName);

  kernel.gridSizeX = extractIntegerFromConstantOp(launchFuncOp.getGridSizeX());
  kernel.gridSizeY = extractIntegerFromConstantOp(launchFuncOp.getGridSizeY());
  kernel.gridSizeZ = extractIntegerFromConstantOp(launchFuncOp.getGridSizeZ());
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

    // Host code does a lot of reinterpreting casts, so we need to find the
    // original operand. This will get constants, allocations and program
    // arguments.
    // Dimensions of operands are not needed as PTX code will iterate over
    // the stored data according to the shape.
    Value operandValue = walkBackToOriginalValue(operand, launchFuncOp);
    AsmState asmState(
        operandValue.getDefiningOp()
            ? operandValue.getDefiningOp()->getParentOfType<ModuleOp>()
            : launchFuncOp->getParentOfType<ModuleOp>());

    operandValue.printAsOperand(operandStream, asmState);
    operandName = operandStream.str();
    inputNames.push_back(operandName);

    TypedAttr constantValue = nullptr;
    Operation *finalDefiningOp = operandValue.getDefiningOp();
    auto constantOp = (finalDefiningOp)
                          ? dyn_cast<arith::ConstantOp>(finalDefiningOp)
                          : nullptr;
    if (constantOp) {
      constantValue = constantOp.getValue();
    }

    auto getGlobalOp = (finalDefiningOp)
                           ? dyn_cast<memref::GetGlobalOp>(finalDefiningOp)
                           : nullptr;
    if (getGlobalOp) {
      auto globalSymbolRef =
          mlir::cast<mlir::FlatSymbolRefAttr>(getGlobalOp->getAttr("name"));

      ModuleOp parentModule = getGlobalOp->getParentOfType<ModuleOp>();
      SymbolTable symbolTable(parentModule);
      auto globalOp = mlir::cast<memref::GlobalOp>(
          symbolTable.lookup(globalSymbolRef.getValue()));

      // Extract the constant value from the global's initial value.
      if (globalOp.getInitialValue().has_value()) {
        constantValue =
            mlir::cast<TypedAttr>(globalOp.getInitialValue().value());
      }
    }

    if (constantValue && !constantMemRefDescMap.count(operandName)) {
      constantMemRefDescMap.insert(
          {operandName, MemRefDesc{operandName, operandValue.getType(),
                                   constantValue, opIndex, opIndex}});
    }

    if (constantValue && constantMemRefDescMap.count(operandName)) {
      auto memRefDesc = constantMemRefDescMap.lookup(operandName);
      memRefDesc.last = opIndex;
      constantMemRefDescMap[operandName] = memRefDesc;
    }

    if (!constantValue && !memRefDescMap.count(operandName)) {
      memRefDescMap.insert(
          {operandName, MemRefDesc{operandName, operandValue.getType(),
                                   constantValue, opIndex, opIndex}});
    }

    if (!constantValue && memRefDescMap.count(operandName)) {
      auto memRefDesc = memRefDescMap.lookup(operandName);
      memRefDesc.last = opIndex;
      memRefDescMap[operandName] = memRefDesc;
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

  return ::cuda::CreateKernel(fbb, nameOffset, ptxOffset, kernel.gridSizeX,
                              kernel.gridSizeY, kernel.gridSizeZ,
                              kernel.blockSizeX, kernel.blockSizeY,
                              kernel.blockSizeZ, inputNamesOffset);
}
std::shared_ptr<void> cudaToFlatbuffer(Operation *op) {
  ModuleOp rootModule = dyn_cast<ModuleOp>(op);
  assert(rootModule && "Expected ModuleOp as top level operation");

  ::flatbuffers::FlatBufferBuilder fbb;

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
  auto kernelMap = kernelMapResult.get();
  llvm::StringMap<MemRefDesc> memRefDescMap;
  llvm::StringMap<MemRefDesc> constantMemRefDescMap;
  std::vector<::cuda::Action> actionTypes;
  std::vector<::flatbuffers::Offset<void>> actionObjects;

  uint64_t opIndex = 0;
  rootModule.walk([&](Operation *op) {
    // Process LaunchFuncOp
    if (auto launchFuncOp = dyn_cast<mlir::gpu::LaunchFuncOp>(op)) {
      auto kernelOffset =
          processLaunchFuncOp(launchFuncOp, kernelMap, fbb, memRefDescMap,
                              constantMemRefDescMap, opIndex);
      actionTypes.push_back(::cuda::Action::Kernel);
      actionObjects.push_back(kernelOffset.Union());
      opIndex++;
    } else if (auto copyOp = dyn_cast<mlir::memref::CopyOp>(op)) {
      auto copyFunction =
          processCopyOp(copyOp, fbb, rootModule, memRefDescMap, opIndex);
      actionTypes.push_back(::cuda::Action::CopyFunction);
      actionObjects.push_back(copyFunction.Union());
      opIndex++;
    }
  });

  std::vector<::flatbuffers::Offset<::cuda::MemRefDesc>> memRefDescs;
  std::vector<::flatbuffers::Offset<::cuda::Constant>> constants;

  processMemRefDescMap(memRefDescMap, fbb, false, memRefDescs, constants);
  processMemRefDescMap(constantMemRefDescMap, fbb, true, memRefDescs,
                       constants);

  auto actionTypesVector = fbb.CreateVector(actionTypes);
  auto actionObjectsVector = fbb.CreateVector(actionObjects);
  auto memrefsVector = fbb.CreateVector(memRefDescs);
  auto constantsVector = fbb.CreateVector(constants);
  auto returnVariableOffset = fbb.CreateString(returnVariableName);

  auto finalProgram = ::cuda::CreateProgram(
      fbb, actionTypesVector, actionObjectsVector, memrefsVector,
      constantsVector, returnVariableOffset);
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
  if (!data) {
    return failure();
  }

  std::size_t size = ::flatbuffers::GetSizePrefixedBufferLength(
      static_cast<const uint8_t *>(data.get()));
  os.write(reinterpret_cast<const char *>(data.get()), size);
  return success();
}

} // namespace mlir::tt::cuda
