// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTKernelToEmitC/TTKernelToEmitC.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TT/IR/Utils.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.h"
#include "ttmlir/Target/TTKernel/TTKernelToCpp.h"
#include "ttmlir/Target/TTMetal/Target.h"
#include "ttmlir/Target/Utils/FlatbufferObjectCache.h"
#include "ttmlir/Target/Utils/MLIRToFlatbuffer.h"
#include "ttmlir/Version.h"

#include "flatbuffers/buffer.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstddef>
#include <memory>

using namespace ttmlir::utils;

namespace mlir::tt::ttmetal {

struct CQBuilder {
  ::flatbuffers::FlatBufferBuilder *fbb;
  const char *name;
  std::vector<::flatbuffers::Offset<::tt::target::metal::BufferRef>> inputs;
  std::vector<::flatbuffers::Offset<::tt::target::metal::BufferRef>> outputs;
  std::vector<::flatbuffers::Offset<::tt::target::metal::Command>> commands;
  OpPrintingFlags printFlags;

  CQBuilder(::flatbuffers::FlatBufferBuilder *fbb) : fbb(fbb) {
    printFlags = printFlags.elideLargeElementsAttrs()
                     .elideLargeResourceString()
                     .skipRegions()
                     .enableDebugInfo();
  }

  std::string getDebugString(mlir::Operation *op) {
    std::string str;
    llvm::raw_string_ostream os(str);
    op->print(os, printFlags);
    return str;
  };

  template <typename CommandT>
  ::flatbuffers::Offset<::tt::target::metal::Command>
  appendCommand(::flatbuffers::Offset<CommandT> commandT, mlir::Operation *op) {
    auto debugString = getDebugString(op);
    commands.push_back(::tt::target::metal::CreateCommandDirect(
        *fbb, ::tt::target::metal::CommandTypeTraits<CommandT>::enum_value,
        commandT.Union(), debugString.c_str()));
    return commands.back();
  }
};

static ::tt::target::MathFidelity toFlatbuffer(ttmetal::MathFidelity mathFidelity) {
  switch (mathFidelity) {
  case ttmetal::MathFidelity::HiFi4:
    return ::tt::target::MathFidelity::HiFi4;
  case ttmetal::MathFidelity::HiFi3:
    return ::tt::target::MathFidelity::HiFi3;
  case ttmetal::MathFidelity::HiFi2:
    return ::tt::target::MathFidelity::HiFi2;
  case ttmetal::MathFidelity::LoFi:
    return ::tt::target::MathFidelity::LoFi;
  }
  assert(false && "Unsupported MathFidelity");
}

static std::vector<::tt::target::metal::UnpackToDestMode>
toFlatbuffer(llvm::ArrayRef<ttmetal::UnpackToDestMode> unpackToDestModes) {
  std::vector<::tt::target::metal::UnpackToDestMode> result;
  result.reserve(unpackToDestModes.size());

  for (auto mode : unpackToDestModes) {
    switch (mode) {
    case ttmetal::UnpackToDestMode::UnpackToDestFp32:
      result.push_back(::tt::target::metal::UnpackToDestMode::UnpackToDestFp32);
      break;
    case ttmetal::UnpackToDestMode::Default:
      result.push_back(::tt::target::metal::UnpackToDestMode::Default);
      break;
    }
  }
  return result;
}

static ::tt::target::metal::EthType toFlatbuffer(ttmetal::EthType ethType) {
  switch (ethType) {
  case ttmetal::EthType::Sender:
    return ::tt::target::metal::EthType::Sender;
  case ttmetal::EthType::Receiver:
    return ::tt::target::metal::EthType::Receiver;
  }
  assert(false && "Unsupported EthType");
}

static ::tt::target::metal::NocIndex toFlatbuffer(ttmetal::NocIndex nocIndex) {
  switch (nocIndex) {
  case ttmetal::NocIndex::Noc0:
    return ::tt::target::metal::NocIndex::Noc0;
  case ttmetal::NocIndex::Noc1:
    return ::tt::target::metal::NocIndex::Noc1;
  }
  assert(false && "Unsupported NocIndex");
}

static ::tt::target::Dim2dRange toFlatbuffer(CoreRangeAttr coreRange) {
  auto offset = coreRange.getOffset();
  auto size = coreRange.getSize();
  return ::tt::target::Dim2dRange(::tt::target::Dim2d(offset[0], offset[1]),
                                  ::tt::target::Dim2d(size[0], size[1]));
}

static flatbuffers::Offset<::tt::target::metal::BufferDesc>
memrefTypeToFlatbuffer(FlatbufferObjectCache &cache, MemRefType memref,
                       DeviceAttr device) {
  std::vector<int32_t> gridShape;
  std::vector<int32_t> shardShape;
  std::vector<::tt::target::Dim2dRange> coreRangeSet;
  std::uint64_t size = 0;

  DataType dtype = DataType::Float32;
  ::tt::target::Dim2d tileShape(1, 1);
  Type elementType = memref.getElementType();
  int64_t elementSize = 0;
  if (isa<TileType>(elementType)) {
    auto tileType = mlir::cast<TileType>(elementType);
    dtype = tileType.getDataType();
    tileShape = ::tt::target::Dim2d(tileType.getHeight(), tileType.getWidth());
    elementSize = tileType.getSizeBytes();
  } else {
    dtype = elementTypeToDataType(elementType);
    elementSize = getElementSizeBytes(dtype);
  }

  if (auto layout = mlir::dyn_cast_if_present<DeviceLayoutInterface>(
          memref.getLayout())) {
    auto arrRefGridShape = layout.getGridShape(memref);
    gridShape = castVec<std::vector<int32_t>>(arrRefGridShape);
    shardShape = castVec<std::vector<int32_t>>(layout.getShardShape(memref));
    coreRangeSet = toFlatbuffer(cache, arrRefGridShape,
                                device.getWorkerGrid().getMapping());
    size = device.getMemrefSizeBytes(memref, 0);
  } else {
    auto memrefShape = memref.getShape();
    shardShape = castVec<std::vector<int32_t>>(memrefShape);
    size = volume(memrefShape, elementSize);
  }

  ::tt::target::MemorySpace memorySpace =
      memref.getMemorySpace()
          ? toFlatbuffer(
                cache,
                mlir::cast<MemorySpaceAttr>(memref.getMemorySpace()).getValue())
          : ::tt::target::MemorySpace::System;

  return ::tt::target::metal::CreateBufferDescDirect(
      *cache.fbb, &gridShape, &shardShape, &coreRangeSet, &tileShape,
      toFlatbuffer(cache, dtype), memorySpace,
      size);
}

static ::flatbuffers::Offset<::tt::target::metal::BufferRef>
bufferValueToFlatbuffer(FlatbufferObjectCache &cache, Value value,
                        uint64_t address, uint64_t size) {
  auto device = lookupDevice(value.getParentBlock()->getParentOp());
  assert(device);
  auto memrefType = mlir::cast<MemRefType>(value.getType());
  auto bufferDesc =
      cache.getOrCreate(memrefType, memrefTypeToFlatbuffer, device);
  return ::tt::target::metal::CreateBufferRef(*cache.fbb, cache.nextGlobalId(),
                                              address, size, bufferDesc);
}

static ::flatbuffers::Offset<::tt::target::metal::KernelArg>
toFlatbuffer(FlatbufferObjectCache &cache, KernelArgAttr kernelArg) {
  ::tt::target::metal::KernelArgType argType;
  ::flatbuffers::Offset<void> arg;
  switch (kernelArg.getType()) {
  case ttkernel::ArgType::CBPort: {
    argType = ::tt::target::metal::KernelArgType::KernelArgCBPort;
    arg = ::tt::target::metal::CreateKernelArgCBPort(
              *cache.fbb, kernelArg.getOperandIndex())
              .Union();
    break;
  }
  case ttkernel::ArgType::BufferAddress: {
    argType = ::tt::target::metal::KernelArgType::KernelArgBufferAddress;
    arg = ::tt::target::metal::CreateKernelArgBufferAddress(
              *cache.fbb, kernelArg.getOperandIndex())
              .Union();
    break;
  }
  case ttkernel::ArgType::Semaphore: {
    argType = ::tt::target::metal::KernelArgType::KernelArgSemaphore;
    arg = ::tt::target::metal::CreateKernelArgSemaphore(*cache.fbb).Union();
    break;
  }
  }

  return kernelArg.isCompileTime()
             ? ::tt::target::metal::CreateKernelArg(*cache.fbb, argType, arg,
                                                    kernelArg.getCtValue())
             : ::tt::target::metal::CreateKernelArg(*cache.fbb, argType, arg);
}

static ::flatbuffers::Offset<::tt::target::metal::KernelArgs>
kernelArgsToFlatbuffer(FlatbufferObjectCache &cache,
                       KernelArgsAttr kernelArgs) {
  auto rtArgs = toFlatbuffer(cache, kernelArgs.getRtArgs());
  auto ctArgs = toFlatbuffer(cache, kernelArgs.getCtArgs());
  return ::tt::target::metal::CreateKernelArgs(*cache.fbb, rtArgs, ctArgs);
}

static ::flatbuffers::Offset<::tt::target::metal::NocConfig>
nocConfigToFlatbuffer(FlatbufferObjectCache &cache,
                      NocConfigAttr nocConfigAttr) {
  return ::tt::target::metal::CreateNocConfig(
      *cache.fbb, toFlatbuffer(nocConfigAttr.getNocIndex()));
}

static ::flatbuffers::Offset<::tt::target::metal::TensixConfig>
tensixConfigToFlatbuffer(FlatbufferObjectCache &cache,
                         TensixConfigAttr tensixConfigAttr) {
  auto unpackToDestModeVec =
        toFlatbuffer(tensixConfigAttr.getUnpackToDestMode());
  return ::tt::target::metal::CreateTensixConfigDirect(
      *cache.fbb, toFlatbuffer(tensixConfigAttr.getMathFidelity()),
      tensixConfigAttr.getFp32DestAccEn(), tensixConfigAttr.getMathApproxMode(),
      &unpackToDestModeVec);
}

static ::flatbuffers::Offset<::tt::target::metal::EthernetConfig>
ethernetConfigToFlatbuffer(FlatbufferObjectCache &cache,
                           EthernetConfigAttr ethernetConfigAttr) {
  return ::tt::target::metal::CreateEthernetConfig(
      *cache.fbb, toFlatbuffer(ethernetConfigAttr.getEthType()),
      toFlatbuffer(ethernetConfigAttr.getNocIndex()));
}

static ::flatbuffers::Offset<::tt::target::metal::KernelConfig>
kernelConfigToFlatbuffer(FlatbufferObjectCache &cache,
                         KernelConfigInterface kernelConfig,
                         SymbolTable const &symbolTable) {
  StringRef kernelSymbol = kernelConfig.getKernelSymbol().getRootReference();
  auto kernelEntry = symbolTable.lookup<func::FuncOp>(kernelSymbol);
  assert(kernelEntry);
  std::string source;
  llvm::raw_string_ostream stream(source);
  LogicalResult result =
      ttkernel::translateKernelFuncToCpp(kernelEntry, stream);
  assert(result.succeeded());
  assert(source.size() > 0 && "empty kernel source");

  std::vector<::tt::target::Dim2dRange> coreRangeSet = {
      toFlatbuffer(mlir::cast<CoreRangeAttr>(kernelConfig.getCoreRange()))};

  ::flatbuffers::Offset<::tt::target::metal::KernelArgs> args =
      cache.getOrCreate(
          mlir::cast<KernelArgsAttr>(kernelConfig.getKernelArgs()),
          kernelArgsToFlatbuffer);

  ::tt::target::metal::KernelConfigType configType;
  ::flatbuffers::Offset<void> configUnion;
  switch (kernelConfig.getThreadType()) {
  case ttkernel::ThreadType::Noc: {
    configType = ::tt::target::metal::KernelConfigType::NocConfig;
    configUnion = cache
                      .getOrCreate(mlir::cast<NocConfigAttr>(kernelConfig),
                                   nocConfigToFlatbuffer)
                      .Union();
    break;
  }
  case ttkernel::ThreadType::Tensix: {
    configType = ::tt::target::metal::KernelConfigType::TensixConfig;
    configUnion = cache
                      .getOrCreate(mlir::cast<TensixConfigAttr>(kernelConfig),
                                   tensixConfigToFlatbuffer)
                      .Union();
    break;
  }
  case ttkernel::ThreadType::Ethernet: {
    configType = ::tt::target::metal::KernelConfigType::EthernetConfig;
    configUnion = cache
                      .getOrCreate(mlir::cast<EthernetConfigAttr>(kernelConfig),
                                   ethernetConfigToFlatbuffer)
                      .Union();
    break;
  }
  }

  return ::tt::target::metal::CreateKernelConfigDirect(
      *cache.fbb, ::tt::target::metal::Kernel::KernelSource,
      ::tt::target::metal::CreateKernelSourceDirect(*cache.fbb, source.c_str())
          .Union(),
      &coreRangeSet, args, configType, configUnion, kernelSymbol.data());
}

static std::shared_ptr<void> translateModuleToFlatbuffer(
    Operation *op, std::unordered_map<std::string, GoldenTensor> goldenMap) {
  ::flatbuffers::FlatBufferBuilder fbb;
  FlatbufferObjectCache cache(&fbb);

  ModuleOp module = dyn_cast<ModuleOp>(op);
  assert(module && "Expected ModuleOp as top level operation");
  SymbolTable symbolTable(module);

  auto systemDesc =
      mlir::cast<tt::SystemDescAttr>(module->getAttr(tt::SystemDescAttr::name));
  ::ttmlir::Version ttmlirVersion = ::ttmlir::getVersion();
  ::tt::target::Version binaryVersion(ttmlirVersion.major, ttmlirVersion.minor,
                                      ttmlirVersion.patch);
  std::vector<::flatbuffers::Offset<::tt::target::metal::Program>> programs;

  module->walk([&](func::FuncOp entry) {
    if (!entry.isPublic()) {
      // skip private functions
      return;
    }

    CQBuilder cqBuilder(&fbb);
    cqBuilder.name = entry.getSymName().data();

    cqBuilder.inputs.reserve(entry.getBody().getArguments().size());
    for (auto &input : entry.getBody().getArguments()) {
      cqBuilder.inputs.push_back(
          cache.getOrCreate(input, bufferValueToFlatbuffer, 0, 0));
    }

    cqBuilder.commands.reserve(entry.getBody().front().getOperations().size());
    entry->walk([&](mlir::Operation *op) {
      if (auto allocOp = dyn_cast_if_present<memref::AllocOp>(op); allocOp) {
        cqBuilder.appendCommand(
            ::tt::target::metal::CreateHostAllocCommand(
                fbb, cache.getOrCreate(allocOp.getResult(),
                                       bufferValueToFlatbuffer, 0, 0)),
            op);
      } else if (auto enqueueProgramOp =
                     dyn_cast_if_present<tt::ttmetal::EnqueueProgramOp>(op);
                 enqueueProgramOp) {
        std::vector<::flatbuffers::Offset<::tt::target::metal::BufferRef>>
            buffers;
        buffers.reserve(enqueueProgramOp.getBuffers().size());
        for (auto buffer : enqueueProgramOp.getBuffers()) {
          buffers.push_back(cache.at<::tt::target::metal::BufferRef>(buffer));
        }

        std::vector<::flatbuffers::Offset<::tt::target::metal::CBRef>> cbs;
        uint32_t port = 0;
        cbs.reserve(enqueueProgramOp.getCbs().size());
        for (auto cb : enqueueProgramOp.getCbs()) {
          auto buffer = cache.at<::tt::target::metal::BufferRef>(cb);
          cbs.push_back(
              ::tt::target::metal::CreateCBRef(*cache.fbb, port, buffer));
        }

        std::vector<::flatbuffers::Offset<::tt::target::metal::KernelConfig>>
            kernelConfigs;
        kernelConfigs.reserve(enqueueProgramOp.getKernelConfigs().size());
        for (Attribute kernelConfig : enqueueProgramOp.getKernelConfigs()) {
          kernelConfigs.push_back(kernelConfigToFlatbuffer(
              cache, mlir::cast<KernelConfigInterface>(kernelConfig),
              symbolTable));
        }
        cqBuilder.appendCommand(
            ::tt::target::metal::CreateEnqueueProgramCommandDirect(
                fbb, &buffers, &cbs,
                ::tt::target::metal::CreateProgramDescDirect(fbb,
                                                             &kernelConfigs)),
            op);
      } else if (auto createBufferOp =
                     dyn_cast_if_present<tt::ttmetal::CreateBufferOp>(op);
                 createBufferOp) {
        cqBuilder.appendCommand(
            ::tt::target::metal::CreateCreateBufferCommand(
                fbb, cache.getOrCreate(createBufferOp.getResult(),
                                       bufferValueToFlatbuffer,
                                       createBufferOp.getAddress(),
                                       createBufferOp.getSize())),
            op);
      } else if (auto deallocateBufferOp =
                     dyn_cast_if_present<tt::ttmetal::DeallocateBufferOp>(op);
                 deallocateBufferOp) {
        cqBuilder.appendCommand(
            ::tt::target::metal::CreateDeallocateBufferCommand(
                fbb, cache.at<::tt::target::metal::BufferRef>(
                         deallocateBufferOp.getInput())),
            op);
      } else if (auto enqueueReadBufferOp =
                     dyn_cast_if_present<tt::ttmetal::EnqueueReadBufferOp>(op);
                 enqueueReadBufferOp) {
        cqBuilder.appendCommand(
            ::tt::target::metal::CreateEnqueueReadBufferCommand(
                fbb,
                cache.at<::tt::target::metal::BufferRef>(
                    enqueueReadBufferOp.getInput()),
                cache.at<::tt::target::metal::BufferRef>(
                    enqueueReadBufferOp.getOutput())),
            op);
      } else if (auto enqueueWriteBufferOp =
                     dyn_cast_if_present<tt::ttmetal::EnqueueWriteBufferOp>(op);
                 enqueueWriteBufferOp) {
        cqBuilder.appendCommand(
            ::tt::target::metal::CreateEnqueueWriteBufferCommand(
                fbb,
                cache.at<::tt::target::metal::BufferRef>(
                    enqueueWriteBufferOp.getInput()),
                cache.at<::tt::target::metal::BufferRef>(
                    enqueueWriteBufferOp.getOutput())),
            op);
      } else if (auto returnOp = dyn_cast_if_present<func::ReturnOp>(op);
                 returnOp) {
        for (auto output : returnOp.getOperands()) {
          cqBuilder.outputs.push_back(
              cache.at<::tt::target::metal::BufferRef>(output));
        }
      }
    });

    std::vector<::flatbuffers::Offset<::tt::target::metal::CommandQueue>>
        commandQueues = {
            ::tt::target::metal::CreateCommandQueueDirect(fbb, cqBuilder.name,
                                                          &cqBuilder.commands),
        };

    std::vector<::flatbuffers::Offset<::tt::target::metal::DeviceProgram>>
        devicePrograms = {
            ::tt::target::metal::CreateDeviceProgramDirect(
                fbb, &cqBuilder.inputs, &cqBuilder.outputs, &commandQueues),
        };
    programs.push_back(::tt::target::metal::CreateProgramDirect(
        fbb, cqBuilder.name, &cqBuilder.inputs, &cqBuilder.outputs,
        &devicePrograms));
  });

  auto binary = ::tt::target::metal::CreateTTMetalBinaryDirect(
      fbb, &binaryVersion, ::ttmlir::getGitHash(),
      toFlatbuffer(cache, systemDesc), &programs);

  FinishSizePrefixedTTMetalBinaryBuffer(fbb, binary);
  ::flatbuffers::Verifier verifier(fbb.GetBufferPointer(), fbb.GetSize());
  ::tt::target::metal::VerifySizePrefixedTTMetalBinaryBuffer(verifier);

  uint8_t *buf = fbb.GetBufferPointer();
  auto size = fbb.GetSize();

  std::shared_ptr<void> serializedBinary =
      std::shared_ptr<void>(std::malloc(size), std::free);
  std::memcpy(serializedBinary.get(), buf, size);

  return serializedBinary;
}

LogicalResult translateTTMetalToFlatbuffer(
    Operation *op, llvm::raw_ostream &os,
    std::unordered_map<std::string, GoldenTensor> goldenMap) {
  std::shared_ptr<void> data = translateModuleToFlatbuffer(op, goldenMap);
  std::size_t size = ::flatbuffers::GetSizePrefixedBufferLength(
      static_cast<const uint8_t *>(data.get()));
  os.write(reinterpret_cast<char const *>(data.get()), size);
  return success();
}

} // namespace mlir::tt::ttmetal
