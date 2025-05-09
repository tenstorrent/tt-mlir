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

namespace mlir::tt::ttmetal {

namespace target = ::tt::target;

struct CQBuilder {
  flatbuffers::FlatBufferBuilder *fbb;
  const char *name;
  std::vector<flatbuffers::Offset<target::metal::BufferRef>> inputs;
  std::vector<flatbuffers::Offset<target::metal::BufferRef>> outputs;
  std::vector<flatbuffers::Offset<target::metal::Command>> commands;
  OpPrintingFlags printFlags;

  CQBuilder(flatbuffers::FlatBufferBuilder *fbb) : fbb(fbb) {
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
  flatbuffers::Offset<target::metal::Command>
  appendCommand(flatbuffers::Offset<CommandT> commandT, mlir::Operation *op) {
    auto debugString = getDebugString(op);
    commands.push_back(target::metal::CreateCommandDirect(
        *fbb, target::metal::CommandTypeTraits<CommandT>::enum_value,
        commandT.Union(), debugString.c_str()));
    return commands.back();
  }
};

static target::MathFidelity toFlatbuffer(ttmetal::MathFidelity mathFidelity) {
  switch (mathFidelity) {
  case ttmetal::MathFidelity::HiFi4:
    return target::MathFidelity::HiFi4;
  case ttmetal::MathFidelity::HiFi3:
    return target::MathFidelity::HiFi3;
  case ttmetal::MathFidelity::HiFi2:
    return target::MathFidelity::HiFi2;
  case ttmetal::MathFidelity::LoFi:
    return target::MathFidelity::LoFi;
  }
  assert(false && "Unsupported MathFidelity");
}

static std::vector<target::metal::UnpackToDestMode>
toFlatbuffer(llvm::ArrayRef<ttmetal::UnpackToDestMode> unpackToDestModes) {
  std::vector<target::metal::UnpackToDestMode> result;
  result.reserve(unpackToDestModes.size());

  for (auto mode : unpackToDestModes) {
    switch (mode) {
    case ttmetal::UnpackToDestMode::Fp32:
      result.push_back(target::metal::UnpackToDestMode::Fp32);
      break;
    case ttmetal::UnpackToDestMode::Default:
      result.push_back(target::metal::UnpackToDestMode::Default);
      break;
    }
  }
  return result;
}

static target::metal::EthType toFlatbuffer(ttmetal::EthType ethType) {
  switch (ethType) {
  case ttmetal::EthType::Sender:
    return target::metal::EthType::Sender;
  case ttmetal::EthType::Receiver:
    return target::metal::EthType::Receiver;
  }
  assert(false && "Unsupported EthType");
}

static target::metal::NocIndex toFlatbuffer(ttmetal::NocIndex nocIndex) {
  switch (nocIndex) {
  case ttmetal::NocIndex::Noc0:
    return target::metal::NocIndex::Noc0;
  case ttmetal::NocIndex::Noc1:
    return target::metal::NocIndex::Noc1;
  }
  assert(false && "Unsupported NocIndex");
}

static target::Dim2dRange toFlatbuffer(CoreRangeAttr coreRange) {
  const auto offset = coreRange.getOffset();
  const auto size = coreRange.getSize();
  return target::Dim2dRange(target::Dim2d(offset[0], offset[1]),
                            target::Dim2d(size[0], size[1]));
}

static std::array<int32_t, 2> calculateCoreRangeSetShapeExtents(
    const std::vector<target::Dim2dRange> &coreRangeSet) {
  std::array<int32_t, 2> extents = {0, 0};
  for (const auto &range : coreRangeSet) {
    extents[0] = std::max(extents[0], range.loc().y() + range.size().y());
    extents[1] = std::max(extents[1], range.loc().x() + range.size().x());
  }
  return extents;
}

static flatbuffers::Offset<target::metal::ShardedBufferConfig>
memrefTypeToShardedBufferConfigFlatbuffer(FlatbufferObjectCache &cache,
                                          MemRefType memref, DeviceAttr device,
                                          target::Dim2d tileShape) {
  auto deviceLayout =
      mlir::dyn_cast_if_present<DeviceLayoutInterface>(memref.getLayout());
  if (!deviceLayout) {
    return 0;
  }

  auto shardLayout = mlir::cast<ShardLayoutAttr>(deviceLayout);
  uint64_t shardSize =
      device.getMemrefSizeBytes(memref, 0, /*includeBuffers=*/true);
  ArrayRef<int64_t> stride = shardLayout.getStride();
  int64_t elementSize = stride[stride.size() - 1];
  auto memrefGridShape = shardLayout.getGridShape(memref);
  auto memrefShardShape = shardLayout.getShardShape(memref);
  std::vector<target::Dim2dRange> coreRangeSet =
      toFlatbuffer(cache, memrefGridShape, device.getWorkerGrid().getMapping());
  std::array<int32_t, 2> gridShapeExtents =
      calculateCoreRangeSetShapeExtents(coreRangeSet);
  uint64_t size = gridShapeExtents[0] * gridShapeExtents[1] * shardSize;

  // Calculate ShardSpec
  assert(stride[stride.size() - 1] % elementSize == 0);
  int32_t shardXTiles = stride[stride.size() - 2] / elementSize;
  assert((memrefShardShape[0] * stride[0] / elementSize) % shardXTiles == 0);
  int32_t collapsedShardYTiles =
      (memrefShardShape[0] * stride[0] / elementSize) / shardXTiles;
  // Shard shape is the fully collapsed shard down to 2D, so:
  //   [d0 * ... * dN-2, dN-1]
  target::Dim2d shardShape(collapsedShardYTiles * tileShape.y(),
                           shardXTiles * tileShape.x());
  auto shardSpec = target::metal::CreateShardSpecDirect(
      *cache.fbb, &coreRangeSet, &shardShape);

  // Calculate ShardSpecBuffer
  target::Dim2d pageShape(tileShape.y(), shardShape.x());
  std::array<int32_t, 2> tensorShape = {gridShapeExtents[0] * shardShape.y(),
                                        gridShapeExtents[1] * shardShape.x()};
  assert(tensorShape[0] % pageShape.y() == 0);
  assert(tensorShape[1] % pageShape.x() == 0);
  target::Dim2d tensorShapeInPages(tensorShape[0] / pageShape.y(),
                                   tensorShape[1] / pageShape.x());
  auto shardSpecBuffer = target::metal::CreateShardSpecBuffer(
      *cache.fbb, shardSpec, &pageShape, &tensorShapeInPages);

  // Calculate ShardedBufferConfig
  assert(pageShape.y() % tileShape.y() == 0);
  assert(pageShape.x() % tileShape.x() == 0);
  std::array<int32_t, 2> pageShapeInTiles = {pageShape.y() / tileShape.y(),
                                             pageShape.x() / tileShape.x()};
  uint64_t pageSize = pageShapeInTiles[0] * pageShapeInTiles[1] * elementSize;
  return target::metal::CreateShardedBufferConfig(*cache.fbb, size, pageSize,
                                                  shardSpecBuffer);
}

static flatbuffers::Offset<target::metal::CircularBufferConfig>
memrefTypeToCircularBufferConfigFlatbuffer(FlatbufferObjectCache &cache,
                                           MemRefType memref, DeviceAttr device,
                                           target::Dim2d tileShape) {
  auto deviceLayout =
      mlir::dyn_cast_if_present<DeviceLayoutInterface>(memref.getLayout());
  if (!deviceLayout) {
    return 0;
  }

  auto shardLayout = mlir::cast<ShardLayoutAttr>(deviceLayout);
  auto memrefGridShape = shardLayout.getGridShape(memref);
  std::vector<target::Dim2dRange> coreRangeSet =
      toFlatbuffer(cache, memrefGridShape, device.getWorkerGrid().getMapping());

  uint64_t shardSize =
      device.getMemrefSizeBytes(memref, 0, /*includeBuffers=*/true);
  uint64_t pageSize = device.getMemrefCBPageSizeBytes(memref);
  uint64_t numBuffers = shardLayout.getBuffers();
  return target::metal::CreateCircularBufferConfigDirect(
      *cache.fbb, &coreRangeSet, /*total_size=*/shardSize,
      /*page_size=*/pageSize, numBuffers);
}

static flatbuffers::Offset<target::metal::BufferDesc>
memrefTypeToFlatbuffer(FlatbufferObjectCache &cache, MemRefType memref,
                       DeviceAttr device) {
  std::vector<int32_t> shape =
      ttmlir::utils::castContainer<std::vector<int32_t>>(memref.getShape());
  target::Dim2d tileShape(1, 1);
  DataType dtype = DataType::Float32;
  target::MemorySpace memorySpace =
      memref.getMemorySpace()
          ? toFlatbuffer(
                cache,
                mlir::cast<MemorySpaceAttr>(memref.getMemorySpace()).getValue())
          : target::MemorySpace::System;

  Type elementType = memref.getElementType();
  if (auto tileType = mlir::dyn_cast<TileType>(elementType)) {
    dtype = tileType.getDataType();
    tileShape = target::Dim2d(tileType.getHeight(), tileType.getWidth());
  } else {
    dtype = elementTypeToDataType(elementType);
  }

  flatbuffers::Offset<target::metal::ShardedBufferConfig> shardedBufferConfig =
      memrefTypeToShardedBufferConfigFlatbuffer(cache, memref, device,
                                                tileShape);

  flatbuffers::Offset<target::metal::CircularBufferConfig>
      circularBufferConfig = memrefTypeToCircularBufferConfigFlatbuffer(
          cache, memref, device, tileShape);

  return target::metal::CreateBufferDescDirect(
      *cache.fbb, &shape, &tileShape, toFlatbuffer(cache, dtype), memorySpace,
      shardedBufferConfig, circularBufferConfig);
}

static flatbuffers::Offset<target::metal::BufferRef>
bufferValueToFlatbuffer(FlatbufferObjectCache &cache, Value value,
                        uint64_t address) {
  auto device = lookupDevice(value.getParentBlock()->getParentOp());
  assert(device);
  auto memrefType = mlir::cast<MemRefType>(value.getType());
  auto bufferDesc =
      cache.getOrCreate(memrefType, memrefTypeToFlatbuffer, device);
  return target::metal::CreateBufferRef(*cache.fbb, cache.nextGlobalId(),
                                        address, bufferDesc);
}

static flatbuffers::Offset<target::metal::TensorRef>
tensorValueToFlatbuffer(FlatbufferObjectCache &cache, Value value) {
  auto device = lookupDevice(value.getParentBlock()->getParentOp());
  assert(device);
  auto memref = mlir::cast<MemRefType>(value.getType());

  Type elementType = memref.getElementType();
  assert(!mlir::isa<TileType>(elementType));
  DataType dtype = elementTypeToDataType(elementType);

  assert(!mlir::isa<DeviceLayoutInterface>(memref.getLayout()));
  std::vector<int32_t> shape =
      ttmlir::utils::castContainer<std::vector<int32_t>>(memref.getShape());
  std::vector<int32_t> meshShape;
  int32_t elementSize = getElementSizeBytes(dtype);
  std::uint64_t size =
      ttmlir::utils::volume(mlir::ArrayRef<int32_t>(shape), elementSize);

  auto memoryDesc =
      target::metal::CreateMemoryDesc(*cache.fbb, toFlatbuffer(cache, dtype));
  auto layoutDesc = target::metal::CreateLayoutDesc(*cache.fbb, memoryDesc);
  auto tensorDesc = target::metal::CreateTensorDescDirect(
      *cache.fbb, &shape, &meshShape, layoutDesc);
  return target::metal::CreateTensorRef(*cache.fbb, size, tensorDesc);
}

static flatbuffers::Offset<target::metal::KernelArg>
toFlatbuffer(FlatbufferObjectCache &cache, KernelArgAttr kernelArg) {
  target::metal::KernelArgType argType;
  flatbuffers::Offset<void> arg;
  switch (kernelArg.getType()) {
  case ttkernel::ArgType::CBPort: {
    argType = target::metal::KernelArgType::KernelArgCBPort;
    arg = target::metal::CreateKernelArgCBPort(*cache.fbb,
                                               kernelArg.getOperandIndex())
              .Union();
    break;
  }
  case ttkernel::ArgType::BufferAddress: {
    argType = target::metal::KernelArgType::KernelArgBufferAddress;
    arg = target::metal::CreateKernelArgBufferAddress(
              *cache.fbb, kernelArg.getOperandIndex())
              .Union();
    break;
  }
  case ttkernel::ArgType::Semaphore: {
    argType = target::metal::KernelArgType::KernelArgSemaphore;
    arg = target::metal::CreateKernelArgSemaphore(*cache.fbb).Union();
    break;
  }
  }

  return target::metal::CreateKernelArg(*cache.fbb, argType, arg);
}

static flatbuffers::Offset<target::metal::KernelArgs>
kernelArgsToFlatbuffer(FlatbufferObjectCache &cache,
                       KernelArgsAttr kernelArgs) {
  auto rtArgs = toFlatbuffer(cache, kernelArgs.getRtArgs());
  auto ctArgs = toFlatbuffer(cache, kernelArgs.getCtArgs());
  return target::metal::CreateKernelArgs(*cache.fbb, rtArgs, ctArgs);
}

static flatbuffers::Offset<target::metal::NocConfig>
nocConfigToFlatbuffer(FlatbufferObjectCache &cache,
                      NocConfigAttr nocConfigAttr) {
  return target::metal::CreateNocConfig(
      *cache.fbb, toFlatbuffer(nocConfigAttr.getNocIndex()));
}

static flatbuffers::Offset<target::metal::ComputeConfig>
computeConfigToFlatbuffer(FlatbufferObjectCache &cache,
                          ComputeConfigAttr computeConfigAttr) {
  auto unpackToDestModeVec =
      toFlatbuffer(computeConfigAttr.getUnpackToDestMode());
  return target::metal::CreateComputeConfigDirect(
      *cache.fbb, toFlatbuffer(computeConfigAttr.getMathFidelity()),
      computeConfigAttr.getFp32DestAccEn(),
      computeConfigAttr.getMathApproxMode(), &unpackToDestModeVec);
}

static flatbuffers::Offset<target::metal::EthernetConfig>
ethernetConfigToFlatbuffer(FlatbufferObjectCache &cache,
                           EthernetConfigAttr ethernetConfigAttr) {
  return target::metal::CreateEthernetConfig(
      *cache.fbb, toFlatbuffer(ethernetConfigAttr.getEthType()),
      toFlatbuffer(ethernetConfigAttr.getNocIndex()));
}

static flatbuffers::Offset<target::metal::KernelConfig>
kernelConfigToFlatbuffer(FlatbufferObjectCache &cache,
                         KernelConfigInterface kernelConfig,
                         const SymbolTable &symbolTable) {
  StringRef kernelSymbol = kernelConfig.getKernelSymbol().getRootReference();
  auto kernelEntry = symbolTable.lookup<func::FuncOp>(kernelSymbol);
  assert(kernelEntry);
  std::string source;
  llvm::raw_string_ostream stream(source);
  LogicalResult result =
      ttkernel::translateKernelFuncToCpp(kernelEntry, stream);
  assert(result.succeeded());
  assert(source.size() > 0 && "empty kernel source");

  std::vector<target::Dim2dRange> coreRangeSet = {
      toFlatbuffer(mlir::cast<CoreRangeAttr>(kernelConfig.getCoreRange()))};

  flatbuffers::Offset<target::metal::KernelArgs> args = cache.getOrCreate(
      mlir::cast<KernelArgsAttr>(kernelConfig.getKernelArgs()),
      kernelArgsToFlatbuffer);

  target::metal::KernelConfigType configType;
  flatbuffers::Offset<void> configUnion;
  switch (kernelConfig.getThreadType()) {
  case ttkernel::ThreadType::Noc: {
    configType = target::metal::KernelConfigType::NocConfig;
    configUnion = cache
                      .getOrCreate(mlir::cast<NocConfigAttr>(kernelConfig),
                                   nocConfigToFlatbuffer)
                      .Union();
    break;
  }
  case ttkernel::ThreadType::Compute: {
    configType = target::metal::KernelConfigType::ComputeConfig;
    configUnion = cache
                      .getOrCreate(mlir::cast<ComputeConfigAttr>(kernelConfig),
                                   computeConfigToFlatbuffer)
                      .Union();
    break;
  }
  case ttkernel::ThreadType::Ethernet: {
    configType = target::metal::KernelConfigType::EthernetConfig;
    configUnion = cache
                      .getOrCreate(mlir::cast<EthernetConfigAttr>(kernelConfig),
                                   ethernetConfigToFlatbuffer)
                      .Union();
    break;
  }
  }

  return target::metal::CreateKernelConfigDirect(
      *cache.fbb, target::metal::Kernel::KernelSource,
      target::metal::CreateKernelSourceDirect(*cache.fbb, source.c_str())
          .Union(),
      &coreRangeSet, args, configType, configUnion, kernelSymbol.data());
}

static std::shared_ptr<void> translateModuleToFlatbuffer(
    Operation *op,
    const std::unordered_map<std::string, GoldenTensor> &goldenMap,
    const std::vector<std::pair<std::string, std::string>> &moduleCache) {
  flatbuffers::FlatBufferBuilder fbb;
  FlatbufferObjectCache cache(&fbb);

  ModuleOp module = dyn_cast<ModuleOp>(op);
  assert(module && "Expected ModuleOp as top level operation");
  SymbolTable symbolTable(module);

  auto systemDesc =
      mlir::cast<tt::SystemDescAttr>(module->getAttr(tt::SystemDescAttr::name));
  ttmlir::Version ttmlirVersion = ttmlir::getVersion();
  target::Version binaryVersion(ttmlirVersion.major, ttmlirVersion.minor,
                                ttmlirVersion.patch);
  std::vector<flatbuffers::Offset<target::metal::Program>> programs;
  std::vector<flatbuffers::Offset<target::metal::TensorRef>> tensorInputs;
  std::vector<flatbuffers::Offset<target::metal::TensorRef>> tensorOutputs;

  module->walk([&](func::FuncOp entry) {
    if (!entry.isPublic()) {
      // Skip private functions.
      return;
    }

    CQBuilder cqBuilder(&fbb);
    cqBuilder.name = entry.getSymName().data();

    cqBuilder.inputs.reserve(entry.getBody().getArguments().size());
    for (auto &input : entry.getBody().getArguments()) {
      cqBuilder.inputs.push_back(
          cache.getOrCreate(input, bufferValueToFlatbuffer, 0));
      tensorInputs.push_back(tensorValueToFlatbuffer(cache, input));
    }

    cqBuilder.commands.reserve(entry.getBody().front().getOperations().size());
    entry->walk([&](mlir::Operation *op) {
      if (auto allocOp = dyn_cast_if_present<memref::AllocOp>(op); allocOp) {
        cqBuilder.appendCommand(
            target::metal::CreateHostAllocCommand(
                fbb, cache.getOrCreate(allocOp.getResult(),
                                       bufferValueToFlatbuffer, 0)),
            op);
      } else if (auto enqueueProgramOp =
                     dyn_cast_if_present<tt::ttmetal::EnqueueProgramOp>(op);
                 enqueueProgramOp) {
        std::vector<flatbuffers::Offset<target::metal::BufferRef>> buffers;
        buffers.reserve(enqueueProgramOp.getBuffers().size());
        for (auto buffer : enqueueProgramOp.getBuffers()) {
          buffers.push_back(cache.at<target::metal::BufferRef>(buffer));
        }

        std::vector<flatbuffers::Offset<target::metal::CBRef>> cbs;
        cbs.reserve(enqueueProgramOp.getCbs().size());
        for (auto [port, cb] : llvm::zip(enqueueProgramOp.getCbPorts(),
                                         enqueueProgramOp.getCbs())) {
          auto buffer = cache.at<target::metal::BufferRef>(cb);
          cbs.push_back(target::metal::CreateCBRef(*cache.fbb, port, buffer));
        }

        std::vector<flatbuffers::Offset<target::metal::KernelConfig>>
            kernelConfigs;
        kernelConfigs.reserve(enqueueProgramOp.getKernelConfigs().size());
        for (Attribute kernelConfig : enqueueProgramOp.getKernelConfigs()) {
          kernelConfigs.push_back(kernelConfigToFlatbuffer(
              cache, mlir::cast<KernelConfigInterface>(kernelConfig),
              symbolTable));
        }
        cqBuilder.appendCommand(
            target::metal::CreateEnqueueProgramCommandDirect(
                fbb, &buffers, &cbs,
                target::metal::CreateProgramDescDirect(fbb, &kernelConfigs)),
            op);
      } else if (auto createBufferOp =
                     dyn_cast_if_present<tt::ttmetal::CreateBufferOp>(op);
                 createBufferOp) {
        cqBuilder.appendCommand(
            target::metal::CreateCreateBufferCommand(
                fbb, cache.getOrCreate(createBufferOp.getResult(),
                                       bufferValueToFlatbuffer,
                                       createBufferOp.getAddress())),
            op);
      } else if (auto deallocateBufferOp =
                     dyn_cast_if_present<tt::ttmetal::DeallocateBufferOp>(op);
                 deallocateBufferOp) {
        cqBuilder.appendCommand(target::metal::CreateDeallocateBufferCommand(
                                    fbb, cache.at<target::metal::BufferRef>(
                                             deallocateBufferOp.getInput())),
                                op);
      } else if (auto enqueueReadBufferOp =
                     dyn_cast_if_present<tt::ttmetal::EnqueueReadBufferOp>(op);
                 enqueueReadBufferOp) {
        cqBuilder.appendCommand(target::metal::CreateEnqueueReadBufferCommand(
                                    fbb,
                                    cache.at<target::metal::BufferRef>(
                                        enqueueReadBufferOp.getInput()),
                                    cache.at<target::metal::BufferRef>(
                                        enqueueReadBufferOp.getOutput())),
                                op);
      } else if (auto enqueueWriteBufferOp =
                     dyn_cast_if_present<tt::ttmetal::EnqueueWriteBufferOp>(op);
                 enqueueWriteBufferOp) {
        cqBuilder.appendCommand(target::metal::CreateEnqueueWriteBufferCommand(
                                    fbb,
                                    cache.at<target::metal::BufferRef>(
                                        enqueueWriteBufferOp.getInput()),
                                    cache.at<target::metal::BufferRef>(
                                        enqueueWriteBufferOp.getOutput())),
                                op);
      } else if (auto returnOp = dyn_cast_if_present<func::ReturnOp>(op);
                 returnOp) {
        assert(cqBuilder.outputs.empty() &&
               "Unexpected multiple func::ReturnOp's");
        for (auto output : returnOp.getOperands()) {
          cqBuilder.outputs.push_back(
              cache.at<target::metal::BufferRef>(output));
          tensorOutputs.push_back(tensorValueToFlatbuffer(cache, output));
        }

        cqBuilder.appendCommand(
            target::metal::CreateReturnCommandDirect(fbb, &cqBuilder.outputs),
            op);
      }
    });

    constexpr uint32_t cqId = 0;
    std::vector<flatbuffers::Offset<target::metal::CommandQueue>>
        commandQueues = {
            target::metal::CreateCommandQueueDirect(fbb, cqBuilder.name, cqId,
                                                    &cqBuilder.commands),
        };

    std::vector<flatbuffers::Offset<target::metal::DeviceProgram>>
        devicePrograms = {
            target::metal::CreateDeviceProgramDirect(
                fbb, &cqBuilder.inputs, &cqBuilder.outputs, &commandQueues),
        };

    flatbuffers::Offset<target::DebugInfo> debugInfo =
        debugInfoToFlatbuffer(fbb, "ttmetal", module, goldenMap, moduleCache);

    programs.push_back(target::metal::CreateProgramDirect(
        fbb, cqBuilder.name, &tensorInputs, &tensorOutputs, &devicePrograms,
        debugInfo));
  });

  auto binary = target::metal::CreateTTMetalBinaryDirect(
      fbb, &binaryVersion, ttmlir::getGitHash(),
      toFlatbuffer(cache, systemDesc), &programs);

  FinishSizePrefixedTTMetalBinaryBuffer(fbb, binary);
  flatbuffers::Verifier verifier(fbb.GetBufferPointer(), fbb.GetSize());
  target::metal::VerifySizePrefixedTTMetalBinaryBuffer(verifier);

  uint8_t *buf = fbb.GetBufferPointer();
  auto size = fbb.GetSize();

  std::shared_ptr<void> serializedBinary =
      std::shared_ptr<void>(std::malloc(size), std::free);
  std::memcpy(serializedBinary.get(), buf, size);

  return serializedBinary;
}

LogicalResult translateTTMetalToFlatbuffer(
    Operation *op, llvm::raw_ostream &os,
    const std::unordered_map<std::string, GoldenTensor> &goldenMap,
    const std::vector<std::pair<std::string, std::string>> &moduleCache) {
  std::shared_ptr<void> data =
      translateModuleToFlatbuffer(op, goldenMap, moduleCache);
  std::size_t size = flatbuffers::GetSizePrefixedBufferLength(
      static_cast<const uint8_t *>(data.get()));
  os.write(reinterpret_cast<const char *>(data.get()), size);
  return success();
}

} // namespace mlir::tt::ttmetal
