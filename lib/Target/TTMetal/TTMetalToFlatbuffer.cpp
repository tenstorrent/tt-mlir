// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTKernelToEmitC/TTKernelToEmitC.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TT/IR/Utils.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.h"
#include "ttmlir/Target/TTMetal/Target.h"
#include "ttmlir/Target/Utils/FlatbufferObjectCache.h"
#include "ttmlir/Target/Utils/MLIRToFlatbuffer.h"
#include "ttmlir/Version.h"

#include "flatbuffers/buffer.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

namespace mlir::tt {
flatbuffers::Offset<::tt::target::metal::MemoryDesc>
memrefAttrToFlatbuffer(FlatbufferObjectCache &cache, MemRefType memref) {
  auto shapeInt64 = memref.getShape();
  std::vector<int32_t> shape(shapeInt64.begin(), shapeInt64.end());
  DataType dtype = DataType::Float32;
  ::tt::target::Dim2d tileShape(1, 1);
  Type elementType = memref.getElementType();
  std::uint64_t elementSize = 0;
  if (isa<TileType>(elementType)) {
    auto tileType = mlir::cast<TileType>(elementType);
    dtype = tileType.getDataType();
    tileShape = ::tt::target::Dim2d(tileType.getHeight(), tileType.getWidth());
    elementSize = tileType.getSizeBytes();
  } else {
    dtype = elementTypeToDataType(elementType);
    elementSize = getElementSizeBytes(dtype);
  }

  std::uint64_t size = elementSize;
  for (auto dim : shapeInt64) {
    size *= dim;
  }

  return ::tt::target::metal::CreateMemoryDescDirect(
      *cache.fbb, &shape, &tileShape, toFlatbuffer(cache, dtype),
      toFlatbuffer(
          cache,
          mlir::cast<MemorySpaceAttr>(memref.getMemorySpace()).getValue()),
      size);
}

flatbuffers::Offset<::tt::target::metal::LayoutDesc>
metalLayoutAttrToFlatbuffer(FlatbufferObjectCache &cache,
                            MetalLayoutAttr metalLayoutAttr,
                            ArrayRef<int64_t> logicalShape,
                            DeviceAttr deviceAttr) {
  auto coreRangeSet = toFlatbuffer(cache, metalLayoutAttr.getGrid(),
                                   deviceAttr.getWorkerGrid());
  return ::tt::target::metal::CreateLayoutDescDirect(
      *cache.fbb, toFlatbuffer(cache, metalLayoutAttr.getOobVal()),
      &coreRangeSet,
      cache.getOrCreate(metalLayoutAttr.getMemref(), memrefAttrToFlatbuffer));
}

} // namespace mlir::tt

namespace mlir::tt::ttmetal {

struct CQBuilder {
  ::flatbuffers::FlatBufferBuilder *fbb;
  const char *name;
  std::vector<::flatbuffers::Offset<::tt::target::metal::TensorRef>> inputs;
  std::vector<::flatbuffers::Offset<::tt::target::metal::TensorRef>> outputs;
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

::tt::target::MathFidelity toFlatbuffer(ttkernel::MathFidelity mathFidelity) {
  switch (mathFidelity) {
  case ttkernel::MathFidelity::HiFi4:
    return ::tt::target::MathFidelity::HiFi4;
  case ttkernel::MathFidelity::HiFi3:
    return ::tt::target::MathFidelity::HiFi3;
  case ttkernel::MathFidelity::HiFi2:
    return ::tt::target::MathFidelity::HiFi2;
  case ttkernel::MathFidelity::LoFi:
    return ::tt::target::MathFidelity::LoFi;
  }
  assert(false && "Unsupported MathFidelity");
}

std::vector<::tt::target::metal::UnpackToDestMode>
toFlatbuffer(llvm::ArrayRef<ttkernel::UnpackToDestMode> unpackToDestModes) {
  std::vector<::tt::target::metal::UnpackToDestMode> result;
  result.reserve(unpackToDestModes.size());

  for (auto mode : unpackToDestModes) {
    switch (mode) {
    case ttkernel::UnpackToDestMode::UnpackToDestFp32:
      result.push_back(::tt::target::metal::UnpackToDestMode::UnpackToDestFp32);
      break;
    case ttkernel::UnpackToDestMode::Default:
      result.push_back(::tt::target::metal::UnpackToDestMode::Default);
      break;
    }
  }
  return result;
}

::tt::target::metal::EthType toFlatbuffer(ttkernel::EthType ethType) {
  switch (ethType) {
  case ttkernel::EthType::Sender:
    return ::tt::target::metal::EthType::Sender;
  case ttkernel::EthType::Receiver:
    return ::tt::target::metal::EthType::Receiver;
  }
  assert(false && "Unsupported EthType");
}

::tt::target::metal::NocIndex toFlatbuffer(ttkernel::NocIndex nocIndex) {
  switch (nocIndex) {
  case ttkernel::NocIndex::Noc0:
    return ::tt::target::metal::NocIndex::Noc0;
  case ttkernel::NocIndex::Noc1:
    return ::tt::target::metal::NocIndex::Noc1;
  }
  assert(false && "Unsupported NocIndex");
}

// Take KernelConfig and return pair of its type and variantized config itself
std::pair<::tt::target::metal::KernelConfig, ::flatbuffers::Offset<void>>
toFlatbuffer(::flatbuffers::FlatBufferBuilder &fbb,
             ttkernel::KernelConfigInterface kernelConfig) {
  ttkernel::ThreadType threadType = kernelConfig.getThreadType();

  switch (threadType) {
  case ttkernel::ThreadType::Noc: {
    auto nocConfigAttr = mlir::dyn_cast<ttkernel::NocConfigAttr>(kernelConfig);
    auto configType = ::tt::target::metal::KernelConfig::NocConfig;
    auto config = ::tt::target::metal::CreateNocConfig(
        fbb, toFlatbuffer(nocConfigAttr.getNocIndex()));
    return std::make_pair(configType, config.Union());
  }
  case ttkernel::ThreadType::Tensix: {
    auto tensixConfigAttr =
        mlir::dyn_cast<ttkernel::TensixConfigAttr>(kernelConfig);
    auto configType = ::tt::target::metal::KernelConfig::TensixConfig;
    auto unpackToDestModeVec =
        toFlatbuffer(tensixConfigAttr.getUnpackToDestMode());
    auto config = ::tt::target::metal::CreateTensixConfigDirect(
        fbb, toFlatbuffer(tensixConfigAttr.getMathFidelity()),
        tensixConfigAttr.getFp32DestAccEn(),
        tensixConfigAttr.getMathApproxMode(), &unpackToDestModeVec);
    return std::make_pair(configType, config.Union());
  }
  case ttkernel::ThreadType::Ethernet: {
    auto ethernetConfigAttr =
        mlir::dyn_cast<ttkernel::EthernetConfigAttr>(kernelConfig);
    auto configType = ::tt::target::metal::KernelConfig::EthernetConfig;
    auto config = ::tt::target::metal::CreateEthernetConfig(
        fbb, toFlatbuffer(ethernetConfigAttr.getEthType()),
        toFlatbuffer(ethernetConfigAttr.getNocIndex()));
    return std::make_pair(configType, config.Union());
  }
  }
}

::tt::target::Dim2dRange toFlatbuffer(CoreRangeAttr coreRange) {
  auto offset = coreRange.getOffset();
  auto size = coreRange.getSize();
  return ::tt::target::Dim2dRange(::tt::target::Dim2d(offset[0], offset[1]),
                                  ::tt::target::Dim2d(size[0], size[1]));
}

::flatbuffers::Offset<::tt::target::metal::CBDesc>
cbTypeToFlatbuffer(FlatbufferObjectCache &cache, ttkernel::CBType cbType) {
  auto memref = cache.getOrCreate(cbType.getMemref(), memrefAttrToFlatbuffer);
  return ::tt::target::metal::CreateCBDesc(
      *cache.fbb, llvm::to_underlying(cbType.getPort()), memref,
      cbType.getPageSize(), cbType.getNumBuffers());
}

std::pair<::tt::target::metal::RuntimeArg, ::flatbuffers::Offset<void>>
toFlatbuffer(FlatbufferObjectCache &cache, ttkernel::SemaphoreType sem) {
  auto runtimeArgType =
      ::tt::target::metal::RuntimeArg::RuntimeArgSemaphoreAddress;
  auto semAddr = ::tt::target::metal::CreateRuntimeArgSemaphoreAddress(
      *cache.fbb, sem.getInitialValue());
  return std::make_pair(runtimeArgType, semAddr.Union());
}

std::pair<::tt::target::metal::HostBuffer, ::flatbuffers::Offset<void>>
hostBufferToFlatbuffer(FlatbufferObjectCache &cache,
                       ElementsAttr elementsAttr) {
  assert(elementsAttr.getElementType().isIntOrIndexOrFloat() &&
         "unsupported elements attr type");
  assert(elementsAttr.isSplat() && "expected a splat elements attr");
  assert(elementsAttr.getElementType().getIntOrFloatBitWidth() == 32 &&
         "unsupported elements attr bit width");
  auto vector = toFlatbuffer(cache, elementsAttr);
  return std::make_pair(
      ::tt::target::metal::HostBuffer::ConstantBuffer32,
      ::tt::target::metal::CreateConstantBuffer32(*cache.fbb, vector).Union());
}

Value getOperandThroughDPSOps(Value value) {
  auto *op = value.getDefiningOp();
  if (!op) {
    return value;
  }
  while (isa<DestinationStyleOpInterface>(op)) {
    assert(op->getResults().size() == 1);
    auto dps = cast<DestinationStyleOpInterface>(op);
    assert(dps.getNumDpsInits() == 1);
    auto *opOperand = dps.getDpsInitOperand(0);
    value = opOperand->get();
    op = value.getDefiningOp();
  }
  return value;
}

static flatbuffers::Offset<::tt::target::metal::LayoutDesc>
encodingToFlatbuffer(FlatbufferObjectCache &cache, Attribute attr,
                     ArrayRef<int64_t> logicalShape, DeviceAttr deviceAttr) {
  assert(isa<MetalLayoutAttr>(attr) && "unsupported layout attr");
  return metalLayoutAttrToFlatbuffer(cache, cast<MetalLayoutAttr>(attr),
                                     logicalShape, deviceAttr);
}

static flatbuffers::Offset<::tt::target::metal::TensorDesc>
tensorTypeToFlatbuffer(FlatbufferObjectCache &cache, Type type,
                       DeviceAttr deviceAttr) {
  auto tensorType = mlir::cast<RankedTensorType>(type);
  auto shapeInt64 = tensorType.getShape();
  std::vector<int32_t> shape(shapeInt64.begin(), shapeInt64.end());
  return ::tt::target::metal::CreateTensorDescDirect(
      *cache.fbb, &shape,
      cache.getOrCreate(tensorType.getEncoding(), encodingToFlatbuffer,
                        shapeInt64, deviceAttr));
}

static flatbuffers::Offset<::tt::target::metal::TensorRef>
tensorValueToFlatbuffer(FlatbufferObjectCache &cache, Value value,
                        uint64_t address, uint64_t size) {
  auto deviceAttr = lookupDevice(value.getParentBlock()->getParentOp());
  assert(deviceAttr);
  auto tensorType = mlir::cast<RankedTensorType>(value.getType());
  auto tensorDesc =
      cache.getOrCreate(tensorType, tensorTypeToFlatbuffer, deviceAttr);
  return ::tt::target::metal::CreateTensorRef(*cache.fbb, cache.global_id++,
                                              address, size, tensorDesc);
}

static std::shared_ptr<void> translateModuleToFlatbuffer(
    Operation *op, std::unordered_map<std::string, GoldenTensor> goldenMap) {
  ::flatbuffers::FlatBufferBuilder fbb;
  FlatbufferObjectCache cache(&fbb);

  ModuleOp module = dyn_cast<ModuleOp>(op);
  assert(module && "Expected ModuleOp as top level operation");

  auto systemDesc =
      mlir::cast<tt::SystemDescAttr>(module->getAttr(tt::SystemDescAttr::name));
  ::ttmlir::Version ttmlirVersion = ::ttmlir::getVersion();
  ::tt::target::Version binaryVersion(ttmlirVersion.major, ttmlirVersion.minor,
                                      ttmlirVersion.patch);
  std::vector<::flatbuffers::Offset<::tt::target::metal::Program>> programs;

  module->walk([&](func::FuncOp entry) {
    CQBuilder cqBuilder(&fbb);
    cqBuilder.name = entry.getSymName().data();

    auto argumentAllocations = mlir::cast<ArrayAttr>(
        entry->getDiscardableAttr(ArgumentAllocationAttr::name));
    assert(argumentAllocations && "expected argument_allocations attribute");
    for (auto &input : entry.getBody().getArguments()) {
      auto argAlloc = mlir::cast<tt::ArgumentAllocationAttr>(
          argumentAllocations[input.getArgNumber()]);
      assert(
          argAlloc.getMemorySpace() ==
              mlir::cast<tt::MetalLayoutAttr>(
                  mlir::cast<RankedTensorType>(input.getType()).getEncoding())
                  .getMemorySpace() &&
          "argument allocation memory space does not match tensor type "
          "memory "
          "space");
      cqBuilder.inputs.push_back(
          cache.getOrCreate(input, tensorValueToFlatbuffer,
                            argAlloc.getAddress(), argAlloc.getSize()));
    }

    entry->walk([&](mlir::Operation *op) {
      if (auto enqueueProgramOp =
              dyn_cast_if_present<tt::ttmetal::EnqueueProgramOp>(op);
          enqueueProgramOp) {
        std::vector<::flatbuffers::Offset<::tt::target::metal::TensorRef>>
            operands;
        for (auto operand : enqueueProgramOp.getOperands()) {
          operands.push_back(cache.at<::tt::target::metal::TensorRef>(
              getOperandThroughDPSOps(operand)));
        }

        std::vector<::flatbuffers::Offset<::tt::target::metal::KernelDesc>>
            kernels;

        llvm::SmallVector<std::string> cppKernels(
            enqueueProgramOp->getNumRegions());
        llvm::LogicalResult success =
            emitEnqueueProgramOpRegionsAsCpp(enqueueProgramOp, cppKernels);
        assert(success.succeeded() &&
               "failed to emit enqueue program op regions as cpp");

        // TODO(jdesousa #2960): This code is commented to allow for compilation
        // after the new TTMetal lowering flow was implemented. This needs to be
        // replaced with lowering for new enqueue program semantics.
        llvm_unreachable(
            "EnqueueProgramOp to Flatbuffer is not yet implemented for "
            "new D2M lowering flow.");
        // for (auto &region : enqueueProgramOp.getRegions()) {
        //   std::vector<::tt::target::Dim2dRange> coreRangeSet = {
        //       toFlatbuffer(mlir::cast<CoreRangeAttr>(
        //           enqueueProgramOp.getCoreRanges()[region.getRegionNumber()]))};
        //   std::vector<::flatbuffers::Offset<::tt::target::metal::CBRef>> cbs;
        //   size_t argNumber = 0;
        //   std::vector<::tt::target::metal::RuntimeArg> runtime_args_type;
        //   std::vector<::flatbuffers::Offset<void>> runtime_args;
        //   for (auto arg : region.getArguments()) {
        //     if (mlir::isa<ttkernel::CBType>(arg.getType())) {
        //       auto cbType = mlir::cast<ttkernel::CBType>(arg.getType());
        //       auto cbDesc = cache.getOrCreate(cbType, cbTypeToFlatbuffer);
        //       auto tensorRef =
        //           argNumber >= operands.size() ? 0 : operands[argNumber++];
        //       cbs.push_back(::tt::target::metal::CreateCBRef(
        //           fbb, cache.global_id++, tensorRef, cbType.getAddress(),
        //           cbDesc));
        //     } else if (mlir::isa<ttkernel::SemaphoreType>(arg.getType())) {
        //       auto semType =
        //       mlir::cast<ttkernel::SemaphoreType>(arg.getType()); auto
        //       [runtime_arg_type, runtime_arg] =
        //           toFlatbuffer(cache, semType);
        //       runtime_args_type.push_back(runtime_arg_type);
        //       runtime_args.push_back(runtime_arg);
        //     } else {
        //       llvm_unreachable(
        //           "Block arguments must be either CBType or SemaphoreType");
        //     }
        //   }

        //   std::string &source = cppKernels[region.getRegionNumber()];
        //   assert(source.size() > 0 && "empty kernel source");

        //   // Get pair of kernel's config type and config itself.
        //   auto kernelConfig =
        //       enqueueProgramOp.getKernelConfigs()[region.getRegionNumber()];
        //   auto [kernelConfigType, kernelConfigUnion] = toFlatbuffer(
        //       fbb,
        //       mlir::cast<ttkernel::KernelConfigInterface>(kernelConfig));

        //   kernels.push_back(::tt::target::metal::CreateKernelDescDirect(
        //       fbb, ::tt::target::metal::Kernel::KernelSource,
        //       ::tt::target::metal::CreateKernelSourceDirect(
        //           fbb, source.c_str(), kernelConfigType, kernelConfigUnion)
        //           .Union(),
        //       &coreRangeSet, &cbs, &runtime_args_type, &runtime_args,
        //       nullptr /*TODO debug info*/));
        // }
        ::flatbuffers::Offset<::tt::target::metal::ProgramDesc> program =
            ::tt::target::metal::CreateProgramDescDirect(fbb, &kernels);

        cqBuilder.appendCommand(
            ::tt::target::metal::CreateEnqueueProgramCommandDirect(
                fbb, &operands, program),
            op);
      } else if (auto createBufferOp =
                     dyn_cast_if_present<tt::ttmetal::CreateBufferOp>(op);
                 createBufferOp) {
        cqBuilder.appendCommand(
            ::tt::target::metal::CreateCreateBufferCommand(
                fbb, cache.getOrCreate(createBufferOp.getResult(),
                                       tensorValueToFlatbuffer,
                                       createBufferOp.getAddress(),
                                       createBufferOp.getSize())),
            op);
      } else if (auto deallocateBufferOp =
                     dyn_cast_if_present<tt::ttmetal::DeallocateBufferOp>(op);
                 deallocateBufferOp) {
        cqBuilder.appendCommand(
            ::tt::target::metal::CreateDeallocateBufferCommand(
                fbb,
                cache.at<::tt::target::metal::TensorRef>(
                    getOperandThroughDPSOps(deallocateBufferOp.getInput()))),
            op);
      } else if (auto enqueueReadBufferOp =
                     dyn_cast_if_present<tt::ttmetal::EnqueueReadBufferOp>(op);
                 enqueueReadBufferOp) {
        cqBuilder.appendCommand(
            ::tt::target::metal::CreateEnqueueReadBufferCommand(
                fbb,
                cache.at<::tt::target::metal::TensorRef>(
                    getOperandThroughDPSOps(enqueueReadBufferOp.getInput())),
                cache.at<::tt::target::metal::TensorRef>(
                    getOperandThroughDPSOps(enqueueReadBufferOp.getOutput()))),
            op);
      } else if (auto enqueueWriteBufferOp =
                     dyn_cast_if_present<tt::ttmetal::EnqueueWriteBufferOp>(op);
                 enqueueWriteBufferOp) {
        cqBuilder.appendCommand(
            ::tt::target::metal::CreateEnqueueWriteBufferCommand(
                fbb,
                cache.at<::tt::target::metal::TensorRef>(
                    getOperandThroughDPSOps(enqueueWriteBufferOp.getInput())),
                cache.at<::tt::target::metal::TensorRef>(
                    getOperandThroughDPSOps(enqueueWriteBufferOp.getOutput()))),
            op);
      } else if (auto returnOp = dyn_cast_if_present<func::ReturnOp>(op);
                 returnOp) {
        for (auto output : returnOp.getOperands()) {
          cqBuilder.outputs.push_back(cache.at<::tt::target::metal::TensorRef>(
              getOperandThroughDPSOps(output)));
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
