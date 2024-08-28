// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include <memory>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LLVM.h>

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.h"
#include "ttmlir/Dialect/TTMetal/Transforms/KernelsToCpp.h"
#include "ttmlir/Dialect/TTMetal/Transforms/Passes.h"
#include "ttmlir/Target/TTMetal/Target.h"
#include "ttmlir/Target/Utils/FlatbufferObjectCache.h"
#include "ttmlir/Target/Utils/MLIRToFlatbuffer.h"
#include "ttmlir/Version.h"

namespace mlir::tt::ttmetal {

struct CQBuilder {
  ::flatbuffers::FlatBufferBuilder *fbb;
  const char *name;
  std::vector<::flatbuffers::Offset<::tt::target::TensorRef>> inputs;
  std::vector<::flatbuffers::Offset<::tt::target::TensorRef>> outputs;
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

::tt::target::metal::SourceType toFlatbuffer(ttkernel::ThreadType threadType) {
  switch (threadType) {
  case ttkernel::ThreadType::Noc0:
    return ::tt::target::metal::SourceType::Noc0;
  case ttkernel::ThreadType::Noc1:
    return ::tt::target::metal::SourceType::Noc1;
  case ttkernel::ThreadType::Tensix:
    return ::tt::target::metal::SourceType::Tensix;
  case ttkernel::ThreadType::Ethernet:
    return ::tt::target::metal::SourceType::Ethernet;
  }
}

::tt::target::Dim2dRange toFlatbuffer(CoreRangeAttr coreRange) {
  auto offset = coreRange.getOffset();
  auto size = coreRange.getSize();
  return ::tt::target::Dim2dRange(::tt::target::Dim2d(offset[0], offset[1]),
                                  ::tt::target::Dim2d(size[0], size[1]));
}

::flatbuffers::Offset<::tt::target::CBDesc>
cbTypeToFlatbuffer(FlatbufferObjectCache &cache, ttkernel::CBType cbType) {
  auto memref = cache.getOrCreate(cbType.getMemref(), memrefAttrToFlatbuffer);
  return ::tt::target::CreateCBDesc(
      *cache.fbb,
      static_cast<std::underlying_type_t<ttkernel::CBPort>>(cbType.getPort()),
      memref, cbType.getPageSize(), cbType.getNumBuffers());
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

static std::shared_ptr<void> translateModuleToFlatbuffer(Operation *op) {
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
              mlir::cast<tt::LayoutAttr>(
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
      if (auto dispatchOp = dyn_cast_or_null<tt::ttmetal::DispatchOp>(op);
          dispatchOp) {
        std::vector<::flatbuffers::Offset<::tt::target::TensorRef>> operands;
        for (auto operand : dispatchOp.getOperands()) {
          operands.push_back(cache.at<::tt::target::TensorRef>(
              getOperandThroughDPSOps(operand)));
        }

        std::vector<::flatbuffers::Offset<::tt::target::metal::KernelDesc>>
            kernels;
        for (auto &region : dispatchOp.getRegions()) {
          std::string source;
          llvm::raw_string_ostream os(source);
          auto result = emitDispatchOpRegionAsCpp(dispatchOp,
                                                  region.getRegionNumber(), os);
          assert(succeeded(result) &&
                 "failed to emit dispatch op region as cpp");
          auto threadType =
              mlir::cast<ttkernel::ThreadTypeAttr>(
                  dispatchOp.getThreadTypes()[region.getRegionNumber()])
                  .getValue();
          std::vector<::tt::target::Dim2dRange> coreRangeSet = {
              toFlatbuffer(mlir::cast<CoreRangeAttr>(
                  dispatchOp.getCoreRanges()[region.getRegionNumber()]))};
          std::vector<::flatbuffers::Offset<::tt::target::CBRef>> cbs;
          for (auto arg : region.getArguments()) {
            assert(arg.getArgNumber() < operands.size());
            auto cbType = mlir::cast<ttkernel::CBType>(arg.getType());
            auto cbDesc = cache.getOrCreate(cbType, cbTypeToFlatbuffer);
            auto tensorRef = operands[arg.getArgNumber()];
            cbs.push_back(
                ::tt::target::CreateCBRef(fbb, cache.global_id++, tensorRef,
                                          cbType.getAddress(), cbDesc));
          }
          kernels.push_back(::tt::target::metal::CreateKernelDescDirect(
              fbb, ::tt::target::metal::Kernel::KernelSource,
              ::tt::target::metal::CreateKernelSourceDirect(
                  fbb, toFlatbuffer(threadType), source.c_str())
                  .Union(),
              &coreRangeSet, &cbs, nullptr /*TODO debug info*/));
        }
        ::flatbuffers::Offset<::tt::target::metal::ProgramDesc> program =
            ::tt::target::metal::CreateProgramDescDirect(fbb, &kernels);

        cqBuilder.appendCommand(
            ::tt::target::metal::CreateEnqueueProgramCommandDirect(
                fbb, &operands, program),
            op);
      } else if (auto allocOp = dyn_cast_or_null<tt::ttmetal::AllocOp>(op);
                 allocOp) {
        cqBuilder.appendCommand(
            ::tt::target::metal::CreateCreateBufferCommand(
                fbb,
                cache.getOrCreate(allocOp.getResult(), tensorValueToFlatbuffer,
                                  allocOp.getAddress(), allocOp.getSize())),
            op);
      } else if (auto deallocOp = dyn_cast_or_null<tt::ttmetal::DeallocOp>(op);
                 deallocOp) {
        cqBuilder.appendCommand(
            ::tt::target::metal::CreateDeallocateBufferCommand(
                fbb, cache.at<::tt::target::TensorRef>(
                         getOperandThroughDPSOps(deallocOp.getInput()))),
            op);
      } else if (auto hostReadOp =
                     dyn_cast_or_null<tt::ttmetal::HostReadOp>(op);
                 hostReadOp) {
        cqBuilder.appendCommand(
            ::tt::target::metal::CreateEnqueueReadBufferCommand(
                fbb,
                cache.at<::tt::target::TensorRef>(
                    getOperandThroughDPSOps(hostReadOp.getInput())),
                cache.at<::tt::target::TensorRef>(
                    getOperandThroughDPSOps(hostReadOp.getOutput()))),
            op);
      } else if (auto hostWriteOp =
                     dyn_cast_or_null<tt::ttmetal::HostWriteOp>(op);
                 hostWriteOp) {
        cqBuilder.appendCommand(
            ::tt::target::metal::CreateEnqueueWriteBufferCommand(
                fbb,
                cache.at<::tt::target::TensorRef>(
                    getOperandThroughDPSOps(hostWriteOp.getInput())),
                cache.at<::tt::target::TensorRef>(
                    getOperandThroughDPSOps(hostWriteOp.getOutput()))),
            op);
      } else if (auto returnOp = dyn_cast_or_null<func::ReturnOp>(op);
                 returnOp) {
        for (auto output : returnOp.getOperands()) {
          cqBuilder.outputs.push_back(cache.at<::tt::target::TensorRef>(
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

LogicalResult translateTTMetalToFlatbuffer(Operation *op,
                                           llvm::raw_ostream &os) {
  std::shared_ptr<void> data = translateModuleToFlatbuffer(op);
  os << data.get();
  return success();
}

} // namespace mlir::tt::ttmetal
