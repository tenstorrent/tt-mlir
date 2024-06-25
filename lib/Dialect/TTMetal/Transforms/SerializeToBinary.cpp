// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fstream>

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
#include "ttmlir/Dialect/TTMetal/Passes.h"
#include "ttmlir/Dialect/TTMetal/Transforms/KernelsToCpp.h"
#include "ttmlir/Target/TTMetal/Target.h"
#include "ttmlir/Target/Utils/FlatbufferObjectCache.h"
#include "ttmlir/Target/Utils/MLIRToFlatbuffer.h"
#include "ttmlir/Version.h"

namespace mlir::tt::ttmetal {

#define GEN_PASS_DEF_TTMETALSERIALIZETOBINARY
#include "ttmlir/Dialect/TTMetal/Passes.h.inc"

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

class TTMetalSerializeToBinary
    : public impl::TTMetalSerializeToBinaryBase<TTMetalSerializeToBinary> {
public:
  using impl::TTMetalSerializeToBinaryBase<
      TTMetalSerializeToBinary>::TTMetalSerializeToBinaryBase;

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

  void runOnOperation() final {
    constexpr uint64_t kHostAllocatedAddress = 0;
    constexpr uint64_t kHostAllocatedSize = 0;

    ::flatbuffers::FlatBufferBuilder fbb;
    FlatbufferObjectCache cache(&fbb);
    CQBuilder cqBuilder(&fbb);

    ModuleOp module = getOperation();
    auto systemDesc =
        module->getAttr(tt::SystemDescAttr::name).cast<tt::SystemDescAttr>();
    func::FuncOp entry = dyn_cast<func::FuncOp>(*module.getRegion().op_begin());
    assert(entry && "expected an entry function");
    cqBuilder.name = entry.getSymName().data();

    for (auto &input : entry.getBody().getArguments()) {
      cqBuilder.inputs.push_back(
          cache.getOrCreate(input, tensorValueToFlatbuffer,
                            kHostAllocatedAddress, kHostAllocatedSize));
    }

    module->walk([&](mlir::Operation *op) {
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
              dispatchOp.getThreadTypes()[region.getRegionNumber()]
                  .cast<ttkernel::ThreadTypeAttr>()
                  .getValue();
          ::tt::target::Dim2dRange core_range =
              toFlatbuffer(dispatchOp.getCoreRanges()[region.getRegionNumber()]
                               .cast<CoreRangeAttr>());
          ::tt::target::Dim2dRange(::tt::target::Dim2d(0, 0),
                                   ::tt::target::Dim2d(0, 0));
          std::vector<::flatbuffers::Offset<::tt::target::CBRef>> cbs;
          kernels.push_back(::tt::target::metal::CreateKernelDescDirect(
              fbb, ::tt::target::metal::Kernel::KernelSource,
              ::tt::target::metal::CreateKernelSourceDirect(
                  fbb, toFlatbuffer(threadType), source.c_str())
                  .Union(),
              &core_range, &cbs, nullptr /*TODO debug info*/));
        }
        std::vector<::flatbuffers::Offset<::tt::target::metal::DispatchProgram>>
            programs = {
                ::tt::target::metal::CreateDispatchProgramDirect(fbb, &kernels),
            };

        cqBuilder.appendCommand(
            ::tt::target::metal::CreateDispatchCommandDirect(fbb, &operands,
                                                             &programs),
            op);
      } else if (auto allocOp = dyn_cast_or_null<tt::ttmetal::AllocOp>(op);
                 allocOp) {
        cqBuilder.appendCommand(
            ::tt::target::metal::CreateHostAllocCommand(
                fbb,
                cache.getOrCreate(allocOp.getResult(), tensorValueToFlatbuffer,
                                  allocOp.getAddress(), allocOp.getSize())),
            op);
      } else if (auto deallocOp = dyn_cast_or_null<tt::ttmetal::DeallocOp>(op);
                 deallocOp) {
        cqBuilder.appendCommand(
            ::tt::target::metal::CreateHostDeallocCommand(
                fbb, cache.at<::tt::target::TensorRef>(
                         getOperandThroughDPSOps(deallocOp.getInput()))),
            op);
      } else if (auto hostReadOp =
                     dyn_cast_or_null<tt::ttmetal::HostReadOp>(op);
                 hostReadOp) {
        cqBuilder.appendCommand(
            ::tt::target::metal::CreateHostReadCommand(
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
            ::tt::target::metal::CreateHostReadCommand(
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

    ::ttmlir::Version ttmlirVersion = ::ttmlir::getVersion();
    ::tt::target::Version binaryVersion(
        ttmlirVersion.major, ttmlirVersion.minor, ttmlirVersion.release);

    std::vector<::flatbuffers::Offset<::tt::target::metal::CommandQueue>>
        commandQueues = {
            ::tt::target::metal::CreateCommandQueueDirect(
                fbb, cqBuilder.name, &cqBuilder.inputs, &cqBuilder.outputs,
                &cqBuilder.commands),
        };
    auto binary = ::tt::target::metal::CreateTTMetalBinaryDirect(
        fbb, &binaryVersion, ::ttmlir::getGitHash(),
        toFlatbuffer(cache, systemDesc), &commandQueues);

    FinishSizePrefixedTTMetalBinaryBuffer(fbb, binary);
    ::flatbuffers::Verifier verifier(fbb.GetBufferPointer(), fbb.GetSize());
    ::tt::target::metal::VerifySizePrefixedTTMetalBinaryBuffer(verifier);

    uint8_t *buf = fbb.GetBufferPointer();
    auto size = fbb.GetSize();

#if 1
    std::ofstream ttb("out.ttb", std::ios::out | std::ios::binary);
    ttb.write(reinterpret_cast<char const *>(buf), size);
    ttb.close();
#endif
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttmetal::TTMetalDialect>();
    registry.insert<mlir::tt::ttkernel::TTKernelDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::emitc::EmitCDialect>();
  }
};

} // namespace mlir::tt::ttmetal
