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
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Passes.h"
#include "ttmlir/Dialect/TTNN/Transforms/TTNNToCpp.h"
#include "ttmlir/Target/TTNN/Target.h"
#include "ttmlir/Target/Utils/FlatbufferObjectCache.h"
#include "ttmlir/Target/Utils/FuncOpToProgram.h"
#include "ttmlir/Target/Utils/MLIRToFlatbuffer.h"
#include "ttmlir/Version.h"

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_TTNNSERIALIZETOBINARY
#include "ttmlir/Dialect/TTNN/Passes.h.inc"

::tt::target::Dim2dRange toFlatbuffer(CoreRangeAttr coreRange) {
  auto offset = coreRange.getOffset();
  auto size = coreRange.getSize();
  return ::tt::target::Dim2dRange(::tt::target::Dim2d(offset[0], offset[1]),
                                  ::tt::target::Dim2d(size[0], size[1]));
}

::flatbuffers::Offset<::tt::target::DeviceRef>
createDeviceRef(FlatbufferObjectCache &cache, Value device) {
  return ::tt::target::CreateDeviceRef(*cache.fbb, cache.nextGlobalId());
}

template <typename OpT>
::flatbuffers::Offset<::tt::target::ttnn::Operation>
createOperation(FlatbufferObjectCache &cache, ::flatbuffers::Offset<OpT> op,
                std::string const &debugString) {
  return CreateOperationDirect(
      *cache.fbb, ::tt::target::ttnn::OpTypeTraits<OpT>::enum_value, op.Union(),
      debugString.c_str());
}

::flatbuffers::Offset<::tt::target::ttnn::OpenDeviceOp>
createOp(FlatbufferObjectCache &cache, OpenDeviceOp op) {
  auto result = op.getResult();
  auto resultType = result.getType().cast<DeviceType>();
  ::tt::target::Dim2d mesh = toFlatbuffer(cache, resultType.getMesh());
  auto chipIds = toFlatbuffer(cache, resultType.getChipIds());
  auto out = cache.getOrCreate(result, createDeviceRef);
  return ::tt::target::ttnn::CreateOpenDeviceOp(*cache.fbb, &mesh, chipIds,
                                                out);
}

::flatbuffers::Offset<::tt::target::ttnn::CloseDeviceOp>
createOp(FlatbufferObjectCache &cache, CloseDeviceOp op) {
  auto device = op.getDevice();
  auto in = cache.at<::tt::target::DeviceRef>(device);
  return ::tt::target::ttnn::CreateCloseDeviceOp(*cache.fbb, in);
}

::flatbuffers::Offset<::tt::target::ttnn::ToMemoryConfigOp>
createOp(FlatbufferObjectCache &cache, ToMemoryConfigOp op) {
  auto input = getOperandThroughDPSOps(op.getInput());
  auto output = getOperandThroughDPSOps(op.getOutput());
  return ::tt::target::ttnn::CreateToMemoryConfigOp(
      *cache.fbb, cache.at<::tt::target::TensorRef>(input),
      cache.at<::tt::target::TensorRef>(output));
}

::flatbuffers::Offset<::tt::target::ttnn::FullOp>
createOp(FlatbufferObjectCache &cache, FullOp op) {
  constexpr uint64_t kHostAllocatedAddress = 0;
  constexpr uint64_t kHostAllocatedSize = 0;
  auto device = getOperandThroughDPSOps(op.getDevice());
  auto fillValue = op.getFillValue().convertToFloat();
  auto output = getOperandThroughDPSOps(op.getResult());
  return ::tt::target::ttnn::CreateFullOp(
      *cache.fbb, cache.at<::tt::target::DeviceRef>(device), fillValue,
      cache.getOrCreate(output, tensorValueToFlatbuffer, kHostAllocatedAddress,
                        kHostAllocatedSize));
}

template <typename EltwiseOp>
::flatbuffers::Offset<::tt::target::ttnn::EltwiseOp>
createEltwiseOp(FlatbufferObjectCache &cache, EltwiseOp op) {
  ::tt::target::ttnn::EltwiseOpType type;
  if constexpr (std::is_same_v<EltwiseOp, AddOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::Add;
  } else if constexpr (std::is_same_v<EltwiseOp, MultiplyOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::Multiply;
  } else {
    llvm_unreachable("unhandled EltwiseOp");
  }
  std::vector<::flatbuffers::Offset<::tt::target::TensorRef>> ins;
  for (auto input : op.getInputs()) {
    ins.push_back(
        cache.at<::tt::target::TensorRef>(getOperandThroughDPSOps(input)));
  }
  assert(op.getOutputs().size() == 1);
  return ::tt::target::ttnn::CreateEltwiseOpDirect(
      *cache.fbb, type, &ins,
      cache.at<::tt::target::TensorRef>(
          getOperandThroughDPSOps(op.getOutputs().front())));
}

::flatbuffers::Offset<::tt::target::ttnn::Operation>
emitTTNNOperation(FlatbufferObjectCache &cache, Operation *op,
                  std::string const &debugString) {
  if (auto openDeviceOp = dyn_cast<OpenDeviceOp>(op); openDeviceOp) {
    return createOperation(cache, createOp(cache, openDeviceOp), debugString);
  }
  if (auto closeDeviceOp = dyn_cast<CloseDeviceOp>(op); closeDeviceOp) {
    return createOperation(cache, createOp(cache, closeDeviceOp), debugString);
  }
  if (auto toMemoryConfigOp = dyn_cast<ToMemoryConfigOp>(op);
      toMemoryConfigOp) {
    return createOperation(cache, createOp(cache, toMemoryConfigOp),
                           debugString);
  }
  if (auto fullOp = dyn_cast<FullOp>(op); fullOp) {
    return createOperation(cache, createOp(cache, fullOp), debugString);
  }
  if (auto addOp = dyn_cast<AddOp>(op); addOp) {
    return createOperation(cache, createEltwiseOp(cache, addOp), debugString);
  }
  if (auto multiplyOp = dyn_cast<MultiplyOp>(op); multiplyOp) {
    return createOperation(cache, createEltwiseOp(cache, multiplyOp),
                           debugString);
  }

  llvm_unreachable("unhandled op in emitTTNNOperation");
}

class TTNNSerializeToBinary
    : public impl::TTNNSerializeToBinaryBase<TTNNSerializeToBinary> {
public:
  using impl::TTNNSerializeToBinaryBase<
      TTNNSerializeToBinary>::TTNNSerializeToBinaryBase;

  void runOnOperation() final {
    ::flatbuffers::FlatBufferBuilder fbb;
    FlatbufferObjectCache cache(&fbb);

    ::ttmlir::Version ttmlirVersion = ::ttmlir::getVersion();
    ::tt::target::Version binaryVersion(
        ttmlirVersion.major, ttmlirVersion.minor, ttmlirVersion.release);

    ModuleOp module = getOperation();
    auto systemDesc = toFlatbuffer(
        cache,
        module->getAttr(tt::SystemDescAttr::name).cast<tt::SystemDescAttr>());

    func::FuncOp entry = dyn_cast<func::FuncOp>(*module.getRegion().op_begin());
    assert(entry && "expected an entry function");
    Program<::tt::target::ttnn::Operation> program =
        funcOpToProgram<::tt::target::ttnn::Operation>(cache, entry,
                                                       emitTTNNOperation);

    auto mlir = toDebugInfo(fbb, "ttnn", module);
    std::string cpp;
    llvm::raw_string_ostream os(cpp);
    auto result = emitTTNNAsCpp(module, os);
    (void)result;

    auto debugInfo =
        ::tt::target::CreateDebugInfoDirect(fbb, mlir, cpp.c_str());
    auto programOffset = ::tt::target::ttnn::CreateProgramDirect(
        fbb, program.name, &program.inputs, &program.outputs, &program.ops,
        debugInfo);
    std::vector<::flatbuffers::Offset<::tt::target::ttnn::Program>> programs = {
        programOffset,
    };
    auto binary = ::tt::target::ttnn::CreateTTNNBinaryDirect(
        fbb, &binaryVersion, ::ttmlir::getGitHash(), systemDesc, &programs);

    ::tt::target::ttnn::FinishSizePrefixedTTNNBinaryBuffer(fbb, binary);
    ::flatbuffers::Verifier verifier(fbb.GetBufferPointer(), fbb.GetSize());
    ::tt::target::ttnn::VerifySizePrefixedTTNNBinaryBuffer(verifier);

    uint8_t *buf = fbb.GetBufferPointer();
    auto size = fbb.GetSize();

#if 1
    std::ofstream ttnn("out.ttnn", std::ios::out | std::ios::binary);
    ttnn.write(reinterpret_cast<char const *>(buf), size);
    ttnn.close();
#endif
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttnn::TTNNDialect>();
    registry.insert<mlir::tt::ttkernel::TTKernelDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::emitc::EmitCDialect>();
  }
};

} // namespace mlir::tt::ttnn
