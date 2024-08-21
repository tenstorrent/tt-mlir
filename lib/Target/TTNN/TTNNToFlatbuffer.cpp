// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <flatbuffers/buffer.h>
#include <fstream>
#include <llvm/Support/Casting.h>

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TT/IR/TTOpsAttrDefs.h.inc"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Transforms/TTNNToCpp.h"
#include "ttmlir/Target/TTNN/TTNNToFlatbuffer.h"
#include "ttmlir/Target/TTNN/Target.h"
#include "ttmlir/Target/TTNN/binary_generated.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttmlir/Target/Utils/FlatbufferObjectCache.h"
#include "ttmlir/Target/Utils/FuncOpToProgram.h"
#include "ttmlir/Target/Utils/MLIRToFlatbuffer.h"
#include "ttmlir/Version.h"
#include "types_generated.h"

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_TTNNSERIALIZETOBINARY
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

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
  auto resultType = mlir::cast<DeviceType>(result.getType());
  ::tt::target::Dim2d grid =
      toFlatbuffer(cache, resultType.getDesc().getGrid());
  auto chipIds = toFlatbuffer(cache, resultType.getDesc().getChipIds());
  auto out = cache.getOrCreate(result, createDeviceRef);
  return ::tt::target::ttnn::CreateOpenDeviceOp(*cache.fbb, &grid, chipIds,
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
  constexpr uint64_t kHostAllocatedAddress = 0;
  constexpr uint64_t kHostAllocatedSize = 0;
  auto input =
      cache.at<::tt::target::TensorRef>(getOperandThroughDPSOps(op.getInput()));
  auto output = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                                  kHostAllocatedAddress, kHostAllocatedSize);
  return ::tt::target::ttnn::CreateToMemoryConfigOp(*cache.fbb, input, output);
}

::flatbuffers::Offset<::tt::target::ttnn::EmptyOp>
createOp(FlatbufferObjectCache &cache, EmptyOp op) {
  constexpr uint64_t kHostAllocatedAddress = 0;
  constexpr uint64_t kHostAllocatedSize = 0;
  auto device = getOperandThroughDPSOps(op.getDevice());
  auto output = getOperandThroughDPSOps(op.getResult());
  return ::tt::target::ttnn::CreateEmptyOp(
      *cache.fbb, cache.at<::tt::target::DeviceRef>(device),
      cache.getOrCreate(output, tensorValueToFlatbuffer, kHostAllocatedAddress,
                        kHostAllocatedSize));
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

// ANCHOR: adding_an_op_matmul_serialize_to_binary
::flatbuffers::Offset<::tt::target::ttnn::MatmulOp>
createOp(FlatbufferObjectCache &cache, MatmulOp op) {
  auto in0 =
      cache.at<::tt::target::TensorRef>(getOperandThroughDPSOps(op.getA()));
  auto in1 =
      cache.at<::tt::target::TensorRef>(getOperandThroughDPSOps(op.getB()));
  auto output = cache.at<::tt::target::TensorRef>(
      getOperandThroughDPSOps(op.getResult()));
  return ::tt::target::ttnn::CreateMatmulOp(*cache.fbb, in0, in1, output);
}
// ANCHOR_END: adding_an_op_matmul_serialize_to_binary

template <typename EltwiseOp>
::flatbuffers::Offset<::tt::target::ttnn::EltwiseOp>
createEltwiseOp(FlatbufferObjectCache &cache, EltwiseOp op) {
  ::tt::target::ttnn::EltwiseOpType type;
  if constexpr (std::is_same_v<EltwiseOp, AddOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::Add;
  } else if constexpr (std::is_same_v<EltwiseOp, MultiplyOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::Multiply;
  } else if constexpr (std::is_same_v<EltwiseOp, SubtractOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::Subtract;
  } else if constexpr (std::is_same_v<EltwiseOp, GreaterEqualOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::GreaterEqual;
  } else if constexpr (std::is_same_v<EltwiseOp, ReluOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::Relu;
  } else if constexpr (std::is_same_v<EltwiseOp, SqrtOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::Sqrt;
  } else if constexpr (std::is_same_v<EltwiseOp, ReciprocalOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::Reciprocal;
  } else if constexpr (std::is_same_v<EltwiseOp, DivOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::Div;
  } else if constexpr (std::is_same_v<EltwiseOp, SigmoidOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::Sigmoid;
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

template <typename ReductionOp>
::flatbuffers::Offset<::tt::target::ttnn::ReductionOp>
createReductionOp(FlatbufferObjectCache &cache, ReductionOp op) {
  ::tt::target::ttnn::ReductionOpType type;
  if constexpr (std::is_same_v<ReductionOp, SumOp>) {
    type = ::tt::target::ttnn::ReductionOpType::Sum;
  } else if constexpr (std::is_same_v<ReductionOp, MeanOp>) {
    type = ::tt::target::ttnn::ReductionOpType::Mean;
  } else {
    llvm_unreachable("unhandled ReductionOp");
  }

  auto in =
      cache.at<::tt::target::TensorRef>(getOperandThroughDPSOps(op.getInput()));
  auto output = cache.at<::tt::target::TensorRef>(
      getOperandThroughDPSOps(op.getResult()));
  auto dim_arg =
      arrayAttrToFlatbuffer<mlir::IntegerAttr, int>(cache, op.getDimArg());

  return ::tt::target::ttnn::CreateReductionOp(*cache.fbb, type, in, output,
                                               dim_arg, op.getKeepDim());
}

template <typename TransposeOp>
::flatbuffers::Offset<::tt::target::ttnn::TransposeOp>
createTransposeOp(FlatbufferObjectCache &cache, TransposeOp op) {
  auto in =
      cache.at<::tt::target::TensorRef>(getOperandThroughDPSOps(op.getInput()));
  auto out = cache.at<::tt::target::TensorRef>(
      getOperandThroughDPSOps(op.getResult()));
  int32_t dim0 = op.getDim0();
  int32_t dim1 = op.getDim1();

  return ::tt::target::ttnn::CreateTransposeOp(*cache.fbb, in, out, dim0, dim1);
}

template <typename EmbeddingOp>
::flatbuffers::Offset<::tt::target::ttnn::EmbeddingOp>
createEmbeddingOp(FlatbufferObjectCache &cache, EmbeddingOp op) {
  auto in0 =
      cache.at<::tt::target::TensorRef>(getOperandThroughDPSOps(op.getInput()));
  auto in1 = cache.at<::tt::target::TensorRef>(
      getOperandThroughDPSOps(op.getWeight()));
  auto output = cache.at<::tt::target::TensorRef>(
      getOperandThroughDPSOps(op.getResult()));
  return ::tt::target::ttnn::CreateEmbeddingOp(*cache.fbb, in0, in1, output);
}

template <typename SoftmaxOp>
::flatbuffers::Offset<::tt::target::ttnn::SoftmaxOp>
createSoftmaxOp(FlatbufferObjectCache &cache, SoftmaxOp op) {
  auto in =
      cache.at<::tt::target::TensorRef>(getOperandThroughDPSOps(op.getInput()));
  auto out = cache.at<::tt::target::TensorRef>(
      getOperandThroughDPSOps(op.getResult()));
  int32_t dimension = op.getDimension();

  return ::tt::target::ttnn::CreateSoftmaxOp(*cache.fbb, in, out, dimension);
}

template <typename GenericOp>
::flatbuffers::Offset<::tt::target::ttnn::GenericOp>
createGenericOp(FlatbufferObjectCache &cache, GenericOp op) {

  // input tensors
  std::vector<::flatbuffers::Offset<::tt::target::TensorRef>> ins;
  for (auto input : op.getInputs()) {
      ins.push_back(
          cache.at<::tt::target::TensorRef>(getOperandThroughDPSOps(input)));
  }

  // output tensors
  std::vector<::flatbuffers::Offset<::tt::target::TensorRef>> outs;
  for (auto output : op.getResults()) {
    outs.push_back(
      cache.getOrCreate(getOperandThroughDPSOps(output), tensorValueToFlatbuffer, 0, 0));
  }

  // circular buffer attributes
  std::vector<::flatbuffers::Offset<::tt::target::CBConfig>> cb_configs;
  for (auto attr : op.getCircularBufferAttributesAttr()) {
    if (const auto& cbAttr = mlir::dyn_cast<CircularBufferAttributesAttr>(attr)) {
      cb_configs.push_back(cache.getOrCreate(cbAttr, &circularBufferAttributesAttrToFlatbuffer));
    }
  }

  // compute kernel atrributes
  std::vector<::flatbuffers::Offset<::tt::target::ComputeKernelAttribute>> compute_kernel_attributes;
  for (auto attr : op.getComputeAttributesAttr()) {
    if (const auto& compute_attr = mlir::dyn_cast<ComputeAttributesAttr>(attr)) {
      compute_kernel_attributes.push_back(cache.getOrCreate(compute_attr, &computeAttributesAttrToFlatbuffer));
    }
  }

  // data movement attributes
  std::vector<::flatbuffers::Offset<::tt::target::DataMovementAttribute>> data_movement_attributes;
  for (auto attr : op.getDataMovementAttributesAttr()) {
    if (const auto& data_movement_attr = mlir::dyn_cast<DataMovementAttributesAttr>(attr)) {
      data_movement_attributes.push_back(cache.getOrCreate(data_movement_attr, &dataMovementAttributesAttrToFlatbuffer));
    }
  }

  return ::tt::target::ttnn::CreateGenericOp(
          *cache.fbb,
          cache.fbb->CreateVector<::flatbuffers::Offset<::tt::target::TensorRef>>(ins),
          cache.fbb->CreateVector<::flatbuffers::Offset<::tt::target::TensorRef>>(outs),
          cache.fbb->CreateVector<::flatbuffers::Offset<::tt::target::ComputeKernelAttribute>>(compute_kernel_attributes), 
          cache.fbb->CreateVector<::flatbuffers::Offset<::tt::target::DataMovementAttribute>>(data_movement_attributes),
          cache.fbb->CreateVector<::flatbuffers::Offset<::tt::target::CBConfig>>(cb_configs));
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
  if (auto emptyOp = dyn_cast<EmptyOp>(op); emptyOp) {
    return createOperation(cache, createOp(cache, emptyOp), debugString);
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
  if (auto subtractOp = dyn_cast<SubtractOp>(op); subtractOp) {
    return createOperation(cache, createEltwiseOp(cache, subtractOp),
                           debugString);
  }
  if (auto geOp = dyn_cast<GreaterEqualOp>(op); geOp) {
    return createOperation(cache, createEltwiseOp(cache, geOp), debugString);
  }
  if (auto reluOp = dyn_cast<ReluOp>(op); reluOp) {
    return createOperation(cache, createEltwiseOp(cache, reluOp), debugString);
  }
  if (auto sqrtOp = dyn_cast<SqrtOp>(op); sqrtOp) {
    return createOperation(cache, createEltwiseOp(cache, sqrtOp), debugString);
  }
  if (auto sigmoidOp = dyn_cast<SigmoidOp>(op); sigmoidOp) {
    return createOperation(cache, createEltwiseOp(cache, sigmoidOp),
                           debugString);
  }
  if (auto reciprocalOp = dyn_cast<ReciprocalOp>(op); reciprocalOp) {
    return createOperation(cache, createEltwiseOp(cache, reciprocalOp),
                           debugString);
  }
  if (auto divOp = dyn_cast<DivOp>(op); divOp) {
    return createOperation(cache, createEltwiseOp(cache, divOp), debugString);
  }
  if (auto matmulOp = dyn_cast<MatmulOp>(op); matmulOp) {
    return createOperation(cache, createOp(cache, matmulOp), debugString);
  }
  if (auto sumOp = dyn_cast<SumOp>(op); sumOp) {
    return createOperation(cache, createReductionOp(cache, sumOp), debugString);
  }
  if (auto meanOp = dyn_cast<MeanOp>(op); meanOp) {
    return createOperation(cache, createReductionOp(cache, meanOp),
                           debugString);
  }
  if (auto embeddingOp = dyn_cast<EmbeddingOp>(op); embeddingOp) {
    return createOperation(cache, createEmbeddingOp(cache, embeddingOp),
                           debugString);
  }
  if (auto softmaxOp = dyn_cast<SoftmaxOp>(op); softmaxOp) {
    return createOperation(cache, createSoftmaxOp(cache, softmaxOp),
                           debugString);
  }
  if (auto transposeOp = dyn_cast<TransposeOp>(op); transposeOp) {
    return createOperation(cache, createTransposeOp(cache, transposeOp),
                           debugString);
  }

  if (auto genericOp = dyn_cast<GenericOp>(op); genericOp) {
    return createOperation(cache, createGenericOp(cache, genericOp), debugString);
  }

  llvm_unreachable("unhandled op in emitTTNNOperation");
}

std::shared_ptr<void> ttnnToFlatbuffer(Operation *op) {
  ModuleOp module = dyn_cast<ModuleOp>(op);
  assert(module && "Expected ModuleOp as top level operation");

  ::flatbuffers::FlatBufferBuilder fbb;
  FlatbufferObjectCache cache(&fbb);

  ::ttmlir::Version ttmlirVersion = ::ttmlir::getVersion();
  ::tt::target::Version binaryVersion(ttmlirVersion.major, ttmlirVersion.minor,
                                      ttmlirVersion.patch);

  auto systemDesc =
      toFlatbuffer(cache, mlir::cast<tt::SystemDescAttr>(
                              module->getAttr(tt::SystemDescAttr::name)));

  auto mlir = toDebugInfo(fbb, "ttnn", module);
  std::string cpp;
  llvm::raw_string_ostream os(cpp);
  auto result = mlir::tt::ttnn::emitTTNNAsCpp(module, os);
  (void)result;

  auto debugInfo = ::tt::target::CreateDebugInfoDirect(fbb, mlir, cpp.c_str());

  std::vector<::flatbuffers::Offset<::tt::target::ttnn::Program>> programs;
  module->walk([&](func::FuncOp func) {
    Program<::tt::target::ttnn::Operation> program =
        funcOpToProgram<::tt::target::ttnn::Operation>(cache, func,
                                                       emitTTNNOperation);
    programs.push_back(::tt::target::ttnn::CreateProgramDirect(
        fbb, program.name, &program.inputs, &program.outputs, &program.ops,
        debugInfo));
  });

  auto binary = ::tt::target::ttnn::CreateTTNNBinaryDirect(
      fbb, &binaryVersion, ::ttmlir::getGitHash(), systemDesc, &programs);

  ::tt::target::ttnn::FinishSizePrefixedTTNNBinaryBuffer(fbb, binary);
  ::flatbuffers::Verifier verifier(fbb.GetBufferPointer(), fbb.GetSize());
  ::tt::target::ttnn::VerifySizePrefixedTTNNBinaryBuffer(verifier);

  uint8_t *buf = fbb.GetBufferPointer();
  std::size_t size = fbb.GetSize();

  std::shared_ptr<void> bufferPtr =
      std::shared_ptr<void>(std::malloc(size), std::free);
  std::memcpy(bufferPtr.get(), buf, size);
  return bufferPtr;
}

LogicalResult translateTTNNToFlatbuffer(Operation *op, llvm::raw_ostream &os) {
  std::shared_ptr<void> data = ttnnToFlatbuffer(op);
  std::size_t size = ::flatbuffers::GetSizePrefixedBufferLength(
      static_cast<const uint8_t *>(data.get()));
  os.write(reinterpret_cast<char const *>(data.get()), size);
  return success();
}
} // namespace mlir::tt::ttnn
