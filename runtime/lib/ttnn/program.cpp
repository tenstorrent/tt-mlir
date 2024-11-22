// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "operations/ccl/all_gather.h"
#include "operations/context/get_device.h"
#include "operations/conv/conv2d.h"
#include "operations/creation/empty.h"
#include "operations/creation/full.h"
#include "operations/data_movement/concat.h"
#include "operations/data_movement/reshape.h"
#include "operations/data_movement/slice.h"
#include "operations/data_movement/transpose.h"
#include "operations/deletion/deallocate.h"
#include "operations/eltwise/binary/binary.h"
#include "operations/eltwise/binary/binary_composite.h"
#include "operations/eltwise/ternary/ternary.h"
#include "operations/eltwise/unary/unary.h"
#include "operations/eltwise/unary/unary_composite.h"
#include "operations/embedding/embedding.h"
#include "operations/layout/from_device.h"
#include "operations/layout/to_device.h"
#include "operations/layout/to_layout.h"
#include "operations/layout/to_memory_config.h"
#include "operations/layout/typecast.h"
#include "operations/matmul/matmul.h"
#include "operations/normalization/softmax.h"
#include "operations/pool/maxpool2d.h"
#include "operations/reduction/reduction.h"
#include "tt/runtime/detail/debug.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/ttnn/types.h"
#include "tt/runtime/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn {
using LogType = ::tt::runtime::logger::LogType;

static ::tt::target::ttnn::TTNNBinary const *getBinary(Flatbuffer binary) {
  bool isTTNN = ::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
      binary.handle.get());
  if (not isTTNN) {
    throw std::runtime_error("Unsupported binary format");
  }
  return ::tt::target::ttnn::GetSizePrefixedTTNNBinary(binary.handle.get());
}

class ProgramExecutor {
public:
  ProgramExecutor(Binary &executableHandle, const TensorMap &liveTensors,
                  const std::unordered_set<uint32_t> &programInputs,
                  const std::unordered_set<uint32_t> &programOutputs,
                  ::ttnn::MeshDevice *meshDevice)
      : executableHandle(executableHandle),
        context(ProgramContext(liveTensors, programInputs, programOutputs,
                               meshDevice)) {}

  void runCallback(Binary &executableHandle,
                   const ::tt::target::ttnn::Operation *opContext,
                   ProgramContext *programContext) {
    if (auto callback = debug::Hooks::get().getOperatorCallback(); callback) {
      std::shared_ptr<void> programContextPtr =
          ::tt::runtime::utils::unsafe_borrow_shared(programContext);
      std::shared_ptr<void> opContextPtr =
          ::tt::runtime::utils::unsafe_borrow_shared(
              const_cast<::tt::target::ttnn::Operation *>(opContext));
      (*callback)(executableHandle,
                  CallbackContext(programContextPtr, DeviceRuntime::TTNN),
                  OpContext(opContextPtr, DeviceRuntime::TTNN));
    }
  }

  void execute(const ::tt::target::ttnn::Program *program) {
    for (const ::tt::target::ttnn::Operation *op : *program->operations()) {
      LOG_DEBUG(LogType::LogRuntimeTTNN,
                "Executing operation: ", op->debug_info()->c_str());
      runOperation(op);
      runCallback(executableHandle, op, &context);
    }
  }

  ProgramContext &getContext() { return context; }

private:
  Binary executableHandle;
  ProgramContext context;
  void runOperation(const ::tt::target::ttnn::Operation *op);
  void runEltwiseOperation(const ::tt::target::ttnn::EltwiseOp *op);
};

void ProgramExecutor::runEltwiseOperation(
    const ::tt::target::ttnn::EltwiseOp *op) {
  auto runUnaryOp = [&]() {
    if (operations::unary::composite::isUnaryCompositeOp(op)) {
      return operations::unary::composite::run(op, context);
    }
    return operations::unary::run(op, context);
  };

  auto runBinaryOp = [&]() {
    if (operations::binary::composite::isBinaryCompositeOp(op)) {
      return operations::binary::composite::run(op, context);
    }
    return operations::binary::run(op, context);
  };

  auto runTernaryOp = [&]() { return operations::ternary::run(op, context); };

  if (operations::unary::isUnaryOp(op)) {
    return runUnaryOp();
  }

  if (operations::binary::isBinaryOp(op)) {
    return runBinaryOp();
  }
  if (operations::ternary::isTernaryOp(op)) {
    return runTernaryOp();
  }

  LOG_FATAL("Unsupported Eltwise operation");
}

void ProgramExecutor::runOperation(const ::tt::target::ttnn::Operation *op) {
  switch (op->type_type()) {
  case ::tt::target::ttnn::OpType::GetDeviceOp: {
    return operations::context::run(op->type_as_GetDeviceOp(), context);
  }
  case ::tt::target::ttnn::OpType::ToMemoryConfigOp: {
    return operations::layout::run(op->type_as_ToMemoryConfigOp(), context);
  }
  case ::tt::target::ttnn::OpType::ToLayoutOp: {
    return operations::layout::run(op->type_as_ToLayoutOp(), context);
  }
  case ::tt::target::ttnn::OpType::TypecastOp: {
    return operations::layout::run(op->type_as_TypecastOp(), context);
  }
  case ::tt::target::ttnn::OpType::ToDeviceOp: {
    return operations::layout::run(op->type_as_ToDeviceOp(), context);
  }
  case ::tt::target::ttnn::OpType::FromDeviceOp: {
    return operations::layout::run(op->type_as_FromDeviceOp(), context);
  }
  case ::tt::target::ttnn::OpType::EmptyOp: {
    return operations::creation::run(op->type_as_EmptyOp(), context);
  }
  case ::tt::target::ttnn::OpType::FullOp: {
    return operations::creation::run(op->type_as_FullOp(), context);
  }
  case ::tt::target::ttnn::OpType::EltwiseOp: {
    return runEltwiseOperation(op->type_as_EltwiseOp());
  }
  case ::tt::target::ttnn::OpType::LinearOp: {
    return operations::matmul::run(op->type_as_LinearOp(), context);
  }
  // ANCHOR: adding_an_op_matmul_runtime_program
  case ::tt::target::ttnn::OpType::MatmulOp: {
    return operations::matmul::run(op->type_as_MatmulOp(), context);
  }
  // ANCHOR_END: adding_an_op_matmul_runtime_program
  case ::tt::target::ttnn::OpType::ReductionOp: {
    return operations::reduction::run(op->type_as_ReductionOp(), context);
  }
  case ::tt::target::ttnn::OpType::EmbeddingOp: {
    return operations::embedding::run(op->type_as_EmbeddingOp(), context);
  }
  case ::tt::target::ttnn::OpType::SoftmaxOp: {
    return operations::normalization::run(op->type_as_SoftmaxOp(), context);
  }
  case ::tt::target::ttnn::OpType::TransposeOp: {
    return operations::data_movement::run(op->type_as_TransposeOp(), context);
  }
  case ::tt::target::ttnn::OpType::ConcatOp: {
    return operations::data_movement::run(op->type_as_ConcatOp(), context);
  }
  case ::tt::target::ttnn::OpType::ReshapeOp: {
    return operations::data_movement::run(op->type_as_ReshapeOp(), context);
  }
  case ::tt::target::ttnn::OpType::SliceOp: {
    return operations::data_movement::run(op->type_as_SliceOp(), context);
  }
  case ::tt::target::ttnn::OpType::Conv2dOp: {
    return operations::conv::run(op->type_as_Conv2dOp(), context);
  }
  case ::tt::target::ttnn::OpType::DeallocateOp: {
    return operations::deletion::run(op->type_as_DeallocateOp(), context);
  }
  case ::tt::target::ttnn::OpType::MaxPool2dOp: {
    return operations::pool::run(op->type_as_MaxPool2dOp(), context);
  }
  case ::tt::target::ttnn::OpType::AllGatherOp: {
    return operations::ccl::run(op->type_as_AllGatherOp(), context);
  }
  default: {
    LOG_FATAL("Unsupported operation type");
  }
  }
}

// Nop is single input, output tensor where input is returned as output.
static bool handleNopProgram(::tt::target::ttnn::Program const *program,
                             std::vector<::ttnn::Tensor *> const &inputs,
                             std::vector<::ttnn::Tensor *> const &outputs) {

  bool isNop = program->inputs()->size() == 1 &&
               program->outputs()->size() == 1 &&
               program->inputs()->Get(0)->global_id() ==
                   program->outputs()->Get(0)->global_id();

  if (isNop) {
    void *src = ::tt::tt_metal::get_raw_host_data_ptr(*inputs.at(0));
    void *dst = ::tt::tt_metal::get_raw_host_data_ptr(*outputs.at(0));
    std::uint32_t size = outputs[0]->volume() * outputs[0]->element_size();
    std::memcpy(dst, src, size);
  }
  return isNop;
}

void runProgram(::ttnn::MeshDevice &meshDevice, Binary &executableHandle,
                std::uint32_t programIndex,
                std::vector<::ttnn::Tensor *> const &inputs,
                std::vector<::ttnn::Tensor *> const &outputs) {
  ::tt::target::ttnn::TTNNBinary const &fbb = *getBinary(executableHandle);
  ::tt::target::ttnn::Program const *program =
      fbb.programs()->Get(programIndex);
  if (handleNopProgram(program, inputs, outputs)) {
    return;
  }
  TensorMap liveTensors;
  std::unordered_set<uint32_t> programInputs;
  int inputIndex = 0;
  LOG_ASSERT(program->inputs()->size() == inputs.size(),
             "Program input size mismatch: ", program->inputs()->size(),
             " != ", inputs.size());
  for (::tt::target::TensorRef const *input : *program->inputs()) {
    auto [iter, inserted] =
        liveTensors.try_emplace(input->global_id(), inputs[inputIndex++]);
    LOG_ASSERT(inserted, "Duplicate input tensor");
    programInputs.emplace(input->global_id());
  }

  int outputIndex = 0;
  std::unordered_set<uint32_t> programOutputs;
  LOG_ASSERT(program->outputs()->size() == outputs.size());
  for (::tt::target::TensorRef const *output : *program->outputs()) {
    auto [iter, inserted] =
        liveTensors.try_emplace(output->global_id(), outputs[outputIndex++]);
    LOG_ASSERT(inserted, "Duplicate output tensor");
    programOutputs.emplace(output->global_id());
  }
  ProgramExecutor executor(executableHandle, liveTensors, programInputs,
                           programOutputs, &meshDevice);
  executor.execute(program);
}

} // namespace tt::runtime::ttnn
