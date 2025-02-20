// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "operations/ccl/all_gather.h"
#include "operations/ccl/mesh_shard.h"
#include "operations/ccl/reduce_scatter.h"
#include "operations/context/get_device.h"
#include "operations/conv/conv2d.h"
#include "operations/conv/conv_transpose2d.h"
#include "operations/creation/arange.h"
#include "operations/creation/empty.h"
#include "operations/creation/full.h"
#include "operations/creation/ones.h"
#include "operations/creation/zeros.h"
#include "operations/data_movement/concat.h"
#include "operations/data_movement/pad.h"
#include "operations/data_movement/permute.h"
#include "operations/data_movement/repeat.h"
#include "operations/data_movement/repeat_interleave.h"
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
#include "operations/embedding/embedding_backward.h"
#include "operations/kv_cache/fill_cache.h"
#include "operations/kv_cache/update_cache.h"
#include "operations/layout/from_device.h"
#include "operations/layout/to_device.h"
#include "operations/layout/to_dtype.h"
#include "operations/layout/to_layout.h"
#include "operations/layout/to_memory_config.h"
#include "operations/layout/typecast.h"
#include "operations/matmul/matmul.h"
#include "operations/moreh/moreh_cumsum.h"
#include "operations/normalization/softmax.h"
#include "operations/pool/maxpool2d.h"
#include "operations/pool/upsample.h"
#include "operations/reduction/prod.h"
#include "operations/reduction/reduction.h"
#include "tt/runtime/detail/debug.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/ttnn/types.h"
#include "tt/runtime/ttnn/utils.h"
#include "tt/runtime/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"

#ifdef TT_RUNTIME_ENABLE_PERF_TRACE
#include "tracy/Tracy.hpp"
#endif

namespace tt::runtime::ttnn {
using LogType = ::tt::runtime::logger::LogType;

static void tracyLogOpLocation(const ::tt::target::ttnn::Operation *op) {
#ifdef TT_RUNTIME_ENABLE_PERF_TRACE
  TracyMessage(op->loc_info()->c_str(), op->loc_info()->size());
#endif
}

static ::tt::target::ttnn::TTNNBinary const *getBinary(Flatbuffer binary) {
  bool isTTNN = ::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
      binary.handle.get());
  LOG_ASSERT(isTTNN, "Unsupported binary format");
  return ::tt::target::ttnn::GetSizePrefixedTTNNBinary(binary.handle.get());
}

class ProgramExecutor {
public:
  ProgramExecutor(
      const Binary &executableHandle,
      const std::unordered_map<uint32_t, ::ttnn::Tensor *> &liveTensors,
      const std::vector<uint32_t> &programInputs,
      const std::vector<uint32_t> &programOutputs,
      ::ttnn::MeshDevice *meshDevice)
      : executableHandle(executableHandle),
        context(ProgramContext(liveTensors, programInputs, programOutputs,
                               meshDevice)) {}

  void runCallback(Binary &executableHandle,
                   const ::tt::target::ttnn::Operation *opContext,
                   ProgramContext *programContext);

  void execute(const ::tt::target::ttnn::Program *program) {
    for (const ::tt::target::ttnn::Operation *op : *program->operations()) {
      LOG_INFO(LogType::LogRuntimeTTNN,
                "Executing operation: ", op->debug_info()->c_str());
      tracyLogOpLocation(op);
      runOperation(op);
      runCallback(executableHandle, op, &context);
    }
  }

  ProgramContext &getContext() { return context; }
  std::vector<Tensor> gatherOutputTensors() {
    return context.getTensorPool().gatherOutputTensors();
  }

private:
  Binary executableHandle;
  ProgramContext context;
  void runOperation(const ::tt::target::ttnn::Operation *op);
  void runEltwiseOperation(const ::tt::target::ttnn::EltwiseOp *op);
};

void ProgramExecutor::runCallback(
    Binary &executableHandle, const ::tt::target::ttnn::Operation *opContext,
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
  case ::tt::target::ttnn::OpType::ToDTypeOp: {
    return operations::layout::run(op->type_as_ToDTypeOp(), context);
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
  case ::tt::target::ttnn::OpType::ZerosOp: {
    return operations::creation::run(op->type_as_ZerosOp(), context);
  }
  case ::tt::target::ttnn::OpType::OnesOp: {
    return operations::creation::run(op->type_as_OnesOp(), context);
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
  case ::tt::target::ttnn::OpType::MorehCumSumOp: {
    return operations::moreh::run(op->type_as_MorehCumSumOp(), context);
  }
  case ::tt::target::ttnn::OpType::ReductionProdOp: {
    return operations::reduction::run(op->type_as_ReductionProdOp(), context);
  }
  case ::tt::target::ttnn::OpType::ReductionOp: {
    return operations::reduction::run(op->type_as_ReductionOp(), context);
  }
  case ::tt::target::ttnn::OpType::EmbeddingOp: {
    return operations::embedding::run(op->type_as_EmbeddingOp(), context);
  }
  case ::tt::target::ttnn::OpType::EmbeddingBackwardOp: {
    return operations::embedding_backward::run(
        op->type_as_EmbeddingBackwardOp(), context);
  }
  case ::tt::target::ttnn::OpType::SoftmaxOp: {
    return operations::normalization::run(op->type_as_SoftmaxOp(), context);
  }
  case ::tt::target::ttnn::OpType::TransposeOp: {
    return operations::data_movement::run(op->type_as_TransposeOp(), context);
  }
  case ::tt::target::ttnn::OpType::PadOp: {
    return operations::data_movement::run(op->type_as_PadOp(), context);
  }
  case ::tt::target::ttnn::OpType::ConcatOp: {
    return operations::data_movement::run(op->type_as_ConcatOp(), context);
  }
  case ::tt::target::ttnn::OpType::PermuteOp: {
    return operations::data_movement::run(op->type_as_PermuteOp(), context);
  }
  case ::tt::target::ttnn::OpType::ReshapeOp: {
    return operations::data_movement::run(op->type_as_ReshapeOp(), context);
  }
  case ::tt::target::ttnn::OpType::SliceOp: {
    return operations::data_movement::run(op->type_as_SliceOp(), context);
  }
  case ::tt::target::ttnn::OpType::RepeatOp: {
    return operations::data_movement::run(op->type_as_RepeatOp(), context);
  }
  case ::tt::target::ttnn::OpType::RepeatInterleaveOp: {
    return operations::data_movement::run(op->type_as_RepeatInterleaveOp(),
                                          context);
  }
  case ::tt::target::ttnn::OpType::Conv2dOp: {
    return operations::conv::run(op->type_as_Conv2dOp(), context);
  }
  case ::tt::target::ttnn::OpType::ConvTranspose2dOp: {
    return operations::conv::run(op->type_as_ConvTranspose2dOp(), context);
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
  case ::tt::target::ttnn::OpType::ReduceScatterOp: {
    return operations::ccl::run(op->type_as_ReduceScatterOp(), context);
  }
  case ::tt::target::ttnn::OpType::MeshShardOp: {
    return operations::ccl::run(op->type_as_MeshShardOp(), context);
  }
  case ::tt::target::ttnn::OpType::ArangeOp: {
    return operations::creation::run(op->type_as_ArangeOp(), context);
  }
  case ::tt::target::ttnn::OpType::UpdateCacheOp: {
    return operations::kv_cache::run(op->type_as_UpdateCacheOp(), context);
  }
  case ::tt::target::ttnn::OpType::FillCacheOp: {
    return operations::kv_cache::run(op->type_as_FillCacheOp(), context);
  }
  case ::tt::target::ttnn::OpType::UpsampleOp: {
    return operations::pool::run(op->type_as_UpsampleOp(), context);
  }
  default: {
    LOG_FATAL("Unsupported operation type");
  }
  }
}

std::vector<Tensor> runProgram(::ttnn::MeshDevice &meshDevice,
                               Binary executableHandle,
                               std::uint32_t programIndex,
                               std::vector<::ttnn::Tensor *> const &inputs) {
  ::tt::target::ttnn::TTNNBinary const &fbb = *getBinary(executableHandle);
  ::tt::target::ttnn::Program const *program =
      fbb.programs()->Get(programIndex);
  std::unordered_map<uint32_t, ::ttnn::Tensor *> liveTensors;
  std::vector<uint32_t> programInputs;
  int inputIndex = 0;
  LOG_ASSERT(program->inputs()->size() == inputs.size(),
             "Program input size mismatch: ", program->inputs()->size(),
             " != ", inputs.size());
  for (::tt::target::TensorRef const *input : *program->inputs()) {
    auto [iter, inserted] =
        liveTensors.try_emplace(input->global_id(), inputs[inputIndex++]);
    LOG_ASSERT(inserted, "Duplicate input tensor");
    programInputs.push_back(input->global_id());
  }
  std::vector<uint32_t> programOutputs;
  for (::tt::target::TensorRef const *output : *program->outputs()) {
    programOutputs.push_back(output->global_id());
  }
  ProgramExecutor executor(executableHandle, liveTensors, programInputs,
                           programOutputs, &meshDevice);
  executor.execute(program);
  std::vector<Tensor> outputTensors = executor.gatherOutputTensors();
  return outputTensors;
}

} // namespace tt::runtime::ttnn
