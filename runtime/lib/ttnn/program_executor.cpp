// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttnn/program_executor.h"

#include "operations/cache/load_cached.h"
#include "operations/ccl/all_gather.h"
#include "operations/ccl/collective_permute.h"
#include "operations/ccl/mesh_shard.h"
#include "operations/ccl/reduce_scatter.h"
#include "operations/context/get_device.h"
#include "operations/conv/conv2d.h"
#include "operations/conv/conv_transpose2d.h"
#include "operations/conv/prepare_conv2d_weights.h"
#include "operations/cpu/cpu.h"
#include "operations/creation/arange.h"
#include "operations/creation/constant.h"
#include "operations/creation/empty.h"
#include "operations/creation/full.h"
#include "operations/creation/full_with.h"
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
#include "operations/eltwise/quantization/quantization.h"
#include "operations/eltwise/ternary/where.h"
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
#include "operations/pool/pool2d.h"
#include "operations/pool/upsample.h"
#include "operations/reduction/argmax.h"
#include "operations/reduction/prod.h"
#include "operations/reduction/reduction.h"
#include "tt/runtime/detail/debug.h"
#include "tt/runtime/detail/ttnn/types.h"
#include "tt/runtime/utils.h"

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

ProgramExecutor::ProgramExecutor(
    const ::tt::target::ttnn::Program *program, const Binary &executableHandle,
    std::vector<::tt::runtime::Tensor> &programInputs,
    std::shared_ptr<::ttnn::MeshDevice> meshDevice, const size_t programIndex)
    : program(program), executableHandle(executableHandle) {
  LOG_ASSERT(program, "Program must be provided for execution");

  std::vector<uint32_t> programInputIds;
  int inputIndex = 0;
  TensorPtrMap liveTensors;
  LOG_ASSERT(program->inputs()->size() == programInputs.size(),
             "Program input size mismatch: ", program->inputs()->size(),
             " != ", programInputs.size());
  for (const ::tt::target::ttnn::TensorRef *input : *program->inputs()) {
    auto [iter, inserted] = liveTensors.try_emplace(
        input->global_id(), &(programInputs[inputIndex++]));
    LOG_ASSERT(inserted, "Duplicate input tensor");
    programInputIds.push_back(input->global_id());
  }

  std::vector<uint32_t> programOutputIds;
  for (const ::tt::target::ttnn::TensorRef *output : *program->outputs()) {
    programOutputIds.push_back(output->global_id());
  }

  context = std::make_unique<ProgramContext>(
      programInputIds, programOutputIds, std::move(liveTensors),
      common::DylibManager(program->dylibs()), std::move(meshDevice),
      executableHandle, programIndex);
}

void ProgramExecutor::runCallback(
    std::optional<debug::Hooks::CallbackFn> callback, Binary &executableHandle,
    const ::tt::target::ttnn::Operation *opContext,
    ProgramContext *programContext) {
  if (callback) {
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

void ProgramExecutor::execute() {
  LOG_DEBUG(LogType::LogRuntimeTTNN,
            "Starting execution of program: ", program->name()->c_str());
  for (const ::tt::target::ttnn::Operation *op : *program->operations()) {
    LOG_DEBUG(LogType::LogRuntimeTTNN,
              "Executing operation: ", op->debug_info()->c_str());
    tracyLogOpLocation(op);
    runCallback(debug::Hooks::get().getPreOperatorCallback(), executableHandle,
                op, context.get());
    runOperation(op);
    runCallback(debug::Hooks::get().getPostOperatorCallback(), executableHandle,
                op, context.get());
    dumpPerfCountersIfNeeded(context->getMeshDevice());
  }
  LOG_DEBUG(LogType::LogRuntimeTTNN,
            "Finished execution of program: ", program->name()->c_str());
}

std::vector<::tt::runtime::Tensor> ProgramExecutor::gatherOutputTensors() {
  return context->getTensorPool().gatherOutputTensors();
}

void ProgramExecutor::dumpPerfCountersIfNeeded(::ttnn::MeshDevice &meshDevice) {
#if defined(TT_RUNTIME_ENABLE_PERF_TRACE)
  static uint32_t counter = 0;
  if (counter++ >= debug::PerfEnv::get().dumpDeviceRate) {
    LOG_DEBUG(LogType::LogRuntimeTTNN, "Dumping device profile results after " +
                                           std::to_string(counter) +
                                           " operations");
    for (::ttnn::IDevice *ttnnDevice : meshDevice.get_devices()) {
      ::tt::tt_metal::detail::DumpDeviceProfileResults(ttnnDevice);
    }
    counter = 0;
  }
#endif
}

void ProgramExecutor::runOperation(const ::tt::target::ttnn::Operation *op) {
  switch (op->type_type()) {
  case ::tt::target::ttnn::OpType::GetDeviceOp: {
    return operations::context::run(op->type_as_GetDeviceOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::ToMemoryConfigOp: {
    return operations::layout::run(op->type_as_ToMemoryConfigOp(),
                                   getContext());
  }
  case ::tt::target::ttnn::OpType::ToLayoutOp: {
    return operations::layout::run(op->type_as_ToLayoutOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::ToDTypeOp: {
    return operations::layout::run(op->type_as_ToDTypeOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::TypecastOp: {
    return operations::layout::run(op->type_as_TypecastOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::ToDeviceOp: {
    return operations::layout::run(op->type_as_ToDeviceOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::FromDeviceOp: {
    return operations::layout::run(op->type_as_FromDeviceOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::EmptyOp: {
    return operations::creation::run(op->type_as_EmptyOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::NamedFullOp: {
    return operations::creation::run(op->type_as_NamedFullOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::FullOp: {
    return operations::creation::run(op->type_as_FullOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::EltwiseBinaryOp: {
    return operations::eltwise::binary::run(op->type_as_EltwiseBinaryOp(),
                                            getContext());
  }
  case ::tt::target::ttnn::OpType::EltwiseBinaryCompositeOp: {
    return operations::eltwise::binary::run(
        op->type_as_EltwiseBinaryCompositeOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::EltwiseTernaryWhereOp: {
    return operations::eltwise::ternary::run(
        op->type_as_EltwiseTernaryWhereOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::EltwiseQuantizationOp: {
    return operations::eltwise::quantization::run(
        op->type_as_EltwiseQuantizationOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::EltwiseUnaryOp: {
    return operations::eltwise::unary::run(op->type_as_EltwiseUnaryOp(),
                                           getContext());
  }
  case ::tt::target::ttnn::OpType::EltwiseUnaryCompositeOp: {
    return operations::eltwise::unary::run(
        op->type_as_EltwiseUnaryCompositeOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::LinearOp: {
    return operations::matmul::run(op->type_as_LinearOp(), getContext());
  }
  // ANCHOR: adding_an_op_matmul_runtime_program
  case ::tt::target::ttnn::OpType::MatmulOp: {
    return operations::matmul::run(op->type_as_MatmulOp(), getContext());
  }
  // ANCHOR_END: adding_an_op_matmul_runtime_program
  case ::tt::target::ttnn::OpType::MorehCumSumOp: {
    return operations::moreh::run(op->type_as_MorehCumSumOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::ReductionArgMaxOp: {
    return operations::reduction::run(op->type_as_ReductionArgMaxOp(),
                                      getContext());
  }
  case ::tt::target::ttnn::OpType::ReductionProdOp: {
    return operations::reduction::run(op->type_as_ReductionProdOp(),
                                      getContext());
  }
  case ::tt::target::ttnn::OpType::ReductionOp: {
    return operations::reduction::run(op->type_as_ReductionOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::EmbeddingOp: {
    return operations::embedding::run(op->type_as_EmbeddingOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::EmbeddingBackwardOp: {
    return operations::embedding_backward::run(
        op->type_as_EmbeddingBackwardOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::SoftmaxOp: {
    return operations::normalization::run(op->type_as_SoftmaxOp(),
                                          getContext());
  }
  case ::tt::target::ttnn::OpType::TransposeOp: {
    return operations::data_movement::run(op->type_as_TransposeOp(),
                                          getContext());
  }
  case ::tt::target::ttnn::OpType::PadOp: {
    return operations::data_movement::run(op->type_as_PadOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::ConcatOp: {
    return operations::data_movement::run(op->type_as_ConcatOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::PermuteOp: {
    return operations::data_movement::run(op->type_as_PermuteOp(),
                                          getContext());
  }
  case ::tt::target::ttnn::OpType::ReshapeOp: {
    return operations::data_movement::run(op->type_as_ReshapeOp(),
                                          getContext());
  }
  case ::tt::target::ttnn::OpType::SliceOp: {
    return operations::data_movement::run(op->type_as_SliceOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::RepeatOp: {
    return operations::data_movement::run(op->type_as_RepeatOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::RepeatInterleaveOp: {
    return operations::data_movement::run(op->type_as_RepeatInterleaveOp(),
                                          getContext());
  }
  case ::tt::target::ttnn::OpType::PrepareConv2dWeightsOp: {
    return operations::conv::run(op->type_as_PrepareConv2dWeightsOp(),
                                 getContext());
  }
  case ::tt::target::ttnn::OpType::Conv2dOp: {
    return operations::conv::run(op->type_as_Conv2dOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::ConvTranspose2dOp: {
    return operations::conv::run(op->type_as_ConvTranspose2dOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::DeallocateOp: {
    return operations::deletion::run(op->type_as_DeallocateOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::Pool2dOp: {
    return operations::pool::run(op->type_as_Pool2dOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::AllGatherOp: {
    return operations::ccl::run(op->type_as_AllGatherOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::ReduceScatterOp: {
    return operations::ccl::run(op->type_as_ReduceScatterOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::CollectivePermuteOp: {
    return operations::ccl::run(op->type_as_CollectivePermuteOp(),
                                getContext());
  }
  case ::tt::target::ttnn::OpType::MeshShardOp: {
    return operations::ccl::run(op->type_as_MeshShardOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::ArangeOp: {
    return operations::creation::run(op->type_as_ArangeOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::UpdateCacheOp: {
    return operations::kv_cache::run(op->type_as_UpdateCacheOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::FillCacheOp: {
    return operations::kv_cache::run(op->type_as_FillCacheOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::UpsampleOp: {
    return operations::pool::run(op->type_as_UpsampleOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::CpuOp: {
    return operations::cpu::run(op->type_as_CpuOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::ConstantOp: {
    return operations::creation::run(op->type_as_ConstantOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::LoadCachedOp: {
    return operations::cache::run(op->type_as_LoadCachedOp(), getContext());
  }
  default: {
    LOG_FATAL("Unsupported operation type: ",
              ::tt::target::ttnn::EnumNameOpType(op->type_type()));
  }
  }
}

} // namespace tt::runtime::ttnn
