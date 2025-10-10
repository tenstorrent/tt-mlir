// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttnn/program_executor.h"

#include "operations/cache/load_cached.h"
#include "operations/ccl/all_gather.h"
#include "operations/ccl/collective_permute.h"
#include "operations/ccl/mesh_shard.h"
#include "operations/ccl/point_to_point.h"
#include "operations/ccl/reduce_scatter.h"
#include "operations/context/get_device.h"
#include "operations/conv/conv2d.h"
#include "operations/conv/conv_transpose2d.h"
#include "operations/conv/prepare_conv2d_bias.h"
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
#include "operations/data_movement/sort.h"
#include "operations/data_movement/transpose.h"
#include "operations/data_movement/write_tensor.h"
#include "operations/deletion/deallocate.h"
#include "operations/eltwise/binary/binary.h"
#include "operations/eltwise/binary/binary_composite.h"
#include "operations/eltwise/quantization/quantization.h"
#include "operations/eltwise/ternary/where.h"
#include "operations/eltwise/unary/unary.h"
#include "operations/eltwise/unary/unary_composite.h"
#include "operations/embedding/embedding.h"
#include "operations/embedding/embedding_backward.h"
#include "operations/generic/generic_op.h"
#include "operations/kv_cache/fill_cache.h"
#include "operations/kv_cache/update_cache.h"
#include "operations/layout/from_device.h"
#include "operations/layout/to_device.h"
#include "operations/layout/to_dtype.h"
#include "operations/layout/to_layout.h"
#include "operations/layout/to_memory_config.h"
#include "operations/layout/typecast.h"
#include "operations/matmul/matmul.h"
#include "operations/mlir_native/func_call.h"
#include "operations/moreh/moreh_cumsum.h"
#include "operations/normalization/batch_norm.h"
#include "operations/normalization/rms_norm.h"
#include "operations/normalization/softmax.h"
#include "operations/pool/pool2d.h"
#include "operations/pool/upsample.h"
#include "operations/rand/rand.h"
#include "operations/reduction/argmax.h"
#include "operations/reduction/prod.h"
#include "operations/reduction/reduction.h"
#include "operations/tensor_serialization/dump_tensor.h"
#include "operations/tensor_serialization/load_tensor.h"
#include "operations/trace/begin_trace_capture.h"
#include "operations/trace/capture_or_execute_trace.h"
#include "operations/trace/end_trace_capture.h"
#include "operations/trace/execute_trace.h"
#include "operations/transformer/concatenate_heads.h"
#include "operations/transformer/nlp_concat_heads.h"
#include "operations/transformer/nlp_concat_heads_decode.h"
#include "operations/transformer/nlp_create_qkv_heads.h"
#include "operations/transformer/nlp_create_qkv_heads_decode.h"
#include "operations/transformer/rotary_embedding_llama.h"
#include "operations/transformer/scaled_dot_product_attention.h"
#include "operations/transformer/scaled_dot_product_attention_decode.h"
#include "tt/runtime/debug.h"
#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/perf.h"
#include "tt/runtime/utils.h"

namespace tt::runtime::ttnn {

using LogType = ::tt::runtime::logger::LogType;

ProgramExecutor::ProgramExecutor(
    ::tt::runtime::Device deviceHandle, ::tt::runtime::Binary &executableHandle,
    const size_t programIndex,
    std::vector<::tt::runtime::Tensor> &programInputs, bool constEvalProgram)
    : program(utils::getProgram(executableHandle, programIndex)),
      executableHandle(executableHandle), constEvalProgram(constEvalProgram) {
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
      common::DylibManager(program->dylibs()), std::move(deviceHandle),
      executableHandle, programIndex);
}

void ProgramExecutor::runCallback(
    std::optional<debug::Hooks::CallbackFn> callback, Binary &executableHandle,
    const ::tt::target::ttnn::Operation *opContext,
    ProgramContext *programContext) {
  if (callback) {
    std::shared_ptr<void> programContextPtr =
        ::tt::runtime::utils::unsafeBorrowShared(programContext);
    std::shared_ptr<void> opContextPtr =
        ::tt::runtime::utils::unsafeBorrowShared(
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
    perf::Env::get().tracyLogOpLocation(std::string(op->loc_info()->c_str()));
    perf::Env::get().tracyLogConstEvalProgram(constEvalProgram);
    perf::Env::get().tracyLogProgramMetadata(
        perf::Env::get().tracyProgramMetadata);
    runCallback(debug::Hooks::get().getPreOperatorCallback(), executableHandle,
                op, context.get());
    runOperation(op);
    runCallback(debug::Hooks::get().getPostOperatorCallback(), executableHandle,
                op, context.get());
    dumpPerfCountersIfNeeded();
  }
  LOG_DEBUG(LogType::LogRuntimeTTNN,
            "Finished execution of program: ", program->name()->c_str());
}

std::vector<::tt::runtime::Tensor> ProgramExecutor::gatherOutputTensors() {
  return context->getTensorPool().gatherOutputTensors();
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
  case ::tt::target::ttnn::OpType::EltwiseBinaryCompositeScalarOp: {
    return operations::eltwise::binary::run(
        op->type_as_EltwiseBinaryCompositeScalarOp(), getContext());
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
  case ::tt::target::ttnn::OpType::FuncCallOp: {
    return operations::mlir_native::run(op->type_as_FuncCallOp(), getContext());
  }
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
  case ::tt::target::ttnn::OpType::ConcatenateHeadsOp: {
    return operations::transformer::run(op->type_as_ConcatenateHeadsOp(),
                                        getContext());
  }
  case ::tt::target::ttnn::OpType::RotaryEmbeddingLlamaOp: {
    return operations::transformer::run(op->type_as_RotaryEmbeddingLlamaOp(),
                                        getContext());
  }
  case ::tt::target::ttnn::OpType::NLPCreateQKVHeadsDecodeOp: {
    return operations::transformer::run(op->type_as_NLPCreateQKVHeadsDecodeOp(),
                                        getContext());
  }
  case ::tt::target::ttnn::OpType::NLPConcatHeadsOp: {
    return operations::transformer::run(op->type_as_NLPConcatHeadsOp(),
                                        getContext());
  }
  case ::tt::target::ttnn::OpType::NLPConcatHeadsDecodeOp: {
    return operations::transformer::run(op->type_as_NLPConcatHeadsDecodeOp(),
                                        getContext());
  }
  case ::tt::target::ttnn::OpType::NLPCreateQKVHeadsOp: {
    return operations::transformer::run(op->type_as_NLPCreateQKVHeadsOp(),
                                        getContext());
  }
  case ::tt::target::ttnn::OpType::WriteTensorOp: {
    return operations::data_movement::run(op->type_as_WriteTensorOp(),
                                          getContext());
  }
  case ::tt::target::ttnn::OpType::PermuteOp: {
    return operations::data_movement::run(op->type_as_PermuteOp(),
                                          getContext());
  }
  case ::tt::target::ttnn::OpType::RandOp: {
    return operations::rand::run(op->type_as_RandOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::ReshapeOp: {
    return operations::data_movement::run(op->type_as_ReshapeOp(),
                                          getContext());
  }
  case ::tt::target::ttnn::OpType::SliceOp: {
    return operations::data_movement::run(op->type_as_SliceOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::SortOp: {
    return operations::data_movement::run(op->type_as_SortOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::RepeatOp: {
    return operations::data_movement::run(op->type_as_RepeatOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::RMSNormOp: {
    return operations::rms_norm::run(op->type_as_RMSNormOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::RepeatInterleaveOp: {
    return operations::data_movement::run(op->type_as_RepeatInterleaveOp(),
                                          getContext());
  }
  case ::tt::target::ttnn::OpType::PrepareConv2dWeightsOp: {
    return operations::conv::run(op->type_as_PrepareConv2dWeightsOp(),
                                 getContext());
  }
  case ::tt::target::ttnn::OpType::PrepareConv2dBiasOp: {
    return operations::conv::run(op->type_as_PrepareConv2dBiasOp(),
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
  case ::tt::target::ttnn::OpType::GlobalAvgPool2dOp: {
    return operations::pool::run(op->type_as_GlobalAvgPool2dOp(), getContext());
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
  case ::tt::target::ttnn::OpType::BatchNormOp: {
    return operations::batch_norm::run(op->type_as_BatchNormOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::DumpTensorOp: {
    return operations::tensor_serialization::run(op->type_as_DumpTensorOp(),
                                                 getContext());
  }
  case ::tt::target::ttnn::OpType::LoadTensorOp: {
    return operations::tensor_serialization::run(op->type_as_LoadTensorOp(),
                                                 getContext());
  }
  case ::tt::target::ttnn::OpType::BeginTraceCaptureOp: {
    return operations::trace::run(op->type_as_BeginTraceCaptureOp(),
                                  getContext());
  }
  case ::tt::target::ttnn::OpType::EndTraceCaptureOp: {
    return operations::trace::run(op->type_as_EndTraceCaptureOp(),
                                  getContext());
  }
  case ::tt::target::ttnn::OpType::ExecuteTraceOp: {
    return operations::trace::run(op->type_as_ExecuteTraceOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::CaptureOrExecuteTraceOp: {
    return operations::trace::run(op->type_as_CaptureOrExecuteTraceOp(),
                                  getContext());
  }
  case ::tt::target::ttnn::OpType::PointToPointOp: {
    return operations::ccl::run(op->type_as_PointToPointOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::GenericOp: {
    return operations::generic_op::run(op->type_as_GenericOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::ScaledDotProductAttentionDecodeOp: {
    return operations::transformer::run(
        op->type_as_ScaledDotProductAttentionDecodeOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::ScaledDotProductAttentionOp: {
    return operations::transformer::run(
        op->type_as_ScaledDotProductAttentionOp(), getContext());
  }
  case ::tt::target::ttnn::OpType::NONE: {
    LOG_FATAL("Unsupported operation type: ",
              ::tt::target::ttnn::EnumNameOpType(op->type_type()));
  }
  }

  LOG_FATAL("Unreachable code path, all operations should be handled in switch "
            "statement");
}

void ProgramExecutor::dumpPerfCountersIfNeeded() {
#if defined(TT_RUNTIME_ENABLE_PERF_TRACE) && TT_RUNTIME_ENABLE_PERF_TRACE == 1
  static uint32_t counter = 0;
  if (++counter >= perf::Env::get().dumpDeviceRate) {
    LOG_DEBUG(LogType::LogRuntimeTTNN, "Dumping device profile results after " +
                                           std::to_string(counter) +
                                           " operations");
    ::tt::tt_metal::ReadMeshDeviceProfilerResults(context->getMeshDevice());
    counter = 0;
  }
#endif
}

} // namespace tt::runtime::ttnn
