// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/trace/capture_or_execute_trace.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/program_executor.h"
#include "tt/runtime/detail/ttnn/types/trace_cache.h"
#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/types.h"
#include "tt/runtime/workarounds.h"

namespace tt::runtime::ttnn::operations::trace {

static std::pair<MainProgramKey, CaptureExecuteProgramKey>
getTraceCacheKeys(const ::tt::target::ttnn::CaptureOrExecuteTraceOp *op,
                  ProgramContext &context) {
  return {MainProgramKey(context.getExecutableHandle().id(),
                         context.getProgramIndex()),
          CaptureExecuteProgramKey(op->capture_program_id(),
                                   op->execute_program_id())};
}

static void copyTensor(const ::tt::target::ttnn::TensorRef *srcTensorDesc,
                       const ::ttnn::Tensor &srcTensor,
                       ::ttnn::Tensor &dstTensor) {

  if (::tt::runtime::ttnn::utils::inSystemMemory(srcTensorDesc)) {
    ::tt::tt_metal::write_tensor(srcTensor, dstTensor, /*blocking=*/false);
    return;
  }

  LOG_ASSERT(::tt::runtime::workaround::Env::get().traceImplicitFromDevice,
             "traceImplicitFromDevice workaround must be enabled.");
  ::ttnn::Tensor hostSrcTensor = ::ttnn::from_device(srcTensor);
  ::tt::tt_metal::write_tensor(hostSrcTensor, dstTensor, /*blocking=*/false);
}

static void runTraceProgramAndCaptureTrace(
    const ::tt::target::ttnn::CaptureOrExecuteTraceOp *op,
    ProgramContext &context, ::tt::runtime::ttnn::TraceCache &traceCache) {

  ProgramTensorPool &tensorPool = context.getTensorPool();
  ::tt::runtime::Device deviceHandle = context.getDeviceHandle();

  std::vector<::tt::runtime::Tensor> inputTensors;
  for (const ::tt::target::ttnn::TensorRef *input : *op->inputs()) {
    ::tt::runtime::Tensor inputTensor =
        tensorPool.getRuntimeTensorAndValidate(input);
    inputTensors.push_back(inputTensor);
  }

  ProgramExecutor executor(deviceHandle, context.getExecutableHandle(),
                           op->capture_program_id(), inputTensors,
                           /*constEvalProgram=*/false);
  executor.execute();
  std::vector<::tt::runtime::Tensor> outputTensors =
      executor.gatherOutputTensors();

  // Outputs will be returned in the order of traceId, actual outputs, trace
  // input slots, trace output slots
  size_t expectedNumOutputs =
      1 + op->outputs()->size() + op->inputs()->size() + op->outputs()->size();
  LOG_ASSERT(outputTensors.size() == expectedNumOutputs,
             "Mismatched number of output tensors, expected: ",
             expectedNumOutputs, " got: ", outputTensors.size());
  size_t currOutputIndex = 0;

  // Handle trace id
  const ::ttnn::Tensor &traceIdTensor =
      ::tt::runtime::ttnn::utils::getTTNNTensorFromRuntimeTensor(
          outputTensors[currOutputIndex++]);
  LOG_ASSERT(traceIdTensor.dtype() == ::ttnn::DataType::UINT32,
             "Trace ID must be UINT32");

  uint32_t traceId =
      ::tt::runtime::ttnn::utils::getScalarFromTensor<uint32_t>(traceIdTensor);
  ::ttnn::MeshTraceId meshTraceId(traceId);

  // Handle trace function outputs
  for (size_t i = 0; i < op->outputs()->size(); i++) {
    tensorPool.insertRuntimeTensorAndValidate(op->outputs()->Get(i),
                                              outputTensors[currOutputIndex++]);
  }

  // Handle trace input slots
  std::vector<::tt::runtime::Tensor> inputSlots;
  for (size_t i = 0; i < op->inputs()->size(); i++) {
    ::tt::runtime::Tensor &inputSlot = outputTensors[currOutputIndex++];
    ::tt::runtime::ttnn::TTNNTensorWrapper &inputSlotWrapper =
        inputSlot.as<::tt::runtime::ttnn::TTNNTensorWrapper>(
            DeviceRuntime::TTNN);

    // input slots need to be retained
    inputSlotWrapper.setRetain(true);
    inputSlots.emplace_back(std::move(inputSlot));
  }

  // Handle trace output slots
  std::vector<::tt::runtime::Tensor> outputSlots;
  for (size_t i = 0; i < op->outputs()->size(); i++) {
    ::tt::runtime::Tensor &outputSlot = outputTensors[currOutputIndex++];
    ::tt::runtime::ttnn::TTNNTensorWrapper &outputSlotWrapper =
        outputSlot.as<::tt::runtime::ttnn::TTNNTensorWrapper>(
            DeviceRuntime::TTNN);

    // output slots need to be retained
    outputSlotWrapper.setRetain(true);
    outputSlots.emplace_back(std::move(outputSlot));
  }

  TraceData traceData{.traceId = meshTraceId,
                      .inputTensors = inputSlots,
                      .outputTensors = outputSlots};
  auto [mainProgramKey, captureExecuteKey] = getTraceCacheKeys(op, context);
  traceCache.insert(mainProgramKey, captureExecuteKey, traceData);
}

static void executeTrace(const ::tt::target::ttnn::CaptureOrExecuteTraceOp *op,
                         ProgramContext &context, TraceData &traceData) {
  ::tt::runtime::Device deviceHandle = context.getDeviceHandle();

  std::vector<::ttnn::Tensor> inputs;
  LOG_ASSERT(op->inputs()->size() == traceData.inputTensors.size(),
             "Mismatched number of inputs, expected: ", op->inputs()->size(),
             " got: ", traceData.inputTensors.size());
  LOG_ASSERT(op->outputs()->size() == traceData.outputTensors.size(),
             "Mismatched number of outputs, expected: ", op->outputs()->size(),
             " got: ", traceData.outputTensors.size());

  for (size_t i = 0; i < op->inputs()->size(); i++) {
    const ::tt::target::ttnn::TensorRef *input = op->inputs()->Get(i);
    const ::tt::runtime::ttnn::TTNNTensorWrapper &inputTensorWrapper =
        context.getTensorPool().getTTNNTensorWrapperAndValidate(input);

    ::tt::runtime::ttnn::TTNNTensorWrapper &inputSlotWrapper =
        traceData.inputTensors[i].as<::tt::runtime::ttnn::TTNNTensorWrapper>(
            DeviceRuntime::TTNN);

    // If the input tensor versions match (i.e. has been constant since the
    // previous trace) then we can skip the copy
    // TODO (#3606): We should model this in the compiler somehow. Currently
    // it's done implicitly by the runtime.
    if (inputTensorWrapper.getVersion() == inputSlotWrapper.getVersion()) {
      continue;
    }

    copyTensor(input, inputTensorWrapper.getTensor(),
               inputSlotWrapper.getTensor());

    // Input slot will now contain identical data as the input tensor
    // Thus we can syncronize their versions
    inputSlotWrapper.syncVersion(inputTensorWrapper);
  }

  ::ttnn::Tensor traceIdTensor =
      ::tt::runtime::ttnn::utils::createTTNNTensor<uint32_t>(
          &traceData.traceId.get(), ::ttnn::Shape(), ::ttnn::DataType::UINT32);

  std::vector<::tt::runtime::Tensor> inputTensors = {
      ::tt::runtime::ttnn::utils::createRuntimeTensorFromTTNN(traceIdTensor)};

  ProgramExecutor executor(deviceHandle, context.getExecutableHandle(),
                           op->execute_program_id(), inputTensors,
                           /*constEvalProgram=*/false);
  executor.execute();

  for (size_t i = 0; i < op->outputs()->size(); i++) {
    const ::tt::target::ttnn::TensorRef *output = op->outputs()->Get(i);
    context.getTensorPool().insertRuntimeTensorAndValidate(
        output, traceData.outputTensors[i]);
  }
}

void run(const ::tt::target::ttnn::CaptureOrExecuteTraceOp *op,
         ProgramContext &context) {
  ::tt::runtime::Device deviceHandle = context.getDeviceHandle();
  ::ttnn::MeshDevice &meshDevice =
      deviceHandle.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);

  LOG_ASSERT(meshDevice.get_program_cache().is_enabled(),
             "Program cache must be enabled");
  LOG_ASSERT(meshDevice.allocator()
                     ->get_statistics(::ttnn::BufferType::TRACE)
                     .total_allocatable_size_bytes > 0,
             "Trace region size must be greater than 0");

  auto traceCache =
      deviceHandle.getTraceCache()
          ->asSharedPtr<::tt::runtime::ttnn::TraceCache>(DeviceRuntime::TTNN);
  LOG_ASSERT(traceCache, "TraceCache must be initialized in DeviceHandle");

  auto [mainProgramKey, captureExecuteKey] = getTraceCacheKeys(op, context);

  if (!traceCache->contains(mainProgramKey, captureExecuteKey)) {
    LOG_DEBUG("Trace cache miss, running program and capturing trace");
    runTraceProgramAndCaptureTrace(op, context, *traceCache);
    debug::Stats::get().incrementStat("TraceCacheMiss");
    debug::Stats::get().incrementStat("CapturedTrace");
    return;
  }

  TraceData *traceData = traceCache->get(mainProgramKey, captureExecuteKey);
  LOG_ASSERT(traceData, "TraceData must be populated in TraceCache");
  LOG_DEBUG("Trace cache hit, executing trace directly");
  executeTrace(op, context, *traceData);
  debug::Stats::get().incrementStat("ExecutedTrace");
}

} // namespace tt::runtime::ttnn::operations::trace
