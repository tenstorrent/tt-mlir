// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/trace/trace.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/program_executor.h"
#include "tt/runtime/detail/ttnn/trace_cache.h"
#include "tt/runtime/detail/ttnn/types.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/types.h"
#include "tt/runtime/workarounds.h"

namespace tt::runtime::ttnn::operations::trace {

static void
copyTensor(const ::tt::target::ttnn::TensorRef *srcTensorDesc,
           const ::ttnn::Tensor &srcTensor, ::ttnn::Tensor &dstTensor,
           const ::ttnn::QueueId &queueId = ::ttnn::DefaultQueueId) {

  if (::tt::runtime::ttnn::utils::inSystemMemory(srcTensorDesc)) {
    ::tt::tt_metal::write_tensor(srcTensor, dstTensor, /*blocking=*/false,
                                 queueId);
    return;
  }

  LOG_ASSERT(::tt::runtime::workaround::Env::get().traceImplicitFromDevice,
             "traceImplicitFromDevice workaround must be enabled.");
  ::ttnn::Tensor hostSrcTensor = ::ttnn::from_device(srcTensor);
  ::tt::tt_metal::write_tensor(hostSrcTensor, dstTensor, /*blocking=*/false,
                               queueId);
}

static void executeTraceProgramAndCaptureTrace(
    const ::tt::target::ttnn::TraceOp *op, ProgramContext &context,
    ::tt::runtime::ttnn::TraceCache &traceCache) {

  ::tt::runtime::Device deviceHandle = context.getDeviceHandle();
  ::ttnn::MeshDevice &meshDevice =
      deviceHandle.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);

  std::vector<::tt::runtime::Tensor> traceInputTensors;
  for (const ::tt::target::ttnn::TensorRef *input : *op->inputs()) {
    // Allocate a tensor buffer on device. This will hold the input data for the
    // trace
    ::ttnn::Tensor traceInputTensorTTNN =
        utils::allocateTensorOnDevice(input, meshDevice);

    // Copy the input data from the runtime tensor pool to the trace tensor
    ::tt::runtime::ttnn::TTNNTensorWrapper &inputTensorWrapper =
        context.getTensorPool().getTTNNTensorWrapperAndValidate(input);
    copyTensor(input, inputTensorWrapper.getTensor(), traceInputTensorTTNN);

    // Store the trace input tensor
    ::tt::runtime::Tensor traceInputTensor =
        ::tt::runtime::ttnn::utils::createRuntimeTensorFromTTNN(
            traceInputTensorTTNN, /*meshEvent=*/std::nullopt, /*retain=*/true);
    ::tt::runtime::ttnn::TTNNTensorWrapper &traceInputTensorWrapper =
        traceInputTensor.as<::tt::runtime::ttnn::TTNNTensorWrapper>(
            DeviceRuntime::TTNN);
    traceInputTensorWrapper.syncVersion(inputTensorWrapper);

    traceInputTensors.push_back(traceInputTensor);
  }

  // Execute the trace program to compile kernels and get compute results
  const size_t calleeProgramIndex = op->callee_program_idx();
  ProgramExecutor traceFuncExecutor(deviceHandle, context.getExecutableHandle(),
                                    calleeProgramIndex, traceInputTensors);
  traceFuncExecutor.execute();
  LOG_DEBUG("Finished execution of trace function: ", op->callee_name()->str());
  std::vector<::tt::runtime::Tensor> outputs =
      traceFuncExecutor.gatherOutputTensors();

  for (size_t i = 0; i < outputs.size(); i++) {
    ::ttnn::Tensor &output =
        ::tt::runtime::ttnn::utils::getTTNNTensorFromRuntimeTensor(outputs[i]);
    context.getTensorPool().insertTTNNTensorAndValidate(op->outputs()->Get(i),
                                                        output);
  }

  // Now that the program has been executed and kernels have been compiled
  // We can capture the trace
  ProgramExecutor traceCaptureExecutor(deviceHandle,
                                       context.getExecutableHandle(),
                                       calleeProgramIndex, traceInputTensors);

  // Capture the trace
  ::ttnn::QueueId ttnnCqId = ::ttnn::QueueId(op->cq_id());
  ::ttnn::MeshTraceId traceId =
      ::ttnn::operations::trace::begin_trace_capture(&meshDevice, ttnnCqId);
  traceCaptureExecutor.execute();
  std::vector<::tt::runtime::Tensor> traceOutputs =
      traceCaptureExecutor.gatherOutputTensors();
  ::ttnn::operations::trace::end_trace_capture(&meshDevice, traceId, ttnnCqId);

  // Store the trace data in the cache
  TraceData traceData{.traceId = traceId,
                      .inputTensors = traceInputTensors,
                      .outputTensors = traceOutputs};

  uint64_t binaryId = context.getExecutableHandle().id();
  uint32_t programId = context.getProgramIndex();
  const std::string &traceFuncName = op->callee_name()->str();

  traceCache.insert(binaryId, programId, traceFuncName, traceData);
}

static void executeTrace(const ::tt::target::ttnn::TraceOp *op,
                         ProgramContext &context, TraceData &traceData) {
  ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();

  std::vector<::ttnn::Tensor> inputs;
  LOG_ASSERT(op->inputs()->size() == traceData.inputTensors.size());

  for (size_t i = 0; i < op->inputs()->size(); i++) {
    const ::tt::target::ttnn::TensorRef *input = op->inputs()->Get(i);
    ::tt::runtime::ttnn::TTNNTensorWrapper &inputTensorWrapper =
        context.getTensorPool().getTTNNTensorWrapperAndValidate(input);

    ::tt::runtime::ttnn::TTNNTensorWrapper &inputSlotWrapper =
        traceData.inputTensors[i].as<::tt::runtime::ttnn::TTNNTensorWrapper>(
            DeviceRuntime::TTNN);

    // If the input tensor versions match (i.e. has been constant since the
    // previous trace) then we can skip the copy
    if (inputTensorWrapper.getVersion() == inputSlotWrapper.getVersion()) {
      continue;
    }

    copyTensor(input, inputTensorWrapper.getTensor(),
               inputSlotWrapper.getTensor());

    // Input slot will now contain identical data as the input tensor
    // Thus we can syncronize their versions
    inputSlotWrapper.syncVersion(inputTensorWrapper);
  }

  ::ttnn::QueueId ttnnCqId = ::ttnn::QueueId(op->cq_id());
  // Execute the trace
  ::ttnn::operations::trace::execute_trace(&meshDevice, traceData.traceId,
                                           ttnnCqId, op->blocking());

  size_t outputIndex = 0;
  for (const ::tt::target::ttnn::TensorRef *output : *op->outputs()) {
    const ::ttnn::Tensor &outputTensor =
        ::tt::runtime::ttnn::utils::getTTNNTensorFromRuntimeTensor(
            traceData.outputTensors[outputIndex++]);
    // These outputs must be retained as they are output slots of the trace
    context.getTensorPool().insertTTNNTensorAndValidate(output, outputTensor,
                                                        /*retain=*/true);
  }
}

void run(const ::tt::target::ttnn::TraceOp *op, ProgramContext &context) {
  ::tt::runtime::Device deviceHandle = context.getDeviceHandle();
  ::ttnn::MeshDevice &meshDevice =
      deviceHandle.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);

  LOG_ASSERT(meshDevice.get_program_cache().is_enabled(),
             "Program cache must be enabled");
  LOG_ASSERT(meshDevice.allocator()->get_config().trace_region_size > 0,
             "Trace region size must be greater than 0");

  Binary &executableHandle = context.getExecutableHandle();

  auto traceCache =
      deviceHandle.getTraceCache()
          ->asSharedPtr<::tt::runtime::ttnn::TraceCache>(DeviceRuntime::TTNN);
  LOG_ASSERT(traceCache, "TraceCache must be initialized in DeviceHandle");

  uint64_t binaryId = executableHandle.id();
  uint32_t programId = context.getProgramIndex();
  const std::string &traceFuncName = op->callee_name()->str();

  if (!traceCache->contains(binaryId, programId, traceFuncName)) {
    executeTraceProgramAndCaptureTrace(op, context, *traceCache);
    debug::Stats::get().incrementStat("TraceCacheMiss");
    debug::Stats::get().incrementStat("CapturedTrace");
    return;
  }

  TraceData *traceData = traceCache->get(binaryId, programId, traceFuncName);
  LOG_ASSERT(traceData, "TraceData must be populated in TraceCache");
  executeTrace(op, context, *traceData);
  debug::Stats::get().incrementStat("ExecutedTrace");
}
} // namespace tt::runtime::ttnn::operations::trace
