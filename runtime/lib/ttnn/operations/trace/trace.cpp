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
compileKernelsAndExecuteTraceProgram(const ::tt::target::ttnn::TraceOp *op,
                                     ProgramContext &context) {

  std::vector<::tt::runtime::Tensor> inputs;
  for (const ::tt::target::ttnn::TensorRef *input : *op->inputs()) {
    inputs.emplace_back(
        context.getTensorPool().getRuntimeTensorAndValidate(input));
  }

  ::tt::runtime::Device deviceHandle = context.getDeviceHandle();

  uint64_t binaryId = context.getExecutableHandle().id();
  uint32_t programId = context.getProgramIndex();
  const std::string &traceFuncName = op->callee_name()->str();

  const size_t calleeProgramIndex = op->callee_program_idx();
  ProgramExecutor traceFuncExecutor(deviceHandle, context.getExecutableHandle(),
                                    calleeProgramIndex, inputs);
  traceFuncExecutor.execute();
  LOG_DEBUG("Finished execution of trace function: ", traceFuncName);
  std::vector<::tt::runtime::Tensor> outputs =
      traceFuncExecutor.gatherOutputTensors();

  for (size_t i = 0; i < outputs.size(); i++) {
    ::ttnn::Tensor &output =
        ::tt::runtime::ttnn::utils::getTTNNTensorFromRuntimeTensor(outputs[i]);
    context.getTensorPool().insertTTNNTensorAndValidate(op->outputs()->Get(i),
                                                        output);
  }

  // Kernels have now been compiled, create trace data
  // for this trace function and set the trace stage
  TraceData traceData{.stage = TraceStage::KERNELS_COMPILED,
                      .traceId = {},
                      .inputTensors = {},
                      .outputTensors = {}};
  ::tt::runtime::ttnn::TraceCache &traceCache =
      deviceHandle.getTraceCache()->as<::tt::runtime::ttnn::TraceCache>(
          DeviceRuntime::TTNN);
  traceCache.insert(binaryId, programId, traceFuncName, traceData);
}

static void captureTraceAndPopulateCache(const ::tt::target::ttnn::TraceOp *op,
                                         ProgramContext &context,
                                         TraceData &traceData) {
  LOG_ASSERT(traceData.stage == TraceStage::KERNELS_COMPILED);
  ::tt::runtime::Device deviceHandle = context.getDeviceHandle();
  ::ttnn::MeshDevice &meshDevice =
      deviceHandle.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  std::vector<::tt::runtime::Tensor> inputs;

  // Allocate device tensors on device. These will be the input slots
  // for our trace
  for (const ::tt::target::ttnn::TensorRef *input : *op->inputs()) {
    ::ttnn::Tensor deviceTensor =
        operations::utils::allocateTensorOnDevice(input, meshDevice);
    ::tt::runtime::Tensor runtimeTensor =
        ::tt::runtime::ttnn::utils::createRuntimeTensorFromTTNN(
            deviceTensor, /*retain=*/true);
    inputs.push_back(runtimeTensor);
  }

  const size_t calleeProgramIndex = op->callee_program_idx();
  ProgramExecutor traceCaptureExecutor(
      deviceHandle, context.getExecutableHandle(), calleeProgramIndex, inputs);

  ::ttnn::QueueId ttnnCqId = ::ttnn::QueueId(op->cq_id());
  ::ttnn::MeshTraceId traceId =
      ::ttnn::operations::trace::begin_trace_capture(&meshDevice, ttnnCqId);
  traceCaptureExecutor.execute();
  std::vector<::tt::runtime::Tensor> outputs =
      traceCaptureExecutor.gatherOutputTensors();
  ::ttnn::operations::trace::end_trace_capture(&meshDevice, traceId, ttnnCqId);

  traceData.stage = TraceStage::TRACE_CAPTURED;
  traceData.traceId = traceId;
  traceData.inputTensors = inputs;
  traceData.outputTensors = outputs;
}

static void executeTrace(const ::tt::target::ttnn::TraceOp *op,
                         ProgramContext &context, TraceData &traceData) {
  LOG_ASSERT(traceData.stage == TraceStage::TRACE_CAPTURED);
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

    // Input slot will now contain identical data as the input tensor
    // Thus we can syncronize their versions
    inputSlotWrapper.syncVersion(inputTensorWrapper);

    if (::tt::runtime::ttnn::utils::inSystemMemory(input)) {
      ::tt::tt_metal::write_tensor(inputTensorWrapper.getTensor(),
                                   inputSlotWrapper.getTensor(),
                                   ::ttnn::DefaultQueueId);
      continue;
    }

    LOG_ASSERT(::tt::runtime::workaround::Env::get().traceImplicitFromDevice,
               "traceImplicitFromDevice workaround must be enabled.");
    ::ttnn::Tensor hostInputTensor =
        ::ttnn::from_device(inputTensorWrapper.getTensor());
    ::tt::tt_metal::write_tensor(hostInputTensor, inputSlotWrapper.getTensor(),
                                 ::ttnn::DefaultQueueId);
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

static inline void
incrementDebugStat(const ::tt::runtime::ttnn::TraceCache &cache,
                   const std::string &statName) {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  cache.incrementDebugStat(statName);
#else
  LOG_WARNING_ONCE("Trace cache debug stats are disabled in release mode. To "
                   "record debug stats, TT_RUNTIME_DEBUG must be set");
#endif
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
    compileKernelsAndExecuteTraceProgram(op, context);
    incrementDebugStat(*traceCache, "cacheMiss");
    return;
  }

  TraceData *traceData = traceCache->get(binaryId, programId, traceFuncName);
  LOG_ASSERT(traceData, "TraceData must be initialized in TraceCache");
  if (traceData->stage == TraceStage::KERNELS_COMPILED) {
    captureTraceAndPopulateCache(op, context, *traceData);
    incrementDebugStat(*traceCache, "capturedTrace");
  }

  executeTrace(op, context, *traceData);
  incrementDebugStat(*traceCache, "executedTrace");
}
} // namespace tt::runtime::ttnn::operations::trace
