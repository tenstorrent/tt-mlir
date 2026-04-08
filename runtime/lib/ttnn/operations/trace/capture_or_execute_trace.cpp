// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/trace/capture_or_execute_trace.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/program_executor.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/types/trace_cache.h"
#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/types.h"
#include "tt/runtime/workarounds.h"
#include "ttnn/tensor/tensor_impl.hpp"

namespace tt::runtime::ttnn::operations::trace {

static std::pair<MainProgramKey, CaptureExecuteProgramKey>
getTraceCacheKeys(const ::tt::target::ttnn::CaptureOrExecuteTraceOp *op,
                  ProgramContext &context) {
  return {MainProgramKey(context.getExecutableHandle().id(),
                         context.getProgramIndex()),
          CaptureExecuteProgramKey(op->capture_program_id(),
                                   op->execute_program_id())};
}

static void copyTensorFromHostToDevice(const ::ttnn::Tensor &srcTensor,
                                       ::ttnn::Tensor &dstTensor) {

  LOG_ASSERT(srcTensor.storage_type() == ::ttnn::StorageType::HOST &&
                 dstTensor.storage_type() == ::ttnn::StorageType::DEVICE,
             "srcTensor must be on host and dstTensor must be on device");

  ::tt::tt_metal::tensor_impl::copy_to_device(srcTensor, dstTensor);
}

static void copyTensorFromDeviceToDevice(const ::ttnn::Tensor &srcTensor,
                                         ::ttnn::Tensor &dstTensor) {
  LOG_ASSERT(srcTensor.storage_type() == ::ttnn::StorageType::DEVICE &&
                 dstTensor.storage_type() == ::ttnn::StorageType::DEVICE,
             "srcTensor must be on device and dstTensor must be on device");
  ::ttnn::Tensor hostSrcTensor = ::ttnn::from_device(srcTensor);
  ::tt::tt_metal::tensor_impl::copy_to_device(hostSrcTensor, dstTensor);
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
                      .inputTensors = std::move(inputSlots),
                      .outputTensors = std::move(outputSlots),
                      .generationId = traceCache.getGenerationId()};
  auto [mainProgramKey, captureExecuteKey] = getTraceCacheKeys(op, context);
  traceCache.insert(mainProgramKey, captureExecuteKey, std::move(traceData));
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

    // Constants/parameters and KV cache tensors live on device and persist
    // across trace executions, so their versions are expected to match.
    // Regular inputs live on host and change between executions, so a version
    // mismatch is the expected case — we copy them from host to device below.
    if (inputTensorWrapper.getVersion() == inputSlotWrapper.getVersion()) {
      continue;
    }

    if (input->desc()->layout()->memory_desc()->storage_type() ==
        ::tt::target::ttnn::StorageType::Device) {
      // Device-resident tensors (constants/parameters/KV cache) can be
      // legitimately updated by the user (e.g. weight updates during
      // training). This is handled by copying the new device tensor into the
      // trace input slot on device.
      LOG_DEBUG("Device-resident tensor version changed "
                "(constant/parameter or KV cache). Input index: ",
                i, ", expected version: ", inputSlotWrapper.getVersion());
      copyTensorFromDeviceToDevice(inputTensorWrapper.getTensor(),
                                   inputSlotWrapper.getTensor());
    } else {
      // Regular inputs reside on host and are copied into their trace input
      // slots on device. A version mismatch is expected here since the host
      // tensor can change.
      copyTensorFromHostToDevice(inputTensorWrapper.getTensor(),
                                 inputSlotWrapper.getTensor());
    }

    // Input slot will now contain identical data as the input tensor
    // Thus we can synchronize their versions
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

  auto traceCache =
      deviceHandle.getTraceCache()
          ->asSharedPtr<::tt::runtime::ttnn::TraceCache>(DeviceRuntime::TTNN);
  LOG_ASSERT(traceCache, "TraceCache must be initialized in DeviceHandle");

  auto [mainProgramKey, captureExecuteKey] = getTraceCacheKeys(op, context);

  if (!traceCache->contains(mainProgramKey, captureExecuteKey)) {
    LOG_DEBUG("Trace cache miss, running program and capturing trace");
    traceCache->incrementGeneration();
    runTraceProgramAndCaptureTrace(op, context, *traceCache);
    debug::Stats::get().incrementStat("TraceCacheMiss");
    debug::Stats::get().incrementStat("CapturedTrace");
    return;
  }

  TraceData *traceData = traceCache->get(mainProgramKey, captureExecuteKey);
  LOG_ASSERT(traceData, "TraceData must be populated in TraceCache");

  // Check if the trace is stale by comparing the generation id of the trace
  // with the current generation id of the cache.
  //
  // If the trace is stale, that means that we have new allocations on the
  // device since the trace was captured, so we need to re-capture it again.
  // Otherwise, we would possibly be overwriting new allocations when replaying
  // (executing) the stale trace.
  if (traceData->generationId < traceCache->getGenerationId()) {
    LOG_DEBUG("Trace is stale (captured at gen ", traceData->generationId,
              ", current gen ", traceCache->getGenerationId(),
              "), invalidating and recapturing");

    // Remove the stale trace from the cache and recapture it.
    traceCache->erase(mainProgramKey, captureExecuteKey);
    runTraceProgramAndCaptureTrace(op, context, *traceCache);

    debug::Stats::get().incrementStat("TraceCacheMiss");
    debug::Stats::get().incrementStat("TraceStaleRecapture");
    return;
  }

  LOG_DEBUG("Trace cache hit, executing trace directly");
  executeTrace(op, context, *traceData);
  debug::Stats::get().incrementStat("ExecutedTrace");
}

} // namespace tt::runtime::ttnn::operations::trace
