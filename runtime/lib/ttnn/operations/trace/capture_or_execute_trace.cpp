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

static void copyTensor(const ::ttnn::Tensor &srcTensor,
                       ::ttnn::Tensor &dstTensor) {

  LOG_ASSERT(srcTensor.storage_type() == ::ttnn::StorageType::HOST &&
                 dstTensor.storage_type() == ::ttnn::StorageType::DEVICE,
             "srcTensor must be on host and dstTensor must be on device");

  ::tt::tt_metal::tensor_impl::copy_to_device(srcTensor, dstTensor);
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
  LOG_INFO(::tt::runtime::logger::LogType::LogAlways,
           "    - Capturing trace input slots:");
  std::vector<::tt::runtime::Tensor> inputSlots;
  for (size_t i = 0; i < op->inputs()->size(); i++) {
    ::tt::runtime::Tensor &inputSlot = outputTensors[currOutputIndex++];
    ::tt::runtime::ttnn::TTNNTensorWrapper &inputSlotWrapper =
        inputSlot.as<::tt::runtime::ttnn::TTNNTensorWrapper>(
            DeviceRuntime::TTNN);

    // Log input slot address
    const auto& slotTensor = inputSlotWrapper.getTensor();
    if (slotTensor.storage_type() == ::ttnn::StorageType::DEVICE) {
      auto *buffer = slotTensor.buffer();
      if (buffer) {
        LOG_INFO(::tt::runtime::logger::LogType::LogAlways,
                 "      Input slot ", i, " captured at device address: 0x",
                 std::hex, buffer->address(), std::dec);
      }
    }

    // input slots need to be retained
    inputSlotWrapper.setRetain(true);
    inputSlots.emplace_back(std::move(inputSlot));
  }

  // Handle trace output slots
  LOG_INFO(::tt::runtime::logger::LogType::LogAlways,
           "    - Capturing trace output slots:");
  std::vector<::tt::runtime::Tensor> outputSlots;
  for (size_t i = 0; i < op->outputs()->size(); i++) {
    ::tt::runtime::Tensor &outputSlot = outputTensors[currOutputIndex++];
    ::tt::runtime::ttnn::TTNNTensorWrapper &outputSlotWrapper =
        outputSlot.as<::tt::runtime::ttnn::TTNNTensorWrapper>(
            DeviceRuntime::TTNN);

    // Log output slot address
    const auto& slotTensor = outputSlotWrapper.getTensor();
    if (slotTensor.storage_type() == ::ttnn::StorageType::DEVICE) {
      auto *buffer = slotTensor.buffer();
      if (buffer) {
        LOG_INFO(::tt::runtime::logger::LogType::LogAlways,
                 "      Output slot ", i, " captured at device address: 0x",
                 std::hex, buffer->address(), std::dec);
      }
    }

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

  LOG_INFO(::tt::runtime::logger::LogType::LogAlways,
           "\n  [TRACE EXECUTE] Starting trace execution");
  LOG_INFO(::tt::runtime::logger::LogType::LogAlways,
           "    - Trace ID: ", traceData.traceId.get());
  LOG_INFO(::tt::runtime::logger::LogType::LogAlways,
           "    - Number of inputs: ", op->inputs()->size());
  LOG_INFO(::tt::runtime::logger::LogType::LogAlways,
           "    - Number of outputs: ", op->outputs()->size());

  std::vector<::ttnn::Tensor> inputs;
  LOG_ASSERT(op->inputs()->size() == traceData.inputTensors.size(),
             "Mismatched number of inputs, expected: ", op->inputs()->size(),
             " got: ", traceData.inputTensors.size());
  LOG_ASSERT(op->outputs()->size() == traceData.outputTensors.size(),
             "Mismatched number of outputs, expected: ", op->outputs()->size(),
             " got: ", traceData.outputTensors.size());

  LOG_DEBUG(::tt::runtime::logger::LogType::LogAlways,
            "    - Processing trace inputs...");

  for (size_t i = 0; i < op->inputs()->size(); i++) {
    const ::tt::target::ttnn::TensorRef *input = op->inputs()->Get(i);
    const ::tt::runtime::ttnn::TTNNTensorWrapper &inputTensorWrapper =
        context.getTensorPool().getTTNNTensorWrapperAndValidate(input);

    ::tt::runtime::ttnn::TTNNTensorWrapper &inputSlotWrapper =
        traceData.inputTensors[i].as<::tt::runtime::ttnn::TTNNTensorWrapper>(
            DeviceRuntime::TTNN);

    // Log device slot address for all inputs
    const auto& deviceSlotTensor = inputSlotWrapper.getTensor();
    if (deviceSlotTensor.storage_type() == ::ttnn::StorageType::DEVICE) {
      auto *buffer = deviceSlotTensor.buffer();
      if (buffer) {
        LOG_INFO(::tt::runtime::logger::LogType::LogAlways,
                 "      Input slot ", i, " device buffer address: 0x",
                 std::hex, buffer->address(), std::dec);
      }
    }

    // By trace convention, tensors already on device are constants or KV cache
    // that persist in their trace slots. Regular inputs require host-to-device
    // copy into their trace input slots on device.
    if (input->desc()->layout()->memory_desc()->storage_type() ==
        ::tt::target::ttnn::StorageType::Device) {
      LOG_ASSERT(inputTensorWrapper.getVersion() ==
                     inputSlotWrapper.getVersion(),
                 "Device trace slots for non-regular inputs (constants and KV "
                 "cache) are persisted across traces, so their versions must "
                 "be the same. Expected version: ",
                 inputSlotWrapper.getVersion());
      LOG_ASSERT(inputTensorWrapper.getTensor().storage_type() ==
                     ::ttnn::StorageType::DEVICE &&
                 "Non-regular inputs must already be on device.");
      LOG_DEBUG("Skipping copy for constant input ", i,
                " since it is already on device and trace input slot is "
                "persisted across traces. Version: ",
                inputTensorWrapper.getVersion());
      continue;
    }

    // For regular inputs, we copy the input tensor from host, if there is a
    // version mismatch between the input tensor and the trace input slot, into
    // the corresponding trace input slot on device. This ensures that the trace
    // input slots always have the most up-to-date data from the host for
    // regular inputs.
    if (inputTensorWrapper.getVersion() == inputSlotWrapper.getVersion()) {
      LOG_DEBUG(::tt::runtime::logger::LogType::LogAlways,
                "      Input ", i, " version matches (",
                inputSlotWrapper.getVersion(), "), skipping copy");
      continue;
    }

    LOG_DEBUG(::tt::runtime::logger::LogType::LogAlways,
              "      Input ", i, " copying from host to device trace slot");

    // Optional: Dump input tensor for debugging
    static const char* dumpTraceInputsEnv = std::getenv("TTMLIR_RUNTIME_DUMP_TRACE_INPUTS");
    if (dumpTraceInputsEnv && std::string(dumpTraceInputsEnv) == "1") {
      std::string dumpPath = "trace_input_" + std::to_string(traceData.traceId.get()) +
                            "_slot_" + std::to_string(i) + ".bin";
      LOG_INFO(::tt::runtime::logger::LogType::LogAlways,
               "      [DEBUG] Dumping trace input ", i, " to ", dumpPath);

      // Note: Would need to implement actual tensor dump here if needed
      // For now, just log that tensor info is available
      // Avoid unused variable warning by removing the tensor declaration
      (void)inputTensorWrapper.getTensor(); // Tensor available for future dumping
      LOG_INFO(::tt::runtime::logger::LogType::LogAlways,
               "        Tensor info available for debugging");
    }

    copyTensor(inputTensorWrapper.getTensor(), inputSlotWrapper.getTensor());

    // Input slot will now contain identical data as the input tensor
    // Thus we can syncronize their versions
    inputSlotWrapper.syncVersion(inputTensorWrapper);

    LOG_DEBUG(::tt::runtime::logger::LogType::LogAlways,
              "      Input ", i, " copy completed, version synced to: ",
              inputSlotWrapper.getVersion());
  }

  LOG_DEBUG(::tt::runtime::logger::LogType::LogAlways,
            "    - All inputs processed");

  ::ttnn::Tensor traceIdTensor =
      ::tt::runtime::ttnn::utils::createTTNNTensor<uint32_t>(
          &traceData.traceId.get(), ::ttnn::Shape(), ::ttnn::DataType::UINT32);

  std::vector<::tt::runtime::Tensor> inputTensors = {
      ::tt::runtime::ttnn::utils::createRuntimeTensorFromTTNN(traceIdTensor)};

  LOG_INFO(::tt::runtime::logger::LogType::LogAlways,
           "    - Executing trace program (ID: ", op->execute_program_id(), ")");

  ProgramExecutor executor(deviceHandle, context.getExecutableHandle(),
                           op->execute_program_id(), inputTensors,
                           /*constEvalProgram=*/false);
  executor.execute();

  LOG_INFO(::tt::runtime::logger::LogType::LogAlways,
           "    - Trace program execution completed");

  LOG_DEBUG(::tt::runtime::logger::LogType::LogAlways,
            "    - Gathering trace outputs...");

  for (size_t i = 0; i < op->outputs()->size(); i++) {
    const ::tt::target::ttnn::TensorRef *output = op->outputs()->Get(i);

    // Log output slot device address for debugging
    const ::tt::runtime::ttnn::TTNNTensorWrapper &outputSlotWrapper =
        traceData.outputTensors[i].as<::tt::runtime::ttnn::TTNNTensorWrapper>(
            DeviceRuntime::TTNN);
    const auto& outputSlotTensor = outputSlotWrapper.getTensor();
    if (outputSlotTensor.storage_type() == ::ttnn::StorageType::DEVICE) {
      auto *buffer = outputSlotTensor.buffer();
      if (buffer) {
        LOG_INFO(::tt::runtime::logger::LogType::LogAlways,
                 "      Output slot ", i, " device buffer address: 0x",
                 std::hex, buffer->address(), std::dec);
      }
    }

    context.getTensorPool().insertRuntimeTensorAndValidate(
        output, traceData.outputTensors[i]);
  }

  LOG_INFO(::tt::runtime::logger::LogType::LogAlways,
           "  [TRACE EXECUTE] Trace execution completed successfully");
}

void run(const ::tt::target::ttnn::CaptureOrExecuteTraceOp *op,
         ProgramContext &context) {
  ::tt::runtime::Device deviceHandle = context.getDeviceHandle();
  ::ttnn::MeshDevice &meshDevice =
      deviceHandle.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);

  LOG_INFO(::tt::runtime::logger::LogType::LogAlways,
           "\n  [TRACE OP] CaptureOrExecuteTraceOp starting");

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
    LOG_INFO(::tt::runtime::logger::LogType::LogAlways,
             "    - Trace cache MISS, capturing new trace");
    LOG_DEBUG("Trace cache miss, running program and capturing trace");
    runTraceProgramAndCaptureTrace(op, context, *traceCache);
    debug::Stats::get().incrementStat("TraceCacheMiss");
    debug::Stats::get().incrementStat("CapturedTrace");
    LOG_INFO(::tt::runtime::logger::LogType::LogAlways,
             "  [TRACE OP] Trace capture completed");
    return;
  }

  TraceData *traceData = traceCache->get(mainProgramKey, captureExecuteKey);
  LOG_ASSERT(traceData, "TraceData must be populated in TraceCache");
  LOG_INFO(::tt::runtime::logger::LogType::LogAlways,
           "    - Trace cache HIT, executing existing trace");
  LOG_DEBUG("Trace cache hit, executing trace directly");
  executeTrace(op, context, *traceData);
  debug::Stats::get().incrementStat("ExecutedTrace");
  LOG_INFO(::tt::runtime::logger::LogType::LogAlways,
           "  [TRACE OP] Trace execution completed");
}

} // namespace tt::runtime::ttnn::operations::trace
