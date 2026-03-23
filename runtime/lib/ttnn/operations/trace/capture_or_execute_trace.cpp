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

static constexpr const char *kCaptureTracePrefix = "run_and_capture_";

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
    ::tt::tt_metal::tensor_impl::copy_to_device(srcTensor, dstTensor);
    return;
  }

  LOG_ASSERT(::tt::runtime::workaround::Env::get().traceImplicitFromDevice,
             "traceImplicitFromDevice workaround must be enabled.");
  ::ttnn::Tensor hostSrcTensor = ::ttnn::from_device(srcTensor);
  ::tt::tt_metal::tensor_impl::copy_to_device(hostSrcTensor, dstTensor);
}

// Find the trace body program index by deriving its name from the capture
// program name. The capture program is named "run_and_capture_trace_N_func"
// and the trace body is named "trace_N_func".
static std::optional<size_t>
findTraceBodyProgramIndex(Binary &executableHandle, size_t captureProgramId) {
  std::string captureProgramName =
      executableHandle.getProgramName(captureProgramId);

  std::string prefix(kCaptureTracePrefix);
  if (captureProgramName.substr(0, prefix.size()) != prefix) {
    LOG_WARNING("Capture program name does not have expected prefix: ",
                captureProgramName);
    return std::nullopt;
  }
  std::string traceBodyName = captureProgramName.substr(prefix.size());

  uint32_t numPrograms = executableHandle.getNumPrograms();
  for (uint32_t i = 0; i < numPrograms; i++) {
    if (executableHandle.getProgramName(i) == traceBodyName) {
      return i;
    }
  }

  LOG_WARNING("Could not find trace body program: ", traceBodyName);
  return std::nullopt;
}

// Run the compiler-generated capture program to allocate input slots.
// The trace it captures is released (it predates the full device picture).
// Function outputs are inserted into the tensor pool so the caller gets
// valid results. Returns the retained input slots.
static std::vector<::tt::runtime::Tensor> allocateSlotsFromCaptureProgram(
    const ::tt::target::ttnn::CaptureOrExecuteTraceOp *op,
    ProgramContext &context) {

  ProgramTensorPool &tensorPool = context.getTensorPool();
  ::tt::runtime::Device deviceHandle = context.getDeviceHandle();
  ::ttnn::MeshDevice &meshDevice =
      deviceHandle.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);

  std::vector<::tt::runtime::Tensor> inputTensors;
  for (const ::tt::target::ttnn::TensorRef *input : *op->inputs()) {
    inputTensors.push_back(tensorPool.getRuntimeTensorAndValidate(input));
  }

  ProgramExecutor executor(deviceHandle, context.getExecutableHandle(),
                           op->capture_program_id(), inputTensors,
                           /*constEvalProgram=*/false);
  executor.execute();
  std::vector<::tt::runtime::Tensor> outputTensors =
      executor.gatherOutputTensors();

  size_t expectedNumOutputs =
      1 + op->outputs()->size() + op->inputs()->size() + op->outputs()->size();
  LOG_ASSERT(outputTensors.size() == expectedNumOutputs,
             "Mismatched number of output tensors, expected: ",
             expectedNumOutputs, " got: ", outputTensors.size());
  size_t currOutputIndex = 0;

  // Extract and release the trace (captured before full device picture)
  const ::ttnn::Tensor &traceIdTensor =
      ::tt::runtime::ttnn::utils::getTTNNTensorFromRuntimeTensor(
          outputTensors[currOutputIndex++]);
  LOG_ASSERT(traceIdTensor.dtype() == ::ttnn::DataType::UINT32,
             "Trace ID must be UINT32");
  uint32_t traceId =
      ::tt::runtime::ttnn::utils::getScalarFromTensor<uint32_t>(traceIdTensor);
  ::ttnn::operations::trace::release_trace(&meshDevice,
                                           ::ttnn::MeshTraceId(traceId));

  // Insert function outputs into tensor pool (valid results for caller)
  for (size_t i = 0; i < op->outputs()->size(); i++) {
    tensorPool.insertRuntimeTensorAndValidate(op->outputs()->Get(i),
                                              outputTensors[currOutputIndex++]);
  }

  // Extract input slots with retain
  std::vector<::tt::runtime::Tensor> inputSlots;
  for (size_t i = 0; i < op->inputs()->size(); i++) {
    ::tt::runtime::Tensor &inputSlot = outputTensors[currOutputIndex++];
    inputSlot.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN)
        .setRetain(true);
    inputSlots.emplace_back(std::move(inputSlot));
  }

  // Output slots from the capture program are discarded — captureTrace
  // will produce new ones during the real capture.
  return inputSlots;
}

// Capture a trace. If existingInputSlots is provided, reuses them (recapture).
// If nullopt, allocates new slots by running the capture program (first capture).
static void captureTrace(
    const ::tt::target::ttnn::CaptureOrExecuteTraceOp *op,
    ProgramContext &context, ::tt::runtime::ttnn::TraceCache &traceCache,
    std::optional<std::vector<::tt::runtime::Tensor>> existingInputSlots =
        std::nullopt) {

  ProgramTensorPool &tensorPool = context.getTensorPool();
  ::tt::runtime::Device deviceHandle = context.getDeviceHandle();
  ::ttnn::MeshDevice &meshDevice =
      deviceHandle.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  Binary &executableHandle = context.getExecutableHandle();

  // Get or allocate input slots
  std::vector<::tt::runtime::Tensor> inputSlots =
      existingInputSlots ? std::move(*existingInputSlots)
                         : allocateSlotsFromCaptureProgram(op, context);

  auto traceBodyIndex =
      findTraceBodyProgramIndex(executableHandle, op->capture_program_id());
  LOG_ASSERT(traceBodyIndex.has_value(),
             "Failed to find trace body program index");

  // Copy current input data into slots (version-based skip)
  for (size_t i = 0; i < op->inputs()->size(); i++) {
    const ::tt::target::ttnn::TensorRef *input = op->inputs()->Get(i);
    const ::tt::runtime::ttnn::TTNNTensorWrapper &inputTensorWrapper =
        tensorPool.getTTNNTensorWrapperAndValidate(input);

    ::tt::runtime::ttnn::TTNNTensorWrapper &slotWrapper =
        inputSlots[i].as<::tt::runtime::ttnn::TTNNTensorWrapper>(
            DeviceRuntime::TTNN);

    if (inputTensorWrapper.getVersion() == slotWrapper.getVersion()) {
      continue;
    }

    copyTensor(input, inputTensorWrapper.getTensor(), slotWrapper.getTensor());
    slotWrapper.syncVersion(inputTensorWrapper);
  }

  // Warmup: run trace body to establish program cache and memory layout
  {
    ProgramExecutor executor(deviceHandle, executableHandle, *traceBodyIndex,
                             inputSlots, /*constEvalProgram=*/false);
    executor.execute();
    std::vector<::tt::runtime::Tensor> warmupOutputs =
        executor.gatherOutputTensors();

    LOG_ASSERT(warmupOutputs.size() == op->outputs()->size(),
               "Mismatched warmup output count");
    for (size_t i = 0; i < op->outputs()->size(); i++) {
      tensorPool.insertRuntimeTensorAndValidate(op->outputs()->Get(i),
                                                warmupOutputs[i]);
    }
  }

  // Capture: run trace body between begin/end trace capture
  ::ttnn::QueueId cqId(0);
  ::ttnn::MeshTraceId meshTraceId =
      ::ttnn::operations::trace::begin_trace_capture(&meshDevice, cqId);

  std::vector<::tt::runtime::Tensor> outputSlots;
  {
    ProgramExecutor executor(deviceHandle, executableHandle, *traceBodyIndex,
                             inputSlots, /*constEvalProgram=*/false);
    executor.execute();
    outputSlots = executor.gatherOutputTensors();
  }

  ::ttnn::operations::trace::end_trace_capture(&meshDevice, meshTraceId, cqId);

  // Retain output slots
  for (auto &outputSlot : outputSlots) {
    outputSlot.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN)
        .setRetain(true);
  }

  // Store in cache
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
    LOG_DEBUG("Trace cache miss, capturing trace");
    traceCache->incrementGeneration();
    captureTrace(op, context, *traceCache);
    debug::Stats::get().incrementStat("TraceCacheMiss");
    debug::Stats::get().incrementStat("CapturedTrace");
    return;
  }

  TraceData *traceData = traceCache->get(mainProgramKey, captureExecuteKey);
  LOG_ASSERT(traceData, "TraceData must be populated in TraceCache");

  // Staleness check: if device generation advanced since capture, new
  // allocations may overlap with this trace's intermediate addresses.
  if (traceData->generationId < traceCache->getGenerationId()) {
    LOG_DEBUG("Trace is stale (captured at gen ",
              traceData->generationId, ", current gen ",
              traceCache->getGenerationId(),
              "), invalidating and recapturing");

    auto staleData =
        traceCache->erase(mainProgramKey, captureExecuteKey);

    LOG_ASSERT(staleData.has_value(),
               "Expected trace data to be present in cache for recapture");

    captureTrace(op, context, *traceCache,
                 std::move((*staleData).inputTensors));

    debug::Stats::get().incrementStat("TraceCacheMiss");
    debug::Stats::get().incrementStat("TraceStaleRecapture");
    return;
  }

  LOG_DEBUG("Trace cache hit, executing trace directly");
  executeTrace(op, context, *traceData);
  debug::Stats::get().incrementStat("ExecutedTrace");
}

} // namespace tt::runtime::ttnn::operations::trace
