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

  ::tt::tt_metal::copy_to_device(srcTensor, dstTensor);
}

static void copyTensorFromDeviceToDevice(const ::ttnn::Tensor &srcTensor,
                                         ::ttnn::Tensor &dstTensor) {
  LOG_ASSERT(srcTensor.storage_type() == ::ttnn::StorageType::DEVICE &&
                 dstTensor.storage_type() == ::ttnn::StorageType::DEVICE,
             "srcTensor must be on device and dstTensor must be on device");
  ::ttnn::Tensor hostSrcTensor = ::ttnn::from_device(srcTensor);
  ::tt::tt_metal::copy_to_device(hostSrcTensor, dstTensor);
}

// Runs the slot allocation program, which allocates the persistent device
// buffers this trace will read from and write to. Runs exactly once per trace.
//
// Its program inputs are the device-resident subset of the op's inputs (they
// act as their own slots); it returns the trace input slots followed by the
// trace output slots.
static std::vector<::tt::runtime::Tensor>
allocateTraceSlots(const ::tt::target::ttnn::CaptureOrExecuteTraceOp *op,
                   ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  ::tt::runtime::Device deviceHandle = context.getDeviceHandle();

  std::vector<::tt::runtime::Tensor> deviceResidentInputs;
  for (const ::tt::target::ttnn::TensorRef *input : *op->inputs()) {
    if (::tt::runtime::ttnn::utils::inDeviceMemory(input)) {
      deviceResidentInputs.push_back(
          tensorPool.getRuntimeTensorAndValidate(input));
    }
  }

  ProgramExecutor executor(deviceHandle, context.getExecutableHandle(),
                           op->allocate_slots_program_id(),
                           deviceResidentInputs,
                           /*constEvalProgram=*/false,
                           /*programSemaphoreInputs=*/{});
  executor.execute();
  std::vector<::tt::runtime::Tensor> slots = executor.gatherOutputTensors();

  size_t expectedNumSlots = op->inputs()->size() + op->outputs()->size();
  LOG_ASSERT(slots.size() == expectedNumSlots,
             "Mismatched number of trace slots, expected: ", expectedNumSlots,
             " got: ", slots.size());

  // The slots outlive every program that touches them; the trace cache owns
  // them from here on.
  for (::tt::runtime::Tensor &slot : slots) {
    slot.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN)
        .setRetain(true);
  }

  return slots;
}

// Copies the current values of the op's inputs into the persistent trace input
// slots.
//
// When deviceResidentOnly is set, only tensors that are on the device
// will be updated. This is used before trace capture, since the capture
// program will copy the host inputs into the slots itself.
//
// TODO(pilkic): we could align `executeTrace` to have the same logic for
// handling host inputs as `captureTrace`. That would simplify the contract,
// such that runtime is only responsible for updating device-resident inputs.
static void
updateTraceInputSlots(const ::tt::target::ttnn::CaptureOrExecuteTraceOp *op,
                      ProgramContext &context,
                      std::vector<::tt::runtime::Tensor> &inputSlots,
                      bool deviceResidentOnly = false) {
  LOG_ASSERT(op->inputs()->size() == inputSlots.size(),
             "Mismatched number of inputs, expected: ", op->inputs()->size(),
             " got: ", inputSlots.size());

  for (size_t i = 0; i < op->inputs()->size(); i++) {
    const ::tt::target::ttnn::TensorRef *input = op->inputs()->Get(i);
    const bool deviceResident =
        ::tt::runtime::ttnn::utils::inDeviceMemory(input);
    if (deviceResidentOnly && !deviceResident) {
      continue;
    }

    const ::tt::runtime::ttnn::TTNNTensorWrapper &inputTensorWrapper =
        context.getTensorPool().getTTNNTensorWrapperAndValidate(input);

    ::tt::runtime::ttnn::TTNNTensorWrapper &inputSlotWrapper =
        inputSlots[i].as<::tt::runtime::ttnn::TTNNTensorWrapper>(
            DeviceRuntime::TTNN);

    // Constants/parameters and KV cache tensors live on device and persist
    // across trace executions, so their versions are expected to match.
    // Regular inputs live on host and change between executions, so a version
    // mismatch is the expected case — we copy them from host to device below.
    if (inputTensorWrapper.getVersion() == inputSlotWrapper.getVersion()) {
      continue;
    }

    if (deviceResident) {
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
}

// Publishes the persistent output slots as the op's outputs.
static void
publishTraceOutputs(const ::tt::target::ttnn::CaptureOrExecuteTraceOp *op,
                    ProgramContext &context,
                    const std::vector<::tt::runtime::Tensor> &outputSlots) {
  LOG_ASSERT(op->outputs()->size() == outputSlots.size(),
             "Mismatched number of outputs, expected: ", op->outputs()->size(),
             " got: ", outputSlots.size());
  for (size_t i = 0; i < op->outputs()->size(); i++) {
    const ::tt::target::ttnn::TensorRef *output = op->outputs()->Get(i);
    context.getTensorPool().insertRuntimeTensorAndValidate(output,
                                                           outputSlots[i]);
  }
}

// Captures the trace against its persistent slots and returns the completed
// TraceData; the caller inserts it into the cache. The cache therefore only
// ever holds fully-captured traces.
//
// This is the single capture path: it is used both for the initial capture and
// to recapture a trace that has gone stale, in the latter case with the very
// same slots. The capture program takes the slots as arguments rather than
// allocating them, so it allocates nothing that outlives its own trace capture.
// That is what lets a recaptured trace simply inherit the current cache
// generation - it cannot have moved any buffer that a sibling cached trace
// baked into its command stream, so no sibling needs to be invalidated.
static TraceData captureTrace(
    const ::tt::target::ttnn::CaptureOrExecuteTraceOp *op,
    ProgramContext &context, std::vector<::tt::runtime::Tensor> inputSlots,
    std::vector<::tt::runtime::Tensor> outputSlots, uint64_t generationId) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  ::tt::runtime::Device deviceHandle = context.getDeviceHandle();

  // Program inputs are, in order: the host-staged inputs, the trace input
  // slots, then the trace output slots - matching the capture function's
  // signature. ProgramExecutor holds pointers into this vector, so it must
  // outlive the executor.
  std::vector<::tt::runtime::Tensor> captureInputs;
  captureInputs.reserve(op->inputs()->size() + inputSlots.size() +
                        outputSlots.size());
  for (const ::tt::target::ttnn::TensorRef *input : *op->inputs()) {
    if (!::tt::runtime::ttnn::utils::inDeviceMemory(input)) {
      captureInputs.push_back(tensorPool.getRuntimeTensorAndValidate(input));
    }
  }
  captureInputs.insert(captureInputs.end(), inputSlots.begin(),
                       inputSlots.end());
  captureInputs.insert(captureInputs.end(), outputSlots.begin(),
                       outputSlots.end());

  std::vector<::tt::runtime::GlobalSemaphore> semaphoreInputs =
      utils::collectSemaphoreInputs(op->semaphore_inputs(), context);

  ProgramExecutor executor(deviceHandle, context.getExecutableHandle(),
                           op->capture_program_id(), captureInputs,
                           /*constEvalProgram=*/false, semaphoreInputs);
  executor.execute();
  std::vector<::tt::runtime::Tensor> outputTensors =
      executor.gatherOutputTensors();

  LOG_ASSERT(outputTensors.size() == 1,
             "Capture program must return exactly the new trace id, got: ",
             outputTensors.size());

  const ::ttnn::Tensor &traceIdTensor =
      ::tt::runtime::ttnn::utils::getTTNNTensorFromRuntimeTensor(
          outputTensors[0]);
  LOG_ASSERT(traceIdTensor.dtype() == ::ttnn::DataType::UINT32,
             "Trace ID must be UINT32");

  TraceData traceData{
      .traceId = ::ttnn::MeshTraceId(
          ::tt::runtime::ttnn::utils::getScalarFromTensor<uint32_t>(
              traceIdTensor)),
      .inputTensors = std::move(inputSlots),
      .outputTensors = std::move(outputSlots),
      // Inherit the caller's current generation: the capture allocated no
      // buffer that outlived it, so it invalidated nothing.
      .generationId = generationId};

  // The capture program replays the trace once, so the output slots already
  // hold this invocation's results.
  publishTraceOutputs(op, context, traceData.outputTensors);

  return traceData;
}

static void executeTrace(const ::tt::target::ttnn::CaptureOrExecuteTraceOp *op,
                         ProgramContext &context, TraceData &traceData) {
  ::tt::runtime::Device deviceHandle = context.getDeviceHandle();

  updateTraceInputSlots(op, context, traceData.inputTensors);

  ::ttnn::Tensor traceIdTensor =
      ::tt::runtime::ttnn::utils::createTTNNTensor<uint32_t>(
          &traceData.traceId.get(), ::ttnn::Shape(), ::ttnn::DataType::UINT32);

  std::vector<::tt::runtime::Tensor> inputTensors = {
      ::tt::runtime::ttnn::utils::createRuntimeTensorFromTTNN(traceIdTensor)};

  // The execute trace program only invokes ttnn.execute_trace(traceId); the
  // semaphores were baked into the captured trace at capture time and are not
  // arguments of the execute program. Passing them here would mismatch
  // program->semaphore_inputs() (which is empty for the execute program).
  ProgramExecutor executor(deviceHandle, context.getExecutableHandle(),
                           op->execute_program_id(), inputTensors,
                           /*constEvalProgram=*/false,
                           /*programSemaphoreInputs=*/{});
  executor.execute();

  publishTraceOutputs(op, context, traceData.outputTensors);
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
    LOG_DEBUG("Trace cache miss, allocating slots and capturing trace");
    traceCache->incrementGeneration();

    // Allocate the persistent slots once, then capture against them.
    std::vector<::tt::runtime::Tensor> slots = allocateTraceSlots(op, context);
    size_t numInputSlots = op->inputs()->size();
    std::vector<::tt::runtime::Tensor> inputSlots(
        slots.begin(), slots.begin() + numInputSlots);
    std::vector<::tt::runtime::Tensor> outputSlots(
        slots.begin() + numInputSlots, slots.end());

    // Fill the input slots before capturing; writes are rejected once the
    // capture window is open. The capture program does this itself for the
    // host-staged inputs it receives, so this only has to cover the
    // device-resident slots, which alias their inputs and are therefore
    // version-matched no-ops. Keeping the call here means the capture path and
    // the execute path establish the slots identically.
    updateTraceInputSlots(op, context, inputSlots,
                          /*deviceResidentOnly=*/true);
    TraceData traceData =
        captureTrace(op, context, std::move(inputSlots), std::move(outputSlots),
                     traceCache->getGenerationId());
    traceCache->insert(mainProgramKey, captureExecuteKey, std::move(traceData));

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
              "), recapturing into the existing slots");

    // Recapture through the very same capture program the initial capture used
    // and against the very same slots: take the entry out of the cache, hand
    // its slots to the recapture, and reinsert the result. The slots are never
    // reallocated, so a recapture cannot invalidate any sibling cached trace
    // and the recaptured trace simply inherits the current generation. Were the
    // slots reallocated instead, a recapture could shift the allocator state
    // other cached traces depend on and force them to be recaptured too - a
    // cycle that recaptures every trace on every iteration.
    TraceData staleTrace = traceCache->take(mainProgramKey, captureExecuteKey);

    // Release the old device-side trace before recapturing.
    ::ttnn::operations::trace::release_trace(&meshDevice, staleTrace.traceId);

    updateTraceInputSlots(op, context, staleTrace.inputTensors,
                          /*deviceResidentOnly=*/true);
    TraceData recapturedTrace = captureTrace(
        op, context, std::move(staleTrace.inputTensors),
        std::move(staleTrace.outputTensors), traceCache->getGenerationId());
    traceCache->insert(mainProgramKey, captureExecuteKey,
                       std::move(recapturedTrace));

    debug::Stats::get().incrementStat("TraceStaleRecapture");
    return;
  }

  LOG_DEBUG("Trace cache hit, executing trace directly");
  executeTrace(op, context, *traceData);
  debug::Stats::get().incrementStat("ExecutedTrace");
}

} // namespace tt::runtime::ttnn::operations::trace
