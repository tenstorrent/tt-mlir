// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/control_flow/sub_program.h"
#include "tt/runtime/detail/ttnn/program_executor.h"
#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::control_flow {

namespace {

// Returns a private view of `tensor`: a fresh wrapper over the same underlying
// ttnn tensor, with its own retain flag.
//
// Retain is not this op's flag to touch on the caller's wrapper, since
// const-eval, trace and host callers all keep tensors retained for their own
// reasons. Setting it on a private view leaves theirs untouched by
// construction.
//
// No data is copied. The view shares the underlying buffer, which bumps its
// refcount, so a non-forced deallocation of either wrapper frees nothing while
// the other is still alive.
::tt::runtime::Tensor view(const ::tt::runtime::Tensor &tensor, bool retain) {
  const ::tt::runtime::ttnn::TTNNTensorWrapper &wrapper =
      tensor.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN);
  // Copied once here, so the view's event and the source's are independent from
  // this point on. Revisit if non-blocking readbacks or multiple command queues
  // become widely used.
  std::optional<::ttnn::MeshEvent> meshEvent = wrapper.getMeshEvent();

  ::tt::runtime::Tensor result =
      ::tt::runtime::ttnn::utils::createRuntimeTensorFromTTNN(
          wrapper.getTensor(), meshEvent, retain);

  // Keep the version aligned with the source so consumers that key off it (the
  // const-eval cache, trace input-slot staleness) cannot mistake the view for a
  // newer value.
  result.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN)
      .syncVersion(wrapper);
  return result;
}

} // namespace

std::vector<::tt::runtime::Tensor>
runSubProgram(uint32_t programId, ProgramContext &context,
              const std::vector<::tt::runtime::Tensor> &sources,
              const std::vector<::tt::runtime::GlobalSemaphore> &semaphores) {
  // Retained views, not the tensors themselves: the sub-program must not
  // deallocate what the caller still needs, and the views die with this vector,
  // so no retain state outlives the call.
  //
  // ProgramExecutor's tensor pool holds raw pointers into this vector, so it
  // has to be a distinct, stable object for the executor's whole lifetime.
  std::vector<::tt::runtime::Tensor> inputs;
  inputs.reserve(sources.size());
  for (const ::tt::runtime::Tensor &tensor : sources) {
    inputs.push_back(view(tensor, /*retain=*/true));
  }

  ProgramExecutor executor(context.getDeviceHandle(),
                           context.getExecutableHandle(), programId, inputs,
                           /*constEvalProgram=*/false, semaphores);
  executor.execute();
  std::vector<::tt::runtime::Tensor> outputs = executor.gatherOutputTensors();

  // A program that yields one of its arguments unchanged hands back the very
  // view that was passed in, because the pool resolves that output id straight
  // to the input slot. That view dies with `inputs` and its retain flag is
  // raised, so it cannot be published as a result; replace it with a fresh,
  // non-retained view of the same tensor.
  //
  // Deliberately not the source tensor itself. The caller sees the source and
  // the result as unrelated values and deallocates each at its own last use, so
  // one wrapper for both would let the first of those frees invalidate the
  // other. A second wrapper bumps the buffer's refcount, which makes the first
  // non-forced deallocation a no-op and leaves the survivor valid.
  for (::tt::runtime::Tensor &output : outputs) {
    const void *outputWrapper =
        &output.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN);
    for (size_t i = 0; i < inputs.size(); i++) {
      const void *inputWrapper =
          &inputs[i].as<::tt::runtime::ttnn::TTNNTensorWrapper>(
              DeviceRuntime::TTNN);
      if (inputWrapper == outputWrapper) {
        output = view(sources[i], /*retain=*/false);
        break;
      }
    }
  }

  return outputs;
}

} // namespace tt::runtime::ttnn::operations::control_flow
