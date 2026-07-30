// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/control_flow/while_op.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/program_executor.h"
#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include <cstdlib>

namespace tt::runtime::ttnn::operations::control_flow {

namespace {

// Backstop against a mis-lowered predicate hanging the device forever. Only
// applies to data-dependent loops; counted loops are bounded by trip_count.
constexpr uint64_t kDefaultMaxIterations = 1000000;

uint64_t getMaxIterations() {
  static const uint64_t maxIterations = [] {
    const char *env = std::getenv("TT_RUNTIME_WHILE_MAX_ITERATIONS");
    if (!env) {
      return kDefaultMaxIterations;
    }
    char *end = nullptr;
    unsigned long long parsed = std::strtoull(env, &end, 10);
    if (end == env || parsed == 0) {
      LOG_WARNING("Ignoring invalid TT_RUNTIME_WHILE_MAX_ITERATIONS=", env);
      return kDefaultMaxIterations;
    }
    return static_cast<uint64_t>(parsed);
  }();
  return maxIterations;
}

// Returns a private, retained view of `tensor`: a fresh wrapper over the same
// underlying ttnn tensor, with its own retain flag already raised.
//
// A nested program must not deallocate the values the loop hands it, and retain
// is the flag that prevents that -- but it is not this op's flag to touch.
// Const-eval keeps its cached outputs retained, trace keeps its input and
// output slots retained, and a host caller can retain a tensor it passes in as
// a program argument. Those are not hypothetical: when the loop bounds are
// const-eval'd, every init and most captures arrive already retained and
// registered in GlobalTensorCache. Raising the flag on a private view leaves
// theirs untouched by construction, instead of relying on this op to correctly
// restore what it found.
//
// No data is copied. The view shares the underlying buffer, which also bumps
// its refcount and so makes any non-forced deallocation of it a no-op.
::tt::runtime::Tensor retainedView(const ::tt::runtime::Tensor &tensor) {
  const ::tt::runtime::ttnn::TTNNTensorWrapper &wrapper =
      tensor.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN);
  std::optional<::ttnn::MeshEvent> meshEvent = wrapper.getMeshEvent();

  ::tt::runtime::Tensor view =
      ::tt::runtime::ttnn::utils::createRuntimeTensorFromTTNN(
          wrapper.getTensor(), meshEvent, /*retain=*/true);

  // Keep the version aligned with the source so consumers that key off it (the
  // const-eval cache, trace input-slot staleness) cannot mistake the view for a
  // newer value. Trace does the same for its input slots.
  view.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN)
      .syncVersion(wrapper);
  return view;
}

std::vector<::tt::runtime::Tensor>
runSubProgram(uint32_t programId, ProgramContext &context,
              const std::vector<::tt::runtime::Tensor> &carried,
              const std::vector<::tt::runtime::Tensor> &captures,
              const std::vector<::tt::runtime::GlobalSemaphore> &semaphores) {
  // Retained views, not the tensors themselves: the sub-program must not
  // deallocate what the loop still needs, and this is the only scope in which
  // that protection is required. The views die with this vector, so no retain
  // state outlives the call and there is nothing to unwind on an early exit.
  //
  // ProgramExecutor's tensor pool holds raw pointers into this vector, so it
  // has to be a distinct, stable object for the executor's whole lifetime.
  // Never reuse or reassign one across iterations.
  std::vector<::tt::runtime::Tensor> inputs;
  inputs.reserve(carried.size() + captures.size());
  for (const ::tt::runtime::Tensor &tensor : carried) {
    inputs.push_back(retainedView(tensor));
  }
  for (const ::tt::runtime::Tensor &tensor : captures) {
    inputs.push_back(retainedView(tensor));
  }

  ProgramExecutor executor(context.getDeviceHandle(),
                           context.getExecutableHandle(), programId, inputs,
                           /*constEvalProgram=*/false, semaphores);
  executor.execute();
  std::vector<::tt::runtime::Tensor> outputs = executor.gatherOutputTensors();

  // A body that yields one of its arguments unchanged hands back the very view
  // that was passed in, because the pool resolves that output id straight to
  // the input slot. Map those back to the tensor the view was made from, so no
  // view escapes this function: a view's retain flag is raised, and publishing
  // one as a loop result would stop the enclosing program from ever freeing it.
  auto sourceOf = [&](size_t index) -> const ::tt::runtime::Tensor & {
    return index < carried.size() ? carried[index]
                                  : captures[index - carried.size()];
  };
  for (::tt::runtime::Tensor &output : outputs) {
    const void *outputWrapper =
        &output.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN);
    for (size_t i = 0; i < inputs.size(); i++) {
      const void *inputWrapper =
          &inputs[i].as<::tt::runtime::ttnn::TTNNTensorWrapper>(
              DeviceRuntime::TTNN);
      if (inputWrapper == outputWrapper) {
        output = sourceOf(i);
        break;
      }
    }
  }

  return outputs;
}

// Evaluates the condition program and reads its result back to host.
//
// This is the synchronization point that makes data-dependent loops expensive:
// one device-to-host transfer per iteration. The compiler guarantees the
// condition is already a host-resident single-element uint32 tensor (see
// TTNNLayoutWhileOpRewriter), so there is nothing to move here.
bool evaluateCondition(
    const ::tt::target::ttnn::WhileOp *op, ProgramContext &context,
    const std::vector<::tt::runtime::Tensor> &carried,
    const std::vector<::tt::runtime::Tensor> &captures,
    const std::vector<::tt::runtime::GlobalSemaphore> &semaphores) {
  std::vector<::tt::runtime::Tensor> outputs = runSubProgram(
      op->cond_program_id(), context, carried, captures, semaphores);
  LOG_ASSERT(outputs.size() == 1,
             "While condition program must return exactly one value, got ",
             outputs.size());

  const ::ttnn::Tensor &conditionTensor =
      outputs[0]
          .as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN)
          .getTensor();
  return ::tt::runtime::ttnn::utils::getScalarFromTensor<uint32_t>(
             conditionTensor) != 0;
}

} // namespace

void run(const ::tt::target::ttnn::WhileOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  std::vector<::tt::runtime::Tensor> carried;
  carried.reserve(op->inits()->size());
  for (const auto *init : *op->inits()) {
    carried.emplace_back(tensorPool.getRuntimeTensorAndValidate(init));
  }

  std::vector<::tt::runtime::Tensor> captures;
  captures.reserve(op->captures()->size());
  for (const auto *capture : *op->captures()) {
    captures.emplace_back(tensorPool.getRuntimeTensorAndValidate(capture));
  }

  std::vector<::tt::runtime::GlobalSemaphore> semaphores =
      utils::collectSemaphoreInputs(op->semaphore_inputs(), context);

  const ::flatbuffers::Optional<uint64_t> tripCount = op->trip_count();
  const uint64_t maxIterations = getMaxIterations();

  for (uint64_t iteration = 0;; ++iteration) {
    if (tripCount) {
      if (iteration >= *tripCount) {
        break;
      }
    } else {
      LOG_ASSERT(iteration < maxIterations,
                 "While loop exceeded ", maxIterations,
                 " iterations; set TT_RUNTIME_WHILE_MAX_ITERATIONS to raise "
                 "the limit");
      if (!evaluateCondition(op, context, carried, captures, semaphores)) {
        break;
      }
    }

    std::vector<::tt::runtime::Tensor> next = runSubProgram(
        op->body_program_id(), context, carried, captures, semaphores);
    LOG_ASSERT(next.size() == carried.size(),
               "While body program returned ", next.size(),
               " values but the loop carries ", carried.size());

    // Dropping the previous iteration's values here releases the last
    // reference to any that the body did not carry forward, which frees them.
    carried = std::move(next);
  }

  // `carried` is either the inits, still owned by the enclosing program, or the
  // body's own outputs -- never a view -- so the results are published exactly
  // as a func call would publish them, with retain untouched.
  LOG_ASSERT(carried.size() == op->outputs()->size(),
             "Number of outputs does not match");
  for (size_t i = 0; i < op->outputs()->size(); i++) {
    tensorPool.insertRuntimeTensorAndValidate(op->outputs()->Get(i),
                                              carried[i]);
  }
}

} // namespace tt::runtime::ttnn::operations::control_flow
