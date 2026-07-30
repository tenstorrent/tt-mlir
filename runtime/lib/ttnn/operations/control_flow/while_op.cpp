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

// A tensor whose retain flag this op raised, together with the value it had
// beforehand.
struct SavedRetain {
  ::tt::runtime::Tensor tensor;
  bool previous;
};

// Keeps tensors alive across a nested program boundary. Without this the
// callee's ttnn.deallocate ops would free values the loop still needs: the
// captures on every subsequent iteration, and the carried values that feed the
// next one.
//
// The previous value has to be put back rather than forced to false, because
// the loop is not the only owner of this flag. Const-eval keeps its cached
// outputs retained, trace keeps its input and output slots retained, and a
// host caller can retain a tensor it passes in as a program argument. Any of
// those can show up as a loop init or capture -- in the common case where the
// loop bounds are const-eval'd, they do. Clearing their claim would let a
// later deallocate free a tensor that is still live.
std::vector<SavedRetain> retain(std::vector<::tt::runtime::Tensor> &tensors) {
  std::vector<SavedRetain> saved;
  saved.reserve(tensors.size());
  for (::tt::runtime::Tensor &tensor : tensors) {
    ::tt::runtime::ttnn::TTNNTensorWrapper &wrapper =
        tensor.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN);
    saved.push_back({tensor, wrapper.shouldRetain()});
    wrapper.setRetain(true);
  }
  return saved;
}

// Undoes a `retain`. Nested sets must be released in reverse order of
// acquisition, since the same tensor can appear in more than one of them (an
// init is frequently also a capture).
void release(std::vector<SavedRetain> &saved) {
  for (SavedRetain &entry : saved) {
    entry.tensor.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN)
        .setRetain(entry.previous);
  }
  saved.clear();
}

std::vector<::tt::runtime::Tensor>
runSubProgram(uint32_t programId, ProgramContext &context,
              const std::vector<::tt::runtime::Tensor> &carried,
              const std::vector<::tt::runtime::Tensor> &captures,
              const std::vector<::tt::runtime::GlobalSemaphore> &semaphores) {
  // ProgramExecutor's tensor pool holds raw pointers into this vector, so it
  // has to be a distinct, stable object for the executor's whole lifetime.
  // Never reuse or reassign one across iterations.
  std::vector<::tt::runtime::Tensor> inputs;
  inputs.reserve(carried.size() + captures.size());
  inputs.insert(inputs.end(), carried.begin(), carried.end());
  inputs.insert(inputs.end(), captures.begin(), captures.end());

  ProgramExecutor executor(context.getDeviceHandle(),
                           context.getExecutableHandle(), programId, inputs,
                           /*constEvalProgram=*/false, semaphores);
  executor.execute();
  return executor.gatherOutputTensors();
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

  // Captures are read by every iteration, and the initial carried values are
  // owned by the enclosing program, so neither may be freed by a sub-program.
  std::vector<SavedRetain> capturesRetained = retain(captures);
  std::vector<SavedRetain> carriedRetained = retain(carried);

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

    // Release the outgoing values before retaining the incoming ones. A body
    // that passes a value straight through yields the tensor it was given, so
    // doing it the other way round would snapshot the retain flag this op had
    // already raised and then clear it again when the old set is released.
    // Nothing executes in between, so the values are never unprotected.
    release(carriedRetained);
    carriedRetained = retain(next);
    carried = std::move(next);
  }

  // Reverse order of acquisition: a tensor is frequently both an init and a
  // capture, and only the outermost release should have the last word.
  release(carriedRetained);
  release(capturesRetained);

  // The enclosing program owns the results from here on, exactly as it would
  // the results of a func call, so their retain flags are left as `release`
  // restored them.
  LOG_ASSERT(carried.size() == op->outputs()->size(),
             "Number of outputs does not match");
  for (size_t i = 0; i < op->outputs()->size(); i++) {
    tensorPool.insertRuntimeTensorAndValidate(op->outputs()->Get(i),
                                              carried[i]);
  }
}

} // namespace tt::runtime::ttnn::operations::control_flow
