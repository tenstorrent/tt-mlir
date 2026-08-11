// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/control_flow/while_op.h"
#include "operations/control_flow/sub_program.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
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

// Both sub-programs take `inits ++ captures` as their inputs, and the loop
// rebinds the carried half every iteration.
std::vector<::tt::runtime::Tensor>
concatenate(const std::vector<::tt::runtime::Tensor> &carried,
            const std::vector<::tt::runtime::Tensor> &captures) {
  std::vector<::tt::runtime::Tensor> sources;
  sources.reserve(carried.size() + captures.size());
  sources.insert(sources.end(), carried.begin(), carried.end());
  sources.insert(sources.end(), captures.begin(), captures.end());
  return sources;
}

// Evaluates the condition program and reads its result back to host, the
// device-to-host synchronization a data-dependent loop pays per iteration. The
// compiler guarantees the condition is already a host-resident single-element
// uint32 tensor (see TTNNLayoutWhileOpRewriter), so nothing is moved here.
bool evaluateCondition(
    const ::tt::target::ttnn::WhileOp *op, ProgramContext &context,
    const std::vector<::tt::runtime::Tensor> &carried,
    const std::vector<::tt::runtime::Tensor> &captures,
    const std::vector<::tt::runtime::GlobalSemaphore> &semaphores) {
  std::vector<::tt::runtime::Tensor> outputs =
      runSubProgram(op->cond_program_id(), context,
                    concatenate(carried, captures), semaphores);
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
      LOG_ASSERT(iteration < maxIterations, "While loop exceeded ",
                 maxIterations,
                 " iterations; set TT_RUNTIME_WHILE_MAX_ITERATIONS to raise "
                 "the limit");
      if (!evaluateCondition(op, context, carried, captures, semaphores)) {
        break;
      }
    }

    std::vector<::tt::runtime::Tensor> next =
        runSubProgram(op->body_program_id(), context,
                      concatenate(carried, captures), semaphores);
    LOG_ASSERT(next.size() == carried.size(), "While body program returned ",
               next.size(), " values but the loop carries ", carried.size());

    // Dropping the previous iteration's values here releases the last
    // reference to any that the body did not carry forward, which frees them.
    carried = std::move(next);
  }

  // `carried` holds the inits, the body's own outputs, or - for a value the
  // body yielded unchanged - a non-retained view of one of those. None of them
  // is retained, so the results are published exactly as a func call would
  // publish them.
  LOG_ASSERT(carried.size() == op->outputs()->size(),
             "Number of outputs does not match");
  for (size_t i = 0; i < op->outputs()->size(); i++) {
    tensorPool.insertRuntimeTensorAndValidate(op->outputs()->Get(i),
                                              carried[i]);
  }
}

} // namespace tt::runtime::ttnn::operations::control_flow
