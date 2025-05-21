// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTNN_PROGRAM_EXECUTOR_H
#define TT_RUNTIME_DETAIL_TTNN_PROGRAM_EXECUTOR_H

#include "tt/runtime/detail/debug.h"
#include "tt/runtime/detail/dylib.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/types.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/tensor_cache.h"
#include "tt/runtime/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn {

class ProgramContext; // Forward declaration

/**
 * ProgramExecutor handles the execution of TTNN programs.
 * It processes operations in sequence and maintains program context.
 */
class ProgramExecutor {
public:
  // Constructor for executing a program
  ProgramExecutor(const Binary &executableHandle,
                  std::vector<::tt::runtime::Tensor> &programInputs,
                  std::shared_ptr<::ttnn::MeshDevice> meshDevice,
                  const size_t programIndex = 0);

  /**
   * Executes pre and post operation callbacks if registered
   */
  void
  runCallback(std::optional<::tt::runtime::debug::Hooks::CallbackFn> callback,
              Binary &executableHandle,
              const ::tt::target::ttnn::Operation *opContext,
              ProgramContext *programContext);

  /**
   * Executes all operations in the program
   */
  void execute();

  /**
   * Returns the program context
   */
  ProgramContext &getContext() { return *context; }

  /**
   * Gathers and returns output tensors from the program
   */
  std::vector<::tt::runtime::Tensor> gatherOutputTensors();

private:
  const ::tt::target::ttnn::Program *program;
  Binary executableHandle;
  std::unique_ptr<ProgramContext> context;

  /**
   * Executes a single operation
   */
  void runOperation(const ::tt::target::ttnn::Operation *op);

  void dumpPerfCountersIfNeeded(::ttnn::MeshDevice &meshDevice,
                                bool counterOverride = false);
};

} // namespace tt::runtime::ttnn

#endif
