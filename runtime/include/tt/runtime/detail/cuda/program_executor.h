// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_CUDA_PROGRAM_EXECUTOR_H
#define TT_RUNTIME_DETAIL_CUDA_PROGRAM_EXECUTOR_H

#include "tt/runtime/debug.h"
#include "tt/runtime/detail/common/dylib.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/tensor_cache.h"
#include "tt/runtime/utils.h"
#include "ttmlir/Target/GPU/program_generated.h"
#include "llvm/ADT/StringMap.h"

namespace tt::runtime::cuda {

/**
 * ProgramExecutor handles the execution of programs compiled for CUDA.
 * It processes kernels in sequence.
 */
class ProgramExecutor {
public:
  // Constructor for executing a program
  ProgramExecutor(::tt::runtime::Binary &executableHandle,
                  std::vector<::tt::runtime::Tensor> &programInputs);

  /**
   * Executes all kernels in the program
   */
  void execute();

private:
  const ::gpu::Program *program;
  ::tt::runtime::Binary executableHandle;
  std::vector<::tt::runtime::Tensor> programInputs;
  llvm::StringMap<void *> tensorMap;
  llvm::StringMap<const ::gpu::MemRefDesc *> memrefDescMap;

  void runKernel(const ::gpu::Kernel *kernel);
};

} // namespace tt::runtime::cuda

#endif
