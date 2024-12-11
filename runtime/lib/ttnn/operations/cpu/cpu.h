// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_CPU_CPU_H
#define RUNTIME_LIB_TTNN_OPERATIONS_CPU_CPU_H

#include "tt/runtime/ttnn/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::cpu {

struct wrapped_tensor {
  float *start;
  float *aligned_start;
  int64_t start_idx;
  size_t rank; // this tells us how many elements will be in sizes_and_strides
  int64_t *sizes_and_strides;
};

// generic signature to call all our funcs; args will be an array of input
// tensors + a counter to tell us how many
using WrappedFunc = wrapped_tensor (*)(wrapped_tensor *, size_t);

void run(const ::tt::target::ttnn::CpuOp *op, ProgramContext &context);

} // namespace tt::runtime::ttnn::operations::cpu

#endif
