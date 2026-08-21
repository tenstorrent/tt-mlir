// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_TTNN_OPERATIONS_CONTROL_FLOW_SUB_PROGRAM_H
#define TT_RUNTIME_TTNN_OPERATIONS_CONTROL_FLOW_SUB_PROGRAM_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/types.h"

#include <vector>

namespace tt::runtime::ttnn::operations::control_flow {

// Runs the region program `programId` with `sources` bound to its inputs, and
// returns its outputs.
//
// Inputs are bound positionally, so the caller's order must match the order the
// region's block arguments were serialized in.
//
// The sources are handed to the program as retained views rather than
// themselves, so that the nested program cannot deallocate values the caller
// still owns. An output that the program forwarded straight from an input is
// mapped back to the source it came from, so that no view escapes.
std::vector<::tt::runtime::Tensor>
runSubProgram(uint32_t programId, ProgramContext &context,
              const std::vector<::tt::runtime::Tensor> &sources,
              const std::vector<::tt::runtime::GlobalSemaphore> &semaphores);

} // namespace tt::runtime::ttnn::operations::control_flow

#endif // TT_RUNTIME_TTNN_OPERATIONS_CONTROL_FLOW_SUB_PROGRAM_H
