// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_DEBUG_H
#define RUNTIME_LIB_TTNN_OPERATIONS_DEBUG_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::debug {

void run(const ::tt::target::ttnn::AnnotateOp *op, ProgramContext &context);
void run(const ::tt::target::ttnn::BreakpointOp *op, ProgramContext &context);
void run(const ::tt::target::ttnn::PrintOp *op, ProgramContext &context);
void run(const ::tt::target::ttnn::MemorySnapshotOp *op, ProgramContext &context);
void run(const ::tt::target::ttnn::RegionStartOp *op, ProgramContext &context);
void run(const ::tt::target::ttnn::RegionEndOp *op, ProgramContext &context);

} // namespace tt::runtime::ttnn::operations::debug

#endif
