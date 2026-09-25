// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_TTNN_OPERATIONS_CONTROL_FLOW_CASE_OP_H
#define TT_RUNTIME_TTNN_OPERATIONS_CONTROL_FLOW_CASE_OP_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/Target.h"

namespace tt::runtime::ttnn::operations::control_flow {

void run(const ::tt::target::ttnn::CaseOp *op, ProgramContext &context);

} // namespace tt::runtime::ttnn::operations::control_flow

#endif // TT_RUNTIME_TTNN_OPERATIONS_CONTROL_FLOW_CASE_OP_H
