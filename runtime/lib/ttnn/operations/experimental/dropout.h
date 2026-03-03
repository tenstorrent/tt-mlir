// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_RUNTIME_TTNN_OPERATIONS_EXPERIMENTAL_DROPOUT_H
#define TTMLIR_RUNTIME_TTNN_OPERATIONS_EXPERIMENTAL_DROPOUT_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/TTNN/program_generated.h"
#pragma clang diagnostic pop
#include "tt/runtime/detail/ttnn/types/types.h"

namespace tt::runtime::ttnn::operations::experimental {

void run(const ::tt::target::ttnn::DropoutOp *op, ProgramContext &context);

} // namespace tt::runtime::ttnn::operations::experimental

#endif // TTMLIR_RUNTIME_TTNN_OPERATIONS_EXPERIMENTAL_DROPOUT_H
