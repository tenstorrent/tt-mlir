// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_RUNTIME_EMPTY_H
#define TTNN_RUNTIME_EMPTY_H

#include "tt/runtime/ttnn/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::creation {

void run(const ::tt::target::ttnn::EmptyOp *op, ProgramContext &context);

} // namespace tt::runtime::ttnn::operations::creation

#endif
