// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_TTML_TTML_H
#define RUNTIME_LIB_TTNN_OPERATIONS_TTML_TTML_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttml_runtime_export.h" // TTNN_RUNTIME_TTML_EXPORT (generated)
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::ttml {

// Single exported entry point for every ttml op.
TTNN_RUNTIME_TTML_EXPORT void run(const ::tt::target::ttnn::Operation *op,
                                  ProgramContext &context);

} // namespace tt::runtime::ttnn::operations::ttml

#endif // RUNTIME_LIB_TTNN_OPERATIONS_TTML_TTML_H
