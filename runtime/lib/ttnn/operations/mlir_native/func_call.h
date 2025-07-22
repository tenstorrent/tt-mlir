// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_TTNN_OPERATIONS_MLIR_NATIVE_FUNC_CALL_H
#define TT_RUNTIME_TTNN_OPERATIONS_MLIR_NATIVE_FUNC_CALL_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/Target.h"

namespace tt::runtime::ttnn::operations::mlir_native {

void run(const ::tt::target::ttnn::FuncCallOp *op, ProgramContext &context);

} // namespace tt::runtime::ttnn::operations::mlir_native

#endif // TT_RUNTIME_TTNN_OPERATIONS_MLIR_NATIVE_FUNC_CALL_H
