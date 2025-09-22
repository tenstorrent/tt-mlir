// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_TTNN_OPERATIONS_TENSOR_SERIALIZATION_LOAD_TENSOR_H
#define TT_RUNTIME_TTNN_OPERATIONS_TENSOR_SERIALIZATION_LOAD_TENSOR_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/Target.h"

namespace tt::runtime::ttnn::operations::tensor_serialization {

void run(const ::tt::target::ttnn::LoadTensorOp *op, ProgramContext &context);

} // namespace tt::runtime::ttnn::operations::tensor_serialization

#endif // TT_RUNTIME_TTNN_OPERATIONS_TENSOR_SERIALIZATION_LOAD_TENSOR_H
