// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_CCL_MESH_PARTITION_H
#define RUNTIME_LIB_TTNN_OPERATIONS_CCL_MESH_PARTITION_H

#include "tt/runtime/detail/ttnn/types/types.h"

namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::MeshPartitionOp *op,
         ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::ccl

#endif
