// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_CCL_ALL_TO_ALL_DISPATCH_H
#define RUNTIME_LIB_TTNN_OPERATIONS_CCL_ALL_TO_ALL_DISPATCH_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include <tt-metalium/mesh_coord.hpp>

namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::AllToAllDispatchOp *op,
         ProgramContext &context);

// Shared state: dispatch saves the original mesh shape so combine
// can flatten and restore.
inline tt::tt_metal::distributed::MeshShape g_originalMeshShape;
inline bool g_meshFlattened = false;
} // namespace tt::runtime::ttnn::operations::ccl

#endif
