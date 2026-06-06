// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Interfaces/TTNNDistributedOpInterface.h"

namespace mlir::tt::ttnn {
#define GET_OP_CLASSES
#include "ttmlir/Dialect/TTNN/Interfaces/TTNNDistributedOpInterface.cpp.inc"
} // namespace mlir::tt::ttnn
