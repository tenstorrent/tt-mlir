// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Interfaces/TTNNDeviceOperandInterface.h"

namespace mlir::tt::ttnn {
#define GET_OP_CLASSES
#include "ttmlir/Dialect/TTNN/Interfaces/TTNNDeviceOperandInterface.cpp.inc"
} // namespace mlir::tt::ttnn
