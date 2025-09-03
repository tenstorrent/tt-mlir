// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Interfaces/TTNNTensorSpecInterface.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"

namespace mlir::tt::ttnn {
#define GET_OP_CLASSES
#include "ttmlir/Dialect/TTNN/Interfaces/TTNNTensorSpecInterface.cpp.inc"
} // namespace mlir::tt::ttnn
