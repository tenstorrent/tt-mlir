// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#ifndef TTMLIR_DIALECT_TTNN_IR_TTNNWORKAROUNDINTERFACE_H
#define TTMLIR_DIALECT_TTNN_IR_TTNNWORKAROUNDINTERFACE_H

#include "ttmlir/Dialect/TTNN/IR/TTNNWorkarounds.h"

#include "mlir/IR/Operation.h"

namespace mlir::tt::ttnn::wa {
// Verifies the TTNNWorkaroundInterface
mlir::LogicalResult verifyTTNNWorkaroundInterface(mlir::Operation *op);
} // namespace mlir::tt::ttnn::wa

#include "ttmlir/Dialect/TTNN/IR/TTNNWorkaroundInterface.h.inc"

#endif
