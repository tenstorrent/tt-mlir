// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_IR_TTNNREQUIRESEXPLICITBROADCASTINTERFACE_H
#define TTMLIR_DIALECT_TTNN_IR_TTNNREQUIRESEXPLICITBROADCASTINTERFACE_H

#include "mlir/IR/Operation.h"

namespace mlir::tt::ttnn {
// Verifies the TTNNRequiresExplicitBroadcastInterface
LogicalResult verifyTTNNRequiresExplicitBroadcastInterface(Operation *op);
} // namespace mlir::tt::ttnn

#endif
