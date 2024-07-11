// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_TTNNTOSERIALIZEDBINARY_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_TTNNTOSERIALIZEDBINARY_H

#include <memory>

#include "mlir/IR/BuiltinOps.h"

namespace mlir::tt::ttnn {
std::shared_ptr<void> emitTTNNAsFlatbuffer(OwningOpRef<ModuleOp> &moduleOp);
} // namespace mlir::tt::ttnn
#endif
