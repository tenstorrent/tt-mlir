// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TT_TRANSFORMS_TRANSFORMS_H
#define TTMLIR_DIALECT_TT_TRANSFORMS_TRANSFORMS_H

#include "mlir/IR/BuiltinOps.h"

#include <string>

namespace mlir::tt {

void registerDevice(ModuleOp module, std::string path = {},
                    ArrayRef<int64_t> meshShape = {});

} // namespace mlir::tt

#endif // TTMLIR_DIALECT_TT_TRANSFORMS_TRANSFORMS_H
