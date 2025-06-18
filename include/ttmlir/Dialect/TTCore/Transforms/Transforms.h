// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTCORE_TRANSFORMS_TRANSFORMS_H
#define TTMLIR_DIALECT_TTCORE_TRANSFORMS_TRANSFORMS_H

#include "mlir/IR/BuiltinOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include <string>

namespace mlir::tt {

void registerDevice(ModuleOp module,
                    tt::Arch mockSystemDescArch = tt::Arch::WormholeB0,
                    ArrayRef<int64_t> meshShape = {});

void registerDevice(ModuleOp module, const std::string &systemDescPath,
                    ArrayRef<int64_t> meshShape = {});

} // namespace mlir::tt

#endif // TTMLIR_DIALECT_TTCORE_TRANSFORMS_TRANSFORMS_H
