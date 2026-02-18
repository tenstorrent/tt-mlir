// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTCORE_TRANSFORMS_TRANSFORMS_H
#define TTMLIR_DIALECT_TTCORE_TRANSFORMS_TRANSFORMS_H

#include "mlir/IR/BuiltinOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include <string>

namespace mlir::tt::ttcore {

void registerDevice(ModuleOp module, Arch mockSystemDescArch = Arch::WormholeB0,
                    ArrayRef<int64_t> meshShape = {},
                    ArrayRef<Topology> meshTopology = {});

void registerDevice(ModuleOp module, const std::string &systemDescPath,
                    ArrayRef<int64_t> meshShape = {},
                    ArrayRef<Topology> meshTopology = {});

} // namespace mlir::tt::ttcore

#endif // TTMLIR_DIALECT_TTCORE_TRANSFORMS_TRANSFORMS_H
