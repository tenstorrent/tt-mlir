// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_INTERFACES_TTNNKERNELINTERFACE_H
#define TTMLIR_DIALECT_TTNN_INTERFACES_TTNNKERNELINTERFACE_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

namespace mlir::tt::ttnn {
class CoreRuntimeArgsAttr;
} // namespace mlir::tt::ttnn

#include "ttmlir/Dialect/TTNN/Interfaces/TTNNKernelInterface.h.inc"

#endif // TTMLIR_DIALECT_TTNN_INTERFACES_TTNNKERNELINTERFACE_H
