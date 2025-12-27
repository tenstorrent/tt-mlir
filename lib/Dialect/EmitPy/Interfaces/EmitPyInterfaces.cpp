// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/EmitPy/Interfaces/EmitPyInterfaces.h"

namespace mlir::tt::emitpy {
#define GET_OP_CLASSES
#include "ttmlir/Dialect/EmitPy/Interfaces/EmitPyInterfaces.cpp.inc"
} // namespace mlir::tt::emitpy
