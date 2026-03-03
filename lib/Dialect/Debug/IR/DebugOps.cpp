// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/Debug/IR/DebugOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "ttmlir/Dialect/Debug/IR/Debug.h"

#define GET_OP_CLASSES
#include "ttmlir/Dialect/Debug/IR/DebugOps.cpp.inc"

namespace mlir::tt::debug {

// Custom verification logic for Debug operations can be added here

} // namespace mlir::tt::debug
