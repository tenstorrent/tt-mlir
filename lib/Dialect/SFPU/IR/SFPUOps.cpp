// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/SFPU/IR/SFPUOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "ttmlir/Dialect/SFPU/IR/SFPU.h"

#define GET_OP_CLASSES
#include "ttmlir/Dialect/SFPU/IR/SFPUOps.cpp.inc"

namespace mlir::tt::sfpu {

// Custom verification logic for SFPU operations can be added here

} // namespace mlir::tt::sfpu