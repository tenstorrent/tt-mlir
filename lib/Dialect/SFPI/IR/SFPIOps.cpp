// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/SFPI/IR/SFPIOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "ttmlir/Dialect/SFPI/IR/SFPI.h"
#include "ttmlir/Dialect/SFPI/IR/SFPIOpsTypes.h"

#define GET_OP_CLASSES
#include "ttmlir/Dialect/SFPI/IR/SFPIOps.cpp.inc"

namespace mlir::tt::sfpi {

// Custom verification logic for SFPI operations can be added here

} // namespace mlir::tt::sfpi
