// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_IR_TTIROPSTYPES_H
#define TTMLIR_DIALECT_TTIR_IR_TTIROPSTYPES_H

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/TTIR/IR/TTIROpsTypeDefs.h.inc"

#endif // TTMLIR_DIALECT_TTIR_IR_TTIROPSTYPES_H
