// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_IR_TTIROPSTYPES_H
#define TTMLIR_DIALECT_TTIR_IR_TTIROPSTYPES_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/TTIR/IR/TTIROpsTypeDefs.h.inc"

#endif // TTMLIR_DIALECT_TTIR_IR_TTIROPSTYPES_H
