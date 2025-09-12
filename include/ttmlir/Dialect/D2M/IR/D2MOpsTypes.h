// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_IR_D2MOPSTYPES_H
#define TTMLIR_DIALECT_D2M_IR_D2MOPSTYPES_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/D2M/IR/D2MOpsTypeDefs.h.inc"

#endif // TTMLIR_DIALECT_D2M_IR_D2MOPSTYPES_H
