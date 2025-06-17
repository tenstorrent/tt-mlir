// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_IR_TTNNOPSTYPES_H
#define TTMLIR_DIALECT_TTNN_IR_TTNNOPSTYPES_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h.inc"

#endif // TTMLIR_DIALECT_TTNN_IR_TTNNOPSTYPES_H
