// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_IR_TTNNOPSATTRS_H
#define TTMLIR_DIALECT_TTNN_IR_TTNNOPSATTRS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrDefs.h.inc"

#endif // TTMLIR_DIALECT_TTNN_IR_TTNNOPSATTRS_H
