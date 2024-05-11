// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTMETAL_TTMETALOPSTYPES_H
#define TTMLIR_TTMLIR_DIALECT_TTMETAL_TTMETALOPSTYPES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "ttmlir/Dialect/TTMetal/TTMetalOpsEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/TTMetal/TTMetalOpsTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "ttmlir/Dialect/TTMetal/TTMetalOpsAttrDefs.h.inc"

#endif
