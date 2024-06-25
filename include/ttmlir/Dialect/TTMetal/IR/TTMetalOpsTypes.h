// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTMETAL_IR_TTMETALOPSTYPES_H
#define TTMLIR_DIALECT_TTMETAL_IR_TTMETALOPSTYPES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"

#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsAttrDefs.h.inc"

#endif
