// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTKERNEL_IR_TTKERNELOPSTYPES_H
#define TTMLIR_DIALECT_TTKERNEL_IR_TTKERNELOPSTYPES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsAttrDefs.h.inc"

#endif
