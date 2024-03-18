// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.     ■ Too many errors emitted, stopping now
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_TTOPSTYPES_H
#define TTMLIR_TTMLIR_TTOPSTYPES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "ttmlir/TTOpsEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "ttmlir/TTOpsTypes.h.inc"

#endif
