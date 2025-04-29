// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTKERNELTOEMITC_TTKERNELTOEMITC_H
#define TTMLIR_CONVERSION_TTKERNELTOEMITC_TTKERNELTOEMITC_H

#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt {
#define GEN_PASS_DECL_CONVERTTTKERNELTOEMITC
#include "ttmlir/Conversion/Passes.h.inc"
} // namespace mlir::tt

#endif
