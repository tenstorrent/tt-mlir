// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTMETAL_TTMETALPASSES_TD
#define TTMLIR_TTMLIR_DIALECT_TTMETAL_TTMETALPASSES_TD

include "mlir/Pass/PassBase.td"

def ApplyHostMemrefCallingConvention
    : Pass<"apply-host-memref-calling-convention", "::mlir::ModuleOp"> {
  let summary = "Apply host memref calling convention.";
  let constructor = "createApplyHostMemrefCallingConventionPass()";
  let dependentDialects = [];
  let description = [{
    We need to annotate host memref types with strides in order to represent its memory alignment.
    The assumption is that memrefs with 3 and more dimensions have rows that are packed and therefore
    don't need tile height alignment and columns are not packed and therefore need tile width
    alignment.

    For example, for a memref with logical shape of 128x2x2 the strides will be [64, 32, 1] to match
    the memory representation of rows of 32 columns with 2 values within each one.
  }];
}

#endif
