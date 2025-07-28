// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTKERNEL_TTKERNELPASSES_TD
#define TTMLIR_TTMLIR_DIALECT_TTKERNEL_TTKERNELPASSES_TD

include "mlir/Pass/PassBase.td"

def TTKernelControlDstSection
    : Pass<"ttkernel-control-dst-section", "::mlir::ModuleOp"> {
  let summary = "Control Dst Register Critical Section";
  let description = [{
    Analyze the graph and insert tile_regs_* ops to control the dst critical section.
  }];
  let dependentDialects = ["mlir::tt::ttkernel::TTKernelDialect"];
}

#endif
