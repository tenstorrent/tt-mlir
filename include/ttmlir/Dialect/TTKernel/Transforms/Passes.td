// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTKERNEL_TTKERNELPASSES_TD
#define TTMLIR_TTMLIR_DIALECT_TTKERNEL_TTKERNELPASSES_TD

include "mlir/Pass/PassBase.td"

def TTKernelControlDstSection: Pass<"ttkernel-control-dst-section", "::mlir::ModuleOp"> {
  let summary = "Control Dst Register Critical Section";
  let description = [{
    Analyze the graph and insert tile_regs_* ops to control the dst critical section.
  }];
  let dependentDialects = ["mlir::tt::ttkernel::TTKernelDialect"];
}

def TTKernelInsertDeviceZoneScopes: Pass<"insert-device-zone-scopes", "::mlir::ModuleOp"> {
  let summary = "Insert DeviceZoneScopedN around selected TTKernel ops";
  let description = [{
    Insert scoping braces and DeviceZoneScopedN statements around TTKernel ops
    marked with DeviceZoneOpTrait, to annotate device zone regions prior to
    lowering to EmitC.
  }];
  let dependentDialects = ["mlir::emitc::EmitCDialect", "mlir::tt::ttkernel::TTKernelDialect"];
}
def TTKernelHoistInits: Pass<"ttkernel-hoist-inits", "::mlir::ModuleOp"> {
  let summary = "Hoist inits";
  let description = [{
    Analyze the graph and maximally hoist init ops to respective loop nests.
  }];
  let dependentDialects = ["mlir::tt::ttkernel::TTKernelDialect"];
}

#endif
