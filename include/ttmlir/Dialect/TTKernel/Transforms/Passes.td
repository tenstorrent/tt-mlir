// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTKERNEL_TTKERNELPASSES_TD
#define TTMLIR_TTMLIR_DIALECT_TTKERNEL_TTKERNELPASSES_TD

include "mlir/Pass/PassBase.td"

def TTKernelControlDstSection: Pass<"ttkernel-control-dst-section", "::mlir::func::FuncOp"> {
  let summary = "Control Dst Register Critical Section";
  let description = [{
    Analyze the graph and insert tile_regs_* ops to control the dst critical section.
  }];
  let dependentDialects = ["mlir::tt::ttkernel::TTKernelDialect"];
}

def TTKernelInsertDeviceZoneScopes: Pass<"insert-device-zone-scopes", "::mlir::func::FuncOp"> {
  let summary = "Insert DeviceZoneScopedN around selected TTKernel ops";
  let description = [{
    Insert scoping braces and DeviceZoneScopedN statements around TTKernel ops
    passed in the traits parameter, to annotate device zone regions prior to
    lowering to EmitC.
  }];
  let dependentDialects = ["mlir::emitc::EmitCDialect", "mlir::tt::ttkernel::TTKernelDialect"];
  let options = [
    ListOption<"traitNames", "traits", "std::string", "TTKernel traits to instrument (fpu, sfpu, init, unary, binary, ternary, device-zone, trid-noc, layout, all). Empty = nothing" >
  ];
}

def TTKernelHoistInits: Pass<"ttkernel-hoist-inits", "::mlir::func::FuncOp"> {
  let summary = "Hoist inits";
  let description = [{
    Analyze the graph and maximally hoist init ops to respective loop nests.
  }];
  let dependentDialects = ["mlir::tt::ttkernel::TTKernelDialect"];
}

def TTKernelDedupInits: Pass<"ttkernel-dedup-inits", "::mlir::func::FuncOp"> {
  let summary = "Eliminate redundant TTKernel init ops in straight-line code";
  let description = [{
    A peephole that removes init ops left redundant by the per-op D2M->TTKernel
    lowering (each compute op emits its own init, so many identical inits pile up
    in straight-line code that ttkernel-hoist-inits cannot reach because it only
    lifts inits out of scf.for loops). Two value-preserving rules:

      R1 (adjacent-duplicate): an init op identical to the op immediately
         preceding it (same op, operands and attributes) is a no-op and is erased.
         Catches the duplicate compute_kernel_hw_startup / init_sfpu calls hoisted
         to the top of the kernel.

      R2 (reduce coalesce): for a run of same-config reduces emitted as
         reduce_init(X); reduce_tile; reduce_uninit; reduce_init(X); reduce_tile;
         reduce_uninit; ... the intermediate reduce_uninit/reduce_init pairs are
         erased, leaving init-once / reduce-many / uninit-once. Safe by the
         reduce_uninit contract: it may be omitted when the next op is a reduce
         across the same dimension.

    Neither rule changes the computed values (PCC-preserving); both only remove
    redundant hardware reconfiguration. Run after ttkernel-hoist-inits.
  }];
  let dependentDialects = ["mlir::tt::ttkernel::TTKernelDialect"];
}

#endif
