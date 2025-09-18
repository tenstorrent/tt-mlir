// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_TRANSFORMS_PASSES_TD
#define TTMLIR_DIALECT_D2M_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def D2MGenericApplyInterchange : Pass<"d2m-generic-apply-interchange", "::mlir::ModuleOp"> {
  let summary = "Apply loop interchange on D2M generic ops.";
  let options = [
    ListOption<"matmulInterchange", "matmul-interchange", "int64_t",
      "Set an interchange for generic ops that match matmul style patterns.">
  ];
}

def D2MGenericTileComputeLoops : Pass<"d2m-generic-tile-compute-loops", "::mlir::ModuleOp"> {
  let summary = "Tile affine compute loops for D2M generic ops.";
  let options = [
    Option<"maxDstRegisterSizeTiles", "max-dst-register-size-tiles", "unsigned", "0",
      "Override max dst size tiles or 0 if unset.">
  ];
}

def D2MInsertDstRegisterAccess : Pass<"d2m-insert-dst-register-access", "::mlir::ModuleOp"> {
  let summary = "Insert destination register access for compute regions.";
  let options = [
    Option<"useTileMatmul", "use-tile-matmul", "bool", "true", "Use tile_matmul lowering path.">
  ];
}

def D2MGenericLinearizeMemref : Pass<"d2m-generic-linearize-memref", "::mlir::ModuleOp"> {
  let summary = "Linearize memref accesses for D2M generic ops.";
}

def D2MGenericGenerateDatamovement : Pass<"d2m-generic-generate-datamovement", "::mlir::ModuleOp"> {
  let summary = "Generate D2M datamovement regions for generic ops.";
  let dependentDialects = ["mlir::tt::d2m::D2MDialect"];
}

def D2MGenericLowerDMAs : Pass<"d2m-generic-lower-dmas", "::mlir::ModuleOp"> {
  let summary = "Lower D2M DMA ops to fully indexed forms.";
}

def D2MGenericHWThreadSelection : Pass<"d2m-generic-hw-thread-selection", "::mlir::ModuleOp"> {
  let summary = "Assign datamovement regions to hardware threads.";
}

def D2MGenericGenerateLoops : Pass<"d2m-generic-generate-loops", "::mlir::ModuleOp"> {
  let summary = "Lower affine regions to imperative loops.";
}

def D2MGenericRegionsToFuncs : Pass<"d2m-generic-regions-to-funcs", "::mlir::ModuleOp"> {
  let summary = "Move D2M generic regions to top-level funcs.";
}

def D2MLowerToLayout : Pass<"d2m-lower-to-layout", "::mlir::ModuleOp"> {
  let summary = "Lower layout transitions to D2M generic ops.";
}

#endif // TTMLIR_DIALECT_D2M_TRANSFORMS_PASSES_TD
