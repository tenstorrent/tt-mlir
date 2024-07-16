// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTIR_TTIRPASSES_TD
#define TTMLIR_TTMLIR_DIALECT_TTIR_TTIRPASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertTosaToTTIR : Pass<"convert-tosa-to-ttir", "::mlir::ModuleOp"> {
  let summary = "";
  let description = [{
    Convert TOSA ops to TTIR ops.
  }];
}

def TTIRGeneric: Pass<"ttir-generic", "::mlir::ModuleOp"> {
  let summary = "";
  let description = [{
    Wrap top level ops in a generic op.
  }];
}

def TTIRGenericRegionOperandsToMemref: Pass<"ttir-generic-region-operands-to-memref", "::mlir::ModuleOp"> {
  let summary = "";
  let description = [{
    Convert generic region operands   to memref.
}];
}

def TTIRLayout : Pass<"ttir-layout", "::mlir::ModuleOp"> {
  let summary = "Tensor tilize all generic ops.";
  let description = [{
    Transition between different tensor layouts.
  }];
}

def TTIRAllocate: Pass<"ttir-allocate", "::mlir::ModuleOp"> {
  let summary = "Allocate tensors.";
  let description = [{
    todo
  }];
}

def TTIRGridSet: Pass<"ttir-grid-set", "::mlir::ModuleOp"> {
  let summary = "Determine grid size for ops.";
  let description = [{
    Go through the ops, set grid size for each op based on grid analysis,
    by updating layout attribute of each op.
  }];
}

#endif
