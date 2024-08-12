// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTIR_TTIRPASSES_TD
#define TTMLIR_TTMLIR_DIALECT_TTIR_TTIRPASSES_TD

include "mlir/Pass/PassBase.td"

def TTIRImplicitDevice: Pass<"ttir-implicit-device", "::mlir::ModuleOp"> {
  let summary = "Create an implicit device";
  let description = [{
    This pass will take a view of the system descriptor and create an implicit
    device around it.
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
    Convert generic region operands to memref.
  }];
}

def TTIRLayout: Pass<"ttir-layout", "::mlir::ModuleOp"> {
  let summary = "Tensor tilize all generic ops.";
  let description = [{
    Transition between different tensor layouts.
  }];

  let options = [
    Option<"initMemorySpace", "init-memory-space",
          "::mlir::tt::MemorySpace",
          /*default=*/"::mlir::tt::MemorySpace::System",
           "Set the initial memory space for tensors to start in">,
  ];
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
  let options = [
    Option<"overrideGridSizes", "override-grid-sizes",
          "llvm::StringMap<SmallVector<int64_t, 2>>",
          /*default=*/"llvm::StringMap<SmallVector<int64_t, 2>>()",
           "Override grid sizes for specific ops.">,
  ];
}

def TTIRLoadSystemDesc: Pass<"ttir-load-system-desc", "::mlir::ModuleOp"> {
  let summary = "Load system desc.";
  let description = [{
    Load system descriptor as a compiler pass.
  }];

  list<Option> options = [
        Option<"path", "path", "std::string", "", "System desc path">,
    ];
}

#endif
