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

  let options = [
    ListOption<"meshShape", "mesh-shape", "int64_t",
               "Set the multi-device mesh shape.">,
  ];
}

def TTIRGenericKernel: Pass<"ttir-generic-kernel", "::mlir::ModuleOp"> {
  let summary = "";
  let description = [{
    Wrap top level kernel ops in a generic op.
  }];
}

def TTIRGenericRegion: Pass<"ttir-generic", "::mlir::ModuleOp"> {
  let summary = "";
  let description = [{
    Wrap top level elementwise ops in a generic op.
  }];
}

def TTIRGenericOpCBs: Pass<"ttir-generic-op-cbs", "::mlir::ModuleOp"> {
  let summary = "";
  let description = [{
    Insert circular buffer operands for generic ops.
  }];
}

def TTIRGenericRegionOperandsToMemref: Pass<"ttir-generic-region-operands-to-memref", "::mlir::ModuleOp"> {
  let summary = "";
  let description = [{
    Convert region operations to work on memref instead of tensors.
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
    Option<"defaultMemorySpace", "default-memory-space",
          "::mlir::tt::MemorySpace",
          /*default=*/"::mlir::tt::MemorySpace::DeviceDRAM",
           "Set the default memory space for layout pass to prefer for operation operands, if not constrained">,
    Option<"defaultDeviceMemoryLayout", "default-device-memory-layout",
          "::mlir::tt::TensorMemoryLayout",
          /*default=*/"::mlir::tt::TensorMemoryLayout::Interleaved",
          "Set the default memory layout for layout pass to prefer for operation operands that are on device, if not constrained">
  ];
}

def TTIRSplitCompoundLayout: Pass<"ttir-split-compound-layout", "::mlir::ModuleOp"> {
  let summary = "Split compound layouts.";
  let description = [{
    A single to_layout op in ttir can simultaneously perform multiple layout transformations
    at once, including changing layout, format, memory space or memory layout. This pass splits each of
    these transformation categories into separate to_layout ops.
  }];
}

def TTIRConstantAsFill: Pass<"ttir-constant-as-fill", "::mlir::ModuleOp"> {
  let summary = "Converts constant ops to empty + fill.";
  let description = [{
    This pass converts constant ops to empty + fill ops to allow for better
    optimization and easier lowering for some backend targets.
  }];
}

def TTIRAllocate: Pass<"ttir-allocate", "::mlir::ModuleOp"> {
  let summary = "Insert allocate/deallocate ops for tensors.";
  let description = [{
    This pass walks through the graph and does the following:
      - Replaces tensor empty ops with allocate ops.
      - Inserts deallocate ops after a tensor value's last use.
      - Allocates storage for graph inputs.

    Currently the allocator is built into the pass itself, but in the future
    this should be replaced with an analysis pass that can make global allocation
    decisions, followed by this pass that mechanically applies those decisions.
  }];
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

def TTIRBroadcastFold: Pass<"ttir-broadcast-fold", "::mlir::ModuleOp"> {
  let summary = "Broadcast operation is folded to all the consumers.";
  let description = [{
    This pass walks through the graph and folds all broadcast instructions since broadcast is supported implicitly by backend ops.
    Example:
      %1 = "ttir.broadcast"(%arg0) (tensor<1xf32>) -> tensor<512xf32>
      %2 = "ttir.maximum"(%1, %arg1) (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>

      This above broadcast is folded as:
      %1 = "ttir.maximum"(%arg0, %arg1) (tensor<1xf32>, tensor<512xf32>) -> tensor<512xf32>
  }];
}

#endif
