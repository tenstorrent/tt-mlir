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
    Option<"forceReload", "force-reload", "bool",
           /*default=*/"false",
           "Force reload the system descriptor.">
  ];
}

def TTIRAttachMetalLayout: Pass<"ttir-attach-metal-layout", "::mlir::ModuleOp"> {
  let summary = "";
  let description = [{
    Attach a TTMetal layout to operands of a TTIR op.
  }];

  let options = [
    Option<"initMemorySpace", "init-memory-space",
          "::mlir::tt::MemorySpace",
          /*default=*/"::mlir::tt::MemorySpace::DeviceL1",
           "set the initial memory space for tensors to start in">,
    Option<"useStreamLayout", "use-stream-layout",
          "bool",
          /*default=*/"false",
           "turn on #tt.stream layout decoration">
  ];
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

def TTIRImplicitBroadcastFold: Pass<"ttir-implicit-broadcast-fold", "::mlir::ModuleOp"> {
  let summary = "Broadcast operation is folded to all the consumers.";
  let description = [{
    This pass walks through the graph and folds broadcasts operations when it is implicitly supported by the operation.

    Example:
    %0 = tensor.empty() : tensor<1x16x32xf32>
    %1 = "ttir.broadcast"(%arg1, %0) <{broadcast_dimensions = array<i32: 1, 16, 1>}> : (tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %2 = tensor.empty() : tensor<1x16x32xf32>
    %3 = "ttir.multiply"(%arg0, %1, %2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x16x32xf32>, tensor<1x16x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>

    Since MultiplyOp supports implicit broadcasting, above broadcast is folded as:
    %0 = tensor.empty() : tensor<1x16x32xf32>
    %1 = "ttir.multiply"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x16x32xf32>, tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32
  }];
}

def TTIRHoistTransform: Pass<"ttir-cpu-hoist-transform", "::mlir::ModuleOp">
{
  let summary = "Transform to perform hoist mechanics on any ops marked to be hoisted for CPU lowering";
  let description = [{
    Transform pass which runs an analysis pass to find ops which should be hoisted, and then hoists those ops.  Currently we only have a manual analysis which requires a commandline list of named locs to hoist--in the future, we will have an automatic analysis as well.

    Example:
    input:
      tt.device_module {
        builtin.module {
          func.func @add(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
            %0 = tensor.empty() : tensor<32x32xbf16>
            %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16> loc("add_op1")
            return %1 : tensor<32x32xbf16>
          }
        }
      }
    output:
      tt.device_module {
        builtin.module {
          func.func @add(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
            %0 = tensor.empty() : tensor<32x32xbf16>
            %1 = call @hoisted_ttir_add_32x32xbf16_32x32xbf16_32x32xbf16_func_decl(%arg0, %arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
            return %1 : tensor<32x32xbf16>
          }
          func.func private @hoisted_ttir_add_32x32xbf16_32x32xbf16_32x32xbf16_func_decl(tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
        }
      }
      tt.cpu_module {
        builtin.module {
          func.func @hoisted_ttir_add_32x32xbf16_32x32xbf16_32x32xbf16_func(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>, %arg2: tensor<32x32xbf16>) -> tensor<32x32xbf16> attributes {arg_ranks = [2, 2, 2, 2]} {
            %0 = "ttir.add"(%arg0, %arg1, %arg2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
            return %0 : tensor<32x32xbf16>
        }
      }

  }];

  let dependentDialects = ["::mlir::tt::TTDialect"];
}

#endif
