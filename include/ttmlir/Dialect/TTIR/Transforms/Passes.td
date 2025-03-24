// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTIR_TTIRPASSES_TD
#define TTMLIR_TTMLIR_DIALECT_TTIR_TTIRPASSES_TD

include "mlir/Pass/PassBase.td"

def TTIRAttachMetalLayout: Pass<"ttir-attach-metal-layout", "::mlir::ModuleOp"> {
  let summary = "";
  let description = [{
    Attach a TTMetal layout to operands of a TTIR op.
  }];

  let options = [
    Option<"initMemorySpace", "init-memory-space",
          "::mlir::tt::MemorySpace",
          /*default=*/"::mlir::tt::MemorySpace::DeviceL1",
           "set the initial memory space for tensors to start in">
  ];
}

def TTIRGeneralizeNamedOps: Pass<"ttir-generalize-named-ops", "::mlir::ModuleOp"> {
  let summary = "Convert named TTIR operations to their ttir.generic form.";
  let description = [{
    This pass converts "named" ops to a nested ttir.generic/linalg.generic structure, with
    ttir.generic denoting the available degrees of parallelism across a grid of cores
    and linalg.generic adding another level of nesting for a single core's data movement/compute
    task. This conversion will do an appropriate decomposition of the original op
    into lower-level tiled/blocked ops.

    Example:
    ```mlir
      %0 = "ttir.add"(%arg0, %arg1, %arg2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<128x96xf32>, tensor<128x96xf32>, tensor<128x96xf32>) -> tensor<128x96xf32>
      %1 = "ttir.sum"(%arg0, %0) <{dim_arg = [-2 : i32], keep_dim = true}> : (tensor<128x96xf32>, tensor<1x96xf32>) -> tensor<1x96xf32>
      %2 = "ttir.matmul"(%arg0, %arg1, %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<128x96xf32>, tensor<96x64xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    ```
    becomes
    ```mlir
      #map = affine_map<(d0, d1) -> (d0, d1)>
      #map1 = affine_map<(d0, d1) -> (0, 0)>
      #map2 = affine_map<(d0, d1) -> (0, d1)>
      #map3 = affine_map<(d0, d1, d2) -> (d0, d2)>
      #map4 = affine_map<(d0, d1, d2) -> (d2, d1)>
      #map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
      #parallel = #tt.iterator_type<parallel>
      #reduction = #tt.iterator_type<reduction>
      #layout = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<4x3x!tt.tile<32x32, f32>, #l1_>>
      #layout1 = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<1x3x!tt.tile<32x32, f32>, #l1_>>
      #layout2 = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<1x1x!tt.tile<32x32, f32>, #l1_>>
      #layout3 = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<3x2x!tt.tile<32x32, f32>, #l1_>>
      #layout4 = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<4x2x!tt.tile<32x32, f32>, #l1_>>

      %0 = "ttir.generic"(%arg0, %arg1, %arg2) <{grid = #tt.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], operandSegmentSizes = array<i32: 2, 1>}> ({
      ^compute(%cb0: memref<4x3x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<4x3x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<4x3x!tt.tile<32x32, f32>, #l1_>):
        linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<4x3x!tt.tile<32x32, f32>, #l1_>, memref<4x3x!tt.tile<32x32, f32>, #l1_>) outs(%cb2 : memref<4x3x!tt.tile<32x32, f32>, #l1_>) {
        ^bb0(%in: !tt.tile<32x32, f32>, %in_0: !tt.tile<32x32, f32>, %out: !tt.tile<32x32, f32>):
          %1 = "ttir.tile_add"(%in, %in_0) : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
          linalg.yield %1 : !tt.tile<32x32, f32>
        }
      }) : (tensor<128x96xf32, #layout>, tensor<128x96xf32, #layout>, tensor<128x96xf32, #layout>) -> tensor<128x96xf32, #layout>
      return %0 : tensor<128x96xf32, #layout>

      %scaler = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<32x32xf32, #layout2>}> : () -> tensor<32x32xf32, #layout2>
      %1 = "ttir.generic"(%arg0, %scaler, %0) <{grid = #tt.grid<1x1>, indexing_maps = [#map, #map1, #map2], iterator_types = [#reduction, #parallel], operandSegmentSizes = array<i32: 2, 1>}> ({
      ^compute(%cb0: memref<4x3x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<1x1x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<1x3x!tt.tile<32x32, f32>, #l1_>):
        linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel"]} ins(%cb0, %cb1 : memref<4x3x!tt.tile<32x32, f32>, #l1_>, memref<1x1x!tt.tile<32x32, f32>, #l1_>) outs(%cb2 : memref<1x3x!tt.tile<32x32, f32>, #l1_>) {
        ^bb0(%in: !tt.tile<32x32, f32>, %in_0: !tt.tile<32x32, f32>, %out: !tt.tile<32x32, f32>):
          %3 = "ttir.tile_reduce_sum"(%in, %in_0) <{reduce_dim = #ttir<reduce_dim R>}> : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
          linalg.yield %3 : !tt.tile<32x32, f32>
        }
      }) : (tensor<128x96xf32, #layout>, tensor<32x32xf32, #layout2>, tensor<1x96xf32, #layout1>) -> tensor<1x96xf32, #layout1>

      %2 = "ttir.generic"(%arg0, %arg1, %arg2) <{grid = #tt.grid<1x1>, indexing_maps = [#map3, #map4, #map5], iterator_types = [#parallel, #parallel, #reduction], operandSegmentSizes = array<i32: 2, 1>}> ({
      ^compute(%cb0: memref<4x3x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<3x2x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<4x2x!tt.tile<32x32, f32>, #l1_>):
        "ttir.tile_matmul_block"(%cb0, %cb1, %cb2) : (memref<4x3x!tt.tile<32x32, f32>, #l1_>, memref<3x2x!tt.tile<32x32, f32>, #l1_>, memref<4x2x!tt.tile<32x32, f32>, #l1_>) -> ()
      }) : (tensor<128x96xf32, #layout>, tensor<96x64xf32, #layout3>, tensor<128x64xf32, #layout4>) -> tensor<128x64xf32, #layout4>
    ```
  }];
}

def TTIRGenericLinearizeMemref: Pass<"ttir-generic-linearize-memref", "::mlir::ModuleOp"> {
  let summary = "Linearize memref operands for generic ops.";
  let description = [{
    This pass takes a nested loop structure over n-dimensional memrefs and linearizes
    them into a single dimension. This is a useful because circular buffers in metal
    are only one-dimensional.

    Example, this pass will convert the following code:
    ```mlir
      affine.for %arg5 = 0 to 2 {
        affine.for %arg6 = 0 to 4 {
          %0 = affine.load %arg2[%arg5, %arg6] : memref<2x4x!tt.tile<32x32, f32>, #l1_>
          %1 = affine.load %arg3[%arg5, %arg6] : memref<2x4x!tt.tile<32x32, f32>, #l1_>
          %2 = "ttir.tile_maximum"(%0, %1) : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
          affine.store %2, %arg4[%arg5, %arg6] : memref<2x4x!tt.tile<32x32, f32>, #l1_>
        }
      }
    ```

    Into:
    ```mlir
      %collapse_shape = memref.collapse_shape %arg2 [[0, 1]] : memref<2x4x!tt.tile<32x32, f32>, #l1_> into memref<8x!tt.tile<32x32, f32>, #l1_>
      %collapse_shape_0 = memref.collapse_shape %arg3 [[0, 1]] : memref<2x4x!tt.tile<32x32, f32>, #l1_> into memref<8x!tt.tile<32x32, f32>, #l1_>
      %collapse_shape_1 = memref.collapse_shape %arg4 [[0, 1]] : memref<2x4x!tt.tile<32x32, f32>, #l1_> into memref<8x!tt.tile<32x32, f32>, #l1_>
      affine.for %arg5 = 0 to 2 {
        affine.for %arg6 = 0 to 4 {
          %0 = affine.load %collapse_shape[%arg5 * 4 + %arg6] : memref<8x!tt.tile<32x32, f32>, #l1_>
          %1 = affine.load %collapse_shape_0[%arg5 * 4 + %arg6] : memref<8x!tt.tile<32x32, f32>, #l1_>
          %2 = "ttir.tile_maximum"(%0, %1) : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
          affine.store %2, %collapse_shape_1[%arg5 * 4 + %arg6] : memref<8x!tt.tile<32x32, f32>, #l1_>
        }
      }
    ```
  }];
}

def TTIROptimizeTensorLayout: Pass<"ttir-optimize-tensor-layout", "::mlir::ModuleOp"> {
  let summary = "";
  let description = [{
    Analyze the graph and select optimal layouts, insert to_layout where needed.
  }];

  list<Option> options = [
        ListOption<"overrideDeviceShape", "override-device-shape", "int64_t", "Override the device shape.">,
    ];
}

def TTIRGenericGenerateDatamovement: Pass<"ttir-generic-generate-datamovement", "::mlir::ModuleOp"> {
  let summary = "Generate generic data movement threads.";
  let description = [{
    This pass makes the following transformation, given a generic compute region:
    ```mlir
    #map = affine_map<(d0, d1) -> (d0, d1)>
    #parallel = #tt.iterator_type<parallel>

    "ttir.generic"(%arg0, %arg1, %alloc) <{indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel]}> ({
    ^bb0(%arg2: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %arg3: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %arg4: memref<2x4x!tt.tile<32x32, f32>, #l1_>):
      affine.for %arg5 = 0 to 2 {
        affine.for %arg6 = 0 to 4 {
          %0 = affine.load %arg2[%arg5, %arg6] : memref<2x4x!tt.tile<32x32, f32>, #l1_>
          %1 = affine.load %arg3[%arg5, %arg6] : memref<2x4x!tt.tile<32x32, f32>, #l1_>
          %2 = "ttir.tile_add"(%0, %1) : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
          affine.store %2, %arg4[%arg5, %arg6] : memref<2x4x!tt.tile<32x32, f32>, #l1_>
        }
      }
    })
    ```

    We generate additional (prepended) regions that correspond to the data movement
    for each operand respectively:
    ```mlir
    "ttir.generic"(%arg0, %arg1, %alloc) <{indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel]}> ({
    ^bb0(%arg2: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %arg3: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %arg4: memref<2x4x!tt.tile<32x32, f32>, #l1_>):
      ttir.yield %arg2 : (memref<2x4x!tt.tile<32x32, f32>, #l1_>)
    }, {
    ^bb0(%arg2: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %arg3: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %arg4: memref<2x4x!tt.tile<32x32, f32>, #l1_>):
      ttir.yield %arg3 : (memref<2x4x!tt.tile<32x32, f32>, #l1_>)
    }, {
    ^bb0(%arg2: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %arg3: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %arg4: memref<2x4x!tt.tile<32x32, f32>, #l1_>):
      ttir.await %arg4 : (memref<2x4x!tt.tile<32x32, f32>, #l1_>)
    }, {
    ^bb0(%arg2: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %arg3: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %arg4: memref<2x4x!tt.tile<32x32, f32>, #l1_>):
      ttir.await %arg2, %arg3 : (memref<2x4x!tt.tile<32x32, f32>, #l1_>, memref<2x4x!tt.tile<32x32, f32>, #l1_>)
      affine.for %arg5 = 0 to 2 {
        affine.for %arg6 = 0 to 4 {
          %0 = affine.load %arg2[%arg5, %arg6] : memref<2x4x!tt.tile<32x32, f32>, #l1_>
          %1 = affine.load %arg3[%arg5, %arg6] : memref<2x4x!tt.tile<32x32, f32>, #l1_>
          %2 = "ttir.tile_add"(%0, %1) : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
          affine.store %2, %arg4[%arg5, %arg6] : memref<2x4x!tt.tile<32x32, f32>, #l1_>
        }
      }
      ttir.yield %arg4 : (memref<2x4x!tt.tile<32x32, f32>, #l1_>)
    })
    ```
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

def TTIRPrepareTensorsForBufferization : Pass<"ttir-prepare-tensors-for-bufferization", "::mlir::ModuleOp"> {
  let summary = "Prepare tensor shapes for bufferization.";
  let description = [{
    Bufferization has a constraint that the tensor -> buffer conversion must maintain the
    same shape and rank. Since we use the tensor encoding to express shape collapse this
    becomes problematic because the tensor's shape will rarely match that of the collapsed
    shape. This function of this pass is to normalize all tensor shapes, such that
    all tensors inherit the shape of their layout encoding so that they can 1-1 match the
    bufferized memref. This pass is required to run before bufferization.
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

def TTIRPlaceholderAllocate : Pass<"ttir-placeholder-allocate", "::mlir::ModuleOp"> {
  let summary = "Placeholder for the eventual full allocate pass.";
  let description = [{
    Currently this pass only does some simple heuristics for forming stream_layout ops
    for the situations that require it for correctness.  In the future, this pass will
    be replaced with a full allocate pass that has to do more sophisticated analysis
    for inserting streams on top of actually allocating.

    Converts generic arguments that require a stream (i.e. local storage
    buffer for circular buffer):

    ```mlir
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x4x8x!tt.tile<32x32, f32>, #l1_>
    %0 = "ttir.view_layout"(%arg0) : (memref<2x4x4x6x!tt.tile<32x32, f32>, #l1_>) -> memref<2x4x4x6x!tt.tile<32x32, f32>, #l1_>
    %1 = "ttir.view_layout"(%arg1) : (memref<4x4x6x8x!tt.tile<32x32, f32>, #l1_>) -> memref<4x4x6x8x!tt.tile<32x32, f32>, #l1_>
    "ttir.generic"(%0, %1, %alloc)
    ```

    Into:
    ```mlir
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x4x8x!tt.tile<32x32, f32>, #l1_>
    %storage_0 = memref.alloc() : memref<2x4x4x6x!tt.tile<32x32, f32>, #l1_>
    %stream = "ttir.stream_layout"(%arg0, %storage_0) : (memref<2x4x4x6x!tt.tile<32x32, f32>, #l1_>, memref<2x4x4x6x!tt.tile<32x32, f32>, #l1_>) -> memref<2x4x4x6x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, #l1_>
    %storage_1 = memref.alloc() : memref<4x4x6x8x!tt.tile<32x32, f32>, #l1_>
    %stream_2 = "ttir.stream_layout"(%arg1, %storage_1) : (memref<4x4x6x8x!tt.tile<32x32, f32>, #l1_>, memref<4x4x6x8x!tt.tile<32x32, f32>, #l1_>) -> memref<4x4x6x8x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, #l1_>
    "ttir.generic"(%stream, %stream_2, %alloc)
    ```
  }];
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
