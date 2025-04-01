// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTIR_TTIRPASSES_TD
#define TTMLIR_TTMLIR_DIALECT_TTIR_TTIRPASSES_TD

include "mlir/Pass/PassBase.td"

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

def TTIRGenericHWThreadSelection : Pass<"ttir-generic-hw-thread-selection", "::mlir::ModuleOp"> {
  let summary = "Assign datamovement regions to hardware threads.";
  let description = [{
    This pass assigns the data movement regions to hardware threads. This usually means
    merging 2 or more data movement regions into a single region that is executed by one
    of the 2 datamovement threads (on wormhole).

    ```mlir
    "ttir.generic"(%arg0, %arg1, %alloc) <{grid = #tt.grid<1x1>, indexing_maps = [#map1, #map2, #map3], iterator_types = [#parallel, #parallel, #reduction], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^datamovement0(%cb0: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<4x2x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!tt.tile<32x32, f32>, #l1_>):
      ttir.yield %cb0 : (memref<2x4x!tt.tile<32x32, f32>, #l1_>)
    }, {
    ^datamovement1(%cb0: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<4x2x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!tt.tile<32x32, f32>, #l1_>):
      ttir.yield %cb1 : (memref<4x2x!tt.tile<32x32, f32>, #l1_>)
    }, {
    ^datamovement2(%cb0: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<4x2x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!tt.tile<32x32, f32>, #l1_>):
      ttir.await %cb2 : (memref<2x2x!tt.tile<32x32, f32>, #l1_>)
    }, {
    ^compute(%cb0: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<4x2x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!tt.tile<32x32, f32>, #l1_>):
      ttir.await %cb0, %cb1 : (memref<2x4x!tt.tile<32x32, f32>, #l1_>, memref<4x2x!tt.tile<32x32, f32>, #l1_>)
      "ttir.tile_matmul_block"(%cb0, %cb1, %cb2) : (memref<2x4x!tt.tile<32x32, f32>, #l1_>, memref<4x2x!tt.tile<32x32, f32>, #l1_>, memref<2x2x!tt.tile<32x32, f32>, #l1_>) -> ()
      ttir.yield %cb2 : (memref<2x2x!tt.tile<32x32, f32>, #l1_>)
    }) : (memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>, memref<1x1x4x2x!tt.tile<32x32, f32>, #l1_>, memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_>) -> ()
    ```

    Might move a trivial output datamovement thread to the compute thread to become:
    ```mlir
    "ttir.generic"(%arg0, %arg1, %alloc) <{grid = #tt.grid<1x1>, indexing_maps = [#map1, #map2, #map3], iterator_types = [#parallel, #parallel, #reduction], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^datamovement0(%cb0: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<4x2x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!tt.tile<32x32, f32>, #l1_>):
      ttir.yield %cb0 : (memref<2x4x!tt.tile<32x32, f32>, #l1_>)
    }, {
    ^datamovement1(%cb0: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<4x2x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!tt.tile<32x32, f32>, #l1_>):
      ttir.yield %cb1 : (memref<4x2x!tt.tile<32x32, f32>, #l1_>)
    }, {
    ^compute(%cb0: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<4x2x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!tt.tile<32x32, f32>, #l1_>):
      ttir.await %cb0, %cb1 : (memref<2x4x!tt.tile<32x32, f32>, #l1_>, memref<4x2x!tt.tile<32x32, f32>, #l1_>)
      "ttir.tile_matmul_block"(%cb0, %cb1, %cb2) : (memref<2x4x!tt.tile<32x32, f32>, #l1_>, memref<4x2x!tt.tile<32x32, f32>, #l1_>, memref<2x2x!tt.tile<32x32, f32>, #l1_>) -> ()
      ttir.yield %cb2 : (memref<2x2x!tt.tile<32x32, f32>, #l1_>)
      ttir.await %cb2 : (memref<2x2x!tt.tile<32x32, f32>, #l1_>)
    }) : (memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>, memref<1x1x4x2x!tt.tile<32x32, f32>, #l1_>, memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_>) -> ()
    ```
  }];
}

def TTIRGenericGenerateLoops : Pass<"ttir-generic-generate-loops", "::mlir::ModuleOp"> {
  let summary = "Generate generic loops.";
  let description = [{
    One of the final lowering forms of ttir generic op. This pass converts the affine declarative
    loops into imperative loops and the affine maps are erased. For example a generic region
    might transform as follows:

    ```mlir
    #lhs = affine_map<(d0, d1, d2) -> (d0, d2)>
    #rhs = affine_map<(d0, d1, d2) -> (d2, d1)>
    #out = affine_map<(d0, d1, d2) -> (d0, d1)>

    grid = #tt.grid<2x4>
    operands : (memref<2x4x4x6x!tt.tile<32x32, f32>>, memref<4x4x6x8x!tt.tile<32x32, f32>>, memref<2x4x4x8x!tt.tile<32x32, f32>>)

    ^compute(%cb0: memref<4x6x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<6x8x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<4x8x!tt.tile<32x32, f32>, #l1_>):
      ttir.await %cb0, %cb1
      "ttir.tile_matmul_block"(%cb0, %cb1, %cb2)
      ttir.yield %cb2
    ```

    Into:
    ```mlir
    ^compute(%cb0: memref<4x6x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<6x8x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<4x8x!tt.tile<32x32, f32>, #l1_>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %arg2 = %c0 to %c1 step %c1 {
        scf.for %arg3 = %c0 to %c1 step %c1 {
          scf.for %arg4 = %c0 to %c4 step %c1 {
            ttir.await %cb0, %cb1
            "ttir.tile_matmul_block"(%cb0, %cb1, %cb2)
            ttir.yield %cb2
          }
        }
      }
    ```
  }];
}

def TTIRGenericLowerDMAs : Pass<"ttir-generic-lower-dmas", "::mlir::ModuleOp"> {
  let summary = "Lower DMA ops from their high level form to fully indexed form.";
  let description = [{
    This pass lowers DMA ops from their high level forms to fully indexed form.

    One important pattern is rewriting their affine form to indexed form. This is useful for doing analysis on the DMA
    ops and lowering them to an optimal loop nest of coalesced transactions.  This is acheived by sampling the affine
    map over the entire parent generic op iterator space. Note that the affine map provided to the DMA op must be
    one of the indexing maps of the parent generic op.

    e.g.
    ```mlir
    %tx = ttir.dma %stream<#map1>, %cb0
    ```

    Might become:
    ```mlir
    %c2 = arith.constant 2
    %iter0 = ttir.iter_index(0)
    %core0 = ttir.core_index(0)
    %0 = arith.muli %core0, %c2
    %1 = arith.addi %0, %iter0
    %iter2 = ttir.iter_index(2)
    %tx = ttir.dma %stream [%1, %iter2], %cb0
    ```
  }];
}

def TTIRGenericRegionsToFuncs : Pass<"ttir-generic-regions-to-funcs", "::mlir::ModuleOp"> {
  let summary = "Move generic regions to top level functions.";
  let description = [{
    This pass moves the generic regions to top level functions. This is a useful prerequisite
    step before lowering because it enables us to better separate kernel program lowering from
    host program lowering.

    ```mlir
    func.func @main(/*...*/) {
      ttir.generic {grid = #tt.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#ttir.thread<compute>]}
          ins(%arg0, %arg1 : memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>, memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>)
          outs(%alloc : memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>)  {
      ^compute0(%cb0: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<2x4x!tt.tile<32x32, f32>, #l1_>):
        // ...compute body...
      }
    }
    ```

    Into (note the new compute function / symbol @compute_kernel0):
    ```mlir
    func.func @main(/*...*/) {
      ttir.generic {grid = #tt.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#ttir.thread<compute, @compute_kernel0>]}
          ins(%arg0, %arg1 : memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>, memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>)
          outs(%alloc : memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>)
    }

    func.func private @compute_kernel0(%arg0: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %arg2: memref<2x4x!tt.tile<32x32, f32>, #l1_>) attributes {ttir.thread_type = 0 : i32} {
      // ...compute body...
      return
    }
    ```
  }];
}

def TTIRLowerToLayout: Pass<"ttir-lower-to-layout", "::mlir::ModuleOp"> {
  let summary = "Tensor tilize all generic ops.";
  let description = [{
    Transition between different tensor layouts.

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
    %0 = ttir.empty() : tensor<1x16x32xf32>
    %1 = "ttir.broadcast"(%arg1, %0) <{broadcast_dimensions = array<i32: 1, 16, 1>}> : (tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %2 = ttir.empty() : tensor<1x16x32xf32>
    %3 = "ttir.multiply"(%arg0, %1, %2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x16x32xf32>, tensor<1x16x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>

    Since MultiplyOp supports implicit broadcasting, above broadcast is folded as:
    %0 = ttir.empty() : tensor<1x16x32xf32>
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
            %0 = ttir.empty() : tensor<32x32xbf16>
            %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16> loc("add_op1")
            return %1 : tensor<32x32xbf16>
          }
        }
      }
    output:
      tt.device_module {
        builtin.module {
          func.func @add(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
            %0 = ttir.empty() : tensor<32x32xbf16>
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

def ElementTypeNormalization: Pass<"ttir-element-type-normalization", "::mlir::ModuleOp">
{
  let summary = "Normalize element types into list of supported types.";
  let description = [{
    "This pass walks through the graph and normalizes the element types into a list of supported types. This is useful for lowering
        to a target that only supports a subset of the element types.
  }];

  let options = [
    Option<"enableFP32", "enable-fp32",
          "bool",
          /*default=*/"true",
          "Enables fp32 type.">
  ];

  let dependentDialects = ["::mlir::tt::TTDialect",  "::mlir::tt::ttir::TTIRDialect", "::mlir::func::FuncDialect"];
}

#endif
