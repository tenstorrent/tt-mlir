// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_D2M_D2MPASSES_TD
#define TTMLIR_TTMLIR_DIALECT_D2M_D2MPASSES_TD

include "mlir/Pass/PassBase.td"

// Elementwise fusion of d2m.generic ops on tensors
def D2MElementwiseFusion : Pass<"d2m-elementwise-fusion", "::mlir::ModuleOp"> {
  let summary = "Fuse chains of compute-only d2m.generic elementwise ops";
  let description = [{
    Performs elementwise fusion for `d2m.generic` producers into consumers when
    safe to do so. The initial implementation targets compute-only, affine-map
    form `d2m.generic` with tensor semantics and matching device attributes
    (grid, threads, block_factors). The pass is intended to run pre-bufferization
    to reduce intermediary tensors and improve locality.
  }];

  list<Option> options = [
    Option<"maxDstPhysicalSizeTiles", "max-dst-physical-size-tiles", "unsigned", "0", "Override the max dst physical size tiles or 0 if unset.">,
  ];

  let dependentDialects = [
    "::mlir::tt::d2m::D2MDialect",
    "::mlir::affine::AffineDialect"
  ];
}

def D2MSFPUTileLoopFission : Pass<"d2m-sfpu-tile-loop-fission", "::mlir::ModuleOp"> {
  let summary = "Perform loop fission inside d2m.generic compute regions for load→compute→store triplets";
  let description = [{
    This pass scans `d2m.generic` compute regions that do not contain any tilize/untilize ops,
    finds nested affine loop nests, and performs loop fission (distribution) by extracting
    sequences of the form `affine.load → d2m tile compute op → affine.store` into separate
    loop nests, preserving the original loop bounds and indices.
  }];

  let dependentDialects = [
    "::mlir::tt::d2m::D2MDialect",
    "::mlir::affine::AffineDialect"
  ];
}

def D2MGenericTileComputeLoops : Pass<"d2m-generic-tile-compute-loops", "::mlir::ModuleOp"> {
  let summary = "";
  let description = [{
    This pass tiles affine compute loops according to the generic's subblock factors.
  }];

  list<Option> options = [
    Option<"maxDstPhysicalSizeTiles", "max-dst-physical-size-tiles", "unsigned", "0", "Override the max dst physical size tiles or 0 if unset.">,
  ];

  let dependentDialects = ["mlir::tt::d2m::D2MDialect"];
}

def D2MInsertDstRegisterAccess : Pass<"d2m-insert-dst-register-access", "::mlir::ModuleOp"> {
  let summary = "Insert dst register access.";
  let description = [{
    This pass inserts a high level representation of the destination
    register, manages load and store accesses to it, and has primitive
    support for register allocation.  It's also capable of tracking accesses
    over affine loop nests and correctly cloning them into copy loop nests.

    Example, this pass will convert the following code:
    ```mlir
    ^compute0(%cb0: memref<3x3x!tt.tile<32x32, f32>, #tt.memory_space<l1>>, %cb1: memref<3x2x!tt.tile<32x32, f32>, #tt.memory_space<l1>>, %cb2: memref<3x2x!tt.tile<32x32, f32>, #tt.memory_space<l1>>):
      affine.for %i2 = 0 to 3 {
        affine.for %i3 = 0 to 2 {
          affine.for %i4 = 0 to 3 {
            %0 = affine.load %cb0[%i2, %i4] : memref<3x3x!tt.tile<32x32, f32>, #tt.memory_space<l1>>
            %1 = affine.load %cb1[%i4, %i3] : memref<3x2x!tt.tile<32x32, f32>, #tt.memory_space<l1>>
            %2 = affine.load %cb2[%i2, %i3] : memref<3x2x!tt.tile<32x32, f32>, #tt.memory_space<l1>>
            %3 = "d2m.tile_matmul"(%0, %1, %2) : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
            affine.store %3, %cb2[%i2, %i3] : memref<3x2x!tt.tile<32x32, f32>, #tt.memory_space<l1>>
          }
        }
      }
    ```

    Into:
    ```mlir
    ^compute0(%cb0: memref<3x3x!tt.tile<32x32, f32>, #l1>, %cb1: memref<3x2x!tt.tile<32x32, f32>, #l1>, %cb2: memref<3x2x!tt.tile<32x32, f32>, #l1>):
      %c0 = arith.constant 0 : index
      %dst = d2m.acquire_dst() : memref<3x2x!tt.tile<32x32, f32>, #dst>
      %iter2 = d2m.iter_index(2) : index
      %0 = arith.cmpi ne, %iter2, %c0 : index
      scf.if %0 {
        affine.for %arg3 = 0 to 3 {
          affine.for %arg4 = 0 to 2 {
            %1 = affine.load %cb2[%arg3, %arg4] : memref<3x2x!tt.tile<32x32, f32>, #l1>
            affine.store %1, %dst[%arg3, %arg4] : memref<3x2x!tt.tile<32x32, f32>, #dst>
          }
        }
      }
      affine.for %arg3 = 0 to 3 {
        affine.for %arg4 = 0 to 2 {
          affine.for %arg5 = 0 to 3 {
            %1 = affine.load %cb0[%arg3, %arg5] : memref<3x3x!tt.tile<32x32, f32>, #l1>
            %2 = affine.load %cb1[%arg5, %arg4] : memref<3x2x!tt.tile<32x32, f32>, #l1>
            %3 = affine.load %dst[%arg3, %arg4] : memref<3x2x!tt.tile<32x32, f32>, #dst>
            %4 = "d2m.tile_matmul"(%1, %2, %3) : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
            affine.store %4, %dst[%arg3, %arg4] : memref<3x2x!tt.tile<32x32, f32>, #dst>
          }
        }
      }
      affine.for %arg3 = 0 to 3 {
        affine.for %arg4 = 0 to 2 {
          %1 = affine.load %dst[%arg3, %arg4] : memref<3x2x!tt.tile<32x32, f32>, #dst>
          affine.store %1, %cb2[%arg3, %arg4] : memref<3x2x!tt.tile<32x32, f32>, #l1>
        }
      }
    ```

    Notes:
       - Operand loads & stores that implicitly uses the DST are replaced with explicit affine loads & stores that go through the DST.
       - Generic op automatically inserts conditional guard for reload.
       - This pass only works on affine ops.
  }];
  let dependentDialects = ["::mlir::tt::d2m::D2MDialect", "::mlir::affine::AffineDialect", "::mlir::func::FuncDialect"];
  let options = [
    Option<"useTileMatmul", "use-tile-matmul", "bool", /*default=*/"false", "Use tile_matmul instead of tile_matmul_block">,
    Option<"maxDstPhysicalSizeTiles", "max-dst-physical-size-tiles", "unsigned", "0", "Override the max dst physical size tiles or 0 if unset.">,
  ];
}

def D2MGenericLinearizeMemref: Pass<"d2m-generic-linearize-memref", "::mlir::ModuleOp"> {
  let summary = "Linearize memref operands for generic ops.";
  let description = [{
    This pass takes a nested loop structure over n-dimensional memrefs and linearizes
    them into a single dimension. This is a useful because circular buffers in metal
    are only one-dimensional.

    Example, this pass will convert the following code:
    ```mlir
      affine.for %arg5 = 0 to 2 {
        affine.for %arg6 = 0 to 4 {
          %0 = affine.load %arg2[%arg5, %arg6] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
          %1 = affine.load %arg3[%arg5, %arg6] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
          %2 = "d2m.tile_maximum"(%0, %1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          affine.store %2, %arg4[%arg5, %arg6] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        }
      }
    ```

    Into:
    ```mlir
      %collapse_shape = memref.collapse_shape %arg2 [[0, 1]] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_> into memref<8x!ttcore.tile<32x32, f32>, #l1_>
      %collapse_shape_0 = memref.collapse_shape %arg3 [[0, 1]] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_> into memref<8x!ttcore.tile<32x32, f32>, #l1_>
      %collapse_shape_1 = memref.collapse_shape %arg4 [[0, 1]] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_> into memref<8x!ttcore.tile<32x32, f32>, #l1_>
      affine.for %arg5 = 0 to 2 {
        affine.for %arg6 = 0 to 4 {
          %0 = affine.load %collapse_shape[%arg5 * 4 + %arg6] : memref<8x!ttcore.tile<32x32, f32>, #l1_>
          %1 = affine.load %collapse_shape_0[%arg5 * 4 + %arg6] : memref<8x!ttcore.tile<32x32, f32>, #l1_>
          %2 = "d2m.tile_maximum"(%0, %1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          affine.store %2, %collapse_shape_1[%arg5 * 4 + %arg6] : memref<8x!ttcore.tile<32x32, f32>, #l1_>
        }
      }
    ```
  }];
}

def D2MGenericGenerateDatamovement: Pass<"d2m-generic-generate-datamovement", "::mlir::ModuleOp"> {
  let summary = "Generate generic data movement threads.";
  let description = [{
    This pass makes the following transformation, given a generic compute region:
    ```mlir
    #map = affine_map<(d0, d1) -> (d0, d1)>
    #parallel = #ttcore.iterator_type<parallel>

    "d2m.generic"(%arg0, %arg1, %alloc) <{indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel]}> ({
    ^bb0(%arg2: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg3: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg4: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>):
      affine.for %arg5 = 0 to 2 {
        affine.for %arg6 = 0 to 4 {
          %0 = affine.load %arg2[%arg5, %arg6] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
          %1 = affine.load %arg3[%arg5, %arg6] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
          %2 = "d2m.tile_add"(%0, %1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          affine.store %2, %arg4[%arg5, %arg6] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        }
      }
    })
    ```

    We generate additional (prepended) regions that correspond to the data movement
    for each operand respectively:
    ```mlir
    "d2m.generic"(%arg0, %arg1, %alloc) <{indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel]}> ({
    ^bb0(%arg2: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg3: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg4: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>):
      d2m.yield %arg2 : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>)
    }, {
    ^bb0(%arg2: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg3: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg4: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>):
      d2m.yield %arg3 : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>)
    }, {
    ^bb0(%arg2: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg3: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg4: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>):
      d2m.await %arg4 : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>)
    }, {
    ^bb0(%arg2: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg3: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg4: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>):
      d2m.await %arg2, %arg3 : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<2x4x!ttcore.tile<32x32, f32>, #l1_>)
      affine.for %arg5 = 0 to 2 {
        affine.for %arg6 = 0 to 4 {
          %0 = affine.load %arg2[%arg5, %arg6] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
          %1 = affine.load %arg3[%arg5, %arg6] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
          %2 = "d2m.tile_add"(%0, %1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          affine.store %2, %arg4[%arg5, %arg6] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        }
      }
      d2m.yield %arg4 : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>)
    })
    ```
  }];
}

def D2MGenericHWThreadSelection : Pass<"d2m-generic-hw-thread-selection", "::mlir::ModuleOp"> {
  let summary = "Assign datamovement regions to hardware threads.";
  let description = [{
    This pass assigns the data movement regions to hardware threads. This usually means
    merging 2 or more data movement regions into a single region that is executed by one
    of the 2 datamovement threads (on wormhole).

    ```mlir
    "d2m.generic"(%arg0, %arg1, %alloc) <{grid = #ttcore.grid<1x1>, indexing_maps = [#map1, #map2, #map3], iterator_types = [#parallel, #parallel, #reduction], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^datamovement0(%cb0: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<4x2x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>):
      d2m.yield %cb0 : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>)
    }, {
    ^datamovement1(%cb0: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<4x2x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>):
      d2m.yield %cb1 : (memref<4x2x!ttcore.tile<32x32, f32>, #l1_>)
    }, {
    ^datamovement2(%cb0: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<4x2x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>):
      d2m.await %cb2 : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
    }, {
    ^compute(%cb0: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<4x2x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>):
      d2m.await %cb0, %cb1 : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<4x2x!ttcore.tile<32x32, f32>, #l1_>)
      "d2m.tile_matmul_block"(%cb0, %cb1, %cb2) : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<4x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> ()
      d2m.yield %cb2 : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
    }) : (memref<1x1x2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<1x1x4x2x!ttcore.tile<32x32, f32>, #l1_>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #l1_>) -> ()
    ```

    Might move a trivial output datamovement thread to the compute thread to become:
    ```mlir
    "d2m.generic"(%arg0, %arg1, %alloc) <{grid = #ttcore.grid<1x1>, indexing_maps = [#map1, #map2, #map3], iterator_types = [#parallel, #parallel, #reduction], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^datamovement0(%cb0: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<4x2x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>):
      d2m.yield %cb0 : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>)
    }, {
    ^datamovement1(%cb0: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<4x2x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>):
      d2m.yield %cb1 : (memref<4x2x!ttcore.tile<32x32, f32>, #l1_>)
    }, {
    ^compute(%cb0: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<4x2x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>):
      d2m.await %cb0, %cb1 : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<4x2x!ttcore.tile<32x32, f32>, #l1_>)
      "d2m.tile_matmul_block"(%cb0, %cb1, %cb2) : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<4x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> ()
      d2m.yield %cb2 : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
      d2m.await %cb2 : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
    }) : (memref<1x1x2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<1x1x4x2x!ttcore.tile<32x32, f32>, #l1_>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #l1_>) -> ()
    ```
  }];
}

def D2MGenericGenerateLoops : Pass<"d2m-generic-generate-loops", "::mlir::ModuleOp"> {
  let summary = "Generate generic loops.";
  let description = [{
    One of the final lowering forms of d2m generic op. This pass converts the affine declarative
    loops into imperative loops and the affine maps are erased. For example a generic region
    might transform as follows:

    ```mlir
    #lhs = affine_map<(d0, d1, d2) -> (d0, d2)>
    #rhs = affine_map<(d0, d1, d2) -> (d2, d1)>
    #out = affine_map<(d0, d1, d2) -> (d0, d1)>

    grid = #ttcore.grid<2x4>
    operands : (memref<2x4x4x6x!ttcore.tile<32x32, f32>>, memref<4x4x6x8x!ttcore.tile<32x32, f32>>, memref<2x4x4x8x!ttcore.tile<32x32, f32>>)

    ^compute(%cb0: memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<4x8x!ttcore.tile<32x32, f32>, #l1_>):
      d2m.await %cb0, %cb1
      "d2m.tile_matmul_block"(%cb0, %cb1, %cb2)
      d2m.yield %cb2
    ```

    Into:
    ```mlir
    ^compute(%cb0: memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<4x8x!ttcore.tile<32x32, f32>, #l1_>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %arg2 = %c0 to %c1 step %c1 {
        scf.for %arg3 = %c0 to %c1 step %c1 {
          scf.for %arg4 = %c0 to %c4 step %c1 {
            d2m.await %cb0, %cb1
            "d2m.tile_matmul_block"(%cb0, %cb1, %cb2)
            d2m.yield %cb2
          }
        }
      }
    ```
  }];
}

def D2MGenericLowerDMAs : Pass<"d2m-generic-lower-dmas", "::mlir::ModuleOp"> {
  let summary = "Lower DMA ops from their high level form to fully indexed form.";
  let description = [{
    This pass lowers DMA ops from their high level forms to fully indexed form.

    One important pattern is rewriting their affine form to indexed form. This is useful for doing analysis on the DMA
    ops and lowering them to an optimal loop nest of coalesced transactions.  This is acheived by sampling the affine
    map over the entire parent generic op iterator space. Note that the affine map provided to the DMA op must be
    one of the indexing maps of the parent generic op.

    e.g.
    ```mlir
    %tx = d2m.dma %stream<#map1>, %cb0
    ```

    Might become:
    ```mlir
    %c2 = arith.constant 2
    %iter0 = d2m.iter_index(0)
    %core0 = d2m.core_index(0)
    %0 = arith.muli %core0, %c2
    %1 = arith.addi %0, %iter0
    %iter2 = d2m.iter_index(2)
    %tx = d2m.dma %stream [%1, %iter2], %cb0
    ```
  }];
}

def D2MGenericRegionsToFuncs : Pass<"d2m-generic-regions-to-funcs", "::mlir::ModuleOp"> {
  let summary = "Move generic regions to top level functions.";
  let description = [{
    This pass moves the generic regions to top level functions. This is a useful prerequisite
    step before lowering because it enables us to better separate kernel program lowering from
    host program lowering.

    ```mlir
    func.func @main(/*...*/) {
      d2m.generic {grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
          ins(%arg0, %arg1 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<1x1x2x4x!ttcore.tile<32x32, f32>, #l1_>)
          outs(%alloc : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #l1_>)  {
      ^compute0(%cb0: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>):
        // ...compute body...
      }
    }
    ```

    Into (note the new compute function / symbol @compute_kernel0):
    ```mlir
    func.func @main(/*...*/) {
      d2m.generic {grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute, @compute_kernel0>]}
          ins(%arg0, %arg1 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<1x1x2x4x!ttcore.tile<32x32, f32>, #l1_>)
          outs(%alloc : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #l1_>)
    }

    func.func private @compute_kernel0(%arg0: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg2: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>) attributes {d2m.thread_type = 0 : i32} {
      // ...compute body...
      return
    }
    ```
  }];
}

def D2MLowerToLayout: Pass<"d2m-lower-to-layout", "::mlir::ModuleOp"> {
  let summary = "Lower layouts to generic ops.";
  let description = [{
    Transition between different tensor layouts.

    A single to_layout op in d2m can simultaneously perform multiple layout transformations
    at once, including changing layout, format, memory space or memory layout. This pass splits each of
    these transformation categories into separate to_layout ops and then lowers them to generic ops.
    There is one exception to the generic lowering and that is to/from device memory, this case lowers
    to a specialized flavor of to_layout that has the hostInfo attribute set.

    For example a compound to layout that goes from host to tilized 8x8 device grid might look like:
    ```
    #layout2 = #ttcore.metal_layout<(d0, d1) -> (d0, d1), undef, <8x8>, memref<1x3x!ttcore.tile<32x32, f32>, #l1_>>
    %0 = d2m.empty() : tensor<256x768xf32, #layout2>
    %1 = d2m.to_layout %arg0, %0 : tensor<256x768xf32> into tensor<256x768xf32, #layout2>
    ```

    Into:
    ```
    #layout = #ttcore.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<256x768xf32, #l1_>>
    #layout1 = #ttcore.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<8x24x!ttcore.tile<32x32, f32>, #l1_>>
    #layout2 = #ttcore.metal_layout<(d0, d1) -> (d0, d1), undef, <8x8>, memref<1x3x!ttcore.tile<32x32, f32>, #l1_>>
    %0 = d2m.empty() : tensor<256x768xf32, #layout2>
    %1 = d2m.empty() : tensor<256x768xf32>
    %2 = d2m.empty() : tensor<256x768xf32, #layout>
    // Move to device
    %3 = d2m.to_layout %arg0, %2 : tensor<256x768xf32> into tensor<256x768xf32, #layout> hostInfo = #layout -> tensor<256x768xf32, #layout>
    %4 = d2m.empty() : tensor<256x768xf32, #layout1>
    // Tilize
    %5 = d2m.generic {grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%3 : tensor<256x768xf32, #layout>)
        outs(%4 : tensor<256x768xf32, #layout1>)  {
    ^compute0(%cb0: memref<256x768xf32, #l1_>, %cb1: memref<8x24x!ttcore.tile<32x32, f32>, #l1_>):
      "d2m.tile_tilize_block"(%cb0, %cb1) : (memref<256x768xf32, #l1_>, memref<8x24x!ttcore.tile<32x32, f32>, #l1_>) -> ()
    } : tensor<256x768xf32, #layout1>
    // Reblock to 8x8 grid
    %view = "d2m.view_layout"(%5) : (tensor<256x768xf32, #layout1>) -> tensor<256x768xf32, #layout2>
    %6 = d2m.generic {grid = #ttcore.grid<8x8>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%view : tensor<256x768xf32, #layout2>)
        outs(%0 : tensor<256x768xf32, #layout2>)  {
    ^compute0(%cb0: memref<1x3x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<1x3x!ttcore.tile<32x32, f32>, #l1_>):
      d2m.yield %cb0 : (memref<1x3x!ttcore.tile<32x32, f32>, #l1_>)
    } : tensor<256x768xf32, #layout2>
    ```
  }];
}

def D2MGenericApplyInterchange : Pass<"d2m-generic-apply-interchange", "::mlir::ModuleOp"> {
  let summary = "Apply loop interchange on generic ops.";
  let description = [{
    For example, the default matmul interchange looks like:
      (m, n, k) -> (m, k)
      (m, n, k) -> (k, n)
      (m, n, k) -> (m, n)

    This pass might choose a different interchange, such as
    making the k dim the outermost loop, given interchange
    (2, 0, 1):
      (k, m, n) -> (m, k)
      (k, m, n) -> (k, n)
      (k, m, n) -> (m, n)
  }];

  list<Option> options = [
      ListOption<"matmulInterchange", "matmul-interchange", "int64_t", "Set an interchange for generic ops that match matmul style indexing maps and iterator types. The interchange indices here always correspond to the innermost 3 dims.">,
  ];
}

defvar D2MStreamingOptions = [
  Option<"numStreamBuffers",
         "num-stream-buffers",
         "unsigned", /*default=*/"2",
         "Number of backing buffers to allocate per stream storage (>=1) Default is 2.">,
  Option<"allowL1OutputSpilling",
         "allow-l1-output-spilling",
         "bool", /*default=*/"false",
         "Make generic outputs eligible for spilling from L1 to DRAM.">,
];

def D2MAllocate: Pass<"d2m-allocate", "::mlir::ModuleOp"> {
  let summary = "Create streams required by generic ops.";
  let description = [{
    This pass handles several related tasks:
      - allocating data streams (stream_layouts) and their associated buffers as required by generic
        op operands;
      - allocating memory addresses/alignments for buffers supporting such streams as well as
        other memref allocs needed by data movement/compute kernel functions;
      - inserting dealloc instructions at the appropriate end-of-scope positions.

    Converts:
    ```mlir
    %alloc = memref.alloc() : memref<6x6x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
    %alloc_0 = memref.alloc() : memref<6x6x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
    %alloc_1 = memref.alloc() : memref<6x6x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
    d2m.generic {block_factors = [1, 1, 6], grid = #ttcore.grid<6x6>, indexing_maps = [#map, #map1, #map2], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<compute>]}
        ins(%alloc, %alloc_0 : memref<6x6x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<6x6x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>)
        outs(%alloc_1 : memref<6x6x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>)  {
    ^compute0(%cb0: memref<1x1x!ttcore.tile<32x32, f32>, #l1>, %cb1: memref<1x1x!ttcore.tile<32x32, f32>, #l1>, %cb2: memref<1x1x!ttcore.tile<32x32, f32>, #l1>):
      linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%cb0, %cb1 : memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>) outs(%cb2 : memref<1x1x!ttcore.tile<32x32, f32>, #l1>) {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %in_4: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %0 = "d2m.tile_matmul"(%in, %in_4, %out) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %0 : !ttcore.tile<32x32, f32>
      }
    }
    ```

    Into:
    ```mlir
    %alloc = memref.alloc() {address = 5120 : i64, alignment = 32 : i64} : memref<6x6x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #dram>
    %alloc_0 = memref.alloc() {address = 1024 : i64, alignment = 32 : i64} : memref<6x6x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #dram>
    %alloc_1 = memref.alloc() {address = 17408 : i64, alignment = 16 : i64} : memref<6x6x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
    %alloc_2 = memref.alloc() {address = 9216 : i64, alignment = 16 : i64} : memref<6x6x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 2>, #l1>
    %stream = "d2m.stream_layout"(%alloc, %alloc_2) : (memref<6x6x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #dram>, memref<6x6x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 2>, #l1>) -> memref<6x6x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>
    %alloc_3 = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<6x6x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 2>, #l1>
    %stream_4 = "d2m.stream_layout"(%alloc_0, %alloc_3) : (memref<6x6x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #dram>, memref<6x6x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 2>, #l1>) -> memref<6x6x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>
    d2m.generic {block_factors = [1, 1, 6], grid = #ttcore.grid<6x6>, indexing_maps = [#map, #map1, #map2], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<compute>]}
        ins(%stream, %stream_4 : memref<6x6x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>, memref<6x6x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>)
        outs(%alloc_1 : memref<6x6x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>)  {
    ^compute0(%cb0: memref<1x1x!ttcore.tile<32x32, f32>, #l1>, %cb1: memref<1x1x!ttcore.tile<32x32, f32>, #l1>, %cb2: memref<1x1x!ttcore.tile<32x32, f32>, #l1>):
      linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%cb0, %cb1 : memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>) outs(%cb2 : memref<1x1x!ttcore.tile<32x32, f32>, #l1>) {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %in_7: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %0 = "d2m.tile_matmul"(%in, %in_7, %out) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %0 : !ttcore.tile<32x32, f32>
      }
    }
    memref.dealloc %alloc_3 : memref<6x6x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 2>, #l1>
    memref.dealloc %alloc_0 : memref<6x6x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #dram>
    memref.dealloc %alloc_2 : memref<6x6x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 2>, #l1>
    ```
  }];
  let dependentDialects = ["::mlir::tt::ttcore::TTCoreDialect", "::mlir::memref::MemRefDialect"];
  let options = D2MStreamingOptions # [
    Option<"testAssumeL1Capacity",
          "test-assume-l1-capacity",
          "std::int64_t", /*default=*/"0",
          "Assume given L1 capacity.">,
    Option<"testBufferSizePolicy",
          "test-buffer-size-policy",
          "std::string", /*default=*/"\"max\"",
          "Set policy for sizing stream buffers ('min', 'max').">,
  ];
}

def D2MInsertStreams: Pass<"d2m-insert-streams", "::mlir::ModuleOp"> {
  let summary = "Always insert streams for D2M generic ops.";
  let description = [{
    This pass is only available for D2M/TTNN integration (i.e.: when ttnn-mode is on). This pass will aggressively
    insert a stream for all inputs/outputs of the TTIR generic op.

    Example:
  }];
  let dependentDialects = ["::mlir::tt::ttcore::TTCoreDialect", "::mlir::memref::MemRefDialect"];
  let options = D2MStreamingOptions;
}

def D2MGlobalDataFormatConversion: Pass<"d2m-global-data-format-conversion", "::mlir::ModuleOp">
{
  let summary = "Convert all tensor data types to a target format.";
  let description = [{
    This pass converts all tensor element types in the function body
    to a specified target format. Function signatures are preserved, and typecast operations
    are automatically inserted at boundaries where type conversions are needed.

    Supported target formats:
      - "f32" (Float32)
      - "bf16" (BFloat16, default)
      - "bfp_bf8" (BFP_BFloat8)

    Example:
      Before:
       func.func @forward(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
        %0 = ttir.empty() : tensor<64x128xf32>
        %1 = "ttir.exp"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
        return %1 : tensor<64x128xf32>
       }

      After (with target-format="bf16"):
        func.func @forward(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
          %0 = ttir.empty() : tensor<64x128xbf16>
          %1 = "ttir.typecast"(%arg0, %0) <{conservative_folding = false}> : (tensor<64x128xf32>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
          %2 = ttir.empty() : tensor<64x128xbf16>
          %3 = "ttir.exp"(%1, %2) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
          %4 = ttir.empty() : tensor<64x128xf32>
          %5 = "ttir.typecast"(%3, %4) <{conservative_folding = false}> : (tensor<64x128xbf16>, tensor<64x128xf32>) -> tensor<64x128xf32>
          return %5 : tensor<64x128xf32>
        }
  }];

  let options = [
    Option<"targetFormat", "target-format", "std::string", /*default=*/"\"bf16\"",
           "Target data format for conversion: f32, bf16 (default), or bfp_bf8">
  ];

  let dependentDialects = ["mlir::tt::ttcore::TTCoreDialect", "mlir::tt::ttir::TTIRDialect"];
}

def D2MGridSelection: Pass<"d2m-grid-selection", "::mlir::ModuleOp"> {
  let summary = "Optimized grid selection for D2M generic ops.";
  let description = [{
    This pass optimizes the grid selection for D2M generic operations. It walks through
    all d2m.generic ops and updates their operand ToLayoutOps with optimal grid sizes
    and proper dimAlignments based on the tensor physical shapes.

    The TTIRToD2M conversion creates all tensors with simple 1x1 grids and minimal
    dimAlignments. This pass then computes the optimal grid for each tensor based on
    its physical shape, finding the largest grid dimensions that evenly divide the shape.
    It also recomputes proper dimAlignments that account for grid-based alignment requirements.

    Note: This pass is not used in TTNN mode, where grids are already correctly set.
    The pipeline conditionally adds this pass only for non-TTNN mode.
  }];
  let dependentDialects = ["::mlir::tt::d2m::D2MDialect", "::mlir::tt::ttcore::TTCoreDialect"];
  let options = [
    ListOption<"overrideDeviceShape", "override-device-shape", "int64_t",
        "Override the device shape for grid optimization.">,
  ];
}

def D2MGenericReplaceGlobals : Pass<"d2m-generic-replace-globals", "::mlir::ModuleOp"> {
  let summary = "Replace `ttcore.get_global` operations with direct operand references within `d2m.generic` operations";
  let description = [{
    This pass replaces `ttcore.get_global` operations within `d2m.generic` operations
    with direct references to the corresponding operands of the parent generic operation.
    When a `ttcore.get_global` operation appears within a `d2m.generic` operation's compute
    region, and the global has an associated index, this pass replaces the get_global
    operation with the corresponding operand from the generic operation's operand list.
    This transformation eliminates unnecessary global symbol lookups and enables better
    optimization opportunities by making data flow more explicit within the generic
    operation's regions.
    Example transformation:
    ```mlir
    // Before:
    d2m.generic ins(%arg0, %arg1 : tensor<...>, tensor<...>) {
    ^bb0(%cb0: memref<...>, %cb1: memref<...>):
      %global_val = ttcore.get_global @my_global : tensor<...>
      // ... use %global_val ...
    }
    // After (assuming @my_global has index 0):
    d2m.generic ins(%arg0, %arg1 : tensor<...>, tensor<...>) {
    ^bb0(%cb0: memref<...>, %cb1: memref<...>):
      // ... use %arg0 directly instead of %global_val ...
    }
    ```
    This pass only operates on `ttcore.get_global` operations that:
    1. Are contained within a `d2m.generic` operation
    2. Reference a global that has a valid index attribute
    3. The index corresponds to a valid operand in the parent generic operation
  }];
}

def D2MMaterializeViewReturns : Pass<"d2m-materialize-view-returns", "::mlir::ModuleOp"> {
  let summary = "Materialize unmaterialized views before return operations";
  let description = [{
    This pass detects when a return value is an unmaterialized tensor view
    (produced by view operations like d2m.view_layout or d2m.stream_layout)
    and emits a d2m.generic op to consume and materialize the view before
    returning it.

    View operations in D2M represent tensor transformations using affine mappings
    without actual data movement. These views are "representational" and are
    normally consumed by generic ops that perform the actual fetching/transformation.
    However, when a view result is directly returned without being consumed by a
    generic op, there's no materialization, which can cause issues downstream.

    Example transformation:
    ```mlir
    // Before:
    func.func @example(%arg0: tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout>)
        -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout> {
      %view = d2m.view_layout %arg0 : tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout>
          -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout>
      return %view : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout>
    }

    // After:
    func.func @example(%arg0: tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout>)
        -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout> {
      %view = d2m.view_layout %arg0 : tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout>
          -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout>
      %empty = d2m.empty() : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout>
      %materialized = d2m.generic {grid = #ttcore.grid<8x8>, ...}
          ins(%view : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout>)
          outs(%empty : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout>) {
        ^bb0(%in: memref<...>, %out: memref<...>):
          d2m.yield %in : memref<...>
      } : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout>
      return %materialized : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout>
    }
    ```

    This pass should run after layout lowering and before bufferization.
  }];
  let dependentDialects = ["::mlir::tt::d2m::D2MDialect", "::mlir::tt::ttcore::TTCoreDialect"];
}
#endif
