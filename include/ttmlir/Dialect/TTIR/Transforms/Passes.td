// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTIR_TTIRPASSES_TD
#define TTMLIR_TTMLIR_DIALECT_TTIR_TTIRPASSES_TD

include "mlir/Pass/PassBase.td"

// Elementwise fusion of ttir.generic ops on tensors
def TTIRElementwiseFusion : Pass<"ttir-elementwise-fusion", "::mlir::ModuleOp"> {
  let summary = "Fuse chains of compute-only ttir.generic elementwise ops";
  let description = [{
    Performs elementwise fusion for `ttir.generic` producers into consumers when
    safe to do so. The initial implementation targets compute-only, affine-map
    form `ttir.generic` with tensor semantics and matching device attributes
    (grid, threads, block_factors). The pass is intended to run pre-bufferization
    to reduce intermediary tensors and improve locality.
  }];

  list<Option> options = [
    Option<"maxDstRegisterSizeTiles", "max-dst-register-size-tiles", "unsigned", "0", "Override the max dst size tiles or 0 if unset.">,
  ];

  let dependentDialects = [
    "::mlir::tt::ttir::TTIRDialect",
    "::mlir::affine::AffineDialect"
  ];
}

def TTIRLinAlgLoopFission : Pass<"ttir-linalg-loop-fission", "::mlir::ModuleOp"> {
  let summary = "Perform loop fission inside ttir.generic compute regions for load→compute→store triplets";
  let description = [{
    This pass scans `ttir.generic` compute regions that do not contain any tilize/untilize ops,
    finds nested affine loop nests, and performs loop fission (distribution) by extracting
    sequences of the form `affine.load → ttir tile compute op → affine.store` into separate
    loop nests, preserving the original loop bounds and indices.
  }];

  let dependentDialects = [
    "::mlir::tt::ttir::TTIRDialect",
    "::mlir::affine::AffineDialect"
  ];
}

def TTIRGenericTileComputeLoops : Pass<"ttir-generic-tile-compute-loops", "::mlir::ModuleOp"> {
  let summary = "";
  let description = [{
    This pass tiles affine compute loops according to the generic's subblock factors.
  }];

  list<Option> options = [
    Option<"maxDstRegisterSizeTiles", "max-dst-register-size-tiles", "unsigned", "0", "Override the max dst size tiles or 0 if unset.">,
  ];

  let dependentDialects = ["mlir::tt::ttir::TTIRDialect"];
}

def TTIRInsertDstRegisterAccess : Pass<"ttir-insert-dst-register-access", "::mlir::ModuleOp"> {
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
            %3 = "ttir.tile_matmul"(%0, %1, %2) : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
            affine.store %3, %cb2[%i2, %i3] : memref<3x2x!tt.tile<32x32, f32>, #tt.memory_space<l1>>
          }
        }
      }
    ```

    Into:
    ```mlir
    ^compute0(%cb0: memref<3x3x!tt.tile<32x32, f32>, #l1>, %cb1: memref<3x2x!tt.tile<32x32, f32>, #l1>, %cb2: memref<3x2x!tt.tile<32x32, f32>, #l1>):
      %c0 = arith.constant 0 : index
      %dst = ttir.acquire_dst() : memref<3x2x!tt.tile<32x32, f32>, #dst>
      %iter2 = ttir.iter_index(2) : index
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
            %4 = "ttir.tile_matmul"(%1, %2, %3) : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
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
       - All loads and stores to L1 are replaced with ones to dst.
       - Generic op automatically inserts conditional guard for reload.
       - This pass only works on affine ops.
  }];
  let dependentDialects = ["::mlir::tt::ttir::TTIRDialect", "::mlir::affine::AffineDialect", "::mlir::func::FuncDialect"];
  let options = [
    Option<"useTileMatmul", "use-tile-matmul", "bool", /*default=*/"false", "Use tile_matmul instead of tile_matmul_block">,
  ];
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
          %0 = affine.load %arg2[%arg5, %arg6] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
          %1 = affine.load %arg3[%arg5, %arg6] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
          %2 = "ttir.tile_maximum"(%0, %1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
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
          %2 = "ttir.tile_maximum"(%0, %1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          affine.store %2, %collapse_shape_1[%arg5 * 4 + %arg6] : memref<8x!ttcore.tile<32x32, f32>, #l1_>
        }
      }
    ```
  }];
}

def TTIRGenericGenerateDatamovement: Pass<"ttir-generic-generate-datamovement", "::mlir::ModuleOp"> {
  let summary = "Generate generic data movement threads.";
  let description = [{
    This pass makes the following transformation, given a generic compute region:
    ```mlir
    #map = affine_map<(d0, d1) -> (d0, d1)>
    #parallel = #ttcore.iterator_type<parallel>

    "ttir.generic"(%arg0, %arg1, %alloc) <{indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel]}> ({
    ^bb0(%arg2: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg3: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg4: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>):
      affine.for %arg5 = 0 to 2 {
        affine.for %arg6 = 0 to 4 {
          %0 = affine.load %arg2[%arg5, %arg6] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
          %1 = affine.load %arg3[%arg5, %arg6] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
          %2 = "ttir.tile_add"(%0, %1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          affine.store %2, %arg4[%arg5, %arg6] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        }
      }
    })
    ```

    We generate additional (prepended) regions that correspond to the data movement
    for each operand respectively:
    ```mlir
    "ttir.generic"(%arg0, %arg1, %alloc) <{indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel]}> ({
    ^bb0(%arg2: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg3: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg4: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>):
      ttir.yield %arg2 : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>)
    }, {
    ^bb0(%arg2: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg3: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg4: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>):
      ttir.yield %arg3 : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>)
    }, {
    ^bb0(%arg2: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg3: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg4: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>):
      ttir.await %arg4 : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>)
    }, {
    ^bb0(%arg2: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg3: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg4: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>):
      ttir.await %arg2, %arg3 : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<2x4x!ttcore.tile<32x32, f32>, #l1_>)
      affine.for %arg5 = 0 to 2 {
        affine.for %arg6 = 0 to 4 {
          %0 = affine.load %arg2[%arg5, %arg6] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
          %1 = affine.load %arg3[%arg5, %arg6] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
          %2 = "ttir.tile_add"(%0, %1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          affine.store %2, %arg4[%arg5, %arg6] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        }
      }
      ttir.yield %arg4 : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>)
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
    "ttir.generic"(%arg0, %arg1, %alloc) <{grid = #ttcore.grid<1x1>, indexing_maps = [#map1, #map2, #map3], iterator_types = [#parallel, #parallel, #reduction], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^datamovement0(%cb0: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<4x2x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>):
      ttir.yield %cb0 : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>)
    }, {
    ^datamovement1(%cb0: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<4x2x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>):
      ttir.yield %cb1 : (memref<4x2x!ttcore.tile<32x32, f32>, #l1_>)
    }, {
    ^datamovement2(%cb0: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<4x2x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>):
      ttir.await %cb2 : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
    }, {
    ^compute(%cb0: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<4x2x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>):
      ttir.await %cb0, %cb1 : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<4x2x!ttcore.tile<32x32, f32>, #l1_>)
      "ttir.tile_matmul_block"(%cb0, %cb1, %cb2) : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<4x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> ()
      ttir.yield %cb2 : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
    }) : (memref<1x1x2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<1x1x4x2x!ttcore.tile<32x32, f32>, #l1_>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #l1_>) -> ()
    ```

    Might move a trivial output datamovement thread to the compute thread to become:
    ```mlir
    "ttir.generic"(%arg0, %arg1, %alloc) <{grid = #ttcore.grid<1x1>, indexing_maps = [#map1, #map2, #map3], iterator_types = [#parallel, #parallel, #reduction], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^datamovement0(%cb0: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<4x2x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>):
      ttir.yield %cb0 : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>)
    }, {
    ^datamovement1(%cb0: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<4x2x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>):
      ttir.yield %cb1 : (memref<4x2x!ttcore.tile<32x32, f32>, #l1_>)
    }, {
    ^compute(%cb0: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<4x2x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>):
      ttir.await %cb0, %cb1 : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<4x2x!ttcore.tile<32x32, f32>, #l1_>)
      "ttir.tile_matmul_block"(%cb0, %cb1, %cb2) : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<4x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> ()
      ttir.yield %cb2 : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
      ttir.await %cb2 : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
    }) : (memref<1x1x2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<1x1x4x2x!ttcore.tile<32x32, f32>, #l1_>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #l1_>) -> ()
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

    grid = #ttcore.grid<2x4>
    operands : (memref<2x4x4x6x!ttcore.tile<32x32, f32>>, memref<4x4x6x8x!ttcore.tile<32x32, f32>>, memref<2x4x4x8x!ttcore.tile<32x32, f32>>)

    ^compute(%cb0: memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<4x8x!ttcore.tile<32x32, f32>, #l1_>):
      ttir.await %cb0, %cb1
      "ttir.tile_matmul_block"(%cb0, %cb1, %cb2)
      ttir.yield %cb2
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
      ttir.generic {grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#ttir.thread<compute>]}
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
      ttir.generic {grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#ttir.thread<compute, @compute_kernel0>]}
          ins(%arg0, %arg1 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<1x1x2x4x!ttcore.tile<32x32, f32>, #l1_>)
          outs(%alloc : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #l1_>)
    }

    func.func private @compute_kernel0(%arg0: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg2: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>) attributes {ttir.thread_type = 0 : i32} {
      // ...compute body...
      return
    }
    ```
  }];
}

def TTIRLowerToLayout: Pass<"ttir-lower-to-layout", "::mlir::ModuleOp"> {
  let summary = "Lower layouts to generic ops.";
  let description = [{
    Transition between different tensor layouts.

    A single to_layout op in ttir can simultaneously perform multiple layout transformations
    at once, including changing layout, format, memory space or memory layout. This pass splits each of
    these transformation categories into separate to_layout ops and then lowers them to generic ops.
    There is one exception to the generic lowering and that is to/from device memory, this case lowers
    to a specialized flavor of to_layout that has the hostInfo attribute set.

    For example a compound to layout that goes from host to tilized 8x8 device grid might look like:
    ```
    #layout2 = #ttcore.metal_layout<(d0, d1) -> (d0, d1), undef, <8x8>, memref<1x3x!ttcore.tile<32x32, f32>, #l1_>>
    %0 = ttir.empty() : tensor<256x768xf32, #layout2>
    %1 = ttir.to_layout %arg0, %0 : tensor<256x768xf32> into tensor<256x768xf32, #layout2>
    ```

    Into:
    ```
    #layout = #ttcore.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<256x768xf32, #l1_>>
    #layout1 = #ttcore.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<8x24x!ttcore.tile<32x32, f32>, #l1_>>
    #layout2 = #ttcore.metal_layout<(d0, d1) -> (d0, d1), undef, <8x8>, memref<1x3x!ttcore.tile<32x32, f32>, #l1_>>
    %0 = ttir.empty() : tensor<256x768xf32, #layout2>
    %1 = ttir.empty() : tensor<256x768xf32>
    %2 = ttir.empty() : tensor<256x768xf32, #layout>
    // Move to device
    %3 = ttir.to_layout %arg0, %2 : tensor<256x768xf32> into tensor<256x768xf32, #layout> hostInfo = #layout -> tensor<256x768xf32, #layout>
    %4 = ttir.empty() : tensor<256x768xf32, #layout1>
    // Tilize
    %5 = ttir.generic {grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#ttir.thread<compute>]}
        ins(%3 : tensor<256x768xf32, #layout>)
        outs(%4 : tensor<256x768xf32, #layout1>)  {
    ^compute0(%cb0: memref<256x768xf32, #l1_>, %cb1: memref<8x24x!ttcore.tile<32x32, f32>, #l1_>):
      "ttir.tile_tilize_block"(%cb0, %cb1) : (memref<256x768xf32, #l1_>, memref<8x24x!ttcore.tile<32x32, f32>, #l1_>) -> ()
    } : tensor<256x768xf32, #layout1>
    // Reblock to 8x8 grid
    %view = "ttir.view_layout"(%5) : (tensor<256x768xf32, #layout1>) -> tensor<256x768xf32, #layout2>
    %6 = ttir.generic {grid = #ttcore.grid<8x8>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#ttir.thread<compute>]}
        ins(%view : tensor<256x768xf32, #layout2>)
        outs(%0 : tensor<256x768xf32, #layout2>)  {
    ^compute0(%cb0: memref<1x3x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<1x3x!ttcore.tile<32x32, f32>, #l1_>):
      ttir.yield %cb0 : (memref<1x3x!ttcore.tile<32x32, f32>, #l1_>)
    } : tensor<256x768xf32, #layout2>
    ```
  }];
}

def TTIRGenericApplyInterchange : Pass<"ttir-generic-apply-interchange", "::mlir::ModuleOp"> {
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

def TTIRAllocate: Pass<"ttir-allocate", "::mlir::ModuleOp"> {
  let summary = "Create streams required by generic ops.";
  let description = [{
    This pass handles several related tasks:
      - allocating data streams (stream_layouts) and their associated buffers as required by generic
        op operands;
      - allocating memory addresses/alignments for buffers supporting such streams as well as
        other memref allocs needed by data movement/compute kernel functions.

    Currently this pass only does some simple heuristics for forming stream_layout ops when required
    for correctness. In the future, this will be augmented with analysis that will consider resource conflicts
    between the number of streams, their buffer sizes, and L1 memory size limits.

    Similarly, memory addresses are currently taken from L1 and never deallocated and so are assigned
    by a simple bumper allocation scheme -- again, this will be modified to do smarter static allocation
    schedules.

    Converts:
    ```mlir
    %alloc = memref.alloc() : memref<1x5x1x5x!ttcore.tile<32x32, f32>, #ttcore.shard<20480x4096>, #l1_>
    "ttir.to_layout"(%arg0, %alloc) : (memref<1x1x1x784xf32, #ttcore.shard<3136x4>, #l1_>, memref<1x5x1x5x!ttcore.tile<32x32, f32>, #ttcore.shard<20480x4096>, #l1_>) -> ()
    %alloc_0 = memref.alloc() : memref<5x8x5x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>
    "ttir.to_layout"(%arg1, %alloc_0) : (memref<1x1x784x256xf32, #ttcore.shard<1024x4>, #l1_>, memref<5x8x5x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>) -> ()
    %alloc_1 = memref.alloc() : memref<1x8x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>
    ttir.generic {grid = #ttcore.grid<1x8>, indexing_maps = [#map, #map1, #map2], iterator_types = [#parallel, #parallel, #reduction], ...}
        ins(%alloc, %alloc_0 : memref<1x5x1x5x!ttcore.tile<32x32, f32>, #ttcore.shard<20480x4096>, #l1_>, memref<5x8x5x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>)
        outs(%alloc_1 : memref<1x8x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>)  {
      ...
    }
    ```

    Into:
    ```mlir
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<1x5x1x5x!ttcore.tile<32x32, f32>, #ttcore.shard<20480x4096>, #l1_>
    "ttir.to_layout"(%arg0, %alloc) : (memref<1x1x1x784xf32, #ttcore.shard<3136x4>, #l1_>, memref<1x5x1x5x!ttcore.tile<32x32, f32>, #ttcore.shard<20480x4096>, #l1_>) -> ()
    %alloc_0 = memref.alloc() {address = 21504 : i64, alignment = 16 : i64} : memref<5x8x5x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>
    "ttir.to_layout"(%arg1, %alloc_0) : (memref<1x1x784x256xf32, #ttcore.shard<1024x4>, #l1_>, memref<5x8x5x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>) -> ()
    %alloc_1 = memref.alloc() {address = 41984 : i64, alignment = 16 : i64} : memref<1x8x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>
    %alloc_2 = memref.alloc() {address = 46080 : i64, alignment = 16 : i64} : memref<1x5x1x5x!ttcore.tile<32x32, f32>, #ttcore.shard<20480x4096>, #l1_>
    %stream = "ttir.stream_layout"(%alloc, %alloc_2) : (memref<1x5x1x5x!ttcore.tile<32x32, f32>, #ttcore.shard<20480x4096>, #l1_>, memref<1x5x1x5x!ttcore.tile<32x32, f32>, #ttcore.shard<20480x4096>, #l1_>)
     -> memref<1x5x1x5x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>
    %alloc_3 = memref.alloc() {address = 66560 : i64, alignment = 16 : i64} : memref<5x8x5x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>
    %stream_4 = "ttir.stream_layout"(%alloc_0, %alloc_3) : (memref<5x8x5x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>, memref<5x8x5x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>)
     -> memref<5x8x5x1x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>
    ttir.generic {grid = #ttcore.grid<1x8>, indexing_maps = [#map, #map1, #map2], iterator_types = [#parallel, #parallel, #reduction], ...}
        ins(%stream, %stream_4 : memref<1x5x1x5x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>, memref<5x8x5x1x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>)
        outs(%alloc_1 : memref<1x8x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>)  {
      ...
    }
    ```
  }];
  let dependentDialects = ["::mlir::tt::ttcore::TTCoreDialect", "::mlir::memref::MemRefDialect"];
  let options = [
    Option<"numStreamBuffers",
           "num-stream-buffers",
           "unsigned", /*default=*/"2",
           "Number of backing buffers to allocate per stream storage (>=1) Default is 2.">,
    Option<"allowOutputSpilling",
           "allow-output-spilling",
          "bool", /*default=*/"false",
          "Make generic outputs eligible for spilling to DRAM.">,
  ];
}

def TTIRImplicitBroadcastFold: Pass<"ttir-implicit-broadcast-fold", "::mlir::ModuleOp"> {
  let summary = "Broadcast operation is folded to all the consumers.";
  let description = [{
    This pass walks through the graph and folds broadcasts operations when it is implicitly supported by the operation.

    Example:
    %0 = ttir.empty() : tensor<1x16x32xf32>
    %1 = "ttir.broadcast"(%arg1, %0) <{broadcast_dimensions = array<i32: 1, 16, 1>}> : (tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %2 = ttir.empty() : tensor<1x16x32xf32>
    %3 = "ttir.multiply"(%arg0, %1, %2) : (tensor<1x16x32xf32>, tensor<1x16x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>

    Since MultiplyOp supports implicit broadcasting, above broadcast is folded as:
    %0 = ttir.empty() : tensor<1x16x32xf32>
    %1 = "ttir.multiply"(%arg0, %arg1, %0) : (tensor<1x16x32xf32>, tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
  }];
}

def TTIRHoistTransform: Pass<"ttir-cpu-hoist-transform", "::mlir::ModuleOp">
{
  let summary = "Transform to perform hoist mechanics on any ops marked to be hoisted for CPU lowering";
  let description = [{
    Transform pass which runs an analysis pass to find ops which should be hoisted, and then hoists those ops.  Currently we only have a manual analysis which requires a commandline list of named locs to hoist--in the future, we will have an automatic analysis as well.

    Example:
    input:
      ttcore.device_module {
        builtin.module {
          func.func @add(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
            %0 = ttir.empty() : tensor<32x32xbf16>
            %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16> loc("add_op1")
            return %1 : tensor<32x32xbf16>
          }
        }
      }
    output:
      ttcore.device_module {
        builtin.module {
          func.func @add(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
            %0 = ttir.empty() : tensor<32x32xbf16>
            %1 = call @hoisted_ttir_add_32x32xbf16_32x32xbf16_32x32xbf16_func_decl(%arg0, %arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
            return %1 : tensor<32x32xbf16>
          }
          func.func private @hoisted_ttir_add_32x32xbf16_32x32xbf16_32x32xbf16_func_decl(tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
        }
      }
      ttcore.cpu_module {
        builtin.module {
          func.func @hoisted_ttir_add_32x32xbf16_32x32xbf16_32x32xbf16_func(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>, %arg2: tensor<32x32xbf16>) -> tensor<32x32xbf16> attributes {arg_ranks = [2, 2, 2, 2]} {
            %0 = "ttir.add"(%arg0, %arg1, %arg2) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
            return %0 : tensor<32x32xbf16>
          }
        }
      }
  }];

  let dependentDialects = ["::mlir::tt::ttcore::TTCoreDialect"];
}

def ElementTypeNormalization: Pass<"ttir-element-type-normalization", "::mlir::ModuleOp">
{
  let summary = "Normalize element types into list of supported types.";
  let description = [{
    "This pass walks through the graph and normalizes the element types into a list of supported types. This is useful for lowering
        to a target that only supports a subset of the element types.
  }];

  list<Option> options = [
    Option<"enableBfp8Conversion", "enable-bfp8-conversion", "bool", "false", "When enabled this pass will convert all bfloat16 types into bfp8_b types.">,
  ];
}

def TTIRFlattenSlidingWindow: Pass<"ttir-flatten-sliding-window", "::mlir::ModuleOp">
{
  let summary = "Flatten sliding window ops.";
  let description = [{
    This is a compatibility pass for converting to the TTNN dialect.
    This pass walks through the graph and flattens sliding window ops (ttir.conv2d, ttir.max_pool2d, ttir.avg_pool2d).

    Example:
      Before:
         %dps = ttir.empty() : tensor<3x15x31x16xbf16>
         %1 = "ttir.conv2d"(%input, %weight, %bias, %dps)
            <{
              stride = 2: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<3x32x64x8xbf16>, tensor<16x8x3x3xbf16>, tensor<1x1x1x16xbf16>, tensor<3x15x31x16xbf16>) -> tensor<3x15x31x16xbf16>

      After:
        %reshape_dps = ttir.empty() : tensor<1x1x6144x8xbf16>
        %0 = "ttir.reshape"(%input, %reshape_dps) <{[i32: 1, i32: 1, i32: 6144, i32: 8]}> : (tensor<3x32x64x8xbf16>, tensor<1x1x6144x8xbf16>) -> tensor<1x1x6144x8xbf16>
        %new_conv_dps = ttir.empty() : tensor<1x1x1395x16xbf16>
        %1 = "ttir.conv2d"(%0, %weight, %bias, %new_conv_dps)
            <{
              stride = 2: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32,
              flattened_compat_info = #ttir<flattened_compat in_channels = 8, out_channels = 16, batch_size = 3, input_height = 32, input_width = 64,>
            }> : (tensor<1x1x6144x8xbf16>, tensor<16x8x3x3xbf16>, tensor<1x1x1x16xbf16>, tensor<1x1x1395x16xbf16>) -> tensor<1x1x1395x16xbf16>
          %output_reshape_dps = ttir.empty() : tensor<3x15x30x16xbf16>
          %2 = "ttir.reshape"(%1, %output_reshape_dps) <{[i32: 3, i32: 15, i32: 31, i32: 16]}> : (tensor<1x1x1395x16xbf16>, tensor<3x15x31x16xbf16>) -> tensor<3x15x31x16xbf16>
  }];
}

def TTIREraseInverseOps: Pass<"ttir-erase-inverse-ops", "::mlir::ModuleOp">
{
  let summary = "Erase inverse ops.";
  let description = [{
    This pass walks through the graph and erases inverse operations.

    For example:
      ttir.permute(0, 1, 3, 2) -> ttir.exp -> ttir.permute(0, 1, 3, 2)

    The above sequence can be reduced to simply: "ttir.exp" as the permutations
    on either end are inverses.
  }];

  let dependentDialects = ["mlir::tt::ttcore::TTCoreDialect", "mlir::tt::ttir::TTIRDialect"];

  list<Option> options = [
    Option<"enableCommuteUpwards", "enable-commute-upwards", "bool", "true", "Enable commuting upwards. This should only be false for testing purposes (i.e you want to test a commute downwards pattern)">,
    Option<"enableCommuteDownwards", "enable-commute-downwards", "bool", "true", "Enable commuting downwards. This should only be false for testing purposes (i.e you want to test a commute upwards pattern)">,
    Option<"maxIterations", "max-iterations", "uint64_t", "100", "Maximum number of iterations to perform commuting. The number of TMs is expected to converge before this limit is reached.">
  ];
}

def TTIRExplicateTMs: Pass<"ttir-explicate-tms", "::mlir::ModuleOp">
{
  let summary = "This pass walks through the graph and explicates implicit broadcasts and reshapes on the graph edges.";
  let description = [{
    This pass walks through the graph and explicates implicit broadcasts and reshapes on the graph edges.
  }];

  let dependentDialects = ["mlir::tt::ttcore::TTCoreDialect", "mlir::tt::ttir::TTIRDialect"];
}

def TTIRQuantDataTypeConversionPass : Pass<"ttir-quant-data-type-conversion", "::mlir::ModuleOp"> {
  let summary = "Convert integer data types in quantized types to a specified bit width";
  let description = [{
    This pass converts all integer data types in quantized types (e.g., i8) to a specified
    bit width (e.g., i32) during ttir-to-ttnn conversion. This is a temporary workaround as
    tt-metal currently only supports i32 quantized types.

    Example:
    Input:
      %0 = "ttir.quantize"(%arg0, %1) : (tensor<1x3x224x224xf32>, tensor<1x3x224x224x!quant.uniform<i8:f32, 1.000000e-01>>) -> tensor<1x3x224x224x!quant.uniform<i8:f32, 1.000000e-01>>

    Output (with quant_bit_width=32):
      %0 = "ttir.quantize"(%arg0, %1) : (tensor<1x3x224x224xf32>, tensor<1x3x224x224x!quant.uniform<i32:f32, 1.000000e-01>>) -> tensor<1x3x224x224x!quant.uniform<i32:f32, 1.000000e-01>>
  }];
  let dependentDialects = ["mlir::tt::ttir::TTIRDialect", "mlir::quant::QuantDialect"];

  list<Option> options = [
    Option<"targetBitWidth", "target-bit-width", "uint32_t", "32", "Target integer bit width for quantized types (8, 16, 32, 64)">
  ];
}

def TTIRQuantDequantConversion : Pass<"ttir-quant-dequant-conversion", "::mlir::ModuleOp">
{
  let summary = "Convert floating-point ops surrounded by quantize/dequantize into TTIR quantized operations.";
  let description = [{
    This pass detects and fuses the (x -> dequantize → floating-point operation)
    pattern into a single TTIR operation that consumes quantized inputs and produces
    quantized outputs. If the target operation implements the QuantizableOpInterface,
    it is rewritten directly into its quantized form. Otherwise, a fallback transformation
    inserts dequantize and quantize nodes around the original op to maintain correctness.
  }];
  let dependentDialects = ["mlir::tt::ttir::TTIRDialect", "mlir::quant::QuantDialect"];
}

def TTIRFusing: Pass<"ttir-fusing", "::mlir::ModuleOp">
{
  let summary = "TTIR fusing pass.";
  let description = "This pass tries to fuse operations together with goal to reduce the number of operations in the graph.";

  let options = [
      Option<"conv2dWithMultiplyEnabled",
             "ttnn-enable-conv2d-with-multiply-pattern",
             "bool", /*default=*/"false",
             "Controls if we should enable the Conv2dWithMultiply pattern">,
      Option<"globalPoolFusingEnabled",
              "ttnn-enable-global-avg-pool-pattern",
              "bool", /*default=*/"false",
              "Controls if we should enable the GlobalAveragePoolingPattern fusing pattern">,
  ];
}

def TTIRMultiDeviceTensorAnnotation: Pass<"ttir-multi-device-tensor-annotation", "::mlir::ModuleOp">
{
  let summary = "TTIR multi-device tensor annotation pass.";
  let description = [{
    This pass walks through the graph with mesh and add multi-device tensor annotations to tensor encoding.

    Example:
      Before:
         %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 0, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1024x2048xf32>) -> tensor<512x512xf32>
        %1 = "ttir.mesh_shard"(%arg1) <{shard_dims = array<i64: 0, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1024x2048xf32>) -> tensor<512x512xf32>
        %2 = ttir.empty() : tensor<512x512xf32>
        %3 = "ttir.add"(%0, %1, %2) : (tensor<512x512xf32>, tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
        %4 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: 0, 1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<512x512xf32>) -> tensor<1024x2048xf32>
        return %4 : tensor<1024x2048xf32>

      After:
        %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 0, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1024x2048xf32>) -> tensor<512x512xf32, #ttcore.tensor_mesh<"mesh">>
        %1 = "ttir.mesh_shard"(%arg1) <{shard_dims = array<i64: 0, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1024x2048xf32>) -> tensor<512x512xf32, #ttcore.tensor_mesh<"mesh">>
        %2 = ttir.empty() : tensor<512x512xf32, #ttcore.tensor_mesh<"mesh">>
        %3 = "ttir.add"(%0, %1, %2) : (tensor<512x512xf32, #ttcore.tensor_mesh<"mesh">>, tensor<512x512xf32, #ttcore.tensor_mesh<"mesh">>, tensor<512x512xf32, #ttcore.tensor_mesh<"mesh">>) -> tensor<512x512xf32, #ttcore.tensor_mesh<"mesh">>
        %4 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: 0, 1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<512x512xf32, #ttcore.tensor_mesh<"mesh">>) -> tensor<1024x2048xf32>
        return %4 : tensor<1024x2048xf32>
  }];

  let dependentDialects = ["mlir::tt::ttcore::TTCoreDialect", "mlir::tt::ttir::TTIRDialect"];
}

#endif
