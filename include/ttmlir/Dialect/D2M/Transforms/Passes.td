// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_D2M_D2MPASSES_TD
#define TTMLIR_TTMLIR_DIALECT_D2M_D2MPASSES_TD

include "mlir/Pass/PassBase.td"

// Add scratch inputs to d2m.generic ops (pre-bufferization)
def D2MAddScratchInputs : Pass<"d2m-add-scratch-inputs", "::mlir::ModuleOp"> {
  let summary = "Add scratch input operand to each d2m.generic op";
  let description = [{
    This pass adds a scratch tensor input to each d2m.generic operation.
    The scratch tensor type is derived from the first input, ensuring type
    consistency. The scratch is marked in the scratch_inputs attribute to
    prevent streaming.

    This pass must run in tensor space (before bufferization) so that the
    scratch tensor flows through bufferization and D2MAllocate like any
    other operand. Only d2m.generic ops in compute-only form (before
    datamovement generation) and where ALL inputs are tiled are modified.
    This excludes tilize/untilize ops which mix scalar and tiled operands.

    Example transformation:
    ```mlir
    // Before:
    %result = d2m.generic ins(%A : tensor<1x1x4x4x!ttcore.tile<32x32, f32>, #layout>)
                          outs(%B : tensor<1x1x4x4x!ttcore.tile<32x32, f32>, #layout>) { ... }

    // After:
    %scratch = d2m.empty() : tensor<1x1x4x4x!ttcore.tile<32x32, f32>, #layout>
    %result = d2m.generic scratch_inputs = [1]
        ins(%A, %scratch : tensor<...>, tensor<...>)
        outs(%B : tensor<...>) {
    ^compute0(%cb_A, %cb_scratch, %cb_B):
      // compute body (can now use %cb_scratch for intermediate storage)
    }
    ```
  }];

  // Note: The scratch tensor type (shape, tile type, layout) is derived from
  // the first tiled operand of each d2m.generic, ensuring type consistency.
  // No configuration options are needed.

  let dependentDialects = [
    "::mlir::tt::d2m::D2MDialect",
    "::mlir::tt::ttcore::TTCoreDialect"
  ];
}

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

def D2MOpScheduler : Pass<"d2m-op-scheduler", "::mlir::ModuleOp"> {
  let summary = "Performs op scheduling within each affine.for loop nest in the d2m.generic block to minimize total required register slices.";
  let description = [{
    This pass scans `d2m.generic` ops that are nontrivially fused (3+ input operands) to
    find nested affine.for loop-nests with attribute "d2m.linalg_root" on the root loop.
    It then builds an operation tree from the ops contained in the lowest level loop
    under the root, and then uses a Sethi-Ullman scheduling algorithm in conjunction
    with the tree to reorder the ops. This is done such that the final sequence of ops
    requires the minimal number of dst register slices to execute. Adds tag "d2m.scheduled"
    to the root loop for use by downstream D2MInsertDstRegisterAccess pass.
  }];

  list<Option> options = [
    Option<"enableOpScheduler", "enable-op-scheduler", "bool", "true",
           "Enable operation scheduling optimization (currently hard-wired to true internally)">,
  ];

  let dependentDialects = [
    "::mlir::tt::d2m::D2MDialect",
    "::mlir::affine::AffineDialect",
    "::mlir::func::FuncDialect"
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
  let summary = "Tile linalg.generic compute loops to fit DST register capacity.";
  let description = [{
    This pass tiles linalg.generic compute loops according to DST register constraints.

    When `enableTwoPhaseDestTiling` is false (default):
      - Tiles to fit within 1×DST capacity (single-phase tiling)

    When `enableTwoPhaseDestTiling` is true:
      - Phase 1: Tiles to fit within 2×DST capacity (outer loops)
      - Phase 2: Tiles the inner linalg.generic to 1×DST (inner loop iterates 2×)

    This two-phase approach is used for FPU fusion with L1 scratch spilling, where
    the inner loop processes 1×DST chunks and intermediate results are spilled
    to L1 scratch between fissioned loop nests.
  }];

  list<Option> options = [
    Option<"maxDstPhysicalSizeTiles", "max-dst-physical-size-tiles", "unsigned", "0",
           "Override the max dst physical size tiles or 0 if unset.">,
    Option<"enableTwoPhaseDestTiling", "enable-two-phase-dest-tiling", "bool", "false",
           "Enable two-phase tiling: first to 2×DST, then to 1×DST. Used for FPU fusion with L1 scratch spilling.">,
  ];

  let dependentDialects = [
    "mlir::tt::d2m::D2MDialect",
    "::mlir::affine::AffineDialect",
    "::mlir::scf::SCFDialect"
  ];
}

def D2MSpillAndScratch : Pass<"d2m-spill-and-scratch", "::mlir::ModuleOp"> {
  let summary = "Replace intermediate allocations with scratch buffers for FPU fusion.";
  let description = [{
    This pass transforms compute regions with multiple `d2m.scratch_space_loop` loop
    nests to use scratch buffers for intermediate results instead of full-sized
    L1 allocations. This enables efficient pipelining by keeping only a few iterations
    worth of intermediate data alive at a time.

    The pass performs:

    1. **Loop Nest Identification**: Finds all `affine.for` loops tagged with
       `d2m.scratch_space_loop` attribute within each `d2m.generic` op.

    2. **Intermediate Detection**: Identifies `memref.alloc` ops that are written
       by one loop nest and read by another (producer/consumer relationship).

    3. **Scratch Buffer Creation**: For each intermediate allocation:
       - Computes scratch buffer shape from loop structure:
         [scratch_space_iterations, inner_dim0, inner_dim1, ...]
       - Creates a `d2m.scratch_allocate` op with the computed shape
       - The shape is NOT hardcoded but derived from the actual loop bounds

    4. **Store/Load Rewriting**: Updates stores and loads to use scratch buffer
       with full indices including the scratch_space_loop iterator:
       - Producer stores: `affine.store %val, %scratch[%i, %j, %k]`
       - Consumer loads: `affine.load %scratch[%i, %j, %k]`

    5. **Cleanup**: Removes old subviews and full-sized allocations that are
       no longer used.

    Example transformation:
    ```mlir
    // Before (separate loop nests with full-sized intermediate):
    %alloc_14 = memref.alloc() : memref<8x8x!ttcore.tile<...>, #l1>  // 64 tiles
    affine.for %i = 0 to 2 {d2m.scratch_space_loop} {  // Producer
      affine.for %j = 0 to 1 {
        affine.for %k = 0 to 8 {
          %result = d2m.tile_add ...
          affine.store %result, %subview_of_alloc_14[%j, %k]
        }
      }
    }
    affine.for %i = 0 to 2 {d2m.scratch_space_loop} {  // Consumer
      affine.for %j = 0 to 1 {
        affine.for %k = 0 to 8 {
          %val = affine.load %subview_of_alloc_14[%j, %k]
          ...
        }
      }
    }

    // After (scratch buffer sized for loop structure):
    %scratch = d2m.scratch_allocate {slot = 0 : i64}
               : memref<2x1x8x!ttcore.tile<...>, #l1>
    affine.for %i = 0 to 2 {d2m.scratch_space_loop} {  // Producer
      affine.for %j = 0 to 1 {
        affine.for %k = 0 to 8 {
          %result = d2m.tile_add ...
          affine.store %result, %scratch[%i, %j, %k]  // Store with full indices
        }
      }
    }
    affine.for %i = 0 to 2 {d2m.scratch_space_loop} {  // Consumer
      affine.for %j = 0 to 1 {
        affine.for %k = 0 to 8 {
          %val = affine.load %scratch[%i, %j, %k]  // Load with full indices
          ...
        }
      }
    }
    ```
  }];

  let dependentDialects = [
    "::mlir::tt::d2m::D2MDialect",
    "::mlir::affine::AffineDialect",
    "::mlir::memref::MemRefDialect"
  ];
}

def D2MFPUScratchFission : Pass<"d2m-fpu-scratch-fission", "::mlir::ModuleOp"> {
  let summary = "Fission loops at scratch store boundaries for FPU fusion.";
  let description = [{
    This pass performs loop fission on compute regions that have been processed
    by the SpillAndScratch pass. It splits a single loop nest (containing multiple
    compute operations with spill/reload) into separate loop nests, one per store.

    The fission is based on finding "cut points" - stores to scratch buffers or
    L1 memory. For each store, we walk backwards on the dependency graph to find
    all operations that contribute to that store, then create a separate cloned
    loop nest containing only those operations.

    This pass should run after `d2m-spill-and-scratch` and before
    `d2m-insert-dst-register-access`.

    Example transformation:
    ```mlir
    // Before (single loop with spill/reload):
    affine.for %i = ... {d2m.scratch_space_loop} {
      affine.for ... {
        %c = affine.load %input_c[...]
        %d = affine.load %input_d[...]
        %E = d2m.tile_add %c, %d
        affine.store %E, %scratch0[...]      // cut point 1

        %a = affine.load %input_a[...]
        %b = affine.load %input_b[...]
        %F = d2m.tile_add %a, %b
        affine.store %F, %scratch1[...]      // cut point 2

        %E_reload = affine.load %scratch0[...]
        %F_reload = affine.load %scratch1[...]
        %G = d2m.tile_add %E_reload, %F_reload
        affine.store %G, %output[...]        // cut point 3
      }
    }

    // After (three separate loop nests):
    affine.for %i = ... {d2m.scratch_space_loop} {
      affine.for ... {
        %c = affine.load %input_c[...]
        %d = affine.load %input_d[...]
        %E = d2m.tile_add %c, %d
        affine.store %E, %scratch0[...]
      }
    }
    affine.for %i = ... {d2m.scratch_space_loop} {
      affine.for ... {
        %a = affine.load %input_a[...]
        %b = affine.load %input_b[...]
        %F = d2m.tile_add %a, %b
        affine.store %F, %scratch1[...]
      }
    }
    affine.for %i = ... {d2m.scratch_space_loop} {
      affine.for ... {
        %E_reload = affine.load %scratch0[...]
        %F_reload = affine.load %scratch1[...]
        %G = d2m.tile_add %E_reload, %F_reload
        affine.store %G, %output[...]
      }
    }
    ```
  }];

  let dependentDialects = [
    "::mlir::tt::d2m::D2MDialect",
    "::mlir::affine::AffineDialect"
  ];
}

def D2MLinalgToAffine : Pass<"d2m-linalg-to-affine", "::mlir::ModuleOp"> {
  let summary = "Convert linalg.generic operations to affine loops";
  let description = [{
    This pass converts all linalg.generic operations within d2m.generic
    compute regions to affine loop nests. This is a prerequisite for
    the DST register allocation passes which operates on affine IR.

    The pass handles:
    - Regular linalg.generic ops -> affine for loops
    - Special tile matmul cases (when useTileMatmul=true)

    Example transformation:
    ```mlir
    // Before:
    linalg.generic {indexing_maps = [#map, #map],
                    iterator_types = ["parallel", "parallel"]}
        ins(%cb0 : memref<2x4x!ttcore.tile<32x32, f32>, #l1>)
        outs(%cb0 : memref<2x4x!ttcore.tile<32x32, f32>, #l1>) {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %0 = "d2m.tile_exp"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %0 : !ttcore.tile<32x32, f32>
    }

    // After:
    affine.for %i = 0 to 2 {
      affine.for %j = 0 to 4 {
        %val = affine.load %cb0[%i, %j] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
        %result = "d2m.tile_exp"(%val) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        affine.store %result, %cb0[%i, %j] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
      }
    }
    ```
  }];
  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::linalg::LinalgDialect"
  ];
  let options = [
    Option<"useTileMatmul", "use-tile-matmul", "bool", "false",
           "Use tile_matmul instead of tile_matmul_block for matmul ops">,
    Option<"markRootLoops", "mark-root-loops", "bool", "true",
           "Mark root loops with d2m.linalg_root attribute for subsequent passes">
  ];
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
    these transformation categories into separate ops and then lowers them to generic ops.

    Host transfers are lowered to dedicated ops:
    - Host to device: d2m.to_device
    - Device to host: d2m.to_host

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
    // Move to device using dedicated to_device op
    %3 = d2m.to_device %arg0, %2 layout = #layout : tensor<256x768xf32> into tensor<256x768xf32, #layout> -> tensor<256x768xf32, #layout>
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
    Option<"streamInsertPolicy",
          "stream-insert-policy",
          "std::string", /*default=*/"\"infer\"",
          "Policy for deciding when to insert operand streams ('always', 'infer').">,
    ListOption<"availableL1AddrRange",
          "available-l1-addr-range",
          "std::int64_t",
          "Assume given L1 addressable range [base, max).">,
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

def D2MDecomposeMasking : Pass<"d2m-decompose-masking", "::mlir::ModuleOp"> {
  let summary = "Decompose block_mask ops into linalg.generic with tile arithmetic";
  let description = [{
    This pass decomposes high-level `block_mask` operations into linalg.generic
    with per-tile masking using primitive tile arithmetic. The block_mask op
    operates at block level (memref of tiles) and masks tiles that are completely
    outside the logical tensor bounds with a specified OOBVal.

    The decomposition creates a linalg.generic that iterates over tiles and:
    1. Get current tile indices from linalg.index
    2. Compute global element offset: tileIdx * 32 (tile size)
    3. Check if entire tile is OOB: globalOffset >= logicalBound
    4. Combine row/col OOB flags with OR (any dimension OOB means tile is OOB)
    5. Blend: result = input * select(oob, 0, 1) + select(oob, fill, 0)

    The pass uses arith.select to avoid inf*0=NaN issues when fill is infinity.

    Example transformation:
    ```mlir
    // Before:
    d2m.block_mask %input, %output, %rows, %cols, <zero>
        : (memref<2x2x!ttcore.tile<32x32, f32>>, memref<2x2x!ttcore.tile<32x32, f32>>)

    // After:
    linalg.generic ... {
      %row = linalg.index 0 : index
      %col = linalg.index 1 : index
      %globalRow = arith.muli %row, %c32 : index
      %globalCol = arith.muli %col, %c32 : index
      %rowOOB = arith.cmpi sge, %globalRow, %rows : index
      %colOOB = arith.cmpi sge, %globalCol, %cols : index
      %oob = arith.ori %rowOOB, %colOOB : i1
      %mulFactor = arith.select %oob, %zero, %one : f32
      %addend = arith.select %oob, %fillVal, %zero : f32
      %scaled = d2m.tile_mul %in, %mulFactor
      %result = d2m.tile_add %scaled, %addend
      linalg.yield %result
    }
    ```

    This pass should run after LowerToLayout and before LinalgToAffine.
  }];
  let dependentDialects = [
    "::mlir::tt::d2m::D2MDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::linalg::LinalgDialect"
  ];
}

def D2MScalarizeConstTensors : Pass<"d2m-scalarize-const-tensors", "::mlir::ModuleOp"> {
  let summary = "Scalarize constant tensor inputs in d2m.generic operations";
  let description = [{
    This pass looks at d2m.generic operations and identifies constant tensor inputs
    (created by d2m.full operations). When a constant tensor is used by operations
    that support scalar operands (operations with supportsTileOrScalarRhs()), the
    pass replaces the tensor block arguments in linalg.generic operations with
    scalar constants.

    This transformation can improve performance by:
    1. Reducing memory usage (no need to store the constant tensor)
    2. Enabling more efficient scalar operations in the compute kernel

    Example transformation:
    ```mlir
    // Before:
    %const = d2m.full {fill_value = 2.5 : f32, shape = array<i32: 128, 128>} : tensor<128x128xf32>
    %const_layout = d2m.to_layout %const, ...
    d2m.generic ins(%input, %const_layout : ...) {
    ^compute0(%cb0: !d2m.cb<tensor<4x4x!ttcore.tile<...>>>, %cb1: !d2m.cb<tensor<4x4x!ttcore.tile<...>>>, ...):
      %t0 = d2m.wait %cb0 : ...
      %t1 = d2m.wait %cb1 : ...
      %t2 = d2m.reserve %cb2 : ...
      %result = linalg.generic ins(%t0, %t1 : ...) outs(%t2 : ...) {
      ^bb0(%in: !ttcore.tile<...>, %in_const: !ttcore.tile<...>, %out: !ttcore.tile<...>):
        %add = d2m.tile_add(%in, %in_const) : (!ttcore.tile<...>, !ttcore.tile<...>) -> !ttcore.tile<...>
        linalg.yield %add : !ttcore.tile<...>
      } -> ...
    }

    // After:
    d2m.generic ins(%input : ...) {
    ^compute0(%cb0: !d2m.cb<tensor<4x4x!ttcore.tile<...>>>, ...):
      %t0 = d2m.wait %cb0 : ...
      %t1 = d2m.reserve %cb1 : ...
      %result = linalg.generic ins(%t0 : ...) outs(%t1 : ...) {
      ^bb0(%in: !ttcore.tile<...>, %out: !ttcore.tile<...>):
        %scalar = arith.constant 2.5 : f32
        %add = d2m.tile_add(%in, %scalar) : (!ttcore.tile<...>, f32) -> !ttcore.tile<...>
        linalg.yield %add : !ttcore.tile<...>
      } -> ...
    }
    ```

    This pass should run early in the D2M pipeline, before memory allocation.
  }];
  let dependentDialects = ["::mlir::tt::d2m::D2MDialect", "::mlir::arith::ArithDialect"];
}

def D2MDecomposeComplexPermute: Pass<"d2m-decompose-complex-permute", "::mlir::ModuleOp"> {
  let summary = "Decompose complex permute operations into inner/outer permutes.";
  let description = [{
    This pass decomposes complex permute operations into a sequence of inner and outer
    permutes. An inner permute swaps only the last two dimensions, while an outer permute
    handles any other permutation on the outer dimensions.

    The permute will at most be decomposed into an Outer -> Inner -> Outer permute.

    Example:
      Before:
        %0 = "ttir.permute"(%input, %output) {permutation = array<i64: 0, 3, 1, 2>}
            : (tensor<2x3x4x5xf32>, tensor<2x5x3x4xf32>) -> tensor<2x5x3x4xf32>

      After:
        // First: inner permute (swap last two dimensions)
        %1 = "ttir.permute"(%input, %tmp1) {permutation = array<i64: 0, 1, 3, 2>}
            : (tensor<2x3x4x5xf32>, tensor<2x3x5x4xf32>) -> tensor<2x3x5x4xf32>
        // Then: outer permute
        %2 = "ttir.permute"(%1, %output) {permutation = array<i64: 0, 3, 1, 2>}
            : (tensor<2x3x5x4xf32>, tensor<2x5x3x4xf32>) -> tensor<2x5x3x4xf32>
  }];
}

def D2MRankNormalization: Pass<"d2m-rank-normalization", "::mlir::ModuleOp"> {
  let summary = "Normalize tensor ranks by promoting low-rank tensors to 2D.";
  let description = [{
    This pass destructively rewrites tensor ranks to ensure all tensors have at
    least rank 2. This is needed because the device compute kernels work with
    2D+ tensors.

    The pass directly modifies types in place - no reshape operations are inserted.
    This is safe because adding 1's to the outer dimensions doesn't change the
    memory layout in any way; a scalar is packed the same way regardless of rank.

    The pass performs the following transformations:

    1. Function argument types are promoted: tensor<N> -> tensor<1xN>,
       tensor<f32> -> tensor<1x1xf32>

    2. All operation result types are promoted similarly.

    3. The function signature is updated to match the new types.

    Example 1 - Scalar tensor:
      Before:
        func.func @main(%a: tensor<f32>) -> tensor<f32> {
          %0 = "ttir.exp"(%a) : (tensor<f32>) -> tensor<f32>
          return %0 : tensor<f32>
        }

      After:
        func.func @main(%a: tensor<1x1xf32>) -> tensor<1x1xf32> {
          %0 = "ttir.exp"(%a) : (tensor<1x1xf32>) -> tensor<1x1xf32>
          return %0 : tensor<1x1xf32>
        }

    Example 2 - 1D tensor:
      Before:
        func.func @forward(%arg0: tensor<32xf32>) -> tensor<32xf32> {
          %0 = "ttir.abs"(%arg0) : (tensor<32xf32>) -> tensor<32xf32>
          return %0 : tensor<32xf32>
        }

      After:
        func.func @forward(%arg0: tensor<1x32xf32>) -> tensor<1x32xf32> {
          %0 = "ttir.abs"(%arg0) : (tensor<1x32xf32>) -> tensor<1x32xf32>
          return %0 : tensor<1x32xf32>
        }

    Example 3 - Chained ops:
      Before:
        func.func @forward(%arg0: tensor<128xf32>) -> tensor<128xf32> {
          %0 = "ttir.abs"(%arg0) : (tensor<128xf32>) -> tensor<128xf32>
          %1 = "ttir.neg"(%0) : (tensor<128xf32>) -> tensor<128xf32>
          return %1 : tensor<128xf32>
        }

      After:
        func.func @forward(%arg0: tensor<1x128xf32>) -> tensor<1x128xf32> {
          %0 = "ttir.abs"(%arg0) : (tensor<1x128xf32>) -> tensor<1x128xf32>
          %1 = "ttir.neg"(%0) : (tensor<1x128xf32>) -> tensor<1x128xf32>
          return %1 : tensor<1x128xf32>
        }

    This pass runs in isolation and requires no changes to existing code.
    It should run early in the D2M pipeline, before layout lowering.
  }];
  let dependentDialects = ["::mlir::tt::d2m::D2MDialect"];
}

def D2MGenerateOuterLoops : Pass<"d2m-generate-outer-loops", "::mlir::ModuleOp"> {
  let summary = "Generate outer loops.";
  let description = [{
    This pass generates outer loops for generic operations.
  }];
  let dependentDialects = ["::mlir::tt::d2m::D2MDialect"];
}

def D2MLowerMulticastLoads : Pass<"d2m-lower-multicast-loads", "::mlir::ModuleOp"> {
  let summary = "Lower high-level multicast remote loads to low-level form.";
  let description = [{
    This pass converts RemoteLoadOp operations from high-level multicast form
    (using mcast[dims]) to low-level multicast form (using mcore[...] mshape[...]).

    The high-level form specifies which grid dimensions participate in multicast
    as dimension indices. The low-level form provides explicit core start coordinates
    and multicast shape values for each grid dimension.

    Example:
    ```mlir
    // Before (high-level form):
    d2m.remote_load %memref[%i, %j] mcast[%c0] : ...

    // After (low-level form, assuming grid shape [4, 8]):
    d2m.remote_load %memref[%i, %j] mcore[%c1, %core_x] mshape[%c3, %c1] : ...
    ```
  }];
  let dependentDialects = ["::mlir::tt::d2m::D2MDialect", "::mlir::arith::ArithDialect"];
}

def D2MSplitUnifiedThread : Pass<"d2m-split-unified-thread", "::mlir::ModuleOp"> {
  let summary = "Split unified thread regions.";
  let description = [{
    This pass splits unified thread regions into separate regions.
  }];
  let dependentDialects = ["::mlir::tt::d2m::D2MDialect"];
}

def D2MPreallocateMcastSemaphores : Pass<"d2m-preallocate-mcast-semaphores", "::mlir::ModuleOp"> {
  let summary = "Pre-allocate semaphores for multicast DMA operations.";
  let description = [{
    This pass pre-allocates semaphores for RemoteLoadOp operations that will
    require multicast synchronization. It runs before d2m-schedule-dma to ensure
    that each multicast operation has its own dedicated semaphores that are
    correctly mapped across all regions of the generic op.

    The pass identifies RemoteLoadOps that have multicast parameters (mcastStartIndex
    and mcastShape) and creates two semaphore block arguments for each:
    - receiversReadySemaphore: used by receivers to signal they are ready
    - senderFinishedSemaphore: used by sender to signal completion

    The indices of these semaphores are stored as attributes on the RemoteLoadOp
    so that downstream passes (d2m-schedule-dma and d2m-lower-load-store-ops-to-dma)
    can correctly use them even after region cloning/splitting.

    Example:
    ```mlir
    // Before:
    d2m.generic {threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]} {
    ^dm0(%cb0: !d2m.cb<...>, %cb1: !d2m.cb<...>):
      d2m.remote_load %cb0, %memref[%i] core[%c0] mcast[%c4] : ...
    }, { ... }

    // After (semaphores pre-allocated):
    d2m.generic {threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]} {
    ^dm0(%cb0: !d2m.cb<...>, %cb1: !d2m.cb<...>, %sem0: !d2m.semaphore, %sem1: !d2m.semaphore):
      d2m.remote_load %cb0, %memref[%i] core[%c0] mcast[%c4]
        {preallocated_semaphores = [2, 3]} : ...
    }, {
    ^compute0(%cb0: !d2m.cb<...>, %cb1: !d2m.cb<...>, %sem0: !d2m.semaphore, %sem1: !d2m.semaphore):
      ...
    }
    ```
  }];
  let dependentDialects = ["::mlir::tt::d2m::D2MDialect"];
}

def D2MScheduleDMA : Pass<"d2m-schedule-dma", "::mlir::ModuleOp"> {
  let summary = "Schedule DMA operations across multiple hardware threads.";
  let description = [{
    This pass takes a single datamovement region from a d2m.generic op and splits
    it into N separate datamovement regions, distributing remote_load and remote_store
    operations across hardware datamovement threads for balanced execution.

    The pass runs after d2m-split-unified-thread which produces a generic op with
    a single datamovement region and a compute region. This pass analyzes the
    remote_load and remote_store operations in the datamovement region and
    distributes them across available hardware datamovement threads (typically 2
    on Wormhole).

    The scheduling strategy aims to balance the workload across threads by:
    1. Counting the number of DMA operations per operand/CB
    2. Assigning each operand's DMA operations to a thread to balance total work
    3. Creating separate regions for each thread's assigned work

    Example transformation:
    ```mlir
    // Before (single datamovement region with 2 remote_loads):
    d2m.generic {threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%stream0, %stream1 : ...) outs(%alloc : ...) {
    ^datamovement0(%cb0, %cb1, %cb2):
      scf.for ... {
        d2m.remote_load %cb0, %stream0[...]  // Load for operand 0
        d2m.remote_load %cb1, %stream1[...]  // Load for operand 1
      }
    }, {
    ^compute0(...):
      // compute ops
    }

    // After (split into 2 datamovement regions):
    d2m.generic {threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%stream0, %stream1 : ...) outs(%alloc : ...) {
    ^datamovement0(%cb0, %cb1, %cb2):
      scf.for ... {
        d2m.remote_load %cb0, %stream0[...]  // Thread 0 handles operand 0
      }
    }, {
    ^datamovement1(%cb0, %cb1, %cb2):
      scf.for ... {
        d2m.remote_load %cb1, %stream1[...]  // Thread 1 handles operand 1
      }
    }, {
    ^compute0(...):
      // compute ops
    }
    ```

    If there are more DMA operands than available threads, operations are grouped
    to balance work. If there are fewer operands than threads, some threads may
    be empty (and can be eliminated by later passes).
  }];
  let dependentDialects = ["::mlir::tt::d2m::D2MDialect"];
}

def D2MLowerLoadStoreOpsToDMA : Pass<"d2m-lower-load-store-ops-to-dma", "::mlir::ModuleOp"> {
  let summary = "Lower D2M load/store operations to DMA operations.";
  let description = [{
    This pass performs lowering of D2M load/store operations to DMA operations.
  }];
  let dependentDialects = ["::mlir::tt::d2m::D2MDialect"];
  let options = [
    Option<"debugCoalescingInference", "debug-d2m-coalescing-inference", "bool",
           /*default=*/"false",
           "Enable debug mode for coalescing inference. Runs both analytical and "
           "sampling-based coalescing checks and prints debug output comparing them.">,
  ];
}

def D2MLowerLoadStoreOpsToExplicitCBForm : Pass<"d2m-lower-load-store-ops-to-explicit-cb-form", "::mlir::ModuleOp"> {
  let summary = "Lower load/store operations to explicit CB form.";
  let description = [{
    This pass converts remote_load and remote_store operations from implicit form
    to explicit CB form.
  }];
  let dependentDialects = ["::mlir::tt::d2m::D2MDialect"];
}

def D2MConvertLocalLoadStoreOpsToAliasedCBs : Pass<"d2m-convert-local-load-store-ops-to-aliased-cbs", "::mlir::ModuleOp"> {
  let summary = "Convert local load/store operations to aliased circular buffers.";
  let description = [{
    This pass converts local load/store operations to aliased circular buffers.
  }];
  let dependentDialects = ["::mlir::tt::d2m::D2MDialect"];
}

def D2MLowerScratchAllocate : Pass<"d2m-lower-scratch-allocate", "::mlir::ModuleOp"> {
  let summary = "Lower d2m.scratch_allocate ops to memref.view into scratch buffer.";
  let description = [{
    This pass lowers `d2m.scratch_allocate` operations to `memref.view`
    operations into the scratch buffer provided by `d2m.scratch_init`.

    The pass performs the following steps:
    1. Find `d2m.scratch_init` to get the pre-allocated scratch byte buffer
    2. Collect all `d2m.scratch_allocate` ops in the function
    3. Compute byte offsets for each allocation (sequential, no overlap)
    4. Validate total size fits within the scratch buffer
    5. Replace each `d2m.scratch_allocate` with `memref.view` at computed offset
    6. Erase the `d2m.scratch_init` op

    Example transformation:
    ```mlir
    // Before:
    func.func @kernel(...) {
      %buf = memref.alloc() : memref<65536xi8, #l1>
      d2m.scratch_init %buf : memref<65536xi8, #l1>
      %s0 = d2m.scratch_allocate {slot = 0} : memref<8x!tile_bf16, #l1>
      %s1 = d2m.scratch_allocate {slot = 1} : memref<8x!tile_bf16, #l1>
      // ...
    }

    // After:
    func.func @kernel(...) {
      %buf = memref.alloc() : memref<65536xi8, #l1>
      %c0 = arith.constant 0 : index
      %s0 = memref.view %buf[%c0][] : memref<65536xi8, #l1> to memref<8x!tile_bf16, #l1>
      %c16384 = arith.constant 16384 : index
      %s1 = memref.view %buf[%c16384][] : memref<65536xi8, #l1> to memref<8x!tile_bf16, #l1>
      // ...
    }
    ```
  }];
  let dependentDialects = [
    "::mlir::tt::d2m::D2MDialect",
    "::mlir::tt::ttcore::TTCoreDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect"
  ];
  let options = [
    Option<"testInsertDummyScratch",
          "test-insert-dummy-scratch",
          "bool", /*default=*/"true",
          "For testing: insert a dummy d2m.scratch_allocate if scratch_init "
          "is present but no scratch_allocate ops exist. Note: the dummy scratch "
          "is in the main function, not a kernel, so it won't exercise the full "
          "D2MToTTKernel lowering path.">,
  ];
}

#endif
