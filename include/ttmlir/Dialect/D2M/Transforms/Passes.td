// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_D2M_D2MPASSES_TD
#define TTMLIR_TTMLIR_DIALECT_D2M_D2MPASSES_TD

include "mlir/Pass/PassBase.td"

// Insert scratch buffers to d2m.generic ops (post-bufferization)
def D2MInsertScratchBuffers : Pass<"d2m-insert-scratch-buffers", "::mlir::ModuleOp"> {
  let summary = "Insert scratch buffers inside fused d2m.generic ops.";
  let description = [{
    This pass inserts a scratch buffer inside fused `d2m.generic` operations
    that may need to spill intermediate results. A generic is considered to
    need scratch when its compute region contains more than one `linalg.generic`
    op (indicating elementwise fusion).

    The scratch buffer is a 128KB L1 buffer whose tile type matches the first
    tiled input operand. Its shard shape is `[1, numTiles]`, where
    `numTiles = 128KB / tile_size_bytes`.

    This pass runs after bufferization (in memref space) and before the
    D2MAllocate pass. It inserts a `memref.alloc` with `CBLayoutAttr`
    and a `d2m.scratch_init` anchor op at the start of the generic's region body.

    Example transformation:
    ```mlir
    // Before:
    d2m.generic { ... }
    ins(%A, %B : memref<...>, memref<...>)
    outs(%C : memref<...>) {
    ^bb0:
      // ...compute body...
    }

    // After:
    d2m.generic { ... }
    ins(%A, %B : memref<...>, memref<...>)
    outs(%C : memref<...>) {
    ^bb0:
      %scratch = memref.alloc() : memref<1x32x!tile_f32, #l1>
      d2m.scratch_init %scratch : memref<1x32x!tile_f32, #l1>
      // ...compute body...
    }
    ```
  }];

  let dependentDialects = [
    "::mlir::tt::d2m::D2MDialect",
    "::mlir::tt::ttcore::TTCoreDialect",
    "::mlir::memref::MemRefDialect"
  ];
}

// Fusion of d2m.generic ops on tensors (elementwise, eltwise->reduction).
def D2MGenericFusion : Pass<"d2m-generic-fusion", "::mlir::ModuleOp"> {
  let summary = "Fuse compute-only d2m.generic ops (elementwise and eltwise->reduction)";
  let description = [{
    Performs fusion for `d2m.generic` producers into consumers when safe to do
    so. The initial implementation targets compute-only, affine-map form
    `d2m.generic` with tensor semantics and matching device attributes (grid,
    threads, block_factors). Elementwise->elementwise fusion is always enabled;
    elementwise->reduction fusion (single reduction dim) is opt-in via
    `enable-eltwise-reduction-fusion`. The pass is intended to run
    pre-bufferization to reduce intermediary tensors and improve locality.
  }];

  list<Option> options = [
    Option<"enableEltwiseReductionFusion", "enable-eltwise-reduction-fusion", "bool", "false",
           "Enable fusion of an elementwise d2m.generic producer into a reduction d2m.generic consumer (single reduction dim).">,
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
    to the root loop for use by downstream D2MInsertDstRegisterAccessScheduled pass.
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
    This pass tiles linalg.generic compute loops according to fusion and DST register constraints.
  }];

  list<Option> options = [
    Option<"maxDstPhysicalSizeTiles", "max-dst-physical-size-tiles", "unsigned", "0",
           "Override the max DST physical size in tiles, upper-bounding the numTilesPerFlip for any op. 0 means use the device descriptor value.">,
  ];

  let dependentDialects = [
    "mlir::tt::d2m::D2MDialect",
    "::mlir::affine::AffineDialect",
    "::mlir::scf::SCFDialect"
  ];
}

def D2MInsertSpillAndScratch : Pass<"d2m-insert-spill-and-scratch", "::mlir::ModuleOp"> {
  let summary = "Replace intermediate allocations with scratch buffers for fusion.";
  let description = [{
    This pass transforms compute regions with multiple `d2m.scratch_space_loop` loop
    nests to use scratch buffers for intermediate results.
    This enables efficient pipelining by keeping only a few iterations
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

def D2MLinalgToAffine : Pass<"d2m-linalg-to-affine", "::mlir::ModuleOp"> {
  let summary = "Convert linalg.generic operations to affine loops";
  let description = [{
    This pass converts all linalg.generic operations within d2m.generic
    compute regions to affine loop nests. This is a prerequisite for
    the DST register allocation passes which operates on affine IR.

    The pass handles:
    - Regular linalg.generic ops -> affine for loops
    - Tile matmul linalg.generic ops -> affine for loops with tile_matmul

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
    Option<"markRootLoops", "mark-root-loops", "bool", "true",
           "Mark root loops with d2m.linalg_root attribute for subsequent passes">
  ];
}

def D2MInsertDstRegisterAccessUnscheduled : Pass<"d2m-insert-dst-register-access-unscheduled", "::mlir::ModuleOp"> {
  let summary = "Insert DST register access for non-scheduled linalg_root loops.";
  let description = [{
    Handles `d2m.linalg_root` loops that do **not** carry `d2m.scheduled`
    (matmul / reduction lowering): bump allocator, cloned affine copy nests,
    optional L1 accumulation for tile matmul.

    Run **before** `d2m-insert-dst-register-access-scheduled` when both kinds
    of roots appear in the same module.  This pass does **not** handle the
    no-`linalg_root` fallback (canonicalized-away root loops); that is owned
    by the scheduled pass.

    Precondition: `--d2m-linalg-to-affine` must have been run beforehand
    (the pass calls `verifyInsertDstRegisterAccessPreconditions`, which walks
    the module and fails if any `linalg.generic` is still present in the
    compute regions).

    Example, this pass will convert the following code:
    ```mlir
    ^compute0(%cb0: memref<3x3x!tt.tile<32x32, f32>, #l1>, %cb1: memref<3x2x!tt.tile<32x32, f32>, #l1>, %cb2: memref<3x2x!tt.tile<32x32, f32>, #l1>):
      affine.for %i2 = 0 to 3 {
        affine.for %i3 = 0 to 2 {
          affine.for %i4 = 0 to 3 {
            %0 = affine.load %cb0[%i2, %i4] : memref<3x3x!tt.tile<32x32, f32>, #l1>
            %1 = affine.load %cb1[%i4, %i3] : memref<3x2x!tt.tile<32x32, f32>, #l1>
            %2 = affine.load %cb2[%i2, %i3] : memref<3x2x!tt.tile<32x32, f32>, #l1>
            %3 = "d2m.tile_matmul"(%0, %1, %2) : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
            affine.store %3, %cb2[%i2, %i3] : memref<3x2x!tt.tile<32x32, f32>, #l1>
          }
        }
      } {d2m.linalg_root}
    ```

    Into (3 separate loop nests: CB->DST loads, compute, DST->CB stores):
    ```mlir
    ^compute0(%cb0: memref<3x3x!tt.tile<32x32, f32>, #l1>, %cb1: memref<3x2x!tt.tile<32x32, f32>, #l1>, %cb2: memref<3x2x!tt.tile<32x32, f32>, #l1>):
      %dst = d2m.acquire_dst() : memref<3x2x!tt.tile<32x32, f32>, #dst>
      affine.for %i2 = 0 to 3 {
        affine.for %i3 = 0 to 2 {
          %0 = affine.load %cb2[%i2, %i3] : memref<3x2x!tt.tile<32x32, f32>, #l1>
          affine.store %0, %dst[%i2, %i3] : memref<3x2x!tt.tile<32x32, f32>, #dst>
        }
      }
      affine.for %i2 = 0 to 3 {
        affine.for %i3 = 0 to 2 {
          affine.for %i4 = 0 to 3 {
            %0 = affine.load %cb0[%i2, %i4] : memref<3x3x!tt.tile<32x32, f32>, #l1>
            %1 = affine.load %cb1[%i4, %i3] : memref<3x2x!tt.tile<32x32, f32>, #l1>
            %2 = affine.load %dst[%i2, %i3] : memref<3x2x!tt.tile<32x32, f32>, #dst>
            %3 = "d2m.tile_matmul"(%0, %1, %2) : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
            affine.store %3, %dst[%i2, %i3] : memref<3x2x!tt.tile<32x32, f32>, #dst>
          }
        }
      }
      affine.for %i2 = 0 to 3 {
        affine.for %i3 = 0 to 2 {
          %0 = affine.load %dst[%i2, %i3] : memref<3x2x!tt.tile<32x32, f32>, #dst>
          affine.store %0, %cb2[%i2, %i3] : memref<3x2x!tt.tile<32x32, f32>, #l1>
        }
      }
    ```
  }];
  let dependentDialects = [
    "::mlir::tt::d2m::D2MDialect",
    "::mlir::affine::AffineDialect",
    "::mlir::func::FuncDialect",
    "::mlir::scf::SCFDialect"
  ];
  let options = [
    Option<"maxDstPhysicalSizeTiles", "max-dst-physical-size-tiles", "unsigned", "0", "Override the max dst physical size tiles or 0 if unset.">,
    Option<"disableL1Acc", "disable-l1-acc", "bool", /*default=*/"false", "Disable accumulating partials in L1 (force reloading from L1 to DST instead).">,
  ];
}

def D2MInsertDstRegisterAccessScheduled : Pass<"d2m-insert-dst-register-access-scheduled", "::mlir::ModuleOp"> {
  let summary = "Insert DST register access for scheduled linalg_root loops and fallback regions.";
  let description = [{
    Handles `d2m.linalg_root` loops that carry `d2m.scheduled` (op-scheduler
    and related passes): stack allocator, in-place CB/DST copies, `memref`
    loads/stores under `scf.for` where applicable.  Strips `d2m.scheduled`
    after processing.

    Also handles the **fallback** when a region has compute but no
    `d2m.linalg_root` loop at region entry (e.g. single-iteration loop folded
    away): treats the region as scheduled-shaped and inserts DST access for
    the flat block.

    Run **after** `d2m-insert-dst-register-access-unscheduled` when both passes
    are needed.

    Precondition: `--d2m-linalg-to-affine` must have been run beforehand
    (see the unscheduled pass for details on how the pre-condition is
    enforced).

    Example, given a `d2m.scheduled` root produced by `--d2m-op-scheduler`
    over a 3-input fused eltwise:
    ```mlir
    ^compute0(%cb0, %cb1, %cb2: memref<2x4x!tt.tile<32x32, f32>, #l1>, %cb3: memref<2x4x!tt.tile<32x32, f32>, #l1>):
      affine.for %i = 0 to 2 {
        affine.for %j = 0 to 4 {
          %0 = affine.load %cb0[%i, %j] : memref<2x4x!tt.tile<32x32, f32>, #l1>
          %1 = affine.load %cb1[%i, %j] : memref<2x4x!tt.tile<32x32, f32>, #l1>
          %2 = affine.load %cb2[%i, %j] : memref<2x4x!tt.tile<32x32, f32>, #l1>
          %3 = "d2m.tile_add"(%0, %1) : (!tt.tile, !tt.tile) -> !tt.tile
          %4 = "d2m.tile_div"(%3, %2) : (!tt.tile, !tt.tile) -> !tt.tile
          affine.store %4, %cb3[%i, %j] : memref<2x4x!tt.tile<32x32, f32>, #l1>
        }
      } {d2m.linalg_root, d2m.scheduled}
    ```

    The pass rewrites loads/stores in place against DST and emits a single
    DST->CB store nest after the compute loop:
    ```mlir
    ^compute0(...):
      %dst = d2m.acquire_dst() : memref<...x!tt.tile<32x32, f32>, #dst>
      affine.for %i = 0 to 2 {
        affine.for %j = 0 to 4 {
          %0 = affine.load %cb0[%i, %j] ... ; -> writes DST slice 0
          %1 = affine.load %cb1[%i, %j] ... ; -> writes DST slice 1
          %2 = affine.load %cb2[%i, %j] ... ; -> writes DST slice 2
          %a = affine.load %dst[...slot 0...] ; ... etc
          %b = affine.load %dst[...slot 1...]
          %3 = "d2m.tile_add"(%a, %b) -> stores into DST
          %c = affine.load %dst[...]
          %d = affine.load %dst[...slot 2...]
          %4 = "d2m.tile_div"(%c, %d) -> stores into DST
        }
      } {d2m.linalg_root}
      affine.for %i = 0 to 2 {
        affine.for %j = 0 to 4 {
          %v = affine.load %dst[...]
          affine.store %v, %cb3[%i, %j] ...
        }
      }
    ```
    Note that the `d2m.scheduled` attribute is consumed by the pass.
  }];
  let dependentDialects = [
    "::mlir::tt::d2m::D2MDialect",
    "::mlir::affine::AffineDialect",
    "::mlir::func::FuncDialect",
    "::mlir::scf::SCFDialect"
  ];
  let options = [
    Option<"maxDstPhysicalSizeTiles", "max-dst-physical-size-tiles", "unsigned", "0", "Override the max dst physical size tiles or 0 if unset.">,
    Option<"disableL1Acc", "disable-l1-acc", "bool", /*default=*/"false", "Disable accumulating partials in L1 (force reloading from L1 to DST instead).">,
  ];
}

def D2MInsertTileMatmulBlock : Pass<"d2m-insert-tile-matmul-block", "::mlir::ModuleOp"> {
  let summary = "Replace tile_matmul compute loops with tile_matmul_block.";
  let description = [{
    This pass finds affine loop nests containing tile_matmul operations
    and replaces them with a single tile_matmul_block operation that
    encapsulates the block-level matmul.

    The pass independently detects eligible matmul compute loops by
    walking the IR for tile_matmul ops, recovers the input/output CB
    memrefs by tracing operands through affine.load ops and a verified
    DST-to-L1 store copy chain (store value must come from affine.load of
    the DST memref), and verifies shape eligibility before replacement.

    When memref.cast or memref.subview has been folded into those loads or
    stores, recovery scans the block before the compute loop for cast/subview
    definitions of the underlying CB and prefers those results so strided
    memref types stay consistent for downstream passes.
  }];
  let dependentDialects = [
    "::mlir::tt::d2m::D2MDialect",
    "::mlir::affine::AffineDialect"
  ];
  let options = [
    Option<"useTileMatmul", "use-tile-matmul", "bool", /*default=*/"false",
           "Use tile_matmul instead of tile_matmul_block. When true, the pass "
           "is a no-op and tile_matmul ops are kept in their loop nests.">,
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
    host program lowering. As the common exit point for generic regions, it also
    resolves remaining virtual-grid metadata needed by kernel ops before
    outlining.

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

defvar D2MNumStreamBuffersOption = [
  Option<"numStreamBuffers",
         "num-stream-buffers",
         "unsigned", /*default=*/"2",
         "Number of backing buffers to allocate per stream storage (>=1). Default is 2.">,
];

def D2MReblockGenerics : Pass<"d2m-reblock-generics", "::mlir::ModuleOp"> {
  let summary = "Rebuild d2m.generic ops using block-factor analysis.";
  let description = [{
    This pass uses
    BlockFactorAnalysis to select block factors for affine blocked d2m.generic
    ops, then rebuilds those generics with the selected factors.
  }];
  let dependentDialects = [
    "::mlir::tt::d2m::D2MDialect",
    "::mlir::tt::ttcore::TTCoreDialect",
    "::mlir::memref::MemRefDialect"
  ];
  let options = D2MNumStreamBuffersOption # [
    Option<"testBufferSizePolicy",
          "test-buffer-size-policy",
          "std::string", /*default=*/"\"auto\"",
          "Set policy for sizing stream buffers ('auto', 'auto-mn', 'bounded', 'min', 'max').">,
  ];
}

def D2MMarkSynchronizedBuffers : Pass<"d2m-mark-synchronized-buffers", "::mlir::ModuleOp"> {
  let summary = "Mark in-generic allocs that require synchronization with d2m.synchronized_buffer attribute.";
  let description = [{
    This pass runs after bufferization and marks `memref.alloc` operations inside
    `d2m.generic` ops that are used for synchronization (i.e., circular buffers)
    with the `d2m.synchronized_buffer` attribute. The attribute value indicates
    the number of stream buffers to allocate.

    This information is used by the HoistCBAllocs pass to properly configure
    the CBLayoutAttr for these buffers.
  }];

  let dependentDialects = [
    "::mlir::tt::d2m::D2MDialect",
    "::mlir::memref::MemRefDialect"
  ];

  let options = D2MNumStreamBuffersOption;
}

def D2MAllocate: Pass<"d2m-allocate", "::mlir::ModuleOp"> {
  let summary = "Create streams required by generic ops.";
  let description = [{
    This pass handles several related tasks:
      - replacing in-generic allocs with CBLayoutAttr-tagged allocs for operands
        that require streaming (circular buffer allocation);
      - allocating memory addresses/alignments for buffers as well as
        other memref allocs needed by data movement/compute kernel functions;
      - inserting dealloc instructions at the appropriate end-of-scope positions.

    Converts:
    ```mlir
    %alloc = memref.alloc() : memref<6x6x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %alloc_0 = memref.alloc() : memref<6x6x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %alloc_1 = memref.alloc() : memref<6x6x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    d2m.generic {block_factors = [1, 1, 6], grid = #ttcore.grid<6x6>, indexing_maps = [#map, #map1, #map2], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<compute>]}
        ins(%alloc, %alloc_0 : memref<...>, memref<...>)
        outs(%alloc_1 : memref<...>)  {
      %cb0 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %cb1 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %cb2 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      "d2m.tile_matmul_block"(%cb0, %cb1, %cb2) : (...) -> ()
    }
    ```

    Into:
    ```mlir
    %alloc = memref.alloc() {address = 5120 : i64, alignment = 32 : i64} : memref<6x6x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>
    %alloc_0 = memref.alloc() {address = 1024 : i64, alignment = 32 : i64} : memref<6x6x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>
    %alloc_1 = memref.alloc() {address = 17408 : i64, alignment = 16 : i64} : memref<6x6x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    d2m.generic {block_factors = [1, 1, 6], grid = #ttcore.grid<6x6>, indexing_maps = [#map, #map1, #map2], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<compute>]}
        ins(%alloc, %alloc_0 : memref<...>, memref<...>)
        outs(%alloc_1 : memref<...>)  {
      %cb0 = memref.alloc() {address = 9216 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2, grid = [6x6]>, #l1>
      %cb1 = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2, grid = [6x6]>, #l1>
      %cb2 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      "d2m.tile_matmul_block"(%cb0, %cb1, %cb2) : (...) -> ()
    }
    memref.dealloc %alloc_0 : memref<...>
    memref.dealloc %alloc : memref<...>
    ```
  }];
  let dependentDialects = ["::mlir::tt::ttcore::TTCoreDialect", "::mlir::memref::MemRefDialect"];
  let options = D2MNumStreamBuffersOption # [
    Option<"allowL1OutputSpilling",
          "allow-l1-output-spilling",
          "bool", /*default=*/"false",
          "Make generic outputs eligible for spilling from L1 to DRAM.">,
    Option<"streamInsertPolicy",
          "stream-insert-policy",
          "std::string", /*default=*/"\"infer\"",
          "Policy for deciding when to insert operand streams ('always', 'infer').">,
    Option<"forceSpillToDramIfLegal",
          "force-spill-to-dram-if-legal",
          "bool", /*default=*/"false",
          "Force all non-bound allocator variables to spill into DRAM. Variables bound to their incoming memspace are left unchanged.">,
    ListOption<"availableL1AddrRange",
          "available-l1-addr-range",
          "std::int64_t",
          "Assume given L1 addressable range [base, max).">,
    Option<"testAssumeL1Capacity",
          "test-assume-l1-capacity",
          "std::int64_t", /*default=*/"0",
          "Assume given L1 capacity.">,
  ];
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
    Option<"ttnnMode", "ttnn-mode", "bool", /*default=*/"false",
        "Enable TTNN mode which hardcodes collapsed intervals and dim alignments.">,
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
    This pass detects when a return value is an unmaterialized tensor or memref view
    (produced by view operations like d2m.view_layout)
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

    This pass can run before or after bufferization. Tensor views are
    materialized with tensor outputs and memref views are materialized with a
    device allocation plus datamovement-only generic.
  }];
  let dependentDialects = ["::mlir::tt::d2m::D2MDialect", "::mlir::tt::ttcore::TTCoreDialect"];
}

def D2MOptimizeMasks : Pass<"d2m-optimize-masks", "::mlir::ModuleOp"> {
  let summary = "Eliminate redundant high-level mask ops";
  let description = [{
    This pass walks the IR in program order and tracks the known effective OOB
    state for values produced by `d2m.mask`.  A later mask is removed when its
    input is already known to satisfy the requested fill value, or when the
    tensor has no maskable padded region.

    Unknown/untracked values are treated as undef for compatibility purposes:
    they satisfy an undef target, but not a concrete fill requirement.  The pass
    is intentionally conservative and is the place where richer arithmetic OOB
    propagation can be added.

    This pass should run after mask creation/canonicalization and before
    `d2m-decompose-masking`.
  }];
  let dependentDialects = [
    "::mlir::tt::d2m::D2MDialect",
    "::mlir::tt::ttcore::TTCoreDialect"
  ];
}

def D2MDecomposeMasking : Pass<"d2m-decompose-masking", "::mlir::ModuleOp"> {
  let summary = "Decompose mask ops into explicit per-tile masking";
  let description = [{
    This pass decomposes high-level `d2m.mask` operations into an explicit
    `d2m.generic` that loads one shard, writes row/column mask tiles in L1,
    and emits tile loops for interior, edge, corner, and fully OOB regions.

    The decomposition uses row/column mask tiles with `d2m.tile_where`, rather
    than multiply-by-zero blending, so infinite fill values do not depend on
    target-specific zero-times-infinity behavior.

    Example transformation:
    ```mlir
    // Before:
    %masked = d2m.mask %input, %output
        logical_shape = [50, 50] fill_value = <zero>
        : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #layout, #l1>
          into memref<1x1x2x2x!ttcore.tile<32x32, f32>, #layout, #l1>
          -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #layout, #l1>

    // After (schematic):
    d2m.generic {grid = #ttcore.grid<1x1>, threads = [#d2m.thread<unified>]}
        ins(%input : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #layout, #l1>)
        outs(%output : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #layout, #l1>) {
      %localInput = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1>
      %localOutput = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1>
      %rowMask = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #cb, #l1>
      %colMask = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #cb, #l1>
      d2m.remote_load %localInput, %input[%coreY, %coreX] : ...
      d2m.write_row_mask_tile(%validRows) to %rowMask : ...
      d2m.write_col_mask_tile(%validCols) to %colMask : ...
      scf.for %row = %rowStart to %rowEnd {
        scf.for %col = %colStart to %colEnd {
          %tile = memref.load %localInput[%row, %col] : ...
          %mask = memref.load %rowMask[%c0, %c0] : ...
          %fill = d2m.tile_fill %fillScalar : ...
          %result = d2m.tile_where %mask, %tile, %fill : ...
          memref.store %result, %localOutput[%row, %col] : ...
        }
      }
      d2m.remote_store %output[%coreY, %coreX], %localOutput : ...
    }
    ```

    This pass should run after bufferization/generic loop generation and before
    allocation.
  }];
  let dependentDialects = [
    "::mlir::tt::d2m::D2MDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect"
  ];
  let options = D2MNumStreamBuffersOption;
}

def D2MDecomposeArange : Pass<"d2m-decompose-arange", "::mlir::ModuleOp"> {
  let summary = "Decompose arange_block ops into low-level tile operations";
  let description = [{
    This pass decomposes high-level `arange_block` operations into low-level
    tile operations for generating arange values.

    The arange_block op operates at block level (memref of tiles) and generates
    values: output[i] = start + step * i, where i is the linear index.

    The decomposition:
    1. Write full linear index tile to scratch CB using FillArangeTileOp
    2. Loop over output tiles:
       a. Compute tile_offset = (tile_row * num_tile_cols * 32 + tile_col) * 32
       b. Load scratch tile for local_index
       c. Compute global_index = tile_add(tile_offset + local_index)
       d. Compute result = start + step * global_index using tiled ops
       e. Store result tile to output

    Example transformation:
    ```mlir
    // Before:
    d2m.arange_block (%full_index_cb, %output) {start = 32, step = 2}
        : (memref<1x1x!ttcore.tile<32x32, f32>>, memref<1x1x!ttcore.tile<32x32, f32>>)

    // After:
    d2m::fill_arange_tile to %full_index_alloc
    scf.for %i = 0 to %num_tile_rows {
      scf.for %j = 0 to %num_tile_cols {
        %tile_offset = ... // (i * num_tile_cols + j) * 1024
        %local_idx = memref.load %full_index_alloc[0, 0]
        %global_idx = d2m.tile_add %offset_tile, %local_idx
        %scaled = d2m.tile_mul %global_idx, step_scalar (in this case 2)
        %result = d2m.tile_add %scaled, start_scalar (in this case 32)
        memref.store %result, %output[%i, %j]
      }
    }
    ```

    This pass should run after bufferization and before LinalgToAffine.
  }];
  let dependentDialects = [
    "::mlir::tt::d2m::D2MDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect"
  ];
}

def D2MDecomposeTopk : Pass<"d2m-decompose-topk", "::mlir::ModuleOp"> {
  let summary = "Decompose topk_block ops into low-level tile operations";
  let description = [{
    This pass decomposes high-level `topk_block` operations into low-level
    tile operations for computing top-K values and indices.

    The topk_block op operates at block level (memref of tiles) and wraps
    the full topk computation: index generation, bitonic sort, merge, and
    rebuild.

    The decomposition:
    1. Allocate an index scratch buffer (buf_idx)
    2. Generate initial indices via arange_block
    3. Bitonic sort each tile independently via tile_topk_local_sort
    4. Merge sorted tile pairs to find top-K via tile_topk_merge
    5. Rebuild final sorted order via tile_topk_rebuild

    Example transformation:
    ```mlir
    // Before:
    %vals, %idx = d2m.topk_block(%input, %scratch, %out_vals, %out_idx)
        {k = 16, num_elements = 64, stable_sort = false}
        : (memref<1x2x!ttcore.tile<32x32, f32>>,
           memref<1x1x!ttcore.tile<32x32, si32>>,
           memref<1x2x!ttcore.tile<32x32, f32>>,
           memref<1x2x!ttcore.tile<32x32, si32>>)
        -> (memref<1x2x!ttcore.tile<32x32, f32>>,
            memref<1x2x!ttcore.tile<32x32, si32>>)

    // After:
    %buf_idx = memref.alloc() : memref<1x2x!ttcore.tile<32x32, si32>, #l1>
    %buf_idx_filled = d2m.arange_block(%scratch, %buf_idx) ...
    %sorted_vals = memref.alloc() ...
    %sorted_idx = memref.alloc() ...
    d2m.tile_topk_local_sort(%input, %buf_idx_filled, %sorted_vals, %sorted_idx) ...
    %merged_vals = memref.alloc() ...
    %merged_idx = memref.alloc() ...
    d2m.tile_topk_merge(%sorted_vals, %sorted_idx, %merged_vals, %merged_idx) ...
    d2m.tile_topk_rebuild(%merged_vals, %merged_idx, %out_vals, %out_idx) ...
    ```

    This pass should run after bufferization and before
    MarkSynchronizedBuffers.
  }];
  let dependentDialects = [
    "::mlir::tt::d2m::D2MDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect"
  ];
}

def D2MScalarizeConstTensors : Pass<"d2m-scalarize-const-tensors", "::mlir::ModuleOp"> {
  let summary = "Scalarize uses of splat fills produced inside d2m.generic (tile_fill pattern)";
  let description = [{
    This pass matches **producer** `d2m.generic` operations whose region contains the
    splat-fill shape produced by lowering `ttir.full`: a single inputless
    `linalg.generic` whose body is `arith.constant` + `d2m.tile_fill` (+ yield).
    It reads the splat value from that pattern, follows the producer result through
    `d2m.to_layout`, `d2m.view_layout`, and `ttir.ttnn_metal_layout_cast` to **consumer**
    `d2m.generic` regions, and rewrites inner `linalg.generic` ops so operands that
    were loaded from the splat tensor become `arith.constant` scalars where the tile op
    implements `supportsTileOrScalarRhs()`.

    It then drops now-unused producer inputs from the outer `d2m.generic` and removes
    redundant `d2m.remote_load` chains where applicable.

    This transformation can improve performance by:
    1. Reducing memory usage (no need to materialize the constant tensor for those uses)
    2. Enabling more efficient scalar operations in the compute kernel

    Example transformation:
    ```mlir
    // Before: splat tensor flows from a fill generic (e.g. after `ttir.full` -> D2M)
    // into a compute generic; %fill below is the fill producer's result after layout.
    %fill = d2m.generic ... { ... linalg.generic ... d2m.tile_fill ... } : ...
    %const_layout = d2m.to_layout %fill, ...
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

def D2MGenerateOuterLoops : Pass<"d2m-generate-outer-loops", "::mlir::ModuleOp"> {
  let summary = "Generate outer loops.";
  let description = [{
    This pass generates outer loops for generic operations.
  }];
  let dependentDialects = ["::mlir::tt::d2m::D2MDialect",
                           "::mlir::affine::AffineDialect",
                           "::mlir::arith::ArithDialect"];
}

def D2MLowerToExplicitForm : Pass<"d2m-generic-lower-to-explicit-form", "::mlir::ModuleOp"> {
  let summary = "Lower outer loop bounds to explicit constants.";
  let description = [{
    This pass replaces affine.for loops marked with the d2m.blocking_loop attribute
    that have bounds defined by d2m.get_block_factor ops with scf.for loops that
    have arith.constant bounds taken from the GenericOp's block_factors attribute.

    All d2m.get_block_factor ops within generic regions are also replaced with
    their corresponding arith.constant values.
  }];
  let dependentDialects = ["::mlir::tt::d2m::D2MDialect",
                           "::mlir::scf::SCFDialect",
                           "::mlir::arith::ArithDialect"];
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

def D2MAssignThreads : Pass<"d2m-assign-threads", "::mlir::ModuleOp"> {
  let summary = "Lower data-movement ops to explicit-CB form and assign each op "
                "in a unified thread region to a thread.";
  let description = [{
    First stage of unified-thread splitting. For each `#d2m.thread<unified>`
    generic region it does two things, without splitting the region:

    - Lowers the streaming data-movement ops (`remote_load`/`remote_store` and
      `local_copy`) from implicit form to explicit-CB form (introducing
      `get_cb`). Aliased remote ops carry no DMA transfer and are left in
      implicit form.
    - Annotates each effectful leaf op with a `d2m.thread` attribute naming the
      thread it will be split into. All remote/`local_copy` ops -- including
      aliased ones -- are assigned to `datamovement` (the verifier requires
      remote ops to live on the datamovement thread); compute ops are assigned
      to `compute`. Ops replicated into both threads (structural ops,
      `semaphore_wait`) are left untagged.

    The annotations are an internal handoff consumed by `d2m-split-threads`.
  }];
  let dependentDialects = ["::mlir::tt::d2m::D2MDialect"];
}

def D2MSplitThreads : Pass<"d2m-split-threads", "::mlir::ModuleOp"> {
  let summary = "Split a unified thread region into datamovement and compute.";
  let description = [{
    Second stage of unified-thread splitting. Mechanically splits a
    `#d2m.thread<unified>` generic into a 2-region generic (`datamovement`,
    `compute`) driven by the `d2m.thread` annotations from `d2m-assign-threads`:
    the region is cloned to both threads and each thread drops the ops that do
    not belong to it. The compute side is tag-driven; the datamovement side
    additionally uses dead-op elimination to remove the untagged, side-effecting
    structural ops of lowered compute nests (`memref.load`/`store` into `#dst`,
    `scf.for`, `acquire_dst`).

    Aliased remote ops stay on the datamovement thread (they have no transfer,
    but the verifier requires remote ops there); `d2m-insert-compute-cb` later
    reads them to add the matching compute-side synchronization. The `d2m.thread`
    annotations are stripped. This pass inserts no CB synchronization ops. It
    also diagnoses compute that spans multiple synchronization scopes (cross-nest
    fan-out), which the single-scope CB protocol cannot express.
  }];
  let dependentDialects = ["::mlir::tt::d2m::D2MDialect"];
}

def D2MInsertComputeCB : Pass<"d2m-insert-compute-cb", "::mlir::ModuleOp"> {
  let summary = "Insert compute-side circular-buffer synchronization ops.";
  let description = [{
    Third stage of unified-thread splitting. For each split generic it analyses
    the compute region and inserts the circular-buffer synchronization ops:
    `wait`/`pop` around the consumers of each CB and `reserve`/`push` around the
    producers, with a single pair per CB (so fan-out -- one CB consumed by
    several compute ops -- collapses to one `wait`/`pop`).

    Compute CB usage is discovered from both the high-level form (`linalg.generic`
    and `tile_*_block`, via `SynchronizableOpInterface`) and the lowered form
    that reaches this pass in the real pipeline (`scf.for` nests with
    `memref.load`/`store` into `#dst` plus tile ops), tracing memref accesses
    through `collapse_shape`/`subview` back to their CB.

    Aliased CBs have no DMA-side producer/consumer; this pass inspects the
    aliased remote markers on the datamovement thread, supplies the missing
    compute-side half (`reserve`/`push` for an aliased load, `wait`/`pop` for an
    aliased store), and then erases those markers.
  }];
  let dependentDialects = ["::mlir::tt::d2m::D2MDialect"];
}

def D2MSplitUnifiedThread : Pass<"d2m-split-unified-thread", "::mlir::ModuleOp"> {
  let summary = "Split unified thread regions into datamovement and compute.";
  let description = [{
    Umbrella pass that runs the three stages of unified-thread splitting in
    sequence: `d2m-assign-threads`, `d2m-split-threads`, and
    `d2m-insert-compute-cb`. The net effect is that each `#d2m.thread<unified>`
    generic becomes a 2-region (`datamovement`, `compute`) generic with the
    data-movement ops in explicit-CB form and the required circular-buffer
    synchronization inserted on the compute side.
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
    ^dm0(%cb0: !d2m.cb<...>, %cb1: !d2m.cb<...>, %sem0: !d2m.local_semaphore, %sem1: !d2m.local_semaphore):
      d2m.remote_load %cb0, %memref[%i] core[%c0] mcast[%c4]
        {preallocated_semaphores = [2, 3]} : ...
    }, {
    ^compute0(%cb0: !d2m.cb<...>, %cb1: !d2m.cb<...>, %sem0: !d2m.local_semaphore, %sem1: !d2m.local_semaphore):
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
    to balance work. If there are fewer operands than threads, only the required
    number of datamovement regions are emitted.
  }];
  list<Option> options = [
    Option<"numDmCores", "num-dm-cores",
           "unsigned", "0",
           "Override the available DM core count. 0 means use the system descriptor.">,
  ];
  let dependentDialects = ["::mlir::tt::d2m::D2MDialect"];
}

def D2MOptimizeDMA : Pass<"d2m-optimize-dma", "::mlir::ModuleOp"> {
  let summary = "Optimize DMA sequences by coalescing barriers and deferring write barriers.";
  let description = [{
    This pass performs DMA-level optimizations on shard-level DMA ops produced by
    D2MLowerLoadStoreOpsToDMA. It runs before D2MLowerDMAToFullyIndexedForm.

    Optimization 1 -- Barrier Coalescing:
    Reorders consecutive same-kind DMA sequences (read or write) within a block
    so that all NOC transactions are issued before any barriers fire, allowing
    the NOC to overlap multiple transfers.

    Optimization 2 -- Write Barrier Deferral (non-mcast):
    Moves write dma_wait and pop from the current loop iteration to the beginning
    of the next iteration, letting the write NOC transfer run concurrently with
    the next iteration's read address generation and NOC calls.
  }];
  let dependentDialects = [
    "::mlir::tt::d2m::D2MDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect"
  ];
}

def D2MLowerLoadStoreOpsToDMA : Pass<"d2m-lower-load-store-ops-to-dma", "::mlir::ModuleOp"> {
  let summary = "Lower D2M load/store operations to shard-level DMA operations.";
  let description = [{
    This pass lowers d2m.remote_load and d2m.remote_store operations to
    shard-level d2m.dma_read and d2m.dma_write operations with CB protocol
    (reserve/push/wait/pop). The resulting DMA ops use grid-level indices
    only. Full address indexing and coalescing is deferred
    to D2MLowerDMAToFullyIndexedForm.
  }];
  let dependentDialects = ["::mlir::tt::d2m::D2MDialect"];
}

def D2MExpandDMAReadCompositeView : Pass<"d2m-expand-dma-read-composite-view", "::mlir::ModuleOp"> {
  let summary = "Expand shard-level DMA reads that act on composite views.";
  let description = [{
    This pass expands GenericOps that contain shard-level DMA reads that act on
    composite views. The original "place holder" composite view operand is
    replaced with the list of the actual inputs. The dma read is expanded to
    an if-else chain of fully-indexed DMA reads that assembles the inputs.
  }];
  let dependentDialects = ["::mlir::tt::d2m::D2MDialect"];
}

def D2MLowerDMAToFullyIndexedForm : Pass<"d2m-lower-dma-to-fully-indexed-form", "::mlir::ModuleOp"> {
  let summary = "Lower shard-level DMA ops to fully indexed form.";
  let description = [{
    This pass expands shard-level d2m.dma_read and d2m.dma_write operations into fully-indexed form with resolved
    byte-offset indices, coalescing loops, and concrete numElems.
  }];
  let dependentDialects = ["::mlir::tt::d2m::D2MDialect"];
  let options = [
    Option<"debugCoalescingInference", "debug-d2m-coalescing-inference", "bool",
           /*default=*/"false",
           "Enable debug mode for coalescing inference. Runs both analytical and "
           "sampling-based coalescing checks and prints debug output comparing them.">,
  ];
}

def D2MHoistCBAllocs : Pass<"d2m-hoist-cb-allocs", "::mlir::ModuleOp"> {
  let summary = "Hoist in-generic CB allocs to external additionalArgs.";
  let description = [{
    This pass scans in-generic memref.alloc ops that carry a CBLayoutAttr
    (set by the Allocate pass to mark circular buffer storage).  For each such
    alloc it creates an external memref.alloc (with ShardLayoutAttr) as an
    additionalArg on the enclosing d2m.generic op.  The in-generic alloc is
    annotated with a `d2m.hoisted_operand_idx` attr pointing to the
    additionalArg's operand index so that downstream passes (getOrCreateCB)
    can wire the CB to the correct external buffer.
  }];
  let dependentDialects = [
    "::mlir::tt::d2m::D2MDialect",
    "::mlir::memref::MemRefDialect"
  ];
}

def D2MNormalizeThreadArgs : Pass<"d2m-normalize-thread-args", "::mlir::ModuleOp"> {
  let summary = "This pass normalizes thread arguments.";
  let description = [{
    This pass normalizes how thread bodies access arguments by inserting
    `d2m.get_arg` operations for thread arguments (including additionalArgs),
    while leaving `d2m.get_cb` operations in place.

    Example transformation:
    ```mlir
    // Before:
    d2m.generic {}
    ^unified0():
      %cb0 = d2m.get_cb(0) operand_index = 0 : <tensor<1x1x!ttcore.tile<32x32, f32>>>
      %cb1 = d2m.get_cb(1) operand_index = 1 : <tensor<1x1x!ttcore.tile<32x32, f32>>>

    // After:
    d2m.generic {}
    ^unified0():
      %5 = d2m.get_arg(2) resolution_stage =  compile : !d2m.global_semaphore
      %6 = d2m.get_cb(0) operand_index = 0 resolution_stage =  compile : <tensor<1x1x!ttcore.tile<32x32, f32>>>
      %7 = d2m.get_cb(1) operand_index = 1 resolution_stage =  compile : <tensor<1x1x!ttcore.tile<32x32, f32>>>
    ```
  }];
  let dependentDialects = ["::mlir::tt::d2m::D2MDialect"];
}

def D2MLowerScratchAllocate : Pass<"d2m-lower-scratch-allocate", "::mlir::ModuleOp"> {
  let summary = "Lower d2m.scratch_allocate ops to memref.subview of the scratch buffer.";
  let description = [{
    This pass lowers `d2m.scratch_allocate` operations inside `d2m.generic` ops
    into rank-reducing `memref.subview` operations of the scratch memref.

    For each `d2m.generic` that contains a `d2m.scratch_init` op, the pass:
    1. Obtains the scratch memref from the `d2m.scratch_init` op.
    2. Collects all `d2m.scratch_allocate` ops in the compute region and sorts
       them by `slot` ID for deterministic layout.
    3. Computes sequential element offsets for each allocation.
    4. Validates that total allocations fit within the scratch buffer capacity.
    5. Replaces each `d2m.scratch_allocate` with a rank-reducing
       `memref.subview` of the scratch memref at the computed offset.
    6. Erases the `d2m.scratch_init` anchor op.

    The scratch buffer has shape `[1, N]` with `CBLayoutAttr` (set by InsertScratchBuffers). Each
    `d2m.scratch_allocate` requests a 1D `memref<M x tile>`, which is lowered to
    `memref.subview [0, offset][1, M][1, 1] : memref<1xN> -> memref<M>`.

    Example transformation:
    ```mlir
    // Before:
    d2m.generic { ... }
    ins(%in : ...) outs(%out : ...) {
    ^bb0:
      %scratch = memref.alloc() : memref<1x32x!tile_f32, #l1>
      d2m.scratch_init %scratch : memref<1x32x!tile_f32, #l1>
      %s0 = d2m.scratch_allocate {slot = 0} : memref<1x!tile_f32, #l1>
      %s1 = d2m.scratch_allocate {slot = 1} : memref<2x!tile_f32, #l1>
      memref.store %val, %s0[%c0] : memref<1x!tile_f32, #l1>
      // ...
    }

    // After:
    d2m.generic { ... }
    ins(%in : ...) outs(%out : ...) {
    ^bb0:
      %scratch = memref.alloc() : memref<1x32x!tile_f32, #l1>
      %sv0 = memref.subview %scratch[0, 0][1, 1][1, 1] : ... to memref<1x!tile_f32, #l1>
      %sv1 = memref.subview %scratch[0, 1][1, 2][1, 1] : ... to memref<2x!tile_f32, strided<[1], offset: 1>, #l1>
      memref.store %val, %sv0[%c0] : memref<1x!tile_f32, #l1>
      // ...
    }
    ```
  }];
  let dependentDialects = [
    "::mlir::tt::d2m::D2MDialect",
    "::mlir::memref::MemRefDialect"
  ];
}

def D2MPythonRewrites : Pass<"d2m-python-rewrites", "::mlir::ModuleOp"> {
  let summary = "Apply Python-defined pattern rewrites via d2m-jit.";
  let description = [{
    Embeds a Python interpreter, loads each user-supplied `.py` file
    listed in `module-path` (which is expected to register one or more
    `@d2m_jit.pattern` decorators on import), and runs
    `d2m_jit.apply_patterns(module)` on the current `ModuleOp`. The
    Python rewrites mutate the same MLIR module the pass owns.

    Adding a new pattern is then a matter of dropping a new `.py` file
    in next to your driver and re-running -- no C++ rebuild required.

    Requires the build to be configured with `-DTTMLIR_ENABLE_D2M_JIT=ON`;
    otherwise the pass fails fast with a diagnostic at run time.
  }];
  let options = [
    ListOption<"modulePaths", "module-path", "std::string",
               "Path(s) to .py files registering d2m-jit patterns. May be "
               "repeated; the pass imports them in order before running.">,
  ];
  // We don't construct any dialect ops directly in C++ — Python handles
  // dialect setup against the host module's existing context. Listing
  // only D2M here so the pass infrastructure has the same expectation
  // as the rest of the suite.
  let dependentDialects = [
    "::mlir::tt::d2m::D2MDialect",
  ];
}

#endif
