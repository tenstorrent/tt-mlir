// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-grid-selection --d2m-build-topk-chain --d2m-materialize-view-returns -o %t %s
// RUN: FileCheck %s --input-file=%t
// RUN: FileCheck %s --input-file=%t --check-prefix=CONSUMED
// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-grid-selection -o %t.plan %s
// RUN: FileCheck %s --input-file=%t.plan --check-prefix=PLAN
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline -o %t.ttmetal %s

// ttir-to-d2m emits only a placeholder leaf; d2m-grid-selection chooses the
// split and folds every buffer it implies onto the leaf, and
// d2m-build-topk-chain builds exactly those ops. So every plan entry is
// consumed and no plan survives the lowering.
// The scan starts at the first function so it skips the ttcore.device
// attribute.
// CONSUMED-LABEL: func.func @topk_dim1_k16
// CONSUMED-NOT: d2m.topk_plan

module {

  // ---- dim=1 (last dim), k<=32 ----

  // 2D topk along dim 1 with k=16 on a 32x64 input (2 tiles wide).
  // The lowering must:
  //   1. Transpose tiles (topk_block sorts down tile columns; dim=1 puts the
  //      sort dim on tile rows).
  //   2. Run topk_block with correct k and num_elements.
  //   3. Extract results with tile_transpose (untranspose).
  // CHECK-LABEL: func @topk_dim1_k16
  // Every topk gets a plan, single core or not: it is the only thing that says
  // what buffers to build.
  // PLAN-LABEL: func.func @topk_dim1_k16
  // PLAN: d2m.topk_plan
  func.func @topk_dim1_k16(%arg0: tensor<32x64xf32>) -> (tensor<32x16xf32>, tensor<32x16xsi32>) {
    // The pre-transpose is a generic op wrapping tile_transpose.
    // CHECK: d2m.generic
    // CHECK: d2m.tile_transpose

    // The TopK generic op contains topk_block.
    // CHECK: d2m.generic
    // CHECK: d2m.topk_block
    // CHECK-SAME: dim = 1
    // CHECK-SAME: k = 16
    // CHECK-SAME: num_elements = 64

    // Extract values using tile_transpose to undo the pre-transpose.
    // CHECK: d2m.generic
    // CHECK: d2m.tile_transpose
    // Extract indices using tile_transpose.
    // CHECK: d2m.generic
    // CHECK: d2m.tile_transpose
    %values, %indices = "ttir.topk"(%arg0) <{k = 16 : i32, dim = -1 : i32, largest = true, sorted = false}> : (tensor<32x64xf32>) -> (tensor<32x16xf32>, tensor<32x16xsi32>)
    return %values, %indices : tensor<32x16xf32>, tensor<32x16xsi32>
  }

  // 2D topk along dim 1 with k=32 on a wider input (8 tiles wide).
  // CHECK-LABEL: func @topk_dim1_k32
  func.func @topk_dim1_k32(%arg0: tensor<32x256xf32>) -> (tensor<32x32xf32>, tensor<32x32xsi32>) {
    // CHECK: d2m.generic
    // CHECK: d2m.tile_transpose
    // CHECK: d2m.generic
    // CHECK: d2m.topk_block
    // CHECK-SAME: dim = 1
    // CHECK-SAME: k = 32
    // CHECK-SAME: num_elements = 256
    // CHECK: d2m.generic
    // CHECK: d2m.tile_transpose
    // CHECK: d2m.generic
    // CHECK: d2m.tile_transpose
    %values, %indices = "ttir.topk"(%arg0) <{k = 32 : i32, dim = -1 : i32, largest = true, sorted = false}> : (tensor<32x256xf32>) -> (tensor<32x32xf32>, tensor<32x32xsi32>)
    return %values, %indices : tensor<32x32xf32>, tensor<32x32xsi32>
  }

  // ---- dim=0, k<=32 ----

  // 2D topk along dim 0 with k=16 on a 64x32 input (2 tiles tall).
  // Nothing is transposed for dim=0: the sort dim already runs down tile
  // columns, which is the orientation topk_block wants. Extract uses
  // tile_typecast, which also converts the si32 indices to the user's type.
  // CHECK-LABEL: func @topk_dim0_k16
  func.func @topk_dim0_k16(%arg0: tensor<64x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xsi32>) {
    // The index buffer is built inside the topk kernel, so no arange runs
    // upstream and nothing is transposed.
    // CHECK-NOT: d2m.tile_transpose
    // CHECK-NOT: d2m.arange_block

    // The TopK generic op contains topk_block.
    // CHECK: d2m.generic
    // CHECK: d2m.topk_block
    // CHECK-SAME: dim = 0
    // CHECK-SAME: k = 16
    // CHECK-SAME: num_elements = 64

    // Extract values using tile_typecast (no transpose needed for dim=0).
    // CHECK: d2m.generic
    // CHECK: d2m.tile_typecast
    // Extract indices using tile_typecast.
    // CHECK: d2m.generic
    // CHECK: d2m.tile_typecast
    %values, %indices = "ttir.topk"(%arg0) <{k = 16 : i32, dim = 0 : i32, largest = true, sorted = false}> : (tensor<64x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xsi32>)
    return %values, %indices : tensor<16x32xf32>, tensor<16x32xsi32>
  }

  // ---- Large k (k>32) ----

  // k=64 spans 2 output tiles; the reduction stays one topk_block here and only
  // d2m-decompose-topk splits it into the large-k left fold.
  // CHECK-LABEL: func @topk_dim1_k64
  func.func @topk_dim1_k64(%arg0: tensor<32x256xf32>) -> (tensor<32x64xf32>, tensor<32x64xsi32>) {
    // Pre-transpose the input tiles.
    // CHECK: d2m.generic
    // CHECK: d2m.tile_transpose
    // The TopK generic op with k=64.
    // CHECK: d2m.generic
    // CHECK: d2m.topk_block
    // CHECK-SAME: dim = 1
    // CHECK-SAME: k = 64
    // CHECK-SAME: num_elements = 256
    // Extract values and indices using tile_transpose for dim=1.
    // CHECK: d2m.generic
    // CHECK: d2m.tile_transpose
    // CHECK: d2m.generic
    // CHECK: d2m.tile_transpose
    %values, %indices = "ttir.topk"(%arg0) <{k = 64 : i32, dim = -1 : i32, largest = true, sorted = false}> : (tensor<32x256xf32>) -> (tensor<32x64xf32>, tensor<32x64xsi32>)
    return %values, %indices : tensor<32x64xf32>, tensor<32x64xsi32>
  }

  // k=64 along dim=0 uses tile_typecast for extract and needs no transpose.
  // CHECK-LABEL: func @topk_dim0_k64
  func.func @topk_dim0_k64(%arg0: tensor<256x32xf32>) -> (tensor<64x32xf32>, tensor<64x32xsi32>) {
    // CHECK-NOT: d2m.tile_transpose
    // CHECK-NOT: d2m.arange_block
    // CHECK: d2m.generic
    // CHECK: d2m.topk_block
    // CHECK-SAME: dim = 0
    // CHECK-SAME: k = 64
    // CHECK-SAME: num_elements = 256
    // CHECK: d2m.generic
    // CHECK: d2m.tile_typecast
    // CHECK: d2m.generic
    // CHECK: d2m.tile_typecast
    %values, %indices = "ttir.topk"(%arg0) <{k = 64 : i32, dim = 0 : i32, largest = true, sorted = false}> : (tensor<256x32xf32>) -> (tensor<64x32xf32>, tensor<64x32xsi32>)
    return %values, %indices : tensor<64x32xf32>, tensor<64x32xsi32>
  }

  // ---- Tile shape validation ----

  // Verify the tiled tensor shapes are correct for dim=1 lowering.
  // 32x64 input maps to 1x2 tiles (Ht=1, Wt=2).
  // CHECK-LABEL: func @topk_dim1_tile_shapes
  func.func @topk_dim1_tile_shapes(%arg0: tensor<32x64xf32>) -> (tensor<32x16xf32>, tensor<32x16xsi32>) {
    // The input is gathered onto one core as 1x2 tiles of f32.
    // CHECK: d2m.to_layout %arg0{{.*}} -> tensor<1x1x1x2x!ttcore.tile<32x32, f32>
    // The pre-transpose buffer is 1x2 tiles of f32.
    // CHECK: d2m.empty() : tensor<1x1x1x2x!ttcore.tile<32x32, f32>
    // The index input is a single si32 scratch tile per core: the kernel
    // derives the whole index buffer from it.
    // CHECK: d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, si32>
    // The TopK values output is 1x2 tiles (full reduction shape before extract).
    // CHECK: d2m.empty() : tensor<1x1x1x2x!ttcore.tile<32x32, f32>
    // The TopK indices output is 1x2 si32 tiles (typecast to the user output
    // happens after extract).
    // CHECK: d2m.empty() : tensor<1x1x1x2x!ttcore.tile<32x32, si32>
    %values, %indices = "ttir.topk"(%arg0) <{k = 16 : i32, dim = -1 : i32, largest = true, sorted = false}> : (tensor<32x64xf32>) -> (tensor<32x16xf32>, tensor<32x16xsi32>)
    return %values, %indices : tensor<32x16xf32>, tensor<32x16xsi32>
  }

  // Verify tiled shapes for dim=0: 64x32 maps to 2x1 tiles.
  // CHECK-LABEL: func @topk_dim0_tile_shapes
  func.func @topk_dim0_tile_shapes(%arg0: tensor<64x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xsi32>) {
    // CHECK: d2m.to_layout %arg0{{.*}} -> tensor<1x1x2x1x!ttcore.tile<32x32, f32>
    // The index input is a single si32 scratch tile per core.
    // CHECK: d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, si32>
    // Values keep the full 2x1 reduction shape ...
    // CHECK: d2m.empty() : tensor<1x1x2x1x!ttcore.tile<32x32, f32>
    // ... and the indices ride along as si32 (dim=0 folds the typecast to the
    // user's index type into the extract itself).
    // CHECK: d2m.empty() : tensor<1x1x2x1x!ttcore.tile<32x32, si32>
    %values, %indices = "ttir.topk"(%arg0) <{k = 16 : i32, dim = 0 : i32, largest = true, sorted = false}> : (tensor<64x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xsi32>)
    return %values, %indices : tensor<16x32xf32>, tensor<16x32xsi32>
  }

  // ---- Non-power-of-2 tile counts (ragged), k<=32 ----

  // 32x544 with k=16: Wt=17, a non-power-of-2 reduction tile count.
  // CHECK-LABEL: func @topk_dim1_k16_nonpow2
  func.func @topk_dim1_k16_nonpow2(%arg0: tensor<32x544xf32>) -> (tensor<32x16xf32>, tensor<32x16xsi32>) {
    // The ragged tile count is the merge tree's problem, not this level's.
    // CHECK: d2m.generic
    // CHECK: d2m.tile_transpose
    // CHECK: d2m.generic
    // CHECK: d2m.topk_block
    // CHECK-SAME: k = 16
    // CHECK-SAME: num_elements = 544
    // CHECK: d2m.generic
    // CHECK: d2m.tile_transpose
    // CHECK: d2m.generic
    // CHECK: d2m.tile_transpose
    %values, %indices = "ttir.topk"(%arg0) <{k = 16 : i32, dim = -1 : i32, largest = true, sorted = false}> : (tensor<32x544xf32>) -> (tensor<32x16xf32>, tensor<32x16xsi32>)
    return %values, %indices : tensor<32x16xf32>, tensor<32x16xsi32>
  }

  // ---- Multi-core (reduction dim split into per-core bands) ----
  //
  // When the reduction dim needs more tiles than one core's budget
  // (kMaxTilesPerCore / nonTargetTiles), grid selection plans numShards bands
  // (one per core) and d2m-build-topk-chain builds them, each running a local
  // topk_block and then narrowing its partial to ceil(k/32) tiles. The bands
  // stay distributed: a merge round gathers them with one composite_view per
  // operand (values and indices need separate generics), re-splitting that
  // grid x shard extent onto the merge grid, then runs one topk_block for every
  // group at once.

  // dim=1 multi-core: 128x512, k=16. Rows=128 (4 non-target tiles), cols=512
  // (16 reduction tiles) -> multi-core band split.
  // CHECK-LABEL: func @topk_dim1_multicore
  // PLAN-LABEL: func.func @topk_dim1_multicore
  // PLAN: d2m.topk_plan
  func.func @topk_dim1_multicore(%arg0: tensor<128x512xf32>) -> (tensor<128x16xf32>, tensor<128x16xsi32>) {
    // Per-band local topk (transpose + topk_block) ...
    // CHECK: d2m.tile_transpose
    // CHECK: d2m.topk_block
    // CHECK-SAME: dim = 1
    // CHECK-SAME: k = 16
    // ... then the merge gathers the per-band partials via composite_view ...
    // CHECK: d2m.composite_view
    // ... and a final merge topk_block selects the global top-k.
    // CHECK: d2m.topk_block
    // CHECK-SAME: dim = 1
    %values, %indices = "ttir.topk"(%arg0) <{k = 16 : i32, dim = -1 : i32, largest = true, sorted = false}> : (tensor<128x512xf32>) -> (tensor<128x16xf32>, tensor<128x16xsi32>)
    return %values, %indices : tensor<128x16xf32>, tensor<128x16xsi32>
  }

  // dim=0 multi-core: 512x128, k=16. Rows=512 (16 reduction tiles), cols=128
  // (4 non-target tiles) -> multi-core band split. Each band core derives its
  // own index slice from its grid coordinate, so no index buffer is built or
  // moved across cores.
  // CHECK-LABEL: func @topk_dim0_multicore
  // PLAN-LABEL: func.func @topk_dim0_multicore
  // PLAN: d2m.topk_plan
  func.func @topk_dim0_multicore(%arg0: tensor<512x128xf32>) -> (tensor<16x128xf32>, tensor<16x128xsi32>) {
    // CHECK-NOT: d2m.tile_transpose
    // CHECK-NOT: d2m.arange_block
    // The per-band local topk ...
    // CHECK: d2m.topk_block
    // CHECK-SAME: dim = 0
    // CHECK-SAME: k = 16
    // ... the merge gathers per-band partials via composite_view ...
    // CHECK: d2m.composite_view
    // ... and a final merge topk_block selects the global top-k.
    // CHECK: d2m.topk_block
    // CHECK-SAME: dim = 0
    %values, %indices = "ttir.topk"(%arg0) <{k = 16 : i32, dim = 0 : i32, largest = true, sorted = false}> : (tensor<512x128xf32>) -> (tensor<16x128xf32>, tensor<16x128xsi32>)
    return %values, %indices : tensor<16x128xf32>, tensor<16x128xsi32>
  }

  // ---- Multi-core (non-target dim split, data-parallel) ----
  //
  // When the NON-TARGET dim alone overflows the per-core budget (64 tiles here
  // against kMaxTilesPerCore = 43), banding the reduction dim cannot help: the
  // whole non-target dim still lives on every band core, leaving it under the
  // two reduction tiles a band needs. topk is independent per slice, so the
  // lowering splits the non-target dim across cores instead and each one runs
  // the entire reduction locally.

  // dim=1 data-parallel: 2048x128, k=8. Rows=2048 (64 non-target tiles) split
  // across cores, cols=128 (4 reduction tiles) kept whole on each.
  // CHECK-LABEL: func @topk_dim1_data_parallel
  // PLAN-LABEL: func.func @topk_dim1_data_parallel
  // PLAN: d2m.topk_plan
  func.func @topk_dim1_data_parallel(%arg0: tensor<2048x128xf32>) -> (tensor<2048x8xf32>, tensor<2048x8xsi32>) {
    // The value input is still pre-transposed for dim=1 ...
    // CHECK: d2m.tile_transpose
    // ... and a lone local topk produces the final answer per slice.
    // CHECK: d2m.topk_block
    // CHECK-SAME: dim = 1
    // CHECK-SAME: k = 8
    // CHECK-NOT: d2m.composite_view
    %values, %indices = "ttir.topk"(%arg0) <{k = 8 : i32, dim = -1 : i32, largest = true, sorted = false}> : (tensor<2048x128xf32>) -> (tensor<2048x8xf32>, tensor<2048x8xsi32>)
    return %values, %indices : tensor<2048x8xf32>, tensor<2048x8xsi32>
  }

  // dim=0 data-parallel: the transpose of the above. Cols=2048 (64 non-target
  // tiles) split across cores, rows=128 (4 reduction tiles) kept whole.
  // CHECK-LABEL: func @topk_dim0_data_parallel
  // PLAN-LABEL: func.func @topk_dim0_data_parallel
  // PLAN: d2m.topk_plan
  func.func @topk_dim0_data_parallel(%arg0: tensor<128x2048xf32>) -> (tensor<8x2048xf32>, tensor<8x2048xsi32>) {
    // Every slice reduces the whole dim locally and builds its own indices, so
    // there is no shared index buffer to broadcast across the slice cores.
    // CHECK-NOT: d2m.tile_transpose
    // CHECK-NOT: d2m.arange_block
    // The per-slice topk.
    // CHECK: d2m.topk_block
    // CHECK-SAME: dim = 0
    // CHECK-SAME: k = 8
    // CHECK-NOT: d2m.composite_view
    %values, %indices = "ttir.topk"(%arg0) <{k = 8 : i32, dim = 0 : i32, largest = true, sorted = false}> : (tensor<128x2048xf32>) -> (tensor<8x2048xf32>, tensor<8x2048xsi32>)
    return %values, %indices : tensor<8x2048xf32>, tensor<8x2048xsi32>
  }
}
