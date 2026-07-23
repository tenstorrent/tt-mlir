// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-materialize-view-returns -o %t %s
// RUN: FileCheck %s --input-file=%t
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline -o %t.ttmetal %s

module {

  // ---- dim=1 (last dim), k<=32 ----

  // 2D topk along dim 1 with k=16 on a 32x64 input (2 tiles wide).
  // The lowering must:
  //   1. Transpose tiles (dim=1 operates on rows, TopkBlockOp on columns).
  //   2. Run topk_block with correct k and num_elements.
  //   3. Extract results with tile_transpose (untranspose).
  // CHECK-LABEL: func @topk_dim1_k16
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
  // The value input needs no pre-transpose for dim=0, but the index buffer is
  // built row-major on a shape-swapped [wt, ht] buffer and full-transposed
  // (grid + tile) back to [ht, wt], so a tile_transpose DOES appear (on the
  // index buffer, before topk_block). Extract uses tile_typecast.
  // CHECK-LABEL: func @topk_dim0_k16
  func.func @topk_dim0_k16(%arg0: tensor<64x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xsi32>) {
    // The index buffer's full-transpose emits a tile_transpose before topk.
    // CHECK: d2m.tile_transpose

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

  // k=64 requires 2 output tiles and a 3-sub-merge tree.
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

  // k=64 along dim=0 uses tile_typecast for extract. The index buffer is
  // built row-major + full-transposed, so a tile_transpose appears before topk.
  // CHECK-LABEL: func @topk_dim0_k64
  func.func @topk_dim0_k64(%arg0: tensor<256x32xf32>) -> (tensor<64x32xf32>, tensor<64x32xsi32>) {
    // The index buffer's full-transpose emits a tile_transpose before topk.
    // CHECK: d2m.tile_transpose
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
    // The input layout is 1x2 tiles of f32.
    // CHECK: d2m.to_layout
    // CHECK-SAME: tensor<1x1x1x2x!ttcore.tile<32x32, f32>
    // The TopK values output is 1x2 tiles (full reduction shape before extract).
    // CHECK: d2m.empty() : tensor<1x1x1x2x!ttcore.tile<32x32, f32>
    // The TopK indices output is 1x2 tiles of f32 (index buffer carried as
    // fp32; typecast to the si32 user output happens after extract).
    // CHECK: d2m.empty() : tensor<1x1x1x2x!ttcore.tile<32x32, f32>
    %values, %indices = "ttir.topk"(%arg0) <{k = 16 : i32, dim = -1 : i32, largest = true, sorted = false}> : (tensor<32x64xf32>) -> (tensor<32x16xf32>, tensor<32x16xsi32>)
    return %values, %indices : tensor<32x16xf32>, tensor<32x16xsi32>
  }

  // Verify tiled shapes for dim=0: 64x32 maps to 2x1 tiles.
  // CHECK-LABEL: func @topk_dim0_tile_shapes
  func.func @topk_dim0_tile_shapes(%arg0: tensor<64x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xsi32>) {
    // CHECK: d2m.to_layout
    // CHECK-SAME: tensor<1x1x2x1x!ttcore.tile<32x32, f32>
    // CHECK: d2m.empty() : tensor<1x1x2x1x!ttcore.tile<32x32, f32>
    // Index buffer is carried as fp32 (typecast to si32 user output after
    // extract).
    // CHECK: d2m.empty() : tensor<1x1x2x1x!ttcore.tile<32x32, f32>
    %values, %indices = "ttir.topk"(%arg0) <{k = 16 : i32, dim = 0 : i32, largest = true, sorted = false}> : (tensor<64x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xsi32>)
    return %values, %indices : tensor<16x32xf32>, tensor<16x32xsi32>
  }

  // ---- Non-power-of-2 tile counts (ragged), k<=32 ----

  // 32x544 with k=16: Wt=17 (non-pow2) now converts successfully.
  // CHECK-LABEL: func @topk_dim1_k16_nonpow2
  func.func @topk_dim1_k16_nonpow2(%arg0: tensor<32x544xf32>) -> (tensor<32x16xf32>, tensor<32x16xsi32>) {
    // Pre-transpose and topk_block emit normally.
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
  // (kMaxTilesPerCore / nonTargetTiles), the lowering splits it into numShards
  // bands (one per core), runs a local topk_block per band, then merges the
  // per-band partials via a composite_view + a final topk_block. A wide
  // non-target dim (4 tiles here) shrinks the per-core reduction budget to
  // 43/4 = 10 tiles, so a 16-reduction-tile input needs >= 2 cores.

  // dim=1 multi-core: 128x512, k=16. Rows=128 (4 non-target tiles), cols=512
  // (16 reduction tiles) -> multi-core band split.
  // CHECK-LABEL: func @topk_dim1_multicore
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
  // (4 non-target tiles) -> multi-core band split. Exercises the row-major
  // arange + full-transpose (grid + tile) index construction under a
  // gridColsx1 band grid.
  // CHECK-LABEL: func @topk_dim0_multicore
  func.func @topk_dim0_multicore(%arg0: tensor<512x128xf32>) -> (tensor<16x128xf32>, tensor<16x128xsi32>) {
    // The index buffer's full-transpose emits a tile_transpose ...
    // CHECK: d2m.tile_transpose
    // ... the per-band local topk ...
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
