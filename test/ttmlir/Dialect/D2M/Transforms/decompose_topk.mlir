// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-materialize-view-returns --ttcore-one-shot-bufferize --d2m-decompose-topk -o %t %s
// RUN: FileCheck %s --input-file=%t
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline -o %t.ttmetal %s

// Verify that d2m-decompose-topk replaces topk_block with arange_block +
// scf.for loops (one per merge iteration) containing
// tile_topk_{local_sort,merge,rebuild} ops.

module {

  // ---- Small k (k<=32), 2 tiles: 1 merge iteration ----

  // 32x64 with k=16: Wt=2, logWt=1 → outer scf.for with nested inner scf.for.
  // CHECK-LABEL: func @decompose_k16_2tiles
  func.func @decompose_k16_2tiles(%arg0: tensor<32x64xf32>) -> (tensor<32x16xf32>, tensor<32x16xsi32>) {
    // The topk_block must be fully replaced.
    // CHECK-NOT: d2m.topk_block

    // Index initialization is done via arange_block.
    // CHECK: d2m.arange_block
    // CHECK-SAME: num_elements = 64

    // logWt=1, so level 0 is also the last level: emitted unrolled (no outer
    // scf.for), with an inner scf.for over tile pairs and an unconditional
    // rebuild.
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 32
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 16{{.*}}logk = 5

    %values, %indices = "ttir.topk"(%arg0) <{k = 16 : i32, dim = -1 : i32, largest = true, sorted = false}> : (tensor<32x64xf32>) -> (tensor<32x16xf32>, tensor<32x16xsi32>)
    return %values, %indices : tensor<32x16xf32>, tensor<32x16xsi32>
  }

  // ---- Small k (k<=32), 8 tiles: 3 merge iterations ----

  // 32x256 with k=16: Wt=8, logWt=3.
  // Single outer scf.for (3 iterations) with nested inner scf.for.
  // CHECK-LABEL: func @decompose_k16_8tiles
  func.func @decompose_k16_8tiles(%arg0: tensor<32x256xf32>) -> (tensor<32x16xf32>, tensor<32x16xsi32>) {
    // CHECK-NOT: d2m.topk_block

    // CHECK: d2m.arange_block
    // CHECK-SAME: num_elements = 256

    // Level 0 is emitted unrolled, levels [1,logWt-1) run in a
    // middle scf.for with no rebuild, and the last level is emitted unrolled
    // with an unconditional rebuild.
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 32
    // Middle level(s): plain scf.for, no rebuild.
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 32
    // Last level: unrolled, unconditional rebuild.
    // CHECK: scf.for
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 32
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 16{{.*}}logk = 5

    %values, %indices = "ttir.topk"(%arg0) <{k = 16 : i32, dim = -1 : i32, largest = true, sorted = false}> : (tensor<32x256xf32>) -> (tensor<32x16xf32>, tensor<32x16xsi32>)
    return %values, %indices : tensor<32x16xf32>, tensor<32x16xsi32>
  }

  // ---- Large k (k>32): left-fold accumulator over reduction tiles ----

  // 32x64 with k=64: 2 reduction tiles. Builds a single 2-tile accumulator
  // (tiles 0,1) with no fold loop, since there are no complete right pairs.
  // CHECK-LABEL: func @decompose_k64_2tiles
  func.func @decompose_k64_2tiles(%arg0: tensor<32x64xf32>) -> (tensor<32x64xf32>, tensor<32x64xsi32>) {
    // CHECK-NOT: d2m.topk_block

    // CHECK: d2m.arange_block
    // CHECK-SAME: num_elements = 64

    // Non-target loop wraps the whole merge tree.
    // CHECK: scf.for
    // Initial accumulator from tiles (0,1): sort-merge-rebuild.
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 64
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 64{{.*}}logk = 6

    %values, %indices = "ttir.topk"(%arg0) <{k = 64 : i32, dim = -1 : i32, largest = true, sorted = false}> : (tensor<32x64xf32>) -> (tensor<32x64xf32>, tensor<32x64xsi32>)
    return %values, %indices : tensor<32x64xf32>, tensor<32x64xsi32>
  }

  // ---- Large k (k>32), 4 tiles: left-fold with one complete right pair ----

  // 32x128 with k=64: 4 reduction tiles. One fold-loop trip combines the
  // accumulator (0,1) with the complete right pair (2,3) via a build plus a
  // 3-sub-merge, keeping the accumulator canonical at (0,1).
  // CHECK-LABEL: func @decompose_k64_4tiles
  func.func @decompose_k64_4tiles(%arg0: tensor<32x128xf32>) -> (tensor<32x64xf32>, tensor<32x64xsi32>) {
    // CHECK-NOT: d2m.topk_block

    // CHECK: d2m.arange_block
    // CHECK-SAME: num_elements = 128

    // Outer scf.for is the non-target-row loop (one row here).
    // CHECK: scf.for
    // Initial accumulator built from tiles (0,1).
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 64
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 64
    // Fold loop over complete right pairs.
    // CHECK: scf.for
    // Build the complete right pair (2,3) from raw input.
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 64
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 64
    // 3-sub-merge combine of acc=(0,1) with the pair.
    // Sub 1: merge winner tiles.
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 64
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 64
    // Sub 2: merge losers.
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 64
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 64
    // Sub 3: merge losers vs winners, writing back to tile 1.
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 64
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 64

    %values, %indices = "ttir.topk"(%arg0) <{k = 64 : i32, dim = -1 : i32, largest = true, sorted = false}> : (tensor<32x128xf32>) -> (tensor<32x64xf32>, tensor<32x64xsi32>)
    return %values, %indices : tensor<32x64xf32>, tensor<32x64xsi32>
  }

  // ---- Large k (k>32), 6 tiles: left-fold with two complete right pairs ----

  // 32x192 with k=64: 6 reduction tiles (even). The fold loop runs twice, once
  // per complete right pair (2,3) and (4,5), each a build plus a 3-sub-merge.
  // No power-of-2 padding is needed and there is no odd tail.
  // CHECK-LABEL: func @decompose_k64_6tiles
  func.func @decompose_k64_6tiles(%arg0: tensor<32x192xf32>) -> (tensor<32x64xf32>, tensor<32x64xsi32>) {
    // CHECK-NOT: d2m.topk_block

    // CHECK: d2m.arange_block
    // CHECK-SAME: num_elements = 192

    // Outer scf.for is the non-target-row loop.
    // CHECK: scf.for
    // Initial accumulator from tiles (0,1).
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 64
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 64
    // Fold loop (2 trips) over complete right pairs.
    // CHECK: scf.for
    // Build the complete right pair.
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 64
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 64
    // 3-sub-merge combine. Sub 1: winners.
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 64
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 64
    // Sub 2: losers.
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 64
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 64
    // Sub 3: losers vs winners.
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 64
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 64

    %values, %indices = "ttir.topk"(%arg0) <{k = 64 : i32, dim = -1 : i32, largest = true, sorted = false}> : (tensor<32x192xf32>) -> (tensor<32x64xf32>, tensor<32x64xsi32>)
    return %values, %indices : tensor<32x64xf32>, tensor<32x64xsi32>
  }

  // ---- Large k (k>32), 5 tiles: odd tile count → tail fold ----

  // 32x160 with k=64: 5 reduction tiles (odd). Seed (0,1), the fold loop runs
  // once for the complete pair (2,3), then tile 4 is folded in as a lone tail
  // with a 2-sub-merge (preceded by a prime that sorts the raw tail tile).
  // CHECK-LABEL: func @decompose_k64_5tiles
  func.func @decompose_k64_5tiles(%arg0: tensor<32x160xf32>) -> (tensor<32x64xf32>, tensor<32x64xsi32>) {
    // CHECK-NOT: d2m.topk_block

    // CHECK: d2m.arange_block
    // CHECK-SAME: num_elements = 160

    // Outer scf.for is the non-target-row loop.
    // CHECK: scf.for
    // Initial accumulator from tiles (0,1).
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 64
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 64
    // Fold loop over the one complete right pair (2,3): build + 3-sub-merge.
    // CHECK: scf.for
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 64
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 64
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 64
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 64
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 64
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 64
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 64
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 64
    // Odd tail (tile 4), emitted after the fold loop. First prime the raw tail
    // tile into a valid sorted run with a standalone local_sort.
    // CHECK: d2m.tile_topk_local_sort
    // 2-sub-merge combine. Step A: winners (0, tail).
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 64
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 64
    // Step B: losers vs tail (1, tail).
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 64
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 64

    %values, %indices = "ttir.topk"(%arg0) <{k = 64 : i32, dim = -1 : i32, largest = true, sorted = false}> : (tensor<32x160xf32>) -> (tensor<32x64xf32>, tensor<32x64xsi32>)
    return %values, %indices : tensor<32x64xf32>, tensor<32x64xsi32>
  }

  // ---- dim=0 decomposition ----

  // 64x32 with k=16, dim=0: Ht=2, logWt=1.
  // CHECK-LABEL: func @decompose_dim0_k16
  func.func @decompose_dim0_k16(%arg0: tensor<64x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xsi32>) {
    // CHECK-NOT: d2m.topk_block

    // CHECK: d2m.arange_block
    // CHECK-SAME: num_elements = 64

    // logWt=1: level 0 is also the last level, so rebuild is emitted
    // unconditionally.
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 32
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 16{{.*}}logk = 5

    %values, %indices = "ttir.topk"(%arg0) <{k = 16 : i32, dim = 0 : i32, largest = true, sorted = false}> : (tensor<64x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xsi32>)
    return %values, %indices : tensor<16x32xf32>, tensor<16x32xsi32>
  }

  // ---- k=32 special case: no rebuild on single iteration ----

  // 32x64 with k=32: Wt=2, logWt=1.
  // Only 1 iteration and k==32, so the merge output is already exactly k
  // elements: needsRebuild is false at compile time, so no rebuild op
  // is emitted at all, and the merge packs the tiles instead.
  // CHECK-LABEL: func @decompose_k32_2tiles
  func.func @decompose_k32_2tiles(%arg0: tensor<32x64xf32>) -> (tensor<32x32xf32>, tensor<32x32xsi32>) {
    // CHECK-NOT: d2m.topk_block

    // CHECK: d2m.arange_block
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 32
    // CHECK-NOT: d2m.tile_topk_rebuild

    %values, %indices = "ttir.topk"(%arg0) <{k = 32 : i32, dim = -1 : i32, largest = true, sorted = false}> : (tensor<32x64xf32>) -> (tensor<32x32xf32>, tensor<32x32xsi32>)
    return %values, %indices : tensor<32x32xf32>, tensor<32x32xsi32>
  }

  // ---- Non-power-of-2 tile counts (ragged merge tree) ----

  // 32x96 with k=16: Wt=3 (odd, ragged), ceilLog2=2 iterations.
  // Level 0 pairs (0,1); tile 2 is odd tail → standalone local_sort at level 0.
  // Level 1 pairs (0,2).
  // CHECK-LABEL: func @decompose_k16_3tiles
  func.func @decompose_k16_3tiles(%arg0: tensor<32x96xf32>) -> (tensor<32x16xf32>, tensor<32x16xsi32>) {
    // CHECK-NOT: d2m.topk_block
    // CHECK: d2m.arange_block
    // CHECK-SAME: num_elements = 96
    // Level 0 (unrolled): local_sort+merge, plus the odd-tail standalone sort
    // for tile 2 (emitted unconditionally at level 0).
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 32
    // CHECK: d2m.tile_topk_local_sort
    // Last level (logWt-1, unrolled): rebuild is unconditional.
    // CHECK: scf.for
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 32
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 16{{.*}}logk = 5

    %values, %indices = "ttir.topk"(%arg0) <{k = 16 : i32, dim = -1 : i32, largest = true, sorted = false}> : (tensor<32x96xf32>) -> (tensor<32x16xf32>, tensor<32x16xsi32>)
    return %values, %indices : tensor<32x16xf32>, tensor<32x16xsi32>
  }

  // 32x192 with k=16: Wt=6 (even, ragged), ceilLog2=3 iterations.
  // All tiles paired at level 0 (no odd tail).
  // CHECK-LABEL: func @decompose_k16_6tiles
  func.func @decompose_k16_6tiles(%arg0: tensor<32x192xf32>) -> (tensor<32x16xf32>, tensor<32x16xsi32>) {
    // CHECK-NOT: d2m.topk_block
    // CHECK: d2m.arange_block
    // CHECK-SAME: num_elements = 192
    // Level 0 (unrolled).
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 32
    // Middle level(s) [1, logWt-1): plain scf.for, no rebuild.
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 32
    // Last level (unrolled): rebuild is unconditional.
    // CHECK: scf.for
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 32
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 16{{.*}}logk = 5

    %values, %indices = "ttir.topk"(%arg0) <{k = 16 : i32, dim = -1 : i32, largest = true, sorted = false}> : (tensor<32x192xf32>) -> (tensor<32x16xf32>, tensor<32x16xsi32>)
    return %values, %indices : tensor<32x16xf32>, tensor<32x16xsi32>
  }

  // 32x544 with k=16: Wt=17 (odd, ragged), ceilLog2=5 iterations. Headline case.
  // CHECK-LABEL: func @decompose_k16_17tiles
  func.func @decompose_k16_17tiles(%arg0: tensor<32x544xf32>) -> (tensor<32x16xf32>, tensor<32x16xsi32>) {
    // CHECK-NOT: d2m.topk_block
    // CHECK: d2m.arange_block
    // CHECK-SAME: num_elements = 544
    // Level 0 (unrolled), plus odd-tail standalone sort for tile 16.
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 32
    // CHECK: d2m.tile_topk_local_sort
    // Middle level(s) [1, logWt-1): plain scf.for, no rebuild.
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 32
    // Last level (unrolled): rebuild is unconditional.
    // CHECK: scf.for
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 32
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 16{{.*}}logk = 5

    %values, %indices = "ttir.topk"(%arg0) <{k = 16 : i32, dim = -1 : i32, largest = true, sorted = false}> : (tensor<32x544xf32>) -> (tensor<32x16xf32>, tensor<32x16xsi32>)
    return %values, %indices : tensor<32x16xf32>, tensor<32x16xsi32>
  }

  // dim=0 non-pow2: 96x32 with k=16, Ht=3 (odd, ragged), ceilLog2=2.
  // CHECK-LABEL: func @decompose_k16_3tiles_dim0
  func.func @decompose_k16_3tiles_dim0(%arg0: tensor<96x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xsi32>) {
    // CHECK-NOT: d2m.topk_block
    // CHECK: d2m.arange_block
    // CHECK-SAME: num_elements = 96
    // Level 0 (unrolled), plus odd-tail standalone sort for tile 2.
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 32
    // CHECK: d2m.tile_topk_local_sort
    // Last level (unrolled): rebuild is unconditional.
    // CHECK: scf.for
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 32
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 16{{.*}}logk = 5

    %values, %indices = "ttir.topk"(%arg0) <{k = 16 : i32, dim = 0 : i32, largest = true, sorted = false}> : (tensor<96x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xsi32>)
    return %values, %indices : tensor<16x32xf32>, tensor<16x32xsi32>
  }

}
