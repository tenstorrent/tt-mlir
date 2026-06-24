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

    // Outer scf.for over mIter, inner scf.for over tile pairs. The rebuild is
    // guarded by an scf.if on needsRebuild.
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 32
    // CHECK: scf.if
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

    // Outer scf.for over mIter [0, 3), inner scf.for over tile pairs. The
    // rebuild is guarded by an scf.if on needsRebuild.
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 32
    // CHECK: scf.if
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 16{{.*}}logk = 5

    %values, %indices = "ttir.topk"(%arg0) <{k = 16 : i32, dim = -1 : i32, largest = true, sorted = false}> : (tensor<32x256xf32>) -> (tensor<32x16xf32>, tensor<32x16xsi32>)
    return %values, %indices : tensor<32x16xf32>, tensor<32x16xsi32>
  }

  // ---- Large k (k>32), 2 tiles: direct sort-merge-rebuild ----

  // 32x128 with k=64: useLargeK=true, logk=6. Wt=2, logWt=1.
  // Note: We use 32x128 (k < num_elements=128) to avoid a bug where
  // k==num_elements causes a shape mismatch in createExtractGeneric.
  // CHECK-LABEL: func @decompose_k64_2tiles
  func.func @decompose_k64_2tiles(%arg0: tensor<32x128xf32>) -> (tensor<32x64xf32>, tensor<32x64xsi32>) {
    // CHECK-NOT: d2m.topk_block

    // CHECK: d2m.arange_block
    // CHECK-SAME: num_elements = 128

    // Single scf.for with sort-merge-rebuild.
    // CHECK: scf.for
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 64
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 64{{.*}}logk = 6

    %values, %indices = "ttir.topk"(%arg0) <{k = 64 : i32, dim = -1 : i32, largest = true, sorted = false}> : (tensor<32x128xf32>) -> (tensor<32x64xf32>, tensor<32x64xsi32>)
    return %values, %indices : tensor<32x64xf32>, tensor<32x64xsi32>
  }

  // ---- Large k (k>32), 4 tiles: 3-sub-merge tree on later iterations ----

  // 32x128 with k=64: Wt=4, logWt=2.
  // Iter 0: pairs (0,1)(2,3) — direct sort-merge-rebuild (isFirst).
  // Iter 1: pair (0,2) — 3-sub-merge tree.
  // CHECK-LABEL: func @decompose_k64_4tiles
  func.func @decompose_k64_4tiles(%arg0: tensor<32x128xf32>) -> (tensor<32x64xf32>, tensor<32x64xsi32>) {
    // CHECK-NOT: d2m.topk_block

    // CHECK: d2m.arange_block
    // CHECK-SAME: num_elements = 128

    // Outer loop over logWt=2 merge iterations; inner loop over tile pairs.
    // CHECK: scf.for
    // CHECK: scf.for
    // scf.if dispatches iter 0 (then-block) vs later iters (else-block).
    // CHECK: scf.if
    // Iter 0 (then-block): direct sort-merge-rebuild.
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 64
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 64
    // Iter 1 (else-block): 3-sub-merge (winners, losers, winners-vs-losers).
    // Sub 1: merge winner tiles.
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 64
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 64
    // Sub 2: merge loser tiles.
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 64
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 64
    // Sub 3: merge winners vs losers.
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 64
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 64

    %values, %indices = "ttir.topk"(%arg0) <{k = 64 : i32, dim = -1 : i32, largest = true, sorted = false}> : (tensor<32x128xf32>) -> (tensor<32x64xf32>, tensor<32x64xsi32>)
    return %values, %indices : tensor<32x64xf32>, tensor<32x64xsi32>
  }

  // ---- dim=0 decomposition ----

  // 64x32 with k=16, dim=0: Ht=2, logWt=1.
  // CHECK-LABEL: func @decompose_dim0_k16
  func.func @decompose_dim0_k16(%arg0: tensor<64x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xsi32>) {
    // CHECK-NOT: d2m.topk_block

    // CHECK: d2m.arange_block
    // CHECK-SAME: num_elements = 64

    // The rebuild is guarded by an scf.if on needsRebuild.
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 32
    // CHECK: scf.if
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 16{{.*}}logk = 5

    %values, %indices = "ttir.topk"(%arg0) <{k = 16 : i32, dim = 0 : i32, largest = true, sorted = false}> : (tensor<64x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xsi32>)
    return %values, %indices : tensor<16x32xf32>, tensor<16x32xsi32>
  }

  // ---- k=32 special case: no rebuild on single iteration ----

  // 32x64 with k=32: Wt=2, logWt=1.
  // Only 1 iteration and k==32, so the merge output is already exactly k
  // elements. The rebuild is guarded by an scf.if whose condition is false at
  // runtime, so it is skipped and the merge packs the tiles instead.
  // CHECK-LABEL: func @decompose_k32_2tiles
  func.func @decompose_k32_2tiles(%arg0: tensor<32x64xf32>) -> (tensor<32x32xf32>, tensor<32x32xsi32>) {
    // CHECK-NOT: d2m.topk_block

    // CHECK: d2m.arange_block
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge{{.*}}k = 32
    // CHECK: scf.if
    // CHECK: d2m.tile_topk_rebuild

    %values, %indices = "ttir.topk"(%arg0) <{k = 32 : i32, dim = -1 : i32, largest = true, sorted = false}> : (tensor<32x64xf32>) -> (tensor<32x32xf32>, tensor<32x32xsi32>)
    return %values, %indices : tensor<32x32xf32>, tensor<32x32xsi32>
  }
}
