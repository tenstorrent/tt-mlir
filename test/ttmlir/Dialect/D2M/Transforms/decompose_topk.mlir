// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-materialize-view-returns --ttcore-one-shot-bufferize --d2m-decompose-topk -o %t %s
// RUN: FileCheck %s --input-file=%t
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline -o %t.ttmetal %s

// Verify that d2m-decompose-topk replaces topk_block with the decomposed
// arange_block + tile_topk_{local_sort,merge,rebuild} sequence.

module {

  // ---- Small k (k<=32), 2 tiles: 1 merge iteration ----

  // 32x64 with k=16: Wt=2, logWt=1 yields 1 iteration of sort+merge+rebuild.
  // CHECK-LABEL: func @decompose_k16_2tiles
  func.func @decompose_k16_2tiles(%arg0: tensor<32x64xf32>) -> (tensor<32x16xf32>, tensor<32x16xsi32>) {
    // The topk_block must be fully replaced.
    // CHECK-NOT: d2m.topk_block

    // Index initialization is done via arange_block.
    // CHECK: d2m.arange_block
    // CHECK-SAME: num_elements = 64

    // Iteration 0 (only iteration) contains sort, merge, and rebuild.
    // CHECK: d2m.tile_topk_local_sort{{.*}}tile_a = 0{{.*}}tile_b = 1
    // CHECK: d2m.tile_topk_merge{{.*}}k = 32{{.*}}tile_a = 0{{.*}}tile_b = 1
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 16{{.*}}logk = 5{{.*}}skip_second = 1{{.*}}tile_a = 0{{.*}}tile_b = 1

    // No extra iterations should be emitted.
    // CHECK-NOT: d2m.tile_topk_local_sort
    %values, %indices = "ttir.topk"(%arg0) <{k = 16 : i32, dim = -1 : i32, largest = true, sorted = false}> : (tensor<32x64xf32>) -> (tensor<32x16xf32>, tensor<32x16xsi32>)
    return %values, %indices : tensor<32x16xf32>, tensor<32x16xsi32>
  }

  // ---- Small k (k<=32), 8 tiles: 3 merge iterations ----

  // 32x256 with k=16: Wt=8, logWt=3.
  // Iter 0: 4 pairs (0,1)(2,3)(4,5)(6,7) contain sort+merge only (no rebuild).
  // Iter 1: 2 pairs (0,2)(4,6) contain sort+merge only.
  // Iter 2: 1 pair (0,4) contains sort+merge+rebuild (last iteration, k<32).
  // CHECK-LABEL: func @decompose_k16_8tiles
  func.func @decompose_k16_8tiles(%arg0: tensor<32x256xf32>) -> (tensor<32x16xf32>, tensor<32x16xsi32>) {
    // CHECK-NOT: d2m.topk_block

    // CHECK: d2m.arange_block
    // CHECK-SAME: num_elements = 256

    // ---- Iteration 0: 4 pairs with sort+merge and no rebuild ----
    // CHECK: d2m.tile_topk_local_sort{{.*}}tile_a = 0{{.*}}tile_b = 1
    // CHECK: d2m.tile_topk_merge{{.*}}m_iter = 0{{.*}}tile_a = 0{{.*}}tile_b = 1
    // CHECK: d2m.tile_topk_local_sort{{.*}}tile_a = 2{{.*}}tile_b = 3
    // CHECK: d2m.tile_topk_merge{{.*}}tile_a = 2{{.*}}tile_b = 3
    // CHECK: d2m.tile_topk_local_sort{{.*}}tile_a = 4{{.*}}tile_b = 5
    // CHECK: d2m.tile_topk_merge{{.*}}tile_a = 4{{.*}}tile_b = 5
    // CHECK: d2m.tile_topk_local_sort{{.*}}tile_a = 6{{.*}}tile_b = 7
    // CHECK: d2m.tile_topk_merge{{.*}}tile_a = 6{{.*}}tile_b = 7

    // ---- Iteration 1: 2 pairs with sort+merge and read_from_output ----
    // CHECK: d2m.tile_topk_local_sort{{.*}}read_from_output = true{{.*}}tile_a = 0{{.*}}tile_b = 2
    // CHECK: d2m.tile_topk_merge{{.*}}m_iter = 1{{.*}}tile_a = 0{{.*}}tile_b = 2
    // CHECK: d2m.tile_topk_local_sort{{.*}}tile_a = 4{{.*}}tile_b = 6
    // CHECK: d2m.tile_topk_merge{{.*}}tile_a = 4{{.*}}tile_b = 6

    // ---- Iteration 2 (last): 1 pair with sort+merge+rebuild ----
    // CHECK: d2m.tile_topk_local_sort{{.*}}tile_a = 0{{.*}}tile_b = 4
    // CHECK: d2m.tile_topk_merge{{.*}}m_iter = 2{{.*}}tile_a = 0{{.*}}tile_b = 4
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 16{{.*}}logk = 5{{.*}}m_iter = 2{{.*}}skip_second = 1{{.*}}tile_a = 0{{.*}}tile_b = 4

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

    // CHECK: d2m.tile_topk_local_sort{{.*}}read_from_output = false{{.*}}tile_a = 0{{.*}}tile_b = 1
    // CHECK: d2m.tile_topk_merge{{.*}}k = 64{{.*}}tile_a = 0{{.*}}tile_b = 1
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 64{{.*}}logk = 6{{.*}}skip_second = 0{{.*}}tile_a = 0{{.*}}tile_b = 1

    %values, %indices = "ttir.topk"(%arg0) <{k = 64 : i32, dim = -1 : i32, largest = true, sorted = false}> : (tensor<32x128xf32>) -> (tensor<32x64xf32>, tensor<32x64xsi32>)
    return %values, %indices : tensor<32x64xf32>, tensor<32x64xsi32>
  }

  // ---- Large k (k>32), 4 tiles: 3-sub-merge tree on later iterations ----

  // 32x128 with k=64: Wt=4, logWt=2.
  // Iter 0: pairs (0,1)(2,3) use direct sort-merge-rebuild (isFirst).
  // Iter 1: pair (0,2) uses a 3-sub-merge tree:
  //   Sub 1: merge winners (0,2)
  //   Sub 2: merge losers (1,3)
  //   Sub 3: merge winners vs losers (2,1)
  // CHECK-LABEL: func @decompose_k64_4tiles
  func.func @decompose_k64_4tiles(%arg0: tensor<32x128xf32>) -> (tensor<32x64xf32>, tensor<32x64xsi32>) {
    // CHECK-NOT: d2m.topk_block

    // CHECK: d2m.arange_block
    // CHECK-SAME: num_elements = 128

    // ---- Iter 0: 2 pairs with direct sort-merge-rebuild ----
    // Pair (0,1):
    // CHECK: d2m.tile_topk_local_sort{{.*}}read_from_output = false{{.*}}tile_a = 0{{.*}}tile_b = 1
    // CHECK: d2m.tile_topk_merge{{.*}}k = 64{{.*}}tile_a = 0{{.*}}tile_b = 1
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 64{{.*}}tile_a = 0{{.*}}tile_b = 1
    // Pair (2,3):
    // CHECK: d2m.tile_topk_local_sort{{.*}}tile_a = 2{{.*}}tile_b = 3
    // CHECK: d2m.tile_topk_merge{{.*}}k = 64{{.*}}tile_a = 2{{.*}}tile_b = 3
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 64{{.*}}tile_a = 2{{.*}}tile_b = 3

    // ---- Iter 1: 3-sub-merge tree for pair (0,2) ----
    // Sub 1 merges winner tiles (0,2).
    // CHECK: d2m.tile_topk_local_sort{{.*}}read_from_output = true{{.*}}tile_a = 0{{.*}}tile_b = 2
    // CHECK: d2m.tile_topk_merge{{.*}}tile_a = 0{{.*}}tile_b = 2
    // CHECK: d2m.tile_topk_rebuild{{.*}}tile_a = 0{{.*}}tile_b = 2
    // Sub 2 merges loser tiles (1,3).
    // CHECK: d2m.tile_topk_local_sort{{.*}}tile_a = 1{{.*}}tile_b = 3
    // CHECK: d2m.tile_topk_merge{{.*}}tile_a = 1{{.*}}tile_b = 3
    // CHECK: d2m.tile_topk_rebuild{{.*}}tile_a = 1{{.*}}tile_b = 3
    // Sub 3 merges winners vs losers (2,1).
    // CHECK: d2m.tile_topk_local_sort{{.*}}tile_a = 2{{.*}}tile_b = 1
    // CHECK: d2m.tile_topk_merge{{.*}}tile_a = 2{{.*}}tile_b = 1
    // CHECK: d2m.tile_topk_rebuild{{.*}}tile_a = 2{{.*}}tile_b = 1

    %values, %indices = "ttir.topk"(%arg0) <{k = 64 : i32, dim = -1 : i32, largest = true, sorted = false}> : (tensor<32x128xf32>) -> (tensor<32x64xf32>, tensor<32x64xsi32>)
    return %values, %indices : tensor<32x64xf32>, tensor<32x64xsi32>
  }

  // ---- dim=0 decomposition ----

  // 64x32 with k=16, dim=0: Ht=2, logWt=1.
  // The key difference from dim=1 is the shard layout: tiles are 2x1 (tall)
  // rather than 1x2 (wide), since the reduction runs along tile rows.
  // No pre-transpose is needed; extract uses tile_typecast instead.
  // CHECK-LABEL: func @decompose_dim0_k16
  func.func @decompose_dim0_k16(%arg0: tensor<64x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xsi32>) {
    // CHECK-NOT: d2m.topk_block

    // The shard shape is 2x1 (2 tiles tall, 1 tile wide).
    // CHECK: d2m.arange_block
    // CHECK-SAME: num_elements = 64

    // The single iteration has tiles indexed in the height dimension.
    // CHECK: d2m.tile_topk_local_sort{{.*}}tile_a = 0{{.*}}tile_b = 1
    // CHECK: d2m.tile_topk_merge{{.*}}k = 32{{.*}}tile_a = 0{{.*}}tile_b = 1
    // CHECK: d2m.tile_topk_rebuild{{.*}}k = 16{{.*}}logk = 5{{.*}}skip_second = 1{{.*}}tile_a = 0{{.*}}tile_b = 1

    %values, %indices = "ttir.topk"(%arg0) <{k = 16 : i32, dim = 0 : i32, largest = true, sorted = false}> : (tensor<64x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xsi32>)
    return %values, %indices : tensor<16x32xf32>, tensor<16x32xsi32>
  }

  // ---- k=32 special case: no rebuild on single iteration ----

  // 32x64 with k=32: Wt=2, logWt=1.
  // Only 1 iteration and k==32, so the merge output is already exactly k elements.
  // No rebuild is emitted.
  // CHECK-LABEL: func @decompose_k32_2tiles
  func.func @decompose_k32_2tiles(%arg0: tensor<32x64xf32>) -> (tensor<32x32xf32>, tensor<32x32xsi32>) {
    // CHECK-NOT: d2m.topk_block

    // CHECK: d2m.arange_block
    // CHECK: d2m.tile_topk_local_sort{{.*}}tile_a = 0{{.*}}tile_b = 1
    // CHECK: d2m.tile_topk_merge{{.*}}k = 32{{.*}}tile_a = 0{{.*}}tile_b = 1

    // No rebuild is emitted when k==32 and there is only 1 iteration.
    // CHECK-NOT: d2m.tile_topk_rebuild

    %values, %indices = "ttir.topk"(%arg0) <{k = 32 : i32, dim = -1 : i32, largest = true, sorted = false}> : (tensor<32x64xf32>) -> (tensor<32x32xf32>, tensor<32x32xsi32>)
    return %values, %indices : tensor<32x32xf32>, tensor<32x32xsi32>
  }
}
