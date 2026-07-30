// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-materialize-view-returns --ttcore-one-shot-bufferize --d2m-decompose-sort -o %t %s
// RUN: FileCheck %s --input-file=%t
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline -o %t.ttmetal %s

// Verify that d2m-decompose-sort replaces sort_block with the bitonic network:
// scf.for loops containing tile_topk_local_sort and tile_topk_merge. Unlike
// topk, no tile_topk_rebuild is emitted -- topk_merge is a plain
// compare-exchange, so both halves of every pair survive.

module {

  // ---- 2 tiles: bitonic sequence formation alone sorts the pair ----

  // 32x64: Wt=2, so stages=1 and the merge network is empty. Only the initial
  // pairwise local sort runs.
  // CHECK-LABEL: func @decompose_sort_2tiles
  func.func @decompose_sort_2tiles(%arg0: tensor<32x64xf32>) -> (tensor<32x64xf32>, tensor<32x64xsi32>) {
    // The sort_block must be fully replaced.
    // CHECK-NOT: d2m.sort_block

    // Index initialization is done via arange_block.
    // CHECK: d2m.arange_block
    // CHECK-SAME: num_elements = 64

    // Outer scf.for over independent non-target rows, inner over tile pairs.
    // CHECK: scf.for
    // CHECK: scf.for
    // A full 64-element sort runs phases 0..5.
    // CHECK: d2m.tile_topk_local_sort
    // CHECK-SAME: i_end_phase = 5

    // No merge network and never a rebuild.
    // CHECK-NOT: d2m.tile_topk_rebuild

    %values, %indices = "ttir.sort"(%arg0) <{dim = -1 : si32, descending = false, stable = false}> : (tensor<32x64xf32>) -> (tensor<32x64xf32>, tensor<32x64xsi32>)
    return %values, %indices : tensor<32x64xf32>, tensor<32x64xsi32>
  }

  // ---- 8 tiles: a real merge network ----

  // 32x256: Wt=8, stages=3. Stages 2 and 3 each run their compare-exchange
  // steps via tile_topk_merge with k=64, then close with a full local sort.
  // CHECK-LABEL: func @decompose_sort_8tiles
  func.func @decompose_sort_8tiles(%arg0: tensor<32x256xf32>) -> (tensor<32x256xf32>, tensor<32x256xsi32>) {
    // CHECK-NOT: d2m.sort_block
    // CHECK: d2m.arange_block
    // CHECK-SAME: num_elements = 256

    // Phase A: pairwise bitonic sequence formation.
    // CHECK: d2m.tile_topk_local_sort
    // CHECK-SAME: i_end_phase = 5

    // Phase B: k=64 makes topk_merge a plain compare-exchange of the two
    // tiles, and the pack destinations are selected at runtime to flip the
    // direction per comparison block.
    // CHECK: arith.select
    // CHECK: d2m.tile_topk_merge
    // CHECK-SAME: k = 64

    // Each stage closes with a full local sort over the adjacent pair.
    // CHECK: d2m.tile_topk_local_sort
    // CHECK-SAME: i_end_phase = 5

    // CHECK-NOT: d2m.tile_topk_rebuild

    %values, %indices = "ttir.sort"(%arg0) <{dim = -1 : si32, descending = false, stable = false}> : (tensor<32x256xf32>) -> (tensor<32x256xf32>, tensor<32x256xsi32>)
    return %values, %indices : tensor<32x256xf32>, tensor<32x256xsi32>
  }

  // ---- non-power-of-two tile count is padded before decomposition ----

  // 32x96: Wt=3 padded to 4, so the network runs with stages=2.
  // CHECK-LABEL: func @decompose_sort_nonpow2
  func.func @decompose_sort_nonpow2(%arg0: tensor<32x96xf32>) -> (tensor<32x96xf32>, tensor<32x96xsi32>) {
    // CHECK-NOT: d2m.sort_block
    // CHECK: d2m.arange_block
    // CHECK-SAME: num_elements = 128
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge
    // CHECK-SAME: k = 64
    // CHECK-NOT: d2m.tile_topk_rebuild
    %values, %indices = "ttir.sort"(%arg0) <{dim = -1 : si32, descending = false, stable = false}> : (tensor<32x96xf32>) -> (tensor<32x96xf32>, tensor<32x96xsi32>)
    return %values, %indices : tensor<32x96xf32>, tensor<32x96xsi32>
  }

  // ---- dim=0 ----

  // The reduction runs down tile rows, so the flat tile stride differs but the
  // emitted network is the same.
  // CHECK-LABEL: func @decompose_sort_dim0
  func.func @decompose_sort_dim0(%arg0: tensor<256x32xf32>) -> (tensor<256x32xf32>, tensor<256x32xsi32>) {
    // CHECK-NOT: d2m.sort_block
    // CHECK: d2m.arange_block
    // CHECK: d2m.tile_topk_local_sort
    // CHECK: d2m.tile_topk_merge
    // CHECK-SAME: k = 64
    // CHECK-NOT: d2m.tile_topk_rebuild
    %values, %indices = "ttir.sort"(%arg0) <{dim = 0 : si32, descending = false, stable = false}> : (tensor<256x32xf32>) -> (tensor<256x32xf32>, tensor<256x32xsi32>)
    return %values, %indices : tensor<256x32xf32>, tensor<256x32xsi32>
  }
}
