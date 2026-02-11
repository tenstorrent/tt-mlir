// RUN: ttmlir-opt --ttcore-register-device --d2m-generic-affine-loop-fusion %s | FileCheck %s

#l1_ = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<() -> ()>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>
#mapL = affine_map<(d0, d1, d2) -> (d0, d2)>
#mapR = affine_map<(d0, d1, d2) -> (d2, d1)>
#mapO = affine_map<(d0, d1, d2) -> (d0, d1)>

// Test: Two generics with matching 2D loop nests and shared intermediate memref.
// Producer stores to %intermediate, consumer loads from it.
// Both have identical indices and matching loop structure.
// CHECK-LABEL: func.func @test_basic_fusion
// CHECK: d2m.generic
// CHECK-SAME: d2m.affine_fused
// Verify producer ops (load from dram stream) precede consumer ops in fused body.
// CHECK: d2m.remote_load{{.*}}#dram
// CHECK: d2m.remote_store
// CHECK: d2m.remote_load
// CHECK: d2m.remote_store
// CHECK-NOT: d2m.generic
func.func @test_basic_fusion(
    %input: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
  %intermediate = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
  %output = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
  %cb_in = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
  %stream_in = "d2m.stream_layout"(%input, %cb_in) : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>

  // Producer: reads from input, applies relu, stores to intermediate
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%stream_in : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>)
      outs(%intermediate : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
  ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
    %bf0 = d2m.get_block_factor(0) : index
    %bf1 = d2m.get_block_factor(1) : index
    affine.for %i = 0 to %bf0 {
      affine.for %j = 0 to %bf1 {
        %idx0 = d2m.block_index(0) : index
        %idx1 = d2m.block_index(1) : index
        %buf_in = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %loaded = d2m.remote_load %buf_in %stream_in[%idx0, %idx1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>

        %buf_out = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        linalg.generic {
          indexing_maps = [#map, #map],
          iterator_types = ["parallel", "parallel"]}
          ins(%loaded : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>)
          outs(%buf_out : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>)
        {
        ^bb0(%in_val: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
          %relu = "d2m.tile_relu"(%in_val) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          linalg.yield %relu : !ttcore.tile<32x32, f32>
        }

        %stored = d2m.remote_store %intermediate[%idx0, %idx1] %buf_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>, memref<2x4x!ttcore.tile<32x32, f32>, #l1_> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
      } {d2m.blocking_loop = 1}
    } {d2m.blocking_loop = 0}
  }

  // Consumer: reads from intermediate, applies exp, stores to output
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%intermediate : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
      outs(%output : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
  ^unified0(%cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb3: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
    %bf0 = d2m.get_block_factor(0) : index
    %bf1 = d2m.get_block_factor(1) : index
    affine.for %i = 0 to %bf0 {
      affine.for %j = 0 to %bf1 {
        %idx0 = d2m.block_index(0) : index
        %idx1 = d2m.block_index(1) : index
        %buf_in = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %loaded = d2m.remote_load %buf_in %intermediate[%idx0, %idx1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>

        %buf_out = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        linalg.generic {
          indexing_maps = [#map, #map],
          iterator_types = ["parallel", "parallel"]}
          ins(%loaded : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>)
          outs(%buf_out : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>)
        {
        ^bb0(%in_val: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
          %exp = "d2m.tile_exp"(%in_val) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          linalg.yield %exp : !ttcore.tile<32x32, f32>
        }

        %stored = d2m.remote_store %output[%idx0, %idx1] %buf_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>, memref<2x4x!ttcore.tile<32x32, f32>, #l1_> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
      } {d2m.blocking_loop = 1}
    } {d2m.blocking_loop = 0}
  }

  return
}

// Test: Multiple readers of intermediate should prevent fusion.
// CHECK-LABEL: func.func @test_no_fusion_multiple_readers
// CHECK: d2m.generic
// CHECK: d2m.generic
// CHECK: d2m.generic
func.func @test_no_fusion_multiple_readers(
    %input: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
  %intermediate = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
  %output1 = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
  %output2 = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
  %cb_in = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
  %stream_in = "d2m.stream_layout"(%input, %cb_in) : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>

  // Producer
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%stream_in : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>)
      outs(%intermediate : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
  ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
    %bf0 = d2m.get_block_factor(0) : index
    %bf1 = d2m.get_block_factor(1) : index
    affine.for %i = 0 to %bf0 {
      affine.for %j = 0 to %bf1 {
        %idx0 = d2m.block_index(0) : index
        %idx1 = d2m.block_index(1) : index
        %buf_in = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %loaded = d2m.remote_load %buf_in %stream_in[%idx0, %idx1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %buf_out = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %stored = d2m.remote_store %intermediate[%idx0, %idx1] %buf_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>, memref<2x4x!ttcore.tile<32x32, f32>, #l1_> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
      } {d2m.blocking_loop = 1}
    } {d2m.blocking_loop = 0}
  }

  // Consumer 1: reads from intermediate
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%intermediate : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
      outs(%output1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
  ^unified0(%cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb3: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
    %bf0 = d2m.get_block_factor(0) : index
    %bf1 = d2m.get_block_factor(1) : index
    affine.for %i = 0 to %bf0 {
      affine.for %j = 0 to %bf1 {
        %idx0 = d2m.block_index(0) : index
        %idx1 = d2m.block_index(1) : index
        %buf_in = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %loaded = d2m.remote_load %buf_in %intermediate[%idx0, %idx1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %buf_out = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %stored = d2m.remote_store %output1[%idx0, %idx1] %buf_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>, memref<2x4x!ttcore.tile<32x32, f32>, #l1_> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
      } {d2m.blocking_loop = 1}
    } {d2m.blocking_loop = 0}
  }

  // Consumer 2: also reads from intermediate — prevents fusion
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%intermediate : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
      outs(%output2 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
  ^unified0(%cb4: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb5: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
    %bf0 = d2m.get_block_factor(0) : index
    %bf1 = d2m.get_block_factor(1) : index
    affine.for %i = 0 to %bf0 {
      affine.for %j = 0 to %bf1 {
        %idx0 = d2m.block_index(0) : index
        %idx1 = d2m.block_index(1) : index
        %buf_in = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %loaded = d2m.remote_load %buf_in %intermediate[%idx0, %idx1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %buf_out = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %stored = d2m.remote_store %output2[%idx0, %idx1] %buf_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>, memref<2x4x!ttcore.tile<32x32, f32>, #l1_> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
      } {d2m.blocking_loop = 1}
    } {d2m.blocking_loop = 0}
  }

  return
}

// Test: Fuse matmul (MNK, 3 blocking loops) producer into binary add (MN, 2 blocking loops) consumer.
// Matmul is superset (3 loops), add is subset (2 loops).
// Fused generic should use matmul's block_factors [1, 1, 2].
// CHECK-LABEL: func.func @test_matmul_add_subset_fusion
// CHECK: d2m.generic
// CHECK-SAME: block_factors = [1, 1, 2]
// CHECK-SAME: d2m.affine_fused
// CHECK-NOT: d2m.generic
func.func @test_matmul_add_subset_fusion(
    %lhs: memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #dram>,
    %rhs: memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #dram>,
    %bias: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) {
  %intermediate = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>
  %output = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>
  %cb_lhs = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>
  %cb_rhs = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>
  %stream_lhs = "d2m.stream_layout"(%lhs, %cb_lhs) : (memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #dram>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) -> memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #dram>
  %stream_rhs = "d2m.stream_layout"(%rhs, %cb_rhs) : (memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #dram>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) -> memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #dram>

  // Producer: matmul with 3 blocking loops (M, N, K)
  d2m.generic {block_factors = [1, 1, 2], grid = #ttcore.grid<1x1>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<unified>]}
      ins(%stream_lhs, %stream_rhs : memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #dram>, memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #dram>)
      outs(%intermediate : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) {
  ^unified0(%cb0: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
    %bf0 = d2m.get_block_factor(0) : index
    %bf1 = d2m.get_block_factor(1) : index
    %bf2 = d2m.get_block_factor(2) : index
    affine.for %m = 0 to %bf0 {
      affine.for %n = 0 to %bf1 {
        %idx0 = d2m.block_index(0) : index
        %idx1 = d2m.block_index(1) : index

        %buf_out = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>

        affine.for %k = 0 to %bf2 {
          %idx2 = d2m.block_index(2) : index
          %buf_lhs = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
          %buf_rhs = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
          %loaded_lhs = d2m.remote_load %buf_lhs %stream_lhs[%idx0, %idx2]      : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #dram> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
          %loaded_rhs = d2m.remote_load %buf_rhs %stream_rhs[%idx2, %idx1]      : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #dram> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
          "d2m.tile_matmul_block"(%loaded_lhs, %loaded_rhs, %buf_out) : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> ()
        } {d2m.blocking_loop = 2}

        %stored = d2m.remote_store %intermediate[%idx0, %idx1] %buf_out : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_> -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>

      } {d2m.blocking_loop = 1}
    } {d2m.blocking_loop = 0}
  }

  // Consumer: binary add with 2 blocking loops (M, N)
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%intermediate, %bias : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>)
      outs(%output : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) {
  ^unified1(%cb3: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb4: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb5: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
    %abf0 = d2m.get_block_factor(0) : index
    %abf1 = d2m.get_block_factor(1) : index
    affine.for %am = 0 to %abf0 {
      affine.for %an = 0 to %abf1 {
        %aidx0 = d2m.block_index(0) : index
        %aidx1 = d2m.block_index(1) : index

        %buf_inter = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
        %loaded_inter = d2m.remote_load %buf_inter %intermediate[%aidx0, %aidx1] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>

        %buf_bias = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
        %loaded_bias = d2m.remote_load %buf_bias %bias[%aidx0, %aidx1] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>

        %buf_add_out = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>

        linalg.generic {
          indexing_maps = [#map, #map, #map],
          iterator_types = ["parallel", "parallel"]}
          ins(%buf_bias, %buf_inter : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
          outs(%buf_add_out : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
        {
        ^bb0(%in1_val: !ttcore.tile<32x32, f32>, %in2_val: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
          %add = "d2m.tile_add"(%in1_val, %in2_val) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          linalg.yield %add : !ttcore.tile<32x32, f32>
        }

        %stored_out = d2m.remote_store %output[%aidx0, %aidx1] %buf_add_out : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_> -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>
      } {d2m.blocking_loop = 1}
    } {d2m.blocking_loop = 0}
  }

  return
}
