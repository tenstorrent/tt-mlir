// RUN: ttmlir-opt --ttcore-register-device "--d2m-reblock-generics=test-buffer-size-policy=max" "--d2m-allocate=test-assume-l1-capacity=8388608" -o %t.max %s
// RUN: FileCheck %s --check-prefix=CHECK-MAX --input-file=%t.max
// RUN: ttmlir-opt --ttcore-register-device --d2m-reblock-generics "--d2m-allocate=test-assume-l1-capacity=8388608" -o %t.auto %s
// RUN: FileCheck %s --check-prefix=CHECK-AUTO --input-file=%t.auto

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#mapL = affine_map<(d0, d1, d2) -> (d0, d2)>
#mapR = affine_map<(d0, d1, d2) -> (d2, d1)>
#mapO = affine_map<(d0, d1, d2) -> (d0, d1)>
#eltwise = affine_map<(d0, d1) -> (d0, d1)>
#eltwisePermuted = affine_map<(d0, d1) -> (d1, d0)>
#eltwise3d = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#tail3d = affine_map<(d0, d1, d2) -> (d1, d2)>
#broadcast = affine_map<(d0, d1) -> (0, 0)>
#reduceIn = affine_map<(d0, d1) -> (d0, d1)>
#reduceOut = affine_map<(d0, d1) -> (d0, 0)>
#multiReduceIn = affine_map<(d0, d1) -> (d0, d1)>
#multiReduceOut = affine_map<(d0, d1) -> (0, 0)>
#swapShards2d = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#swapShards3d = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d5, d4)>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>

module {
  // Auto reblocks reduction dimension, max leaves block factors unchanged.
  // CHECK-MAX-LABEL: func.func @matmul_auto_vs_max()
  // CHECK-MAX: memref.alloc() {{.*}} : memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>
  // CHECK-MAX: memref.alloc() {{.*}} : memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>
  // CHECK-MAX: d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>
  // CHECK-AUTO-LABEL: func.func @matmul_auto_vs_max()
  // CHECK-AUTO: d2m.view_layout {{.*}} -> memref<1x16x16x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
  // CHECK-AUTO: d2m.view_layout {{.*}} -> memref<16x1x1x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>
  // CHECK-AUTO: d2m.generic {block_factors = [1, 1, 16], grid = #ttcore.grid<1x1>
  // CHECK-AUTO: memref.alloc() {{.*}} : memref<16x1x!ttcore.tile<32x32, f32>, {{.*}}#l1>
  // CHECK-AUTO: memref.alloc() {{.*}} : memref<1x16x!ttcore.tile<32x32, f32>, {{.*}}#l1>
  func.func @matmul_auto_vs_max() -> memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1> {
    %lhs = memref.alloc() : memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>
    %rhs = memref.alloc() : memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>
    %out = memref.alloc() : memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>
    d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<unified>]}
        ins(%lhs, %rhs : memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>, memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>)
        outs(%out : memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>) {
    ^unified0():
      %bf0 = d2m.get_block_factor(0) : index
      %bf1 = d2m.get_block_factor(1) : index
      %bf2 = d2m.get_block_factor(2) : index
      affine.for %iter0 = 0 to %bf0 {
        affine.for %iter1 = 0 to %bf1 {
          affine.for %iter2 = 0 to %bf2 {
            %0 = memref.alloc() {d2m.synchronized_buffer = 2} : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
            d2m.remote_load %0 %lhs[%iter0, %iter2] : memref<16x16x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>
            %1 = memref.alloc() {d2m.synchronized_buffer = 2} : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
            d2m.remote_load %1 %rhs[%iter2, %iter1] : memref<16x16x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>
            %2 = memref.alloc() {d2m.synchronized_buffer = 2} : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0, %1 : memref<16x16x!ttcore.tile<32x32, f32>, #l1>, memref<16x16x!ttcore.tile<32x32, f32>, #l1>) outs(%2 : memref<16x16x!ttcore.tile<32x32, f32>, #l1>) {
            ^bb0(%lhs_elem: !ttcore.tile<32x32, f32>, %rhs_elem: !ttcore.tile<32x32, f32>, %out_elem: !ttcore.tile<32x32, f32>):
              %9 = "d2m.tile_matmul"(%lhs_elem, %rhs_elem, %out_elem) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              linalg.yield %9 : !ttcore.tile<32x32, f32>
            }
            d2m.remote_store %out[%iter0, %iter1] %2 : memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>, memref<16x16x!ttcore.tile<32x32, f32>, #l1>
          } {d2m.blocking_loop = 2}
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return %out : memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>
  }

  // Non-matmul single reduction still exercises the single-reduction auto path.
  // CHECK-MAX-LABEL: func.func @single_reduction_non_matmul()
  // CHECK-MAX: d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>
  // CHECK-AUTO-LABEL: func.func @single_reduction_non_matmul()
  // CHECK-AUTO: d2m.generic {block_factors = [1, 8], grid = #ttcore.grid<1x1>
  func.func @single_reduction_non_matmul() -> memref<1x1x4x1x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1> {
    %in = memref.alloc() : memref<1x1x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x32768, 1>, #l1>
    %broadcast_in = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %out = memref.alloc() : memref<1x1x4x1x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#reduceIn, #broadcast, #reduceOut], iterator_types = [#parallel, #reduction], threads = [#d2m.thread<unified>]}
        ins(%in, %broadcast_in : memref<1x1x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x32768, 1>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
        outs(%out : memref<1x1x4x1x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    ^unified0():
      %bf0 = d2m.get_block_factor(0) : index
      %bf1 = d2m.get_block_factor(1) : index
      affine.for %iter0 = 0 to %bf0 {
        affine.for %iter1 = 0 to %bf1 {
          %tmp_in = memref.alloc() {d2m.synchronized_buffer = 2} : memref<4x8x!ttcore.tile<32x32, f32>, #l1>
          d2m.remote_load %tmp_in %in[%iter0, %iter1] : memref<4x8x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x32768, 1>, #l1>
          %tmp_broadcast = memref.alloc() {d2m.synchronized_buffer = 2} : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
          d2m.remote_load %tmp_broadcast %broadcast_in[%iter0, %iter1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
          %tmp_out = memref.alloc() {d2m.synchronized_buffer = 2} : memref<4x1x!ttcore.tile<32x32, f32>, #l1>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, 0)>, affine_map<(d0, d1) -> (d0, 0)>], iterator_types = ["parallel", "reduction"]} ins(%tmp_in, %tmp_broadcast : memref<4x8x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>) outs(%tmp_out : memref<4x1x!ttcore.tile<32x32, f32>, #l1>) {
          ^bb0(%in0: !ttcore.tile<32x32, f32>, %in1: !ttcore.tile<32x32, f32>, %out_elem: !ttcore.tile<32x32, f32>):
            %add = "d2m.tile_add"(%in0, %in1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %add : !ttcore.tile<32x32, f32>
          }
          d2m.remote_store %out[%iter0, %iter1] %tmp_out : memref<1x1x4x1x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>, memref<4x1x!ttcore.tile<32x32, f32>, #l1>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return %out : memref<1x1x4x1x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  }

  // Reduction auto-reblocking is capped by minimum shard volume of 4 tiles (block factors selected are [1, 1, 2]).
  // CHECK-AUTO-LABEL: func.func @auto_clamps_reduction_factor_at_four_tiles()
  // CHECK-AUTO: d2m.view_layout {{.*}} -> memref<1x2x1x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  // CHECK-AUTO: d2m.view_layout {{.*}} -> memref<2x1x4x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
  // CHECK-AUTO: d2m.generic {block_factors = [1, 1, 2], grid = #ttcore.grid<1x1>
  // CHECK-AUTO: memref.alloc() {{.*}} : memref<1x4x!ttcore.tile<32x32, f32>, {{.*}}#l1>
  // CHECK-AUTO: memref.alloc() {{.*}} : memref<4x1x!ttcore.tile<32x32, f32>, {{.*}}#l1>
  func.func @auto_clamps_reduction_factor_at_four_tiles() -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> {
    %lhs = memref.alloc() : memref<1x1x1x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1>
    %rhs = memref.alloc() : memref<1x1x8x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x32768, 1>, #l1>
    %out = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<unified>]}
        ins(%lhs, %rhs : memref<1x1x1x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1>, memref<1x1x8x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x32768, 1>, #l1>)
        outs(%out : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>) {
    ^unified0():
      %bf0 = d2m.get_block_factor(0) : index
      %bf1 = d2m.get_block_factor(1) : index
      %bf2 = d2m.get_block_factor(2) : index
      affine.for %iter0 = 0 to %bf0 {
        affine.for %iter1 = 0 to %bf1 {
          affine.for %iter2 = 0 to %bf2 {
            %0 = memref.alloc() {d2m.synchronized_buffer = 2} : memref<1x8x!ttcore.tile<32x32, f32>, #l1>
            d2m.remote_load %0 %lhs[%iter0, %iter2] : memref<1x8x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x1x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1>
            %1 = memref.alloc() {d2m.synchronized_buffer = 2} : memref<8x1x!ttcore.tile<32x32, f32>, #l1>
            d2m.remote_load %1 %rhs[%iter2, %iter1] : memref<8x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x8x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x32768, 1>, #l1>
            %2 = memref.alloc() {d2m.synchronized_buffer = 2} : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0, %1 : memref<1x8x!ttcore.tile<32x32, f32>, #l1>, memref<8x1x!ttcore.tile<32x32, f32>, #l1>) outs(%2 : memref<1x1x!ttcore.tile<32x32, f32>, #l1>) {
            ^bb0(%lhs_elem: !ttcore.tile<32x32, f32>, %rhs_elem: !ttcore.tile<32x32, f32>, %out_elem: !ttcore.tile<32x32, f32>):
              %9 = "d2m.tile_matmul"(%lhs_elem, %rhs_elem, %out_elem) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              linalg.yield %9 : !ttcore.tile<32x32, f32>
            }
            d2m.remote_store %out[%iter0, %iter1] %2 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>
          } {d2m.blocking_loop = 2}
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return %out : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
  }

  // Multi reduction generics are currently not part of auto reblocking.
  // CHECK-AUTO-LABEL: func.func @multi_reduction_unchanged_under_auto()
  // CHECK-AUTO: d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>
  func.func @multi_reduction_unchanged_under_auto() -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> {
    %in = memref.alloc() : memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
    %out = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#multiReduceIn, #multiReduceOut], iterator_types = [#reduction, #reduction], threads = [#d2m.thread<unified>]}
        ins(%in : memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
        outs(%out : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>) {
    ^unified0():
      %bf0 = d2m.get_block_factor(0) : index
      %bf1 = d2m.get_block_factor(1) : index
      affine.for %iter0 = 0 to %bf0 {
        affine.for %iter1 = 0 to %bf1 {
          %tmp_in = memref.alloc() {d2m.synchronized_buffer = 2} : memref<4x2x!ttcore.tile<32x32, f32>, #l1>
          d2m.remote_load %tmp_in %in[%iter0, %iter1] : memref<4x2x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
          %tmp_out = memref.alloc() {d2m.synchronized_buffer = 2} : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, 0)>], iterator_types = ["reduction", "reduction"]} ins(%tmp_in : memref<4x2x!ttcore.tile<32x32, f32>, #l1>) outs(%tmp_out : memref<1x1x!ttcore.tile<32x32, f32>, #l1>) {
          ^bb0(%in_elem: !ttcore.tile<32x32, f32>, %out_elem: !ttcore.tile<32x32, f32>):
            %abs = "d2m.tile_abs"(%in_elem) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %abs : !ttcore.tile<32x32, f32>
          }
          d2m.remote_store %out[%iter0, %iter1] %tmp_out : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return %out : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
  }

  // Auto reblocks all-parallel eltwise with CB-eligible output (block factors of [8, 2]).
  // CHECK-AUTO-LABEL: func.func @eltwise_auto_reblocks_with_output_view()
  // CHECK-AUTO: d2m.generic {block_factors = [8, 2], grid = #ttcore.grid<1x1>
  // CHECK-AUTO-COUNT-3: memref.alloc(){{.*}} : memref<1x4x!ttcore.tile<32x32, f32>, {{.*}}#l1>
  func.func @eltwise_auto_reblocks_with_output_view() -> memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #dram> {
    %lhs = memref.alloc() : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #l1>
    %rhs = memref.alloc() : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #l1>
    %out = memref.alloc() : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #dram>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#eltwise, #eltwise, #eltwise], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%lhs, %rhs : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #l1>, memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #l1>)
        outs(%out : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #dram>) {
    ^unified0():
      %bf0 = d2m.get_block_factor(0) : index
      %bf1 = d2m.get_block_factor(1) : index
      affine.for %iter0 = 0 to %bf0 {
        affine.for %iter1 = 0 to %bf1 {
          %tmp_in0 = memref.alloc() {d2m.synchronized_buffer = 2} : memref<8x8x!ttcore.tile<32x32, f32>, #l1>
          d2m.remote_load %tmp_in0 %lhs[%iter0, %iter1] : memref<8x8x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #l1>
          %tmp_in1 = memref.alloc() {d2m.synchronized_buffer = 2} : memref<8x8x!ttcore.tile<32x32, f32>, #l1>
          d2m.remote_load %tmp_in1 %rhs[%iter0, %iter1] : memref<8x8x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #l1>
          %tmp_out = memref.alloc() {d2m.synchronized_buffer = 2} : memref<8x8x!ttcore.tile<32x32, f32>, #l1>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%tmp_in0, %tmp_in1 : memref<8x8x!ttcore.tile<32x32, f32>, #l1>, memref<8x8x!ttcore.tile<32x32, f32>, #l1>) outs(%tmp_out : memref<8x8x!ttcore.tile<32x32, f32>, #l1>) {
          ^bb0(%in0: !ttcore.tile<32x32, f32>, %in1: !ttcore.tile<32x32, f32>, %out_elem: !ttcore.tile<32x32, f32>):
            %add = "d2m.tile_add"(%in0, %in1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %add : !ttcore.tile<32x32, f32>
          }
          d2m.remote_store %out[%iter0, %iter1] %tmp_out : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #dram>, memref<8x8x!ttcore.tile<32x32, f32>, #l1>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return %out : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #dram>
  }

  // Auto reblocks to [8, 2] while preserving the L1 output alias.
  // CHECK-AUTO-LABEL: func.func @eltwise_auto_preserves_aliased_output_by_default()
  // CHECK-AUTO: d2m.generic {block_factors = [8, 2], grid = #ttcore.grid<1x1>
  // CHECK-AUTO-COUNT-2: memref.alloc(){{.*}} : memref<1x4x!ttcore.tile<32x32, f32>, {{.*}}#l1>
  func.func @eltwise_auto_preserves_aliased_output_by_default() -> memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #l1> {
    %lhs = memref.alloc() : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #l1>
    %rhs = memref.alloc() : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #l1>
    %out = memref.alloc() : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#eltwise, #eltwise, #eltwise], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%lhs, %rhs : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #l1>, memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #l1>)
        outs(%out : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #l1>) {
    ^unified0():
      %bf0 = d2m.get_block_factor(0) : index
      %bf1 = d2m.get_block_factor(1) : index
      affine.for %iter0 = 0 to %bf0 {
        affine.for %iter1 = 0 to %bf1 {
          %tmp_in0 = memref.alloc() {d2m.synchronized_buffer = 2} : memref<8x8x!ttcore.tile<32x32, f32>, #l1>
          d2m.remote_load %tmp_in0 %lhs[%iter0, %iter1] : memref<8x8x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #l1>
          %tmp_in1 = memref.alloc() {d2m.synchronized_buffer = 2} : memref<8x8x!ttcore.tile<32x32, f32>, #l1>
          d2m.remote_load %tmp_in1 %rhs[%iter0, %iter1] : memref<8x8x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #l1>
          %tmp_out = memref.alloc() {d2m.synchronized_buffer = 2} : memref<8x8x!ttcore.tile<32x32, f32>, #l1>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%tmp_in0, %tmp_in1 : memref<8x8x!ttcore.tile<32x32, f32>, #l1>, memref<8x8x!ttcore.tile<32x32, f32>, #l1>) outs(%tmp_out : memref<8x8x!ttcore.tile<32x32, f32>, #l1>) {
          ^bb0(%in0: !ttcore.tile<32x32, f32>, %in1: !ttcore.tile<32x32, f32>, %out_elem: !ttcore.tile<32x32, f32>):
            %add = "d2m.tile_add"(%in0, %in1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %add : !ttcore.tile<32x32, f32>
          }
          d2m.remote_store %out[%iter0, %iter1] %tmp_out : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #l1>, memref<8x8x!ttcore.tile<32x32, f32>, #l1>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return %out : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #l1>
  }

  // Auto reblocking applies to unary eltwise ops too.
  // CHECK-AUTO-LABEL: func.func @light_eltwise_auto_reblocks()
  // CHECK-AUTO: d2m.generic {block_factors = [8, 2], grid = #ttcore.grid<1x1>
  func.func @light_eltwise_auto_reblocks() -> memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #dram> {
    %in = memref.alloc() : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #l1>
    %out = memref.alloc() : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #dram>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#eltwise, #eltwise], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%in : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #l1>)
        outs(%out : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #dram>) {
    ^unified0():
      %bf0 = d2m.get_block_factor(0) : index
      %bf1 = d2m.get_block_factor(1) : index
      affine.for %iter0 = 0 to %bf0 {
        affine.for %iter1 = 0 to %bf1 {
          %tmp_in = memref.alloc() {d2m.synchronized_buffer = 2} : memref<8x8x!ttcore.tile<32x32, f32>, #l1>
          d2m.remote_load %tmp_in %in[%iter0, %iter1] : memref<8x8x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #l1>
          %tmp_out = memref.alloc() {d2m.synchronized_buffer = 2} : memref<8x8x!ttcore.tile<32x32, f32>, #l1>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%tmp_in : memref<8x8x!ttcore.tile<32x32, f32>, #l1>) outs(%tmp_out : memref<8x8x!ttcore.tile<32x32, f32>, #l1>) {
          ^bb0(%in_elem: !ttcore.tile<32x32, f32>, %out_elem: !ttcore.tile<32x32, f32>):
            %abs = "d2m.tile_abs"(%in_elem) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %abs : !ttcore.tile<32x32, f32>
          }
          d2m.remote_store %out[%iter0, %iter1] %tmp_out : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #dram>, memref<8x8x!ttcore.tile<32x32, f32>, #l1>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return %out : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #dram>
  }

  // Check 4 tile restriction for reblocking of eltwise.
  // CHECK-AUTO-LABEL: func.func @eltwise_auto_clamps_blocking_at_four_tiles()
  // CHECK-AUTO: d2m.generic {block_factors = [4, 1], grid = #ttcore.grid<1x1>
  func.func @eltwise_auto_clamps_blocking_at_four_tiles() -> memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x16384, 1>, #dram> {
    %lhs = memref.alloc() : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x16384, 1>, #l1>
    %rhs = memref.alloc() : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x16384, 1>, #l1>
    %out = memref.alloc() : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x16384, 1>, #dram>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#eltwise, #eltwise, #eltwise], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%lhs, %rhs : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x16384, 1>, #l1>, memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x16384, 1>, #l1>)
        outs(%out : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x16384, 1>, #dram>) {
    ^unified0():
      %bf0 = d2m.get_block_factor(0) : index
      %bf1 = d2m.get_block_factor(1) : index
      affine.for %iter0 = 0 to %bf0 {
        affine.for %iter1 = 0 to %bf1 {
          %tmp_in0 = memref.alloc() {d2m.synchronized_buffer = 2} : memref<4x4x!ttcore.tile<32x32, f32>, #l1>
          d2m.remote_load %tmp_in0 %lhs[%iter0, %iter1] : memref<4x4x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x16384, 1>, #l1>
          %tmp_in1 = memref.alloc() {d2m.synchronized_buffer = 2} : memref<4x4x!ttcore.tile<32x32, f32>, #l1>
          d2m.remote_load %tmp_in1 %rhs[%iter0, %iter1] : memref<4x4x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x16384, 1>, #l1>
          %tmp_out = memref.alloc() {d2m.synchronized_buffer = 2} : memref<4x4x!ttcore.tile<32x32, f32>, #l1>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%tmp_in0, %tmp_in1 : memref<4x4x!ttcore.tile<32x32, f32>, #l1>, memref<4x4x!ttcore.tile<32x32, f32>, #l1>) outs(%tmp_out : memref<4x4x!ttcore.tile<32x32, f32>, #l1>) {
          ^bb0(%in0: !ttcore.tile<32x32, f32>, %in1: !ttcore.tile<32x32, f32>, %out_elem: !ttcore.tile<32x32, f32>):
            %add = "d2m.tile_add"(%in0, %in1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %add : !ttcore.tile<32x32, f32>
          }
          d2m.remote_store %out[%iter0, %iter1] %tmp_out : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x16384, 1>, #dram>, memref<4x4x!ttcore.tile<32x32, f32>, #l1>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return %out : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x16384, 1>, #dram>
  }

  // Only operands whose indexing maps depend on blocked dims are affected by the candidate.
  // CHECK-AUTO-LABEL: func.func @eltwise_only_operands_using_blocked_dims_reblock()
  // CHECK-AUTO: d2m.generic {block_factors = [4, 1, 1], grid = #ttcore.grid<1x1x1>
  func.func @eltwise_only_operands_using_blocked_dims_reblock() -> memref<1x1x1x4x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x8192x8192, 1>, #dram> {
    %full = memref.alloc() : memref<1x1x1x4x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x8192x8192, 1>, #l1>
    %tail = memref.alloc() : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1>
    %out = memref.alloc() : memref<1x1x1x4x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x8192x8192, 1>, #dram>
    d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1x1>, indexing_maps = [#eltwise3d, #tail3d, #eltwise3d], iterator_types = [#parallel, #parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%full, %tail : memref<1x1x1x4x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x8192x8192, 1>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1>)
        outs(%out : memref<1x1x1x4x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x8192x8192, 1>, #dram>) {
    ^unified0():
      %bf0 = d2m.get_block_factor(0) : index
      %bf1 = d2m.get_block_factor(1) : index
      %bf2 = d2m.get_block_factor(2) : index
      affine.for %iter0 = 0 to %bf0 {
        affine.for %iter1 = 0 to %bf1 {
          affine.for %iter2 = 0 to %bf2 {
            %tmp_in0 = memref.alloc() {d2m.synchronized_buffer = 2} : memref<4x2x2x!ttcore.tile<32x32, f32>, #l1>
            d2m.remote_load %tmp_in0 %full[%iter0, %iter1, %iter2] : memref<4x2x2x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x1x4x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x8192x8192, 1>, #l1>
            %tmp_in1 = memref.alloc() {d2m.synchronized_buffer = 2} : memref<2x2x!ttcore.tile<32x32, f32>, #l1>
            d2m.remote_load %tmp_in1 %tail[%iter1, %iter2] : memref<2x2x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1>
            %tmp_out = memref.alloc() {d2m.synchronized_buffer = 2} : memref<4x2x2x!ttcore.tile<32x32, f32>, #l1>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%tmp_in0, %tmp_in1 : memref<4x2x2x!ttcore.tile<32x32, f32>, #l1>, memref<2x2x!ttcore.tile<32x32, f32>, #l1>) outs(%tmp_out : memref<4x2x2x!ttcore.tile<32x32, f32>, #l1>) {
            ^bb0(%in0: !ttcore.tile<32x32, f32>, %in1: !ttcore.tile<32x32, f32>, %out_elem: !ttcore.tile<32x32, f32>):
              %add = "d2m.tile_add"(%in0, %in1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              linalg.yield %add : !ttcore.tile<32x32, f32>
            }
            d2m.remote_store %out[%iter0, %iter1, %iter2] %tmp_out : memref<1x1x1x4x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x8192x8192, 1>, #dram>, memref<4x2x2x!ttcore.tile<32x32, f32>, #l1>
          } {d2m.blocking_loop = 2}
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return %out : memref<1x1x1x4x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x8192x8192, 1>, #dram>
  }
}
