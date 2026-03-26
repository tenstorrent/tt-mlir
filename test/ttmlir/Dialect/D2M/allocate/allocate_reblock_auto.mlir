// RUN: ttmlir-opt --ttcore-register-device "--d2m-allocate=test-assume-l1-capacity=8388608 test-buffer-size-policy=max" -o %t.max %s
// RUN: FileCheck %s --check-prefix=CHECK-MAX --input-file=%t.max
// RUN: ttmlir-opt --ttcore-register-device "--d2m-allocate=test-assume-l1-capacity=8388608" -o %t.auto %s
// RUN: FileCheck %s --check-prefix=CHECK-AUTO --input-file=%t.auto
// RUN: ttmlir-opt --ttcore-register-device "--d2m-allocate=test-assume-l1-capacity=8388608 test-allow-aliased-eltwise-blocking=true" -o %t.override %s
// RUN: FileCheck %s --check-prefix=CHECK-OVERRIDE --input-file=%t.override

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#mapL = affine_map<(d0, d1, d2) -> (d0, d2)>
#mapR = affine_map<(d0, d1, d2) -> (d2, d1)>
#mapO = affine_map<(d0, d1, d2) -> (d0, d1)>
#eltwise = affine_map<(d0, d1) -> (d0, d1)>
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
  // CHECK-AUTO: memref.alloc() {{.*}} : memref<16x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
  // CHECK-AUTO: memref.alloc() {{.*}} : memref<1x16x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<65536x4096, 2>, #l1>
  func.func @matmul_auto_vs_max() -> memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1> {
    %lhs = memref.alloc() : memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>
    %rhs = memref.alloc() : memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>
    %out = memref.alloc() : memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>
    d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<compute>]}
        ins(%lhs, %rhs : memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>, memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>)
        outs(%out : memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>) {
    ^compute0():
      %0 = memref.alloc() : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
      %1 = memref.alloc() : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
      %2 = memref.alloc() : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
      "d2m.tile_matmul_block"(%0, %1, %2) : (memref<16x16x!ttcore.tile<32x32, f32>, #l1>, memref<16x16x!ttcore.tile<32x32, f32>, #l1>, memref<16x16x!ttcore.tile<32x32, f32>, #l1>) -> ()
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
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#reduceIn, #broadcast, #reduceOut], iterator_types = [#parallel, #reduction], threads = [#d2m.thread<compute>]}
        ins(%in, %broadcast_in : memref<1x1x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x32768, 1>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
        outs(%out : memref<1x1x4x1x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    ^compute0():
      %tmp_in = memref.alloc() : memref<4x8x!ttcore.tile<32x32, f32>, #l1>
      %tmp_broadcast = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %tmp_out = memref.alloc() : memref<4x1x!ttcore.tile<32x32, f32>, #l1>
      %c0 = arith.constant 0 : index
      %t0 = memref.load %tmp_in[%c0, %c0] : memref<4x8x!ttcore.tile<32x32, f32>, #l1>
      %t1 = memref.load %tmp_broadcast[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %add = "d2m.tile_add"(%t0, %t1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    }
    return %out : memref<1x1x4x1x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  }

  // Reduction auto-reblocking is capped by minimum shard volume of 4 tiles (block factors selected are [1, 1, 2]).
  // CHECK-AUTO-LABEL: func.func @auto_clamps_reduction_factor_at_four_tiles()
  // CHECK-AUTO: d2m.view_layout {{.*}} -> memref<1x2x1x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  // CHECK-AUTO: d2m.view_layout {{.*}} -> memref<2x1x4x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
  // CHECK-AUTO: d2m.generic {block_factors = [1, 1, 2], grid = #ttcore.grid<1x1>
  // CHECK-AUTO: memref.alloc() {{.*}} : memref<1x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
  // CHECK-AUTO: memref.alloc() {{.*}} : memref<4x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
  func.func @auto_clamps_reduction_factor_at_four_tiles() -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> {
    %lhs = memref.alloc() : memref<1x1x1x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1>
    %rhs = memref.alloc() : memref<1x1x8x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x32768, 1>, #l1>
    %out = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<compute>]}
        ins(%lhs, %rhs : memref<1x1x1x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1>, memref<1x1x8x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x32768, 1>, #l1>)
        outs(%out : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>) {
    ^compute0():
      %0 = memref.alloc() : memref<1x8x!ttcore.tile<32x32, f32>, #l1>
      %1 = memref.alloc() : memref<8x1x!ttcore.tile<32x32, f32>, #l1>
      %2 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      "d2m.tile_matmul_block"(%0, %1, %2) : (memref<1x8x!ttcore.tile<32x32, f32>, #l1>, memref<8x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>) -> ()
    }
    return %out : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
  }

  // Multi reduction generics are currently not part of auto reblocking.
  // CHECK-AUTO-LABEL: func.func @multi_reduction_unchanged_under_auto()
  // CHECK-AUTO: d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>
  func.func @multi_reduction_unchanged_under_auto() -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> {
    %in = memref.alloc() : memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
    %out = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#multiReduceIn, #multiReduceOut], iterator_types = [#reduction, #reduction], threads = [#d2m.thread<compute>]}
        ins(%in : memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
        outs(%out : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>) {
    ^compute0():
      %tmp_in = memref.alloc() : memref<4x2x!ttcore.tile<32x32, f32>, #l1>
      %tmp_out = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %c0 = arith.constant 0 : index
      %t0 = memref.load %tmp_in[%c0, %c0] : memref<4x2x!ttcore.tile<32x32, f32>, #l1>
      %abs = "d2m.tile_abs"(%t0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    }
    return %out : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
  }

  // Auto reblocks all-parallel eltwise with CB-eligible output (block factors of [8, 2]).
  // CHECK-AUTO-LABEL: func.func @eltwise_auto_reblocks_with_output_view()
  // CHECK-AUTO: d2m.generic {block_factors = [8, 2], grid = #ttcore.grid<1x1>
  // CHECK-AUTO-COUNT-3: memref.alloc() {{.*}} : memref<1x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
  func.func @eltwise_auto_reblocks_with_output_view() -> memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #dram> {
    %lhs = memref.alloc() : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #l1>
    %rhs = memref.alloc() : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #l1>
    %out = memref.alloc() : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #dram>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#eltwise, #eltwise, #eltwise], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%lhs, %rhs : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #l1>, memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #l1>)
        outs(%out : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #dram>) {
    ^compute0():
      %tmp_in0 = memref.alloc() : memref<8x8x!ttcore.tile<32x32, f32>, #l1>
      %tmp_in1 = memref.alloc() : memref<8x8x!ttcore.tile<32x32, f32>, #l1>
      %tmp_out = memref.alloc() : memref<8x8x!ttcore.tile<32x32, f32>, #l1>
      %c0 = arith.constant 0 : index
      %t0 = memref.load %tmp_in0[%c0, %c0] : memref<8x8x!ttcore.tile<32x32, f32>, #l1>
      %t1 = memref.load %tmp_in1[%c0, %c0] : memref<8x8x!ttcore.tile<32x32, f32>, #l1>
      %add = "d2m.tile_add"(%t0, %t1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    }
    return %out : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #dram>
  }

  // Auto does not change block factors because reblocking would require turning aliased output into dedicated CB-backed output.
  // In override case, we enable block factors of [8, 2].
  // CHECK-AUTO-LABEL: func.func @eltwise_auto_preserves_aliased_output_by_default()
  // CHECK-AUTO: d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>
  // CHECK-OVERRIDE-LABEL: func.func @eltwise_auto_preserves_aliased_output_by_default()
  // CHECK-OVERRIDE: d2m.generic {block_factors = [8, 2], grid = #ttcore.grid<1x1>
  // CHECK-OVERRIDE-COUNT-3: memref.alloc() {{.*}} : memref<1x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
  func.func @eltwise_auto_preserves_aliased_output_by_default() -> memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #l1> {
    %lhs = memref.alloc() : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #l1>
    %rhs = memref.alloc() : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #l1>
    %out = memref.alloc() : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#eltwise, #eltwise, #eltwise], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%lhs, %rhs : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #l1>, memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #l1>)
        outs(%out : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #l1>) {
    ^compute0():
      %tmp_in0 = memref.alloc() : memref<8x8x!ttcore.tile<32x32, f32>, #l1>
      %tmp_in1 = memref.alloc() : memref<8x8x!ttcore.tile<32x32, f32>, #l1>
      %tmp_out = memref.alloc() : memref<8x8x!ttcore.tile<32x32, f32>, #l1>
      %c0 = arith.constant 0 : index
      %t0 = memref.load %tmp_in0[%c0, %c0] : memref<8x8x!ttcore.tile<32x32, f32>, #l1>
      %t1 = memref.load %tmp_in1[%c0, %c0] : memref<8x8x!ttcore.tile<32x32, f32>, #l1>
      %add = "d2m.tile_add"(%t0, %t1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    }
    return %out : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #l1>
  }

  // Auto reblocking applies to unary eltwise ops too.
  // CHECK-AUTO-LABEL: func.func @light_eltwise_auto_reblocks()
  // CHECK-AUTO: d2m.generic {block_factors = [8, 2], grid = #ttcore.grid<1x1>
  func.func @light_eltwise_auto_reblocks() -> memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #dram> {
    %in = memref.alloc() : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #l1>
    %out = memref.alloc() : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #dram>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#eltwise, #eltwise], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%in : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #l1>)
        outs(%out : memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x32768, 1>, #dram>) {
    ^compute0():
      %tmp_in = memref.alloc() : memref<8x8x!ttcore.tile<32x32, f32>, #l1>
      %tmp_out = memref.alloc() : memref<8x8x!ttcore.tile<32x32, f32>, #l1>
      %c0 = arith.constant 0 : index
      %t0 = memref.load %tmp_in[%c0, %c0] : memref<8x8x!ttcore.tile<32x32, f32>, #l1>
      %abs = "d2m.tile_abs"(%t0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
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
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#eltwise, #eltwise, #eltwise], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%lhs, %rhs : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x16384, 1>, #l1>, memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x16384, 1>, #l1>)
        outs(%out : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x16384, 1>, #dram>) {
    ^compute0():
      %tmp_in0 = memref.alloc() : memref<4x4x!ttcore.tile<32x32, f32>, #l1>
      %tmp_in1 = memref.alloc() : memref<4x4x!ttcore.tile<32x32, f32>, #l1>
      %tmp_out = memref.alloc() : memref<4x4x!ttcore.tile<32x32, f32>, #l1>
      %c0 = arith.constant 0 : index
      %t0 = memref.load %tmp_in0[%c0, %c0] : memref<4x4x!ttcore.tile<32x32, f32>, #l1>
      %t1 = memref.load %tmp_in1[%c0, %c0] : memref<4x4x!ttcore.tile<32x32, f32>, #l1>
      %add = "d2m.tile_add"(%t0, %t1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
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
    d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1x1>, indexing_maps = [#eltwise3d, #tail3d, #eltwise3d], iterator_types = [#parallel, #parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%full, %tail : memref<1x1x1x4x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x8192x8192, 1>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1>)
        outs(%out : memref<1x1x1x4x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x8192x8192, 1>, #dram>) {
    ^compute0():
      %tmp_in0 = memref.alloc() : memref<4x2x2x!ttcore.tile<32x32, f32>, #l1>
      %tmp_in1 = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1>
      %tmp_out = memref.alloc() : memref<4x2x2x!ttcore.tile<32x32, f32>, #l1>
      %c0 = arith.constant 0 : index
      %t0 = memref.load %tmp_in0[%c0, %c0, %c0] : memref<4x2x2x!ttcore.tile<32x32, f32>, #l1>
      %t1 = memref.load %tmp_in1[%c0, %c0] : memref<2x2x!ttcore.tile<32x32, f32>, #l1>
      %add = "d2m.tile_add"(%t0, %t1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    }
    return %out : memref<1x1x1x4x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x8192x8192, 1>, #dram>
  }
}
