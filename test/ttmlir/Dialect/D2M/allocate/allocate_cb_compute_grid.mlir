// RUN: ttmlir-opt --ttcore-register-device "--d2m-allocate=stream-insert-policy=always" -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verify that CBLayoutAttr on internal allocs uses the generic's compute
// grid shape, not the per-operand stream buffer grid shape.
// The matmul has a 2x4 compute grid.  LHS is 2x6 (grid 2x1, shard 4x6),
// RHS is 6x4 (grid 1x4, shard 6x8).  After stream insertion, the CB allocs
// for operands 0 and 1 should both carry grid = [2x4] (compute grid).

#l1 = #ttcore.memory_space<l1>
#mapL = affine_map<(d0, d1, d2) -> (d0, d2)>
#mapR = affine_map<(d0, d1, d2) -> (d2, d1)>
#mapO = affine_map<(d0, d1, d2) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>

module {
  // CHECK-LABEL: func @matmul_cb_compute_grid
  func.func @matmul_cb_compute_grid() {
    %lhs = memref.alloc() : memref<2x6x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096, 1>, #l1>
    %rhs = memref.alloc() : memref<6x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1>
    %out = memref.alloc() : memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1>

    // CHECK: d2m.generic
    // Both input CB allocs should use the compute grid [2x4]:
    // CHECK: cb_layout<{{[0-9x]+}}, 2, grid = [2x4]>
    // CHECK: cb_layout<{{[0-9x]+}}, 2, grid = [2x4]>
    d2m.generic {block_factors = [1, 1, 6], grid = #ttcore.grid<2x4>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<unified>]}
        ins(%lhs, %rhs : memref<2x6x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096, 1>, #l1>, memref<6x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1>)
        outs(%out : memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1>) {
    ^unified0():
      %bf0 = d2m.get_block_factor(0) : index
      %bf1 = d2m.get_block_factor(1) : index
      %bf2 = d2m.get_block_factor(2) : index
      affine.for %i = 0 to %bf0 {
        affine.for %j = 0 to %bf1 {
          affine.for %k = 0 to %bf2 {
            %buf_lhs = memref.alloc() : memref<4x6x!ttcore.tile<32x32, f32>, #l1>
            %0 = d2m.remote_load %buf_lhs %lhs[%i, %k] : memref<4x6x!ttcore.tile<32x32, f32>, #l1>, memref<2x6x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096, 1>, #l1> -> memref<4x6x!ttcore.tile<32x32, f32>, #l1>
            %buf_rhs = memref.alloc() : memref<6x8x!ttcore.tile<32x32, f32>, #l1>
            %1 = d2m.remote_load %buf_rhs %rhs[%k, %j] : memref<6x8x!ttcore.tile<32x32, f32>, #l1>, memref<6x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1> -> memref<6x8x!ttcore.tile<32x32, f32>, #l1>
            %buf_out = memref.alloc() : memref<4x8x!ttcore.tile<32x32, f32>, #l1>
            "d2m.tile_matmul_block"(%0, %1, %buf_out) : (memref<4x6x!ttcore.tile<32x32, f32>, #l1>, memref<6x8x!ttcore.tile<32x32, f32>, #l1>, memref<4x8x!ttcore.tile<32x32, f32>, #l1>) -> ()
            %2 = d2m.remote_store %out[%i, %j] %buf_out : memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1>, memref<4x8x!ttcore.tile<32x32, f32>, #l1> -> memref<4x8x!ttcore.tile<32x32, f32>, #l1>
          } {d2m.blocking_loop = 2}
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }
}
