// Post-loop-generation isolation test. Input is hand-crafted post-fusion +
// post-bufferization + post-scratch + post-interchange IR. The test
// exercises ONLY `--d2m-generate-outer-loops`, which materializes the
// per-block-factor `affine.for` nests around the body of the fused
// d2m.generic. The K-reduction loop must contain the row-major load,
// `tile_tilize_block`, and `linalg.generic` (which still hosts
// `tile_matmul`) in that order.

// RUN: ttmlir-opt --ttcore-register-device --d2m-generate-outer-loops -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1 = #ttcore.memory_space<l1>
#mapA = affine_map<(d0, d1, d2) -> (d0, d2)>
#mapB = affine_map<(d0, d1, d2) -> (d2, d1)>
#mapC = affine_map<(d0, d1, d2) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>

// CHECK-LABEL: func.func @rm_fed_matmul_post_loops
// Three `affine.for` nests materialized (M / N / K block factors).
// CHECK: affine.for
// CHECK: affine.for
// CHECK: affine.for
// Row-major `remote_load` → `tile_tilize_block` → `linalg.generic` /
// `tile_matmul` must stack inside the innermost loop, in that order, so
// the unpacker re-tilizes the K iteration's row-major block before each
// matmul tile call.
// CHECK: d2m.remote_load
// CHECK-SAME: memref<128x96xf32>
// CHECK: d2m.tile_tilize_block
// CHECK-SAME: (memref<128x96xf32>, memref<4x3x!ttcore.tile<32x32, f32>>)
// CHECK: linalg.generic
// CHECK: d2m.tile_matmul
func.func @rm_fed_matmul_post_loops(%arg0: memref<1x1x128x96xf32, #ttcore.shard<384x4, 1>, #l1>) -> memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1> {
  %alloc = memref.alloc() : memref<1x1x3x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
  %alloc_0 = memref.alloc() : memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
  d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#mapA, #mapB, #mapC], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<unified>]}
      ins(%arg0, %alloc : memref<1x1x128x96xf32, #ttcore.shard<384x4, 1>, #l1>, memref<1x1x3x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
      outs(%alloc_0 : memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
   {
    %b0 = d2m.block_index(0) : index
    %b1 = d2m.block_index(1) : index
    %b2 = d2m.block_index(2) : index
    %rm_local = memref.alloc() {alignment = 64 : i64} : memref<128x96xf32>
    d2m.remote_load %rm_local %arg0[%b0, %b2] : memref<128x96xf32>, memref<1x1x128x96xf32, #ttcore.shard<384x4, 1>, #l1>
    %tile_a = memref.alloc() {alignment = 64 : i64} : memref<4x3x!ttcore.tile<32x32, f32>>
    "d2m.tile_tilize_block"(%rm_local, %tile_a) : (memref<128x96xf32>, memref<4x3x!ttcore.tile<32x32, f32>>) -> ()
    %tile_b = memref.alloc() {alignment = 64 : i64} : memref<3x2x!ttcore.tile<32x32, f32>>
    d2m.remote_load %tile_b %alloc[%b2, %b1] : memref<3x2x!ttcore.tile<32x32, f32>>, memref<1x1x3x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
    %tile_c = memref.alloc() {alignment = 64 : i64} : memref<4x2x!ttcore.tile<32x32, f32>>
    linalg.generic {indexing_maps = [#mapA, #mapB, #mapC], iterator_types = ["parallel", "parallel", "reduction"]} ins(%tile_a, %tile_b : memref<4x3x!ttcore.tile<32x32, f32>>, memref<3x2x!ttcore.tile<32x32, f32>>) outs(%tile_c : memref<4x2x!ttcore.tile<32x32, f32>>) {
    ^bb0(%in: !ttcore.tile<32x32, f32>, %in_b: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
      %0 = "d2m.tile_matmul"(%in, %in_b, %out) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      linalg.yield %0 : !ttcore.tile<32x32, f32>
    }
    %bo0 = d2m.block_index(0) : index
    %bo1 = d2m.block_index(1) : index
    d2m.remote_store %alloc_0[%bo0, %bo1] %tile_c : memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>, memref<4x2x!ttcore.tile<32x32, f32>>
  }
  return %alloc_0 : memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
}
