// RUN: ttmlir-opt --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>

func.func @matmul_pure_tensors(%arg0: tensor<2x4x!ttcore.tile<32x32, f32>>, %arg1: tensor<4x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %0 = d2m.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  // No uses of %3, so it should be removed.
  // CHECK-NOT: d2m.generic
  %3 = "d2m.generic"(%arg0, %arg1, %0) <{block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map1, #map2], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<compute>], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg2: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg3: memref<4x2x!ttcore.tile<32x32, f32>, #l1_>, %arg4: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>):
    "d2m.tile_matmul_block"(%arg2, %arg3, %arg4) : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<4x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> ()
    d2m.yield %arg4 : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
  }) : (tensor<2x4x!ttcore.tile<32x32, f32>>, tensor<4x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

func.func @to_layout_pure_tensors(%arg0: tensor<2x4x!ttcore.tile<32x32, f32>>) -> tensor<2x4x!ttcore.tile<32x32, f32>> {
  %0 = d2m.empty() : tensor<2x4x!ttcore.tile<32x32, f32>>
  // No uses of %3, so it should be removed.
  // CHECK-NOT: d2m.to_layout
  %3 = "d2m.to_layout"(%arg0, %0) : (tensor<2x4x!ttcore.tile<32x32, f32>>, tensor<2x4x!ttcore.tile<32x32, f32>>) -> tensor<2x4x!ttcore.tile<32x32, f32>>
  return %0 : tensor<2x4x!ttcore.tile<32x32, f32>>
}

func.func @matmul_memref(%arg0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>, %arg1: memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
  // Ensure that the generic op is not removed.
  // CHECK: d2m.generic
  "d2m.generic"(%arg0, %arg1, %alloc) <{block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map1, #map2], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<compute>], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg2: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg3: memref<4x2x!ttcore.tile<32x32, f32>, #l1_>, %arg4: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>):
    "d2m.tile_matmul_block"(%arg2, %arg3, %arg4) : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<4x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> ()
  }) : (memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>, memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> ()
  return %alloc : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
}

func.func @to_layout_memref(%arg0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>) -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>
  // Ensure that the layout op is not removed.
  // CHECK: d2m.to_layout
  "d2m.to_layout"(%arg0, %alloc) : (memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>, memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>) -> ()
  return %alloc : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>
}
