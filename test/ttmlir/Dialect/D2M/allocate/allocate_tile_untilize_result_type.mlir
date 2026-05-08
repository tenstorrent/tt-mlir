// RUN: ttmlir-opt --ttcore-register-device "--d2m-allocate=stream-insert-policy=infer test-buffer-size-policy=max" -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

module {
  // CHECK-LABEL: func.func @untilize_result_matches_cb_output
  // CHECK: %[[SCALAR:.*]] = memref.alloc(){{.*}} : memref<32x32xf32, #ttcore.cb_layout<{{[^>]+}}>, #l1>
  // CHECK: "d2m.tile_untilize_block"(%{{.*}}, %[[SCALAR]]) : (memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<{{[^>]+}}>, #l1>, memref<32x32xf32, #ttcore.cb_layout<{{[^>]+}}>, #l1>) -> memref<32x32xf32, #ttcore.cb_layout<{{[^>]+}}>, #l1>
  func.func @untilize_result_matches_cb_output(
      %arg0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>,
      %arg1: memref<1x1x32x32xf32, #ttcore.shard<128x4, 1>, #dram>) {
    d2m.generic {
        block_factors = [1, 1],
        grid = #ttcore.grid<1x1>,
        indexing_maps = [#map, #map],
        iterator_types = [#parallel, #parallel],
        threads = [#d2m.thread<unified>]}
        ins(%arg0 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>)
        outs(%arg1 : memref<1x1x32x32xf32, #ttcore.shard<128x4, 1>, #dram>) {
    ^bb0:
      %bf0 = d2m.get_block_factor(0) : index
      %bf1 = d2m.get_block_factor(1) : index
      affine.for %i = 0 to %bf0 {
        affine.for %j = 0 to %bf1 {
          %tile = memref.alloc() {alignment = 64 : i64} : memref<1x1x!ttcore.tile<32x32, f32>>
          %loaded = d2m.remote_load %tile %arg0[%i, %j] : memref<1x1x!ttcore.tile<32x32, f32>>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>

          %scalar = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          %untilized = "d2m.tile_untilize_block"(%tile, %scalar) : (memref<1x1x!ttcore.tile<32x32, f32>>, memref<32x32xf32>) -> memref<32x32xf32>

          d2m.remote_store %arg1[%i, %j] %scalar : memref<1x1x32x32xf32, #ttcore.shard<128x4, 1>, #dram>, memref<32x32xf32> -> memref<32x32xf32>
        } {d2m.blocking_loop = 1 : i64}
      } {d2m.blocking_loop = 0 : i64}
    }
    return
  }
}
