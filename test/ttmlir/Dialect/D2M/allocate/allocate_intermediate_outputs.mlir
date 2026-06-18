// RUN: ttmlir-opt --ttcore-register-device "--d2m-reblock-generics=test-buffer-size-policy=max" "--d2m-allocate=force-spill-to-dram-if-legal=true" -o %t.no_spill %s
// RUN: FileCheck %s --check-prefix=NO-SPILL --input-file=%t.no_spill
// RUN: ttmlir-opt --ttcore-register-device "--d2m-reblock-generics=test-buffer-size-policy=max" "--d2m-allocate=allow-l1-output-spilling=true test-assume-l1-capacity=12288" -o %t.spill %s
// RUN: FileCheck %s --check-prefix=SPILL --input-file=%t.spill
// RUN: ttmlir-opt --ttcore-register-device "--d2m-reblock-generics=test-buffer-size-policy=max" "--d2m-allocate=allow-l1-output-spilling=true test-assume-l1-capacity=24576" -o %t.l1 %s
// RUN: FileCheck %s --check-prefix=L1 --input-file=%t.l1

// Validate the two allocator decisions needed for internal generic outputs:
//
//   * By default, generic outputs are not spillable. Even with forced spilling,
//     the intermediate remains in L1 and uses the old aliasing path.
//   * With allow-l1-output-spilling=true, an intermediate generic output becomes
//     spillable while the terminal output remains in L1.
//   * If that intermediate spills to DRAM, the remote load/store path must stay
//     materialized as synchronized local buffers. If it stays in L1, the
//     allocator should restore operand aliases after planning.

// NO-SPILL-LABEL: func.func @intermediate_chain
// NO-SPILL: %[[TMP:[A-Za-z0-9_]+]] = memref.alloc() {{.*}} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
// NO-SPILL: %[[OUT:[A-Za-z0-9_]+]] = memref.alloc() {{.*}} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
// NO-SPILL: outs(%[[TMP]] : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
// NO-SPILL: d2m.operand_alias %[[TMP]] : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
// NO-SPILL: ins(%[[TMP]] : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
// NO-SPILL: d2m.operand_alias %[[TMP]] : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
// NO-SPILL: return %[[OUT]] : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>

// SPILL-LABEL: func.func @intermediate_chain
// SPILL: %[[TMP:[A-Za-z0-9_]+]] = memref.alloc() {{.*}} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>
// SPILL: %[[OUT:[A-Za-z0-9_]+]] = memref.alloc() {{.*}} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
// SPILL: outs(%[[TMP]] : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>)
// SPILL-NOT: d2m.operand_alias %[[TMP]]
// SPILL: %[[PRODUCER_BUF:[A-Za-z0-9_]+]] = memref.alloc() {{.*}}d2m.synchronized_buffer = 2{{.*}} : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
// SPILL: d2m.remote_store %[[TMP]]{{.*}} %[[PRODUCER_BUF]] : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>
// SPILL: ins(%[[TMP]] : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>)
// SPILL-NOT: d2m.operand_alias %[[TMP]]
// SPILL: %[[CONSUMER_BUF:[A-Za-z0-9_]+]] = memref.alloc() {{.*}}d2m.synchronized_buffer = 2{{.*}} : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
// SPILL: d2m.remote_load %[[CONSUMER_BUF]] %[[TMP]]{{.*}} : memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>
// SPILL: return %[[OUT]] : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>

// L1-LABEL: func.func @intermediate_chain
// L1: %[[TMP:[A-Za-z0-9_]+]] = memref.alloc() {{.*}} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
// L1: %[[OUT:[A-Za-z0-9_]+]] = memref.alloc() {{.*}} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
// L1: outs(%[[TMP]] : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
// L1: d2m.operand_alias %[[TMP]] : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
// L1: ins(%[[TMP]] : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
// L1: d2m.operand_alias %[[TMP]] : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
// L1: return %[[OUT]] : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>

#l1 = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
!tile_f32 = !ttcore.tile<32x32, f32>
!tensor_l1 = memref<1x1x1x1x!tile_f32, #ttcore.shard<4096x4096, 1>, #l1>
!tile_l1 = memref<1x1x!tile_f32, #l1>

module {
  func.func @intermediate_chain(%input: !tensor_l1) -> !tensor_l1 {
    %tmp = memref.alloc() : !tensor_l1
    %out = memref.alloc() : !tensor_l1

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%input : !tensor_l1)
        outs(%tmp : !tensor_l1)  {
    ^unified0:
      %bf0 = d2m.get_block_factor(0) : index
      %bf1 = d2m.get_block_factor(1) : index
      affine.for %iter0 = 0 to %bf0 {
        affine.for %iter1 = 0 to %bf1 {
          %in_buf = memref.alloc() {d2m.synchronized_buffer = 2} : !tile_l1
          d2m.remote_load %in_buf %input[%iter0, %iter1] : !tile_l1, !tensor_l1
          %out_buf = memref.alloc() {d2m.synchronized_buffer = 2} : !tile_l1
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%in_buf : !tile_l1) outs(%out_buf : !tile_l1) {
          ^bb0(%in_elem: !tile_f32, %out_elem: !tile_f32):
            %abs = "d2m.tile_abs"(%in_elem) : (!tile_f32) -> !tile_f32
            linalg.yield %abs : !tile_f32
          }
          d2m.remote_store %tmp[%iter0, %iter1] %out_buf : !tensor_l1, !tile_l1
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%tmp : !tensor_l1)
        outs(%out : !tensor_l1)  {
    ^unified0:
      %bf0 = d2m.get_block_factor(0) : index
      %bf1 = d2m.get_block_factor(1) : index
      affine.for %iter0 = 0 to %bf0 {
        affine.for %iter1 = 0 to %bf1 {
          %in_buf = memref.alloc() {d2m.synchronized_buffer = 2} : !tile_l1
          d2m.remote_load %in_buf %tmp[%iter0, %iter1] : !tile_l1, !tensor_l1
          %out_buf = memref.alloc() {d2m.synchronized_buffer = 2} : !tile_l1
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%in_buf : !tile_l1) outs(%out_buf : !tile_l1) {
          ^bb0(%in_elem: !tile_f32, %out_elem: !tile_f32):
            %neg = "d2m.tile_negative"(%in_elem) : (!tile_f32) -> !tile_f32
            linalg.yield %neg : !tile_f32
          }
          d2m.remote_store %out[%iter0, %iter1] %out_buf : !tensor_l1, !tile_l1
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }

    return %out : !tensor_l1
  }
}
