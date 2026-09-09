// RUN: ttmlir-opt --ttcore-register-device "--d2m-reblock-generics=test-buffer-size-policy=min" "--d2m-allocate=test-assume-l1-capacity=3407872" -o %t %s
// RUN: FileCheck %s --input-file=%t

// This test uses a tight L1 capacity limit but succeeds by using min-sized stream buffers.

// CHECK-LABEL: func.func @main()
// CHECK: memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64}
// CHECK: d2m.generic {block_factors = [1, 1, 16], grid = #ttcore.grid<1x1>
// CHECK: memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64, d2m.synchronized_buffer = 2 : i64} : memref<16x1x!ttcore.tile<32x32, f32>, #l1>
// CHECK: memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64, d2m.synchronized_buffer = 2 : i64} : memref<1x16x!ttcore.tile<32x32, f32>, #l1>
// CHECK: d2m.operand_alias

#l1 = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#mapL = affine_map<(d0, d1, d2) -> (d0, d2)>
#mapR = affine_map<(d0, d1, d2) -> (d2, d1)>
#mapO = affine_map<(d0, d1, d2) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>

module {
  func.func @main() -> memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1> {
    %lhs = memref.alloc() : memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>
    %rhs = memref.alloc() : memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>
    %r = memref.alloc() : memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>
    d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<unified>]}
        ins(%lhs, %rhs : memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>, memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>)
        outs(%r : memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>)  {
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
              %result = "d2m.tile_matmul"(%lhs_elem, %rhs_elem, %out_elem) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              linalg.yield %result : !ttcore.tile<32x32, f32>
            }
            d2m.remote_store %r[%iter0, %iter1] %2 : memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>, memref<16x16x!ttcore.tile<32x32, f32>, #l1>
          } {d2m.blocking_loop = 2}
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return %r : memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>
  }
}
