// RUN: ttmlir-opt --ttcore-register-device "--d2m-reblock-generics=test-buffer-size-policy=max" "--d2m-allocate=stream-insert-policy=infer" -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>

module {

  // CHECK-LABEL: func.func @test_generic_insert_missing_streams()
  // CHECK: %[[LHS:.*]] = memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64} : memref<1x1x2x3x!ttcore.tile<32x32, f32>, #ttcore.shard<12288x4096, 1>, #dram>
  // CHECK: %[[RHS:.*]] = memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64} : memref<1x1x3x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>
  // CHECK: d2m.generic
  // CHECK: %[[CB_LHS:.*]] = memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64{{.*}}} : memref<2x3x!ttcore.tile<32x32, f32>, {{.*}}#l1>
  // CHECK: d2m.remote_load %[[CB_LHS]] %[[LHS]]{{.*}}
  // CHECK: %[[CB_RHS:.*]] = memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64{{.*}}} : memref<3x4x!ttcore.tile<32x32, f32>, {{.*}}#l1>
  // CHECK: d2m.remote_load %[[CB_RHS]] %[[RHS]]{{.*}}
  func.func @test_generic_insert_missing_streams() {
    %lhs = memref.alloc() : memref<1x1x2x3x!ttcore.tile<32x32, f32>, #ttcore.shard<12288x4096, 1>, #dram>
    %rhs = memref.alloc() : memref<1x1x3x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>
    %out = memref.alloc() : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>

    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%lhs, %rhs :
            memref<1x1x2x3x!ttcore.tile<32x32, f32>, #ttcore.shard<12288x4096, 1>, #dram>,
            memref<1x1x3x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>)
        outs(%out : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)  {
    ^unified0:
      %c0 = arith.constant 0 : index
      %buffer_lhs = memref.alloc() {d2m.synchronized_buffer = 2} : memref<2x3x!ttcore.tile<32x32, f32>, #l1>
      d2m.remote_load %buffer_lhs %lhs[%c0, %c0] : memref<2x3x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x2x3x!ttcore.tile<32x32, f32>, #ttcore.shard<12288x4096, 1>, #dram>
      %buffer_rhs = memref.alloc() {d2m.synchronized_buffer = 2} : memref<3x4x!ttcore.tile<32x32, f32>, #l1>
      d2m.remote_load %buffer_rhs %rhs[%c0, %c0] : memref<3x4x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x3x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>
      %buffer_out = memref.alloc() {d2m.synchronized_buffer = 2} : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
      linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%buffer_lhs, %buffer_rhs : memref<2x3x!ttcore.tile<32x32, f32>, #l1>, memref<3x4x!ttcore.tile<32x32, f32>, #l1>) outs(%buffer_out : memref<2x4x!ttcore.tile<32x32, f32>, #l1>) {
      ^bb0(%lhs_elem: !ttcore.tile<32x32, f32>, %rhs_elem: !ttcore.tile<32x32, f32>, %out_elem: !ttcore.tile<32x32, f32>):
        %result = "d2m.tile_matmul"(%lhs_elem, %rhs_elem, %out_elem) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %result : !ttcore.tile<32x32, f32>
      }
      d2m.remote_store %out[%c0, %c0] %buffer_out : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #l1>
    }
    return
  }

} // module
