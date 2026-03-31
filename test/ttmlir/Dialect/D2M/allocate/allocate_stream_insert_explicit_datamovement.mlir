// RUN: ttmlir-opt --ttcore-register-device "--d2m-allocate=stream-insert-policy=infer test-buffer-size-policy=max" -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>

module {

  // CHECK-LABEL: func.func @test_generic_insert_missing_streams()
  // CHECK: %[[LHS:.*]] = memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64} : memref<1x1x2x3x!ttcore.tile<32x32, f32>, #ttcore.shard<12288x4096, 1>, #dram>
  // CHECK: %[[CB_LHS:.*]] = memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64} : memref<2x3x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<12288x4096, 2>, #l1>
  // CHECK: d2m.remote_load %[[CB_LHS]] %[[LHS]]{{.*}}
  // CHECK: %[[CB_RHS:.*]] = memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64} : memref<3x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
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
      %cb0 = d2m.get_cb(0) : !d2m.cb<memref<2x3x!ttcore.tile<32x32, f32>, #l1>>
      %cb1 = d2m.get_cb(1) : !d2m.cb<memref<3x4x!ttcore.tile<32x32, f32>, #l1>>
      %cb2 = d2m.get_cb(2) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %buffer_lhs = memref.alloc() : memref<2x3x!ttcore.tile<32x32, f32>, #l1>
      %0 = d2m.remote_load %buffer_lhs %lhs[%c0, %c0] : memref<2x3x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x2x3x!ttcore.tile<32x32, f32>, #ttcore.shard<12288x4096, 1>, #dram> -> memref<2x3x!ttcore.tile<32x32, f32>, #l1>
      %buffer_rhs = memref.alloc() : memref<3x4x!ttcore.tile<32x32, f32>, #l1>
      %1 = d2m.remote_load %buffer_rhs %rhs[%c0, %c0] : memref<3x4x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x3x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<3x4x!ttcore.tile<32x32, f32>, #l1>
      %buffer_out = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
      "d2m.tile_matmul_block"(%0, %1, %buffer_out) : (memref<2x3x!ttcore.tile<32x32, f32>, #l1>, memref<3x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #l1>) -> ()
      %result = d2m.remote_store %out[%c0, %c0] %buffer_out : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #l1> -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    }
    return
  }

} // module
