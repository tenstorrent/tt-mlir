// RUN: ttmlir-opt --d2m-mark-synchronized-buffers %s | FileCheck %s

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>

module {
  // CHECK-LABEL: func.func @test_matmul_output_stream_single_buffer()
  // CHECK: d2m.generic
  // CHECK: %[[CB_LHS:.*]] = memref.alloc() {d2m.synchronized_buffer = 2 : i32} : memref<2x3x!ttcore.tile<32x32, f32>, #l1>
  // CHECK: %[[CB_RHS:.*]] = memref.alloc() {d2m.synchronized_buffer = 2 : i32} : memref<3x4x!ttcore.tile<32x32, f32>, #l1>
  // CHECK: %[[CB_OUT:.*]] = memref.alloc() {d2m.synchronized_buffer = 1 : i32} : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
  // CHECK: linalg.generic
  // CHECK: "d2m.tile_matmul"
  func.func @test_matmul_output_stream_single_buffer() {
    %lhs = memref.alloc() : memref<1x1x2x3x!ttcore.tile<32x32, f32>, #ttcore.shard<12288x4096, 1>, #dram>
    %rhs = memref.alloc() : memref<1x1x3x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>
    %out = memref.alloc() : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>

    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%lhs, %rhs :
            memref<1x1x2x3x!ttcore.tile<32x32, f32>, #ttcore.shard<12288x4096, 1>, #dram>,
            memref<1x1x3x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>)
        outs(%out : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>)  {
    ^unified0:
      %c0 = arith.constant 0 : index
      %buffer_lhs = memref.alloc() : memref<2x3x!ttcore.tile<32x32, f32>, #l1>
      d2m.remote_load %buffer_lhs %lhs[%c0, %c0] : memref<2x3x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x2x3x!ttcore.tile<32x32, f32>, #ttcore.shard<12288x4096, 1>, #dram>
      %buffer_rhs = memref.alloc() : memref<3x4x!ttcore.tile<32x32, f32>, #l1>
      d2m.remote_load %buffer_rhs %rhs[%c0, %c0] : memref<3x4x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x3x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>
      %buffer_out = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
      linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%buffer_lhs, %buffer_rhs : memref<2x3x!ttcore.tile<32x32, f32>, #l1>, memref<3x4x!ttcore.tile<32x32, f32>, #l1>) outs(%buffer_out : memref<2x4x!ttcore.tile<32x32, f32>, #l1>) {
      ^bb0(%a: !ttcore.tile<32x32, f32>, %b: !ttcore.tile<32x32, f32>, %acc: !ttcore.tile<32x32, f32>):
        %r = "d2m.tile_matmul"(%a, %b, %acc) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %r : !ttcore.tile<32x32, f32>
      }
      d2m.remote_store %out[%c0, %c0] %buffer_out : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x!ttcore.tile<32x32, f32>, #l1>
    }
    return
  }

  // CHECK-LABEL: func.func @test_reduce_output_stream_single_buffer()
  // CHECK: d2m.generic
  // CHECK: %[[CB_IN:.*]] = memref.alloc() {d2m.synchronized_buffer = 2 : i32} : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
  // CHECK: d2m.remote_load %[[CB_IN]]
  // CHECK: %[[CB_OUT:.*]] = memref.alloc() {d2m.synchronized_buffer = 1 : i32} : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
  // CHECK: linalg.generic
  // CHECK: "d2m.tile_reduce_sum"
  func.func @test_reduce_output_stream_single_buffer() {
    %in = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>
    %out = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>

    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%in : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>)
        outs(%out : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>)  {
    ^unified0:
      %c0 = arith.constant 0 : index
      %buffer_in = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      d2m.remote_load %buffer_in %in[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>
      %buffer_out = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%buffer_in : memref<1x1x!ttcore.tile<32x32, f32>, #l1>) outs(%buffer_out : memref<1x1x!ttcore.tile<32x32, f32>, #l1>) {
      ^bb0(%a: !ttcore.tile<32x32, f32>, %acc: !ttcore.tile<32x32, f32>):
        %r = "d2m.tile_reduce_sum"(%a, %a, %acc) <{reduce_dim = #d2m<reduce_dim R>}> : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %r : !ttcore.tile<32x32, f32>
      }
      d2m.remote_store %out[%c0, %c0] %buffer_out : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>
    }
    return
  }

  // CHECK-LABEL: func.func @test_fanout_consumer_buffer()
  // CHECK: %[[CB_IN:.*]] = memref.alloc() {d2m.synchronized_buffer = 2 : i32} : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
  // CHECK: d2m.remote_load %[[CB_IN]]
  // CHECK: linalg.generic
  // CHECK-SAME: %[[CB_IN]]
  // CHECK: linalg.generic
  // CHECK-SAME: %[[CB_IN]]
  func.func @test_fanout_consumer_buffer() {
    %in = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>
    %out = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>

    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%in : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>)
        outs(%out : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>)  {
    ^unified0:
      %c0 = arith.constant 0 : index
      %buffer_in = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      d2m.remote_load %buffer_in %in[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>
      %buffer_tmp = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%buffer_in : memref<1x1x!ttcore.tile<32x32, f32>, #l1>) outs(%buffer_tmp : memref<1x1x!ttcore.tile<32x32, f32>, #l1>) {
      ^bb0(%in_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
        linalg.yield %in_tile : !ttcore.tile<32x32, f32>
      }
      %buffer_out = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%buffer_tmp, %buffer_in : memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>) outs(%buffer_out : memref<1x1x!ttcore.tile<32x32, f32>, #l1>) {
      ^bb0(%tmp_tile: !ttcore.tile<32x32, f32>, %in_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
        linalg.yield %tmp_tile : !ttcore.tile<32x32, f32>
      }
      d2m.remote_store %out[%c0, %c0] %buffer_out : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>
    }
    return
  }
}
