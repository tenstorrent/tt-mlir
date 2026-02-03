// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

#l1_ = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>

// Metal layout for remote source tensor: 2x4 grid with 2x4 shard
#remote_layout = #ttcore.metal_layout<logical_shape = 4x8, dim_alignments = 2x4, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, dram, sharded, index_map = map(0)>

//===----------------------------------------------------------------------===//
// RemoteLoadOp Negative Tests
//===----------------------------------------------------------------------===//

func.func @test_remote_load_cb_with_tensor(
    %remote_src: tensor<2x4x2x4x!ttcore.tile<32x32, f32>, #remote_layout>,
    %cb: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: error: 'd2m.remote_load' op tensor parameters are not allowed in explicit CB form; memref operand must be a memref type
  d2m.remote_load %remote_src[%c0, %c1] into %cb : tensor<2x4x2x4x!ttcore.tile<32x32, f32>, #remote_layout> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>

  return
}

// -----

#l1_ = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>

// Metal layout for remote source tensor: 2x4 grid with 2x4 shard
#remote_layout = #ttcore.metal_layout<logical_shape = 4x8, dim_alignments = 2x4, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, dram, sharded, index_map = map(0)>

//===----------------------------------------------------------------------===//
// RemoteStoreOp Negative Tests
//===----------------------------------------------------------------------===//

func.func @test_remote_store_cb_with_tensor_memref(
    %remote_dst: tensor<2x4x2x4x!ttcore.tile<32x32, f32>, #remote_layout>,
    %cb: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: error: 'd2m.remote_store' op tensor parameters are not allowed in explicit CB form; memref operand must be a memref type
  d2m.remote_store %remote_dst[%c0, %c1] from %cb : tensor<2x4x2x4x!ttcore.tile<32x32, f32>, #remote_layout> from !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>

  return
}
