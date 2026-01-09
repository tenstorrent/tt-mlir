// RUN: ttmlir-opt %s | FileCheck %s

// This test verifies that various remote_load and remote_store assembly syntaxes
// are parsed correctly for both CB-mode and result-mode variants using tensor values.
// Tensor variants use MetalLayout attributes to encode device layout information.

#l1_ = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// Metal layout for remote source tensor: 2x4 grid with 2x4 shard
// logical_shape = 4x8 (2*2 x 4*2), grid = 2x4, shard = 2x4
#remote_layout = #ttcore.metal_layout<logical_shape = 4x8, dim_alignments = 2x4, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, dram, sharded, index_map = map(0)>

// Metal layout for local tensor: just logical dimensions without sharding
#local_layout = #ttcore.metal_layout<logical_shape = 2x4, dim_alignments = 2x4, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded, index_map = map(0)>

//===----------------------------------------------------------------------===//
// RemoteLoadOp Tests - Tensor Variants
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_remote_load_with_cb_tensor
func.func @test_remote_load_with_cb_tensor(
    %remote_src: tensor<2x4x2x4x!ttcore.tile<32x32, f32>, #remote_layout>,
    %cb: !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // RemoteLoadOp with CB (no result) - basic case, tensor variant
  // CHECK: d2m.remote_load %{{.*}}[%{{.*}}, %{{.*}}] into %{{.*}} : tensor<{{.*}}> into !d2m.cb<tensor<{{.*}}>
  d2m.remote_load %remote_src[%c0, %c1] into %cb : tensor<2x4x2x4x!ttcore.tile<32x32, f32>, #remote_layout> into !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>>>

  return
}

// CHECK-LABEL: @test_remote_load_with_cb_multicast_tensor
func.func @test_remote_load_with_cb_multicast_tensor(
    %remote_src: tensor<2x4x2x4x!ttcore.tile<32x32, f32>, #remote_layout>,
    %cb: !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  // RemoteLoadOp with CB and multicast, tensor variant
  // CHECK: d2m.remote_load %{{.*}}[%{{.*}}, %{{.*}}] into %{{.*}} core[%{{.*}}, %{{.*}}] mcast[%{{.*}}, %{{.*}}] : tensor<{{.*}}> into !d2m.cb<tensor<{{.*}}>
  d2m.remote_load %remote_src[%c0, %c1] into %cb core[%c0, %c0] mcast[%c1, %c2] : tensor<2x4x2x4x!ttcore.tile<32x32, f32>, #remote_layout> into !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>>>

  return
}

// CHECK-LABEL: @test_remote_load_with_result_tensor
func.func @test_remote_load_with_result_tensor(
    %remote_src: tensor<2x4x2x4x!ttcore.tile<32x32, f32>, #remote_layout>) -> tensor<2x4x!ttcore.tile<32x32, f32>> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // RemoteLoadOp with result (no CB), tensor variant
  // CHECK: %{{.*}} = d2m.remote_load %{{.*}}[%{{.*}}, %{{.*}}] : tensor<{{.*}}> -> tensor<{{.*}}>
  %result = d2m.remote_load %remote_src[%c0, %c1] : tensor<2x4x2x4x!ttcore.tile<32x32, f32>, #remote_layout> -> tensor<2x4x!ttcore.tile<32x32, f32>>

  return %result : tensor<2x4x!ttcore.tile<32x32, f32>>
}

// CHECK-LABEL: @test_remote_load_with_result_multicast_tensor
func.func @test_remote_load_with_result_multicast_tensor(
    %remote_src: tensor<2x4x2x4x!ttcore.tile<32x32, f32>, #remote_layout>) -> tensor<2x4x!ttcore.tile<32x32, f32>> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  // RemoteLoadOp with result and multicast, tensor variant
  // CHECK: %{{.*}} = d2m.remote_load %{{.*}}[%{{.*}}, %{{.*}}] core[%{{.*}}, %{{.*}}] mcast[%{{.*}}, %{{.*}}] : tensor<{{.*}}> -> tensor<{{.*}}>
  %result = d2m.remote_load %remote_src[%c0, %c1] core[%c0, %c0] mcast[%c1, %c2] : tensor<2x4x2x4x!ttcore.tile<32x32, f32>, #remote_layout> -> tensor<2x4x!ttcore.tile<32x32, f32>>

  return %result : tensor<2x4x!ttcore.tile<32x32, f32>>
}

//===----------------------------------------------------------------------===//
// RemoteStoreOp Tests - Form I (local buffer) - Tensor Variants
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_remote_store_with_local_buffer_tensor
func.func @test_remote_store_with_local_buffer_tensor(
    %remote_dst: tensor<2x4x2x4x!ttcore.tile<32x32, f32>, #remote_layout>,
    %local_buffer: tensor<2x4x!ttcore.tile<32x32, f32>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // RemoteStoreOp with local buffer (Form I) - basic case, tensor variant
  // CHECK: d2m.remote_store %{{.*}}[%{{.*}}, %{{.*}}] %{{.*}} : tensor<{{.*}}>, tensor<{{.*}}>
  d2m.remote_store %remote_dst[%c0, %c1] %local_buffer : tensor<2x4x2x4x!ttcore.tile<32x32, f32>, #remote_layout>, tensor<2x4x!ttcore.tile<32x32, f32>>

  return
}

// CHECK-LABEL: @test_remote_store_with_local_buffer_multicast_tensor
func.func @test_remote_store_with_local_buffer_multicast_tensor(
    %remote_dst: tensor<2x4x2x4x!ttcore.tile<32x32, f32>, #remote_layout>,
    %local_buffer: tensor<2x4x!ttcore.tile<32x32, f32>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  // RemoteStoreOp with local buffer and multicast (Form I), tensor variant
  // CHECK: d2m.remote_store %{{.*}}[%{{.*}}, %{{.*}}] %{{.*}} core[%{{.*}}, %{{.*}}] mcast[%{{.*}}, %{{.*}}] : tensor<{{.*}}>, tensor<{{.*}}>
  d2m.remote_store %remote_dst[%c0, %c1] %local_buffer core[%c0, %c0] mcast[%c1, %c2] : tensor<2x4x2x4x!ttcore.tile<32x32, f32>, #remote_layout>, tensor<2x4x!ttcore.tile<32x32, f32>>

  return
}

//===----------------------------------------------------------------------===//
// RemoteStoreOp Tests - Form II (circular buffer) - Tensor Variants
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_remote_store_with_cb_tensor
func.func @test_remote_store_with_cb_tensor(
    %remote_dst: tensor<2x4x2x4x!ttcore.tile<32x32, f32>, #remote_layout>,
    %cb: !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // RemoteStoreOp with CB (Form II) - basic case, tensor variant
  // CHECK: d2m.remote_store %{{.*}}[%{{.*}}, %{{.*}}] from %{{.*}} : tensor<{{.*}}> from !d2m.cb<tensor<{{.*}}>
  d2m.remote_store %remote_dst[%c0, %c1] from %cb : tensor<2x4x2x4x!ttcore.tile<32x32, f32>, #remote_layout> from !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>>>

  return
}

// CHECK-LABEL: @test_remote_store_with_cb_multicast_tensor
func.func @test_remote_store_with_cb_multicast_tensor(
    %remote_dst: tensor<2x4x2x4x!ttcore.tile<32x32, f32>, #remote_layout>,
    %cb: !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  // RemoteStoreOp with CB and multicast (Form II), tensor variant
  // CHECK: d2m.remote_store %{{.*}}[%{{.*}}, %{{.*}}] from %{{.*}} core[%{{.*}}, %{{.*}}] mcast[%{{.*}}, %{{.*}}] : tensor<{{.*}}> from !d2m.cb<tensor<{{.*}}>
  d2m.remote_store %remote_dst[%c0, %c1] from %cb core[%c0, %c0] mcast[%c1, %c2] : tensor<2x4x2x4x!ttcore.tile<32x32, f32>, #remote_layout> from !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>>>

  return
}
