// RUN: ttmlir-opt %s | FileCheck %s
// RUN: ttmlir-opt --ttcore-one-shot-bufferize %s | FileCheck %s --check-prefix=BUFFERIZE

// This test verifies that various remote_load and remote_store assembly syntaxes
// are parsed correctly for both CB-mode and result-mode variants using tensor values.
// Tensor variants use MetalLayout attributes to encode device layout information.
// The BUFFERIZE checks verify that the operations work correctly after bufferization.

#l1_ = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// Metal layout for remote source tensor: 2x4 grid with 2x4 shard
// logical_shape = 4x8 (2*2 x 4*2), grid = 2x4, shard = 2x4
#remote_layout = #ttcore.metal_layout<logical_shape = 4x8, dim_alignments = 2x4, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, dram, sharded, index_map = map(0)>

// Metal layout for local tensor: just logical dimensions without sharding
#local_layout = #ttcore.metal_layout<logical_shape = 2x4, dim_alignments = 2x4, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded, index_map = map(0)>

// BUFFERIZE: #[[DRAM:.*]] = #ttcore.memory_space<dram>

//===----------------------------------------------------------------------===//
// RemoteLoadOp Tests - Tensor Variants
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_remote_load_with_result_tensor
// BUFFERIZE-LABEL: @test_remote_load_with_result_tensor
func.func @test_remote_load_with_result_tensor(
    %remote_src: tensor<2x4x2x4x!ttcore.tile<32x32, f32>, #remote_layout>) -> tensor<2x4x!ttcore.tile<32x32, f32>> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %buffer = tensor.empty() : tensor<2x4x!ttcore.tile<32x32, f32>>

  // RemoteLoadOp with result (no CB), tensor variant
  // CHECK: %{{.*}} = d2m.remote_load %{{.*}} %{{.*}}[%{{.*}}, %{{.*}}] : tensor<{{.*}}>, tensor<{{.*}}> -> tensor<{{.*}}>
  // BUFFERIZE: %[[ALLOC:[a-zA-Z0-9]+]] = memref.alloc() {alignment = 64 : i64} : memref<2x4x!ttcore.tile<32x32, f32>>
  // BUFFERIZE: %{{.*}} = d2m.remote_load %[[ALLOC]] %{{.*}}[%{{.*}}, %{{.*}}] : memref<2x4x!ttcore.tile<32x32, f32>>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #[[DRAM]]> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
  // BUFFERIZE: return %[[ALLOC]] : memref<2x4x!ttcore.tile<32x32, f32>>
  %result = d2m.remote_load %buffer %remote_src[%c0, %c1] : tensor<2x4x!ttcore.tile<32x32, f32>>, tensor<2x4x2x4x!ttcore.tile<32x32, f32>, #remote_layout> -> tensor<2x4x!ttcore.tile<32x32, f32>>

  return %result : tensor<2x4x!ttcore.tile<32x32, f32>>
}

// CHECK-LABEL: @test_remote_load_with_result_multicast_tensor
// BUFFERIZE-LABEL: @test_remote_load_with_result_multicast_tensor
func.func @test_remote_load_with_result_multicast_tensor(
    %remote_src: tensor<2x4x2x4x!ttcore.tile<32x32, f32>, #remote_layout>) -> tensor<2x4x!ttcore.tile<32x32, f32>> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %buffer = tensor.empty() : tensor<2x4x!ttcore.tile<32x32, f32>>

  // RemoteLoadOp with result and multicast (low-level multicast), tensor variant
  // CHECK: %{{.*}} = d2m.remote_load %{{.*}} %{{.*}}[%{{.*}}, %{{.*}}] mcore[%{{.*}}, %{{.*}}] mshape[%{{.*}}, %{{.*}}] : tensor<{{.*}}>, tensor<{{.*}}> -> tensor<{{.*}}>
  // BUFFERIZE: %[[ALLOC_MCAST:[a-zA-Z0-9]+]] = memref.alloc() {alignment = 64 : i64} : memref<2x4x!ttcore.tile<32x32, f32>>
  // BUFFERIZE: %{{.*}} = d2m.remote_load %[[ALLOC_MCAST]] %{{.*}}[%{{.*}}, %{{.*}}] mcore[%{{.*}}, %{{.*}}] mshape[%{{.*}}, %{{.*}}] : memref<2x4x!ttcore.tile<32x32, f32>>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #[[DRAM]]> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
  // BUFFERIZE: return %[[ALLOC_MCAST]] : memref<2x4x!ttcore.tile<32x32, f32>>
  %result = d2m.remote_load %buffer %remote_src[%c0, %c1] mcore[%c0, %c0] mshape[%c1, %c2] : tensor<2x4x!ttcore.tile<32x32, f32>>, tensor<2x4x2x4x!ttcore.tile<32x32, f32>, #remote_layout> -> tensor<2x4x!ttcore.tile<32x32, f32>>

  return %result : tensor<2x4x!ttcore.tile<32x32, f32>>
}

//===----------------------------------------------------------------------===//
// RemoteStoreOp Tests - Tensor Variants (Implicit Form)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_remote_store_with_local_buffer_tensor
// BUFFERIZE-LABEL: @test_remote_store_with_local_buffer_tensor
func.func @test_remote_store_with_local_buffer_tensor(
    %remote_dst: tensor<2x4x2x4x!ttcore.tile<32x32, f32>, #remote_layout>,
    %local_buffer: tensor<2x4x!ttcore.tile<32x32, f32>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // RemoteStoreOp with local buffer (implicit form) - basic case, tensor variant
  // Result is required for implicit form. For tensors it can be used, but after bufferization
  // to memrefs it must be unused.
  // CHECK: %{{.*}} = d2m.remote_store %{{.*}}[%{{.*}}, %{{.*}}] %{{.*}} : tensor<{{.*}}>, tensor<{{.*}}> -> tensor<{{.*}}>
  // BUFFERIZE: %{{.*}} = d2m.remote_store %{{.*}}[%{{.*}}, %{{.*}}] %{{.*}} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #[[DRAM]]>, memref<2x4x!ttcore.tile<32x32, f32>> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #[[DRAM]]>
  %result = d2m.remote_store %remote_dst[%c0, %c1] %local_buffer : tensor<2x4x2x4x!ttcore.tile<32x32, f32>, #remote_layout>, tensor<2x4x!ttcore.tile<32x32, f32>> -> tensor<2x4x2x4x!ttcore.tile<32x32, f32>, #remote_layout>

  return
}
