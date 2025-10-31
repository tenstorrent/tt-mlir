// RUN: ttmlir-opt --ttcore-register-device --ttcore-one-shot-bufferize -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// Test that ttcore-one-shot-bufferize properly handles d2m.stream_layout operations
// with MetalLayoutAttr encoding that has no explicit index_map set. The fix ensures
// that getIndexAffineMapOrIdentity is used instead of getIndexAffineMap to handle
// the case where the index map is empty or null.

#layout = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>

// CHECK: #[[MEM_SPACE:.*]] = #ttcore.memory_space<l1>
// CHECK: module attributes {ttcore.system_desc = #system_desc} {
// CHECK:   ttcore.device @default_device
// CHECK:   func.func @stream_layout_empty_index_map_test(%[[ARG0:.*]]: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #[[MEM_SPACE]]>) -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #[[MEM_SPACE]]> {
// CHECK-NEXT:    %[[ALLOC:.*]] = memref.alloc() : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #[[MEM_SPACE]]>
// CHECK-NEXT:    %[[STREAM:.*]] = "d2m.stream_layout"(%[[ARG0]], %[[ALLOC]]) : (memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #[[MEM_SPACE]]>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #[[MEM_SPACE]]>) -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #[[MEM_SPACE]]>
// CHECK-NEXT:    %[[ALLOC_OUT:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #[[MEM_SPACE]]>
// CHECK-NEXT:    memref.copy %[[STREAM]], %[[ALLOC_OUT]] : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #[[MEM_SPACE]]> to memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #[[MEM_SPACE]]>
// CHECK-NEXT:    return %[[ALLOC_OUT]] : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #[[MEM_SPACE]]>
module {
  func.func @stream_layout_empty_index_map_test(
    %arg0: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
  ) -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout> {
    %0 = d2m.empty() : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
    %stream = "d2m.stream_layout"(%arg0, %0) : (tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>) -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
    return %stream : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
  }
}

