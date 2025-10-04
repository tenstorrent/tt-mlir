// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1 = #ttnn.buffer_type<l1>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>>
#layout = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>, undef, l1>


module {
  // cast(ttnn -> metal) followed by cast(metal -> ttnn) should be eliminated
  // CHECK-LABEL: func.func @test_ttnn_metal_ttnn_canonicalize
  func.func @test_ttnn_metal_ttnn_canonicalize(%arg0: tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout> {
    // CHECK-NOT: ttir.ttnn_metal_layout_cast
    // CHECK: return %arg0 : tensor<32x32xf32, #ttnn_layout>
    %0 = ttir.ttnn_metal_layout_cast %arg0 : tensor<32x32xf32, #ttnn_layout> -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %1 = ttir.ttnn_metal_layout_cast %0 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<32x32xf32, #ttnn_layout>
    return %1 : tensor<32x32xf32, #ttnn_layout>
  }

  // cast(metal -> ttnn) followed by cast(ttnn -> metal) should be eliminated
  // CHECK-LABEL: func.func @test_metal_ttnn_metal_canonicalize
  func.func @test_metal_ttnn_metal_canonicalize(%arg0: tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>) -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> {
    // CHECK-NOT: ttir.ttnn_metal_layout_cast
    // CHECK: return %arg0 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %0 = ttir.ttnn_metal_layout_cast %arg0 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<32x32xf32, #ttnn_layout>
    %1 = ttir.ttnn_metal_layout_cast %0 : tensor<32x32xf32, #ttnn_layout> -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    return %1 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
  }

  // multiple casts in a row. Should eliminate greedily.
  // CHECK-LABEL: func.func @test_many_casts_canonicalize
  func.func @test_many_casts_canonicalize(%arg0: tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout> {
    // CHECK-NOT: ttir.ttnn_metal_layout_cast
    // CHECK: return %arg0 : tensor<32x32xf32, #ttnn_layout>
    %0 = ttir.ttnn_metal_layout_cast %arg0 : tensor<32x32xf32, #ttnn_layout> -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %1 = ttir.ttnn_metal_layout_cast %0 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<32x32xf32, #ttnn_layout>
    %2 = ttir.ttnn_metal_layout_cast %1 : tensor<32x32xf32, #ttnn_layout> -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %3 = ttir.ttnn_metal_layout_cast %2 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<32x32xf32, #ttnn_layout>
    return %3 : tensor<32x32xf32, #ttnn_layout>
  }

  // multiple casts in a row. Should eliminate greedily.
  // CHECK-LABEL: func.func @test_many_casts_canonicalize_1
  func.func @test_many_casts_canonicalize_1(%arg0: tensor<32x32xf32, #ttnn_layout>) -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> {
    // CHECK: %[[CAST:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<32x32xf32, #ttnn_layout> -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    // CHECK: return %[[CAST]] : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %0 = ttir.ttnn_metal_layout_cast %arg0 : tensor<32x32xf32, #ttnn_layout> -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %1 = ttir.ttnn_metal_layout_cast %0 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<32x32xf32, #ttnn_layout>
    %2 = ttir.ttnn_metal_layout_cast %1 : tensor<32x32xf32, #ttnn_layout> -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %3 = ttir.ttnn_metal_layout_cast %2 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<32x32xf32, #ttnn_layout>
    %4 = ttir.ttnn_metal_layout_cast %3 : tensor<32x32xf32, #ttnn_layout> -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    return %4 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
  }

  // CHECK-LABEL: func.func @test_single_cast_not_canonicalized
  func.func @test_single_cast_not_canonicalized(%arg0: tensor<32x32xf32, #ttnn_layout>) -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> {
    // CHECK: %[[CAST:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<32x32xf32, #ttnn_layout> -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    // CHECK: return %[[CAST]] : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %0 = ttir.ttnn_metal_layout_cast %arg0 : tensor<32x32xf32, #ttnn_layout> -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    return %0 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
  }

  // CHECK-LABEL: func.func @test_single_cast_not_canonicalized_1
  func.func @test_single_cast_not_canonicalized_1(%arg0: tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>) -> tensor<32x32xf32, #ttnn_layout> {
    // CHECK: %[[CAST:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<32x32xf32, #ttnn_layout>
    // CHECK: return %[[CAST]] : tensor<32x32xf32, #ttnn_layout>
    %0 = ttir.ttnn_metal_layout_cast %arg0 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<32x32xf32, #ttnn_layout>
    return %0 : tensor<32x32xf32, #ttnn_layout>
  }
}
