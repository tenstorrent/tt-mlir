// RUN: ttmlir-opt -o %t %s
// RUN: FileCheck %s --input-file=%t

func.func @test_basic_top_k(%input: tensor<2x3x32x128xf32>) -> (tensor<2x3x32x5xf32>, tensor<2x3x32x5xi32>) {
  // CHECK: %[[VALUES:[a-zA-Z0-9_]+]], %[[INDICES:[a-zA-Z0-9_]+]] = "ttir.top_k"(%arg0) <{dim = -1 : i32, k = 5 : i32, largest = true, sorted = true}> : (tensor<2x3x32x128xf32>) -> (tensor<2x3x32x5xf32>, tensor<2x3x32x5xi32>)
  %values, %indices = "ttir.top_k"(%input) { k = 5 : i32} : (tensor<2x3x32x128xf32>) -> (tensor<2x3x32x5xf32>, tensor<2x3x32x5xi32>)
  return %values, %indices : tensor<2x3x32x5xf32>, tensor<2x3x32x5xi32>
}

func.func @test_topk_explicit_dim(%input: tensor<2x8x4x256xf32>) -> (tensor<2x3x4x5xf32>, tensor<2x3x4x5xi32>) {
  // CHECK: %[[VALUES:[a-zA-Z0-9_]+]], %[[INDICES:[a-zA-Z0-9_]+]] = "ttir.top_k"(%arg0) <{dim = -1 : i32, k = 5 : i32, largest = true, sorted = true}> : (tensor<2x8x4x256xf32>) -> (tensor<2x3x4x5xf32>, tensor<2x3x4x5xi32>)
  %values, %indices = "ttir.top_k"(%input) {k = 5: i32, dim = -1: i32} : (tensor<2x8x4x256xf32>) -> (tensor<2x3x4x5xf32>, tensor<2x3x4x5xi32>)
  return %values, %indices : tensor<2x3x4x5xf32>, tensor<2x3x4x5xi32>
}

// CHECK-LABEL: test_topk_smallest
func.func @test_topk_smallest(%input: tensor<2x64x4x64xf32>) -> (tensor<2x3x4x8xf32>, tensor<2x3x4x8xi32>) {
  // CHECK: %[[VALUES:[a-zA-Z0-9_]+]], %[[INDICES:[a-zA-Z0-9_]+]] = "ttir.top_k"(%arg0) <{dim = -1 : i32, k = 8 : i32, largest = false, sorted = true}> : (tensor<2x64x4x64xf32>) -> (tensor<2x3x4x8xf32>, tensor<2x3x4x8xi32>)
  %values, %indices = "ttir.top_k"(%input) {k = 8: i32, largest = false} : (tensor<2x64x4x64xf32>) -> (tensor<2x3x4x8xf32>, tensor<2x3x4x8xi32>)
  return %values, %indices : tensor<2x3x4x8xf32>, tensor<2x3x4x8xi32>
}

// CHECK-LABEL: @test_top_k_unsorted
func.func @test_top_k_unsorted(%input: tensor<128x3x4x32xf32>) -> (tensor<128x3x4x5xf32>, tensor<128x3x4x5xi32>) {
  // CHECK: %[[VALUES:[a-zA-Z0-9_]+]], %[[INDICES:[a-zA-Z0-9_]+]] = "ttir.top_k"(%arg0) <{dim = -1 : i32, k = 5 : i32, largest = true, sorted = false}> : (tensor<128x3x4x32xf32>) -> (tensor<128x3x4x5xf32>, tensor<128x3x4x5xi32>)
  %values, %indices = "ttir.top_k"(%input) {k = 5 : i32, sorted = false} : (tensor<128x3x4x32xf32>) -> (tensor<128x3x4x5xf32>, tensor<128x3x4x5xi32>)
  return %values, %indices : tensor<128x3x4x5xf32>, tensor<128x3x4x5xi32>
}
