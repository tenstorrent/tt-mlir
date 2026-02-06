// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

func.func @top_k_1d(%arg0 : tensor<16xf32>) -> (tensor<8xf32>, tensor<8xi32>) {
  // CHECK: %{{.*}}, %{{.*}} = "ttir.topk"(%arg0) <{dim = -1 : i32, k = 8 : i32, largest = true, sorted = true}> : (tensor<16xf32>) -> (tensor<8xf32>, tensor<8xi32>)
  %0:2 = stablehlo.custom_call @mhlo.topk(%arg0) {
    mhlo.attributes = {k = 8 : i64, largest = true},
    mhlo.version = 1 : i64
  } : (tensor<16xf32>) -> (tensor<8xf32>, tensor<8xi32>)
  return %0#0, %0#1 : tensor<8xf32>, tensor<8xi32>
}

func.func @top_k_nd(%arg0 : tensor<16x16xf32>) -> (tensor<16x8xf32>, tensor<16x8xi32>) {
  // CHECK: %{{.*}}, %{{.*}} = "ttir.topk"(%arg0) <{dim = -1 : i32, k = 8 : i32, largest = false, sorted = true}> : (tensor<16x16xf32>) -> (tensor<16x8xf32>, tensor<16x8xi32>)
  %0:2 = stablehlo.custom_call @mhlo.topk(%arg0) {
    mhlo.attributes = {k = 8 : i64, largest = false},
    mhlo.version = 1 : i64
  } : (tensor<16x16xf32>) -> (tensor<16x8xf32>, tensor<16x8xi32>)
  return %0#0, %0#1 : tensor<16x8xf32>, tensor<16x8xi32>
}

func.func @top_k_large_k(%arg0 : tensor<262144xf32>) -> (tensor<19999xf32>, tensor<19999xi32>) {
  // CHECK: %{{.*}}, %{{.*}} = "ttir.topk"(%arg0) <{dim = -1 : i32, k = 19999 : i32, largest = true, sorted = true}> : (tensor<262144xf32>) -> (tensor<19999xf32>, tensor<19999xi32>)
  %0:2 = stablehlo.custom_call @mhlo.topk(%arg0) {
    mhlo.attributes = {k = 19999 : i64, largest = true},
    mhlo.version = 1 : i64
  } : (tensor<262144xf32>) -> (tensor<19999xf32>, tensor<19999xi32>)
  return %0#0, %0#1 : tensor<19999xf32>, tensor<19999xi32>
}
