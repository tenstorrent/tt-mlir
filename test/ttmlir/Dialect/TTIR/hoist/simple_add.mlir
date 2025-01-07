// RUN: ttmlir-opt --ttir-cpu-hoist-transform %s | FileCheck %s
module {
func.func @add(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  %0 = tensor.empty() : tensor<32x32xbf16>
  // CHECK: %{{.*}} = call @hoisted_ttir.add_32x32xbf16_32x32xbf16_32x32xbf16_func
  %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> {should_hoist} : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %1 : tensor<32x32xbf16>
}

func.func @add2(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = tensor.empty() : tensor<32x32xf32>
  // CHECK: %{{.*}} = call @hoisted_ttir.add_32x32xf32_32x32xf32_32x32xf32_func
  %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> {should_hoist} : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}

func.func @add3(%arg0: tensor<32x3xf32>, %arg1: tensor<32x3xf32>) -> tensor<32x3xf32> {
  %0 = tensor.empty() : tensor<32x3xf32>
// CHECK: %{{.*}} = call @hoisted_ttir.add_32x3xf32_32x3xf32_32x3xf32_func
  %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> {should_hoist} : (tensor<32x3xf32>, tensor<32x3xf32>, tensor<32x3xf32>) -> tensor<32x3xf32>
  return %1 : tensor<32x3xf32>
}

func.func @add4(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  %0 = tensor.empty() : tensor<32x32xbf16>
  // CHECK: %{{.*}} = call @hoisted_ttir.add_32x32xbf16_32x32xbf16_32x32xbf16_func
  %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> {should_hoist} : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %1 : tensor<32x32xbf16>
}
// CHECK: module @cpu_module attributes {ttir.cpu_module}
// CHECK: func.func @hoisted_ttir.add_32x32xbf16_32x32xbf16_32x32xbf16_func
// CHECK: func.func @hoisted_ttir.add_32x32xf32_32x32xf32_32x32xf32_func
// CHECK: func.func @hoisted_ttir.add_32x3xf32_32x3xf32_32x3xf32_func

// CHECK: func.func private @hoisted_ttir.add_32x32xbf16_32x32xbf16_32x32xbf16_func
// CHECK: func.func private @hoisted_ttir.add_32x32xf32_32x32xf32_32x32xf32_func
// CHECK: func.func private @hoisted_ttir.add_32x3xf32_32x3xf32_32x3xf32_func
}