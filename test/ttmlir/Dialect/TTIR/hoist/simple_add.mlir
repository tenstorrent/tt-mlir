// RUN: ttmlir-opt --tt-wrap-device-module --ttir-cpu-hoist-transform %s | FileCheck %s

// CHECK: tt.device_module {
// CHECK: builtin.module {

// CHECK-DAG: func.func @add1
func.func @add1(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  %0 = tensor.empty() : tensor<32x32xbf16>
  // CHECK-DAG: %{{.*}} = call @hoisted_ttir_add_32x32xbf16_32x32xbf16_32x32xbf16_func_decl
  %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> {should_hoist} : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %1 : tensor<32x32xbf16>
}

// CHECK-DAG: func.func @add2
func.func @add2(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = tensor.empty() : tensor<32x32xf32>
  // CHECK-DAG: %{{.*}} = call @hoisted_ttir_add_32x32xf32_32x32xf32_32x32xf32_func_decl
  %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> {should_hoist} : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}

// CHECK-DAG: func.func @add3
func.func @add3(%arg0: tensor<32x3xf32>, %arg1: tensor<32x3xf32>) -> tensor<32x3xf32> {
  %0 = tensor.empty() : tensor<32x3xf32>
  // CHECK-DAG: %{{.*}} = call @hoisted_ttir_add_32x3xf32_32x3xf32_32x3xf32_func_decl
  %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> {should_hoist} : (tensor<32x3xf32>, tensor<32x3xf32>, tensor<32x3xf32>) -> tensor<32x3xf32>
  return %1 : tensor<32x3xf32>
}

// CHECK-DAG: func.func @add4
func.func @add4(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  %0 = tensor.empty() : tensor<32x32xbf16>
  // CHECK-DAG: %{{.*}} = call @hoisted_ttir_add_32x32xbf16_32x32xbf16_32x32xbf16_func_decl
  %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> {should_hoist} : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %1 : tensor<32x32xbf16>
}
// CHECK-DAG: func.func private @hoisted_ttir_add_32x32xbf16_32x32xbf16_32x32xbf16_func_decl
// CHECK-DAG: func.func private @hoisted_ttir_add_32x32xf32_32x32xf32_32x32xf32_func_decl
// CHECK-DAG: func.func private @hoisted_ttir_add_32x3xf32_32x3xf32_32x3xf32_func_decl

// CHECK: tt.cpu_module {
// CHECK: builtin.module {
// CHECK-DAG: func.func @hoisted_ttir_add_32x32xbf16_32x32xbf16_32x32xbf16_func
// CHECK-DAG: func.func @hoisted_ttir_add_32x32xf32_32x32xf32_32x32xf32_func
// CHECK-DAG: func.func @hoisted_ttir_add_32x3xf32_32x3xf32_32x3xf32_func
