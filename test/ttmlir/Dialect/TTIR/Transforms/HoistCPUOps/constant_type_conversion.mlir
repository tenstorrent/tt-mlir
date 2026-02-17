// RUN: ttmlir-opt --ttcore-wrap-device-module --cpu-hoist-manually-tagged --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test that constant ops with f64 and i64 types are properly converted to f32 and i32
// during CPU hoisting to match the converted result types.

// CHECK: ttcore.device_module {
// CHECK: builtin.module {

// CHECK: func.func @constant_f64
func.func @constant_f64(%arg0: tensor<4x4xbf16>) -> tensor<4x4xbf16> {
  // f64 constant that will be hoisted and converted to f32
  // CHECK: call @cpu_hoisted_ttir_constant_{{.*}}
  %0 = "ttir.constant"() <{value = dense<3.14159265358979> : tensor<4x4xf64>}> {ttir.should_hoist} : () -> tensor<4x4xf64>
  %1 = "ttir.typecast"(%0) <{dtype = bf16}> : (tensor<4x4xf64>) -> tensor<4x4xbf16>
  %2 = "ttir.add"(%arg0, %1) : (tensor<4x4xbf16>, tensor<4x4xbf16>) -> tensor<4x4xbf16>
  return %2 : tensor<4x4xbf16>
}

// CHECK: func.func @constant_i64
func.func @constant_i64(%arg0: tensor<4x4xbf16>) -> tensor<4x4xbf16> {
  // i64 constant that will be hoisted and converted to i32
  // CHECK: call @cpu_hoisted_ttir_constant_{{.*}}
  %0 = "ttir.constant"() <{value = dense<42> : tensor<4x4xi64>}> {ttir.should_hoist} : () -> tensor<4x4xi64>
  %1 = "ttir.typecast"(%0) <{dtype = bf16}> : (tensor<4x4xi64>) -> tensor<4x4xbf16>
  %2 = "ttir.add"(%arg0, %1) : (tensor<4x4xbf16>, tensor<4x4xbf16>) -> tensor<4x4xbf16>
  return %2 : tensor<4x4xbf16>
}

// Verify f64 constant is converted to f32 in CPU hoisted function declaration
// CHECK: func.func private @cpu_hoisted_ttir_constant_{{.*}} -> tensor<4x4xf32>
// Verify i64 constant is converted to i32 in CPU hoisted function declaration
// CHECK: func.func private @cpu_hoisted_ttir_constant_{{.*}} -> tensor<4x4xi32>

// CHECK: ttcore.cpu_module {
// CHECK: builtin.module {
// Verify the CPU module contains the hoisted constant functions with converted types
// f64 constant converted to f32
// CHECK: func.func @cpu_hoisted_ttir_constant_{{.*}} -> tensor<4x4xf32>
// CHECK: ttir.full
// CHECK-SAME: tensor<4x4xf32>
// i64 constant converted to i32
// CHECK: func.func @cpu_hoisted_ttir_constant_{{.*}} -> tensor<4x4xi32>
// CHECK: ttir.full
// CHECK-SAME: tensor<4x4xi32>

// Verify original types are not present anywhere in the output
// CHECK-NOT: tensor<4x4xf64>
// CHECK-NOT: tensor<4x4xi64>
