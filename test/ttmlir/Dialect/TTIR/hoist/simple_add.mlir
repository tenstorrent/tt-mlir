// RUN: ttmlir-opt --ttcore-wrap-device-module --cpu-hoist-transform --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// CHECK: ttcore.device_module {
// CHECK: builtin.module {

// CHECK: func.func @add1
func.func @add1(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  // CHECK: %{{.*}} = ttir.to_layout %{{.*}}, %{{.*}}
  // CHECK: %{{.*}} = ttir.to_layout %{{.*}}, %{{.*}}
  // CHECK: %{{.*}} = call @cpu_hoisted_ttir_add_{{.*}}
  %1 = "ttir.add"(%arg0, %arg1) {ttir.should_hoist} : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  // CHECK: %{{.*}} = ttir.to_layout %{{.*}}, %{{.*}}
  return %1 : tensor<32x32xbf16>
}

// CHECK: func.func @add2
func.func @add2(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK: %{{.*}} = call @cpu_hoisted_ttir_add_{{.*}}
  %1 = "ttir.add"(%arg0, %arg1) {ttir.should_hoist} : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}

// CHECK: func.func @add3
func.func @add3(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK: %{{.*}} = call @cpu_hoisted_ttir_add_{{.*}}
  %1 = "ttir.add"(%arg0, %arg1) {ttir.should_hoist} : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}

// CHECK: func.func @add4
func.func @add4(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  // CHECK: %{{.*}} = ttir.to_layout %{{.*}}, %{{.*}}
  // CHECK: %{{.*}} = ttir.to_layout %{{.*}}, %{{.*}}
  // CHECK: %{{.*}} = call @cpu_hoisted_ttir_add_{{.*}}
  %1 = "ttir.add"(%arg0, %arg1) {ttir.should_hoist} : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  // CHECK: %{{.*}} = ttir.to_layout %{{.*}}, %{{.*}}
  return %1 : tensor<32x32xbf16>
}

// All add operations should share the same CPU-hoisted function since they all
// convert to f32 before calling. Verify there is only ONE declaration and ONE
// definition.
// CHECK-COUNT-1: func.func private @cpu_hoisted_ttir_add_{{.*}}

// CHECK: ttcore.cpu_module {
// CHECK: builtin.module {
// CHECK-COUNT-1: func.func @cpu_hoisted_ttir_add_{{[^_]*}} {
