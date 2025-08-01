// RUN: ttmlir-opt %s | ttmlir-opt | FileCheck %s
// Tests for SFPU dialect operations

// CHECK-LABEL: func.func @sfpi_basic_ops
func.func @sfpi_basic_ops(%arg0: vector<4x8xf32>, %arg1: vector<4x8xf32>) -> vector<4x8xf32> {
  // CHECK: sfpi.add %{{.*}}, %{{.*}} : vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>
  %0 = sfpi.add %arg0, %arg1 : vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>
  
  // CHECK: sfpi.mul %{{.*}}, %{{.*}} : vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>
  %1 = sfpi.mul %0, %arg1 : vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>
  
  // CHECK: sfpi.abs %{{.*}} : vector<4x8xf32> -> vector<4x8xf32>
  %2 = sfpi.abs %1 : vector<4x8xf32> -> vector<4x8xf32>
  
  return %2 : vector<4x8xf32>
}

// CHECK-LABEL: func.func @sfpi_mad_op
func.func @sfpi_mad_op(%arg0: vector<4x8xf32>, %arg1: vector<4x8xf32>, %arg2: vector<4x8xf32>) -> vector<4x8xf32> {
  // CHECK: sfpi.mad %{{.*}}, %{{.*}}, %{{.*}} : vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>
  %0 = sfpi.mad %arg0, %arg1, %arg2 : vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>
  return %0 : vector<4x8xf32>
}

// CHECK-LABEL: func.func @sfpi_comparison_ops
func.func @sfpi_comparison_ops(%arg0: vector<4x8xf32>, %arg1: vector<4x8xf32>) -> vector<4x8xi32> {
  // CHECK: sfpi.gt %{{.*}}, %{{.*}} : vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xi32>
  %0 = sfpi.gt %arg0, %arg1 : vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xi32>
  return %0 : vector<4x8xi32>
}

// CHECK-LABEL: func.func @sfpi_bitwise_ops
func.func @sfpi_bitwise_ops(%arg0: vector<4x8xf32>, %arg1: vector<4x8xf32>) -> vector<4x8xf32> {
  // CHECK: sfpi.and %{{.*}}, %{{.*}} : vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>
  %0 = sfpi.and %arg0, %arg1 : vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>
  
  // CHECK: sfpi.or %{{.*}}, %{{.*}} : vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>
  %1 = sfpi.or %0, %arg1 : vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>
  
  // CHECK: sfpi.not %{{.*}} : vector<4x8xf32> -> vector<4x8xf32>
  %2 = sfpi.not %1 : vector<4x8xf32> -> vector<4x8xf32>
  
  return %2 : vector<4x8xf32>
}

// CHECK-LABEL: func.func @sfpi_live_variants
func.func @sfpi_live_variants(%arg0: vector<4x8xf32>, %arg1: vector<4x8xf32>) -> vector<4x8xf32> {
  // CHECK: sfpi.add_lv %{{.*}}, %{{.*}} : vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>
  %0 = sfpi.add_lv %arg0, %arg1 : vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>
  
  // CHECK: sfpi.mul_lv %{{.*}}, %{{.*}} : vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>
  %1 = sfpi.mul_lv %0, %arg1 : vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>
  
  return %1 : vector<4x8xf32>
}

// CHECK-LABEL: func.func @sfpi_fp_manipulation
func.func @sfpi_fp_manipulation(%arg0: vector<4x8xf32>, %arg1: vector<4x8xi32>) -> (vector<4x8xi32>, vector<4x8xf32>) {
  // CHECK: sfpi.exexp %{{.*}} : vector<4x8xf32> -> vector<4x8xi32>
  %0 = sfpi.exexp %arg0 : vector<4x8xf32> -> vector<4x8xi32>
  
  // CHECK: sfpi.setexp %{{.*}}, %{{.*}} : vector<4x8xf32>, vector<4x8xi32> -> vector<4x8xf32>
  %1 = sfpi.setexp %arg0, %arg1 : vector<4x8xf32>, vector<4x8xi32> -> vector<4x8xf32>
  
  return %0, %1 : vector<4x8xi32>, vector<4x8xf32>
}

// CHECK-LABEL: func.func @sfpi_specialized_ops
func.func @sfpi_specialized_ops(%arg0: vector<4x8xf32>) -> vector<4x8xf32> {
  // CHECK: sfpi.arecip %{{.*}} : vector<4x8xf32> -> vector<4x8xf32>
  %0 = sfpi.arecip %arg0 : vector<4x8xf32> -> vector<4x8xf32>
  
  // CHECK: sfpi.stochrnd %{{.*}} : vector<4x8xf32> -> vector<4x8xf32>
  %1 = sfpi.stochrnd %0 : vector<4x8xf32> -> vector<4x8xf32>
  
  return %1 : vector<4x8xf32>
}