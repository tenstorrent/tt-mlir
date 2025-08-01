// RUN: ttmlir-opt %s | ttmlir-opt | FileCheck %s
// Tests for SFPU dialect operations

// CHECK-LABEL: func.func @sfpu_basic_ops
func.func @sfpu_basic_ops(%arg0: vector<64xf32>, %arg1: vector<64xf32>) -> vector<64xf32> {
  // CHECK: sfpu.add %{{.*}}, %{{.*}} : vector<64xf32>
  %0 = sfpu.add %arg0, %arg1 : vector<64xf32>
  
  // CHECK: sfpu.mul %{{.*}}, %{{.*}} : vector<64xf32>
  %1 = sfpu.mul %0, %arg1 : vector<64xf32>
  
  // CHECK: sfpu.abs %{{.*}} : vector<64xf32>
  %2 = sfpu.abs %1 : vector<64xf32>
  
  return %2 : vector<64xf32>
}

// CHECK-LABEL: func.func @sfpu_mad_op
func.func @sfpu_mad_op(%arg0: vector<64xf32>, %arg1: vector<64xf32>, %arg2: vector<64xf32>) -> vector<64xf32> {
  // CHECK: sfpu.mad %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>
  %0 = sfpu.mad %arg0, %arg1, %arg2 : vector<64xf32>
  return %0 : vector<64xf32>
}

// CHECK-LABEL: func.func @sfpu_comparison_ops
func.func @sfpu_comparison_ops(%arg0: vector<64xf32>, %arg1: vector<64xf32>) -> vector<64xi32> {
  // CHECK: sfpu.gt %{{.*}}, %{{.*}} : vector<64xf32> -> vector<64xi32>
  %0 = sfpu.gt %arg0, %arg1 : vector<64xf32> -> vector<64xi32>
  return %0 : vector<64xi32>
}

// CHECK-LABEL: func.func @sfpu_bitwise_ops
func.func @sfpu_bitwise_ops(%arg0: vector<64xf32>, %arg1: vector<64xf32>) -> vector<64xf32> {
  // CHECK: sfpu.and %{{.*}}, %{{.*}} : vector<64xf32>
  %0 = sfpu.and %arg0, %arg1 : vector<64xf32>
  
  // CHECK: sfpu.or %{{.*}}, %{{.*}} : vector<64xf32>
  %1 = sfpu.or %0, %arg1 : vector<64xf32>
  
  // CHECK: sfpu.not %{{.*}} : vector<64xf32>
  %2 = sfpu.not %1 : vector<64xf32>
  
  return %2 : vector<64xf32>
}

// CHECK-LABEL: func.func @sfpu_live_variants
func.func @sfpu_live_variants(%arg0: vector<64xf32>, %arg1: vector<64xf32>) -> vector<64xf32> {
  // CHECK: sfpu.add_lv %{{.*}}, %{{.*}} : vector<64xf32>
  %0 = sfpu.add_lv %arg0, %arg1 : vector<64xf32>
  
  // CHECK: sfpu.mul_lv %{{.*}}, %{{.*}} : vector<64xf32>
  %1 = sfpu.mul_lv %0, %arg1 : vector<64xf32>
  
  return %1 : vector<64xf32>
}

// CHECK-LABEL: func.func @sfpu_fp_manipulation
func.func @sfpu_fp_manipulation(%arg0: vector<64xf32>, %arg1: vector<64xi32>) -> (vector<64xi32>, vector<64xf32>) {
  // CHECK: sfpu.exexp %{{.*}} : vector<64xf32> -> vector<64xi32>
  %0 = sfpu.exexp %arg0 : vector<64xf32> -> vector<64xi32>
  
  // CHECK: sfpu.setexp %{{.*}}, %{{.*}} : vector<64xf32>, vector<64xi32> -> vector<64xf32>
  %1 = sfpu.setexp %arg0, %arg1 : vector<64xf32>, vector<64xi32> -> vector<64xf32>
  
  return %0, %1 : vector<64xi32>, vector<64xf32>
}

// CHECK-LABEL: func.func @sfpu_specialized_ops
func.func @sfpu_specialized_ops(%arg0: vector<64xf32>) -> vector<64xf32> {
  // CHECK: sfpu.arecip %{{.*}} : vector<64xf32>
  %0 = sfpu.arecip %arg0 : vector<64xf32>
  
  // CHECK: sfpu.stochrnd %{{.*}} : vector<64xf32>
  %1 = sfpu.stochrnd %0 : vector<64xf32>
  
  return %1 : vector<64xf32>
}