// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttcore-register-device --ttcore-wrap-device-module --cpu-hoist-const-eval --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// CHECK: ttcore.device_module {

// CHECK-LABEL: func.func private @pure_data_movement_single_segment
// CHECK: ttir.permute
// CHECK: ttir.reshape
// CHECK-NOT: call @cpu_hoisted
// CHECK: return
func.func private @pure_data_movement_single_segment(
    %arg0: tensor<32x64xbf16>
) -> tensor<32x64xbf16> attributes {tt.function_type = "const_eval"} {
  %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 1, 0>}> : (tensor<32x64xbf16>) -> tensor<64x32xbf16>
  %1 = "ttir.reshape"(%0) <{shape = [32 : i32, 64 : i32]}> : (tensor<64x32xbf16>) -> tensor<32x64xbf16>
  return %1 : tensor<32x64xbf16>
}

// CHECK-LABEL: func.func private @data_movement_before_ccl_arith_after
// CHECK: ttir.permute
// CHECK-NOT: call @cpu_hoisted
// CHECK: ttir.all_gather
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: return
func.func private @data_movement_before_ccl_arith_after(
    %arg0: tensor<32x64xbf16>, %arg1: tensor<64x32xbf16>
) -> tensor<128x32xbf16> attributes {tt.function_type = "const_eval"} {
  %p = "ttir.permute"(%arg0) <{permutation = array<i64: 1, 0>}> : (tensor<32x64xbf16>) -> tensor<64x32xbf16>
  %g = "ttir.all_gather"(%p) <{all_gather_dim = 0 : si32, cluster_axis = 0 : ui32}> : (tensor<64x32xbf16>) -> tensor<128x32xbf16>
  %e = "ttir.all_gather"(%arg1) <{all_gather_dim = 0 : si32, cluster_axis = 0 : ui32}> : (tensor<64x32xbf16>) -> tensor<128x32xbf16>
  %a = "ttir.add"(%g, %e) : (tensor<128x32xbf16>, tensor<128x32xbf16>) -> tensor<128x32xbf16>
  %m = "ttir.multiply"(%a, %g) : (tensor<128x32xbf16>, tensor<128x32xbf16>) -> tensor<128x32xbf16>
  return %m : tensor<128x32xbf16>
}

// CHECK-LABEL: func.func private @arith_before_ccl_data_movement_after
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: ttir.all_gather
// CHECK: ttir.permute
// CHECK-NOT: call @cpu_hoisted
// CHECK: return
func.func private @arith_before_ccl_data_movement_after(
    %arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>
) -> tensor<32x64xbf16> attributes {tt.function_type = "const_eval"} {
  %a = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %m = "ttir.multiply"(%a, %arg0) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %g = "ttir.all_gather"(%m) <{all_gather_dim = 0 : si32, cluster_axis = 0 : ui32}> : (tensor<32x32xbf16>) -> tensor<64x32xbf16>
  %p = "ttir.permute"(%g) <{permutation = array<i64: 1, 0>}> : (tensor<64x32xbf16>) -> tensor<32x64xbf16>
  %r = "ttir.reshape"(%p) <{shape = [32 : i32, 64 : i32]}> : (tensor<32x64xbf16>) -> tensor<32x64xbf16>
  return %r : tensor<32x64xbf16>
}

// CHECK-LABEL: func.func private @data_movement_with_arith_same_segment
// CHECK-NOT: ttir.permute
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: return
func.func private @data_movement_with_arith_same_segment(
    %arg0: tensor<32x64xbf16>, %arg1: tensor<64x32xbf16>
) -> tensor<64x32xbf16> attributes {tt.function_type = "const_eval"} {
  %p = "ttir.permute"(%arg0) <{permutation = array<i64: 1, 0>}> : (tensor<32x64xbf16>) -> tensor<64x32xbf16>
  %a = "ttir.add"(%p, %arg1) : (tensor<64x32xbf16>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
  return %a : tensor<64x32xbf16>
}

// CHECK-LABEL: func.func private @typecast_only_segment
// CHECK-NOT: ttir.permute
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: return
func.func private @typecast_only_segment(
    %arg0: tensor<32x32xbf16>
) -> tensor<32x32xf32> attributes {tt.function_type = "const_eval"} {
  %0 = "ttir.typecast"(%arg0) : (tensor<32x32xbf16>) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// CHECK-LABEL: func.func private @data_movement_with_typecast
// CHECK: ttir.permute
// CHECK: ttir.typecast
// CHECK-NOT: call @cpu_hoisted
// CHECK: return
func.func private @data_movement_with_typecast(
    %arg0: tensor<32x64xbf16>
) -> tensor<64x32xf32> attributes {tt.function_type = "const_eval"} {
  %p = "ttir.permute"(%arg0) <{permutation = array<i64: 1, 0>}> : (tensor<32x64xbf16>) -> tensor<64x32xbf16>
  %t = "ttir.typecast"(%p) : (tensor<64x32xbf16>) -> tensor<64x32xf32>
  return %t : tensor<64x32xf32>
}

// CHECK-LABEL: func.func private @multiple_data_movement_segments
// CHECK: ttir.permute
// CHECK: ttir.all_gather
// CHECK: ttir.reshape
// CHECK: ttir.permute
// CHECK-NOT: call @cpu_hoisted
// CHECK: return
func.func private @multiple_data_movement_segments(
    %arg0: tensor<32x64xbf16>
) -> tensor<128x32xbf16> attributes {tt.function_type = "const_eval"} {
  %p = "ttir.permute"(%arg0) <{permutation = array<i64: 1, 0>}> : (tensor<32x64xbf16>) -> tensor<64x32xbf16>
  %g = "ttir.all_gather"(%p) <{all_gather_dim = 0 : si32, cluster_axis = 0 : ui32}> : (tensor<64x32xbf16>) -> tensor<128x32xbf16>
  %r = "ttir.reshape"(%g) <{shape = [64 : i32, 64 : i32]}> : (tensor<128x32xbf16>) -> tensor<64x64xbf16>
  %p2 = "ttir.permute"(%r) <{permutation = array<i64: 1, 0>}> : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
  %r2 = "ttir.reshape"(%p2) <{shape = [128 : i32, 32 : i32]}> : (tensor<64x64xbf16>) -> tensor<128x32xbf16>
  return %r2 : tensor<128x32xbf16>
}

// CHECK: ttcore.cpu_module {
