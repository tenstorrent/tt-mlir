// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// Tests for ttir.gather_dim lowering to ttnn.gather with si32 indices.
// The runtime (gather.cpp) typecasts INT32 -> UINT32 before calling ::ttnn::gather.
// All indices here are non-negative; the runtime assumes that contract.

// Basic gather along dim 0 with si32 indices.
func.func @gather_dim0_si32(%input: tensor<3x4xbf16>, %idx: tensor<2x4xi32>) -> tensor<2x4xbf16> {
  // CHECK-LABEL: func.func @gather_dim0_si32
  // CHECK: "ttnn.gather"({{.*}}) <{dim = 0 : i32}>
  // CHECK-SAME: (tensor<3x4xbf16, {{.*}}>, tensor<2x4xsi32, {{.*}}>) -> tensor<2x4xbf16, {{.*}}>
  %0 = "ttir.gather_dim"(%input, %idx) <{dim = 0 : i32}> : (tensor<3x4xbf16>, tensor<2x4xi32>) -> tensor<2x4xbf16>
  return %0 : tensor<2x4xbf16>
}

// Gather along dim 1 with si32 indices.
func.func @gather_dim1_si32(%input: tensor<3x4xf32>, %idx: tensor<3x2xi32>) -> tensor<3x2xf32> {
  // CHECK-LABEL: func.func @gather_dim1_si32
  // CHECK: "ttnn.gather"({{.*}}) <{dim = 1 : i32}>
  // CHECK-SAME: tensor<3x2xsi32, {{.*}}>
  %0 = "ttir.gather_dim"(%input, %idx) <{dim = 1 : i32}> : (tensor<3x4xf32>, tensor<3x2xi32>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// Higher-rank gather with si32 indices.
func.func @gather_4d_si32(%input: tensor<2x3x4x5xbf16>, %idx: tensor<2x3x2x5xi32>) -> tensor<2x3x2x5xbf16> {
  // CHECK-LABEL: func.func @gather_4d_si32
  // CHECK: "ttnn.gather"({{.*}}) <{dim = 2 : i32}>
  // CHECK-SAME: tensor<2x3x2x5xsi32, {{.*}}>
  %0 = "ttir.gather_dim"(%input, %idx) <{dim = 2 : i32}> : (tensor<2x3x4x5xbf16>, tensor<2x3x2x5xi32>) -> tensor<2x3x2x5xbf16>
  return %0 : tensor<2x3x2x5xbf16>
}

// Constant si32 indices: input [10..33], expected output rows {0,1} = [[10,21,32,13],[20,31,12,23]].
// Const-evaluated, so the gather op lands in a hoisted `_const_eval_0` function.
func.func @gather_const_si32() -> tensor<2x4xbf16> {
  // CHECK-LABEL: func.func private @gather_const_si32_const_eval_0
  // CHECK: "ttnn.gather"
  // CHECK-SAME: tensor<2x4xsi32, {{.*}}>
  // CHECK-LABEL: func.func @gather_const_si32(
  %input = "ttir.constant"() <{value = dense<[
    [1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01],
    [2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01],
    [3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01]
  ]> : tensor<3x4xbf16>}> : () -> tensor<3x4xbf16>
  %idx = "ttir.constant"() <{value = dense<[
    [0, 1, 2, 0],
    [1, 2, 0, 1]
  ]> : tensor<2x4xi32>}> : () -> tensor<2x4xi32>
  %0 = "ttir.gather_dim"(%input, %idx) <{dim = 0 : i32}> : (tensor<3x4xbf16>, tensor<2x4xi32>) -> tensor<2x4xbf16>
  return %0 : tensor<2x4xbf16>
}

// Constant si32 indices at the upper bound of the gather dim (size-1).
// Verifies max valid index does not trip the typecast wrap.
func.func @gather_const_si32_max_idx() -> tensor<2x4xbf16> {
  // CHECK-LABEL: func.func private @gather_const_si32_max_idx_const_eval_0
  // CHECK: "ttnn.gather"
  // CHECK-SAME: tensor<2x4xsi32, {{.*}}>
  // CHECK-LABEL: func.func @gather_const_si32_max_idx(
  %input = "ttir.constant"() <{value = dense<[
    [1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01],
    [2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01],
    [3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01]
  ]> : tensor<3x4xbf16>}> : () -> tensor<3x4xbf16>
  %idx = "ttir.constant"() <{value = dense<[
    [2, 2, 2, 2],
    [2, 2, 2, 2]
  ]> : tensor<2x4xi32>}> : () -> tensor<2x4xi32>
  %0 = "ttir.gather_dim"(%input, %idx) <{dim = 0 : i32}> : (tensor<3x4xbf16>, tensor<2x4xi32>) -> tensor<2x4xbf16>
  return %0 : tensor<2x4xbf16>
}

// Sanity: ui32 indices should still work (no typecast inserted at runtime).
func.func @gather_dim0_ui32(%input: tensor<3x4xbf16>, %idx: tensor<2x4xui32>) -> tensor<2x4xbf16> {
  // CHECK-LABEL: func.func @gather_dim0_ui32
  // CHECK: "ttnn.gather"({{.*}}) <{dim = 0 : i32}>
  // CHECK-SAME: tensor<2x4xui32, {{.*}}>
  %0 = "ttir.gather_dim"(%input, %idx) <{dim = 0 : i32}> : (tensor<3x4xbf16>, tensor<2x4xui32>) -> tensor<2x4xbf16>
  return %0 : tensor<2x4xbf16>
}
