// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  // Reduction over a unit dimension leaves the values unchanged.
  func.func @sum_unit_dim() -> tensor<16x1xbf16> {
    // CHECK-LABEL: @sum_unit_dim
    // CHECK: "ttir.ones"() <{shape = array<i32: 16, 1>}>
    // CHECK-NOT: "ttir.sum"
    %0 = "ttir.ones"() <{shape = array<i32: 16, 1, 1>}> : () -> tensor<16x1x1xbf16>
    %1 = "ttir.sum"(%0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<16x1x1xbf16>) -> tensor<16x1xbf16>
    return %1 : tensor<16x1xbf16>
  }

  // Reduction of a splat integer constant.
  func.func @sum_splat_int() -> tensor<2xsi32> {
    // CHECK-LABEL: @sum_splat_int
    // CHECK: "ttir.full"() <{fill_value = 12 : i32
    // CHECK-NOT: "ttir.sum"
    %0 = "ttir.constant"() <{value = dense<3> : tensor<2x4xsi32>}> : () -> tensor<2x4xsi32>
    %1 = "ttir.sum"(%0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<2x4xsi32>) -> tensor<2xsi32>
    return %1 : tensor<2xsi32>
  }

  // Reduction of a splat float constant.
  func.func @sum_splat_float() -> tensor<2xf32> {
    // CHECK-LABEL: @sum_splat_float
    // CHECK: "ttir.full"() <{fill_value = 1.000000e+01 : f32
    // CHECK-NOT: "ttir.sum"
    %0 = "ttir.constant"() <{value = dense<2.5> : tensor<2x4xf32>}> : () -> tensor<2x4xf32>
    %1 = "ttir.sum"(%0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<2x4xf32>) -> tensor<2xf32>
    return %1 : tensor<2xf32>
  }

  // keep_dim = true retains the reduced dimension as size 1.
  func.func @sum_keepdim_splat() -> tensor<2x1xsi32> {
    // CHECK-LABEL: @sum_keepdim_splat
    // CHECK: "ttir.full"() <{fill_value = 12 : i32, shape = array<i32: 2, 1>}>
    // CHECK-NOT: "ttir.sum"
    %0 = "ttir.constant"() <{value = dense<3> : tensor<2x4xsi32>}> : () -> tensor<2x4xsi32>
    %1 = "ttir.sum"(%0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<2x4xsi32>) -> tensor<2x1xsi32>
    return %1 : tensor<2x1xsi32>
  }

  // keep_dim = true over a unit dimension is a no-op on both values and shape.
  func.func @sum_keepdim_unit() -> tensor<16x1x1xbf16> {
    // CHECK-LABEL: @sum_keepdim_unit
    // CHECK: "ttir.ones"() <{shape = array<i32: 16, 1, 1>}>
    // CHECK-NOT: "ttir.sum"
    %0 = "ttir.ones"() <{shape = array<i32: 16, 1, 1>}> : () -> tensor<16x1x1xbf16>
    %1 = "ttir.sum"(%0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<16x1x1xbf16>) -> tensor<16x1x1xbf16>
    return %1 : tensor<16x1x1xbf16>
  }

  // A non-splat real reduction is not folded.
  func.func @sum_nonsplat(%arg0: tensor<2x4xf32>) -> tensor<2xf32> {
    // CHECK-LABEL: @sum_nonsplat
    // CHECK: "ttir.sum"
    %1 = "ttir.sum"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<2x4xf32>) -> tensor<2xf32>
    return %1 : tensor<2xf32>
  }

  // A non-splat constant is not folded even over a unit dimension: folding is
  // restricted to splats to avoid materializing large constants.
  func.func @sum_nonsplat_unit_dim() -> tensor<3xsi32> {
    // CHECK-LABEL: @sum_nonsplat_unit_dim
    // CHECK: "ttir.sum"
    %0 = "ttir.constant"() <{value = dense<[[1], [2], [3]]> : tensor<3x1xsi32>}> : () -> tensor<3x1xsi32>
    %1 = "ttir.sum"(%0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<3x1xsi32>) -> tensor<3xsi32>
    return %1 : tensor<3xsi32>
  }
}
