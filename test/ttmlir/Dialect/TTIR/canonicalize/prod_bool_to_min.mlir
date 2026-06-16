// RUN: ttmlir-opt --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // A product reduction over a boolean (comparison) input is equivalent to a
  // logical AND, i.e. a min reduction. It should be rewritten to ttir.min to
  // avoid the memory-heavy ttnn::prod permute on large tensors.
  // CHECK-LABEL: func.func @prod_bool_to_min
  func.func @prod_bool_to_min(%arg0: tensor<1x4x32x32xf32>, %arg1: tensor<1x4x32x32xf32>) -> tensor<1x4x32x1xf32> {
    // CHECK: "ttir.eq"
    // CHECK: "ttir.min"
    // CHECK-NOT: "ttir.prod"
    %0 = "ttir.eq"(%arg0, %arg1) : (tensor<1x4x32x32xf32>, tensor<1x4x32x32xf32>) -> tensor<1x4x32x32xf32>
    %1 = "ttir.prod"(%0) <{dim_arg = [3 : i32], keep_dim = true}> : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x1xf32>
    return %1 : tensor<1x4x32x1xf32>
  }

  // A product reduction over a non-boolean input must be left untouched.
  // CHECK-LABEL: func.func @prod_nonbool_unchanged
  func.func @prod_nonbool_unchanged(%arg0: tensor<1x4x32x32xf32>) -> tensor<1x4x32x1xf32> {
    // CHECK: "ttir.prod"
    // CHECK-NOT: "ttir.min"
    %0 = "ttir.prod"(%arg0) <{dim_arg = [3 : i32], keep_dim = true}> : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x1xf32>
    return %0 : tensor<1x4x32x1xf32>
  }
}
