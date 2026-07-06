// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// topk with k=1 over a dimension of size 1 is a no-op: values == input and
// indices are always zero. Verify it folds away.
func.func @topk_fold_width_one(%arg0: tensor<8x1xbf16>) -> (tensor<8x1xbf16>, tensor<8x1xui16>) {
  // CHECK-LABEL: func.func @topk_fold_width_one
  // CHECK-NOT: "ttir.topk"
  // CHECK: %[[ZEROS:.*]] = "ttir.zeros"() <{shape = array<i32: 8, 1>}> : () -> tensor<8x1xui16>
  // CHECK: return %arg0, %[[ZEROS]]
  %values, %indices = "ttir.topk"(%arg0) {k = 1 : i32, dim = -1 : i32} : (tensor<8x1xbf16>) -> (tensor<8x1xbf16>, tensor<8x1xui16>)
  return %values, %indices : tensor<8x1xbf16>, tensor<8x1xui16>
}

// Same case but with the reduced dim spelled as a positive index.
func.func @topk_fold_width_one_positive_dim(%arg0: tensor<4x1xf32>) -> (tensor<4x1xf32>, tensor<4x1xi32>) {
  // CHECK-LABEL: func.func @topk_fold_width_one_positive_dim
  // CHECK-NOT: "ttir.topk"
  // CHECK: %[[ZEROS:.*]] = "ttir.zeros"() <{shape = array<i32: 4, 1>}> : () -> tensor<4x1xi32>
  // CHECK: return %arg0, %[[ZEROS]]
  %values, %indices = "ttir.topk"(%arg0) {k = 1 : i32, dim = 1 : i32} : (tensor<4x1xf32>) -> (tensor<4x1xf32>, tensor<4x1xi32>)
  return %values, %indices : tensor<4x1xf32>, tensor<4x1xi32>
}

// Negative: reduced dim has size > 1, must NOT fold.
func.func @topk_no_fold_wide_dim(%arg0: tensor<8x128xbf16>) -> (tensor<8x1xbf16>, tensor<8x1xui16>) {
  // CHECK-LABEL: func.func @topk_no_fold_wide_dim
  // CHECK: "ttir.topk"
  %values, %indices = "ttir.topk"(%arg0) {k = 1 : i32, dim = -1 : i32} : (tensor<8x128xbf16>) -> (tensor<8x1xbf16>, tensor<8x1xui16>)
  return %values, %indices : tensor<8x1xbf16>, tensor<8x1xui16>
}
