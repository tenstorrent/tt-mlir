// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-materialize-view-returns -o %t %s
// RUN: FileCheck %s --input-file=%t

module {

  // 3D sum reducing last dim (R): tensor<32x1x8192xf32> -> tensor<32x1x1xf32>
  // CHECK-LABEL: func @sum_3d_reduce_last
  func.func @sum_3d_reduce_last(%arg0: tensor<32x1x8192xf32>) -> tensor<32x1x1xf32> {
    // CHECK: d2m.full
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel, #reduction]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel", "reduction"]
    // CHECK: d2m.tile_reduce_sum{{.+}}d2m<reduce_dim R>
    %0 = "ttir.sum"(%arg0) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<32x1x8192xf32>) -> tensor<32x1x1xf32>
    return %0 : tensor<32x1x1xf32>
  }

  // 3D sum reducing second-to-last dim (C): tensor<32x8192x1xf32> -> tensor<32x1x1xf32>
  // CHECK-LABEL: func @sum_3d_reduce_second_to_last
  func.func @sum_3d_reduce_second_to_last(%arg0: tensor<32x8192x1xf32>) -> tensor<32x1x1xf32> {
    // CHECK: d2m.full
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #reduction, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "reduction", "parallel"]
    // CHECK: d2m.tile_reduce_sum{{.+}}d2m<reduce_dim C>
    %0 = "ttir.sum"(%arg0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<32x8192x1xf32>) -> tensor<32x1x1xf32>
    return %0 : tensor<32x1x1xf32>
  }

  // 3D sum reducing both inner dims (RC): tensor<32x128x96xf32> -> tensor<32x1x1xf32>
  // CHECK-LABEL: func @sum_3d_reduce_both_inner
  func.func @sum_3d_reduce_both_inner(%arg0: tensor<32x128x96xf32>) -> tensor<32x1x1xf32> {
    // CHECK: d2m.full
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #reduction, #reduction]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "reduction", "reduction"]
    // CHECK: d2m.tile_reduce_sum{{.+}}d2m<reduce_dim RC>
    %0 = "ttir.sum"(%arg0) <{dim_arg = [1 : i32, 2 : i32], keep_dim = true}> : (tensor<32x128x96xf32>) -> tensor<32x1x1xf32>
    return %0 : tensor<32x1x1xf32>
  }

  // 3D sum reducing last dim with negative index: tensor<32x1x8192xf32> -> tensor<32x1x1xf32>
  // CHECK-LABEL: func @sum_3d_reduce_last_neg
  func.func @sum_3d_reduce_last_neg(%arg0: tensor<32x1x8192xf32>) -> tensor<32x1x1xf32> {
    // CHECK: d2m.full
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel, #reduction]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel", "reduction"]
    // CHECK: d2m.tile_reduce_sum{{.+}}d2m<reduce_dim R>
    %0 = "ttir.sum"(%arg0) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<32x1x8192xf32>) -> tensor<32x1x1xf32>
    return %0 : tensor<32x1x1xf32>
  }

}
