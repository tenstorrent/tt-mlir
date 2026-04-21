// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-materialize-view-returns -o %t %s
// RUN: FileCheck %s --input-file=%t

// Integer reductions lower to the SFPU tile_sfpu_reduce_* ops (no scaler tile).

module {

  // i32 sum on last dim → d2m.tile_sfpu_reduce_sum with reduce_dim R; no scaler tile_fill.
  // CHECK-LABEL: func @sum_i32_reduce_R
  // CHECK-NOT: d2m.tile_fill
  // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #reduction]
  // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "reduction"]
  // CHECK: d2m.tile_sfpu_reduce_sum{{.+}}d2m<reduce_dim R>
  // CHECK-NOT: d2m.tile_reduce_sum
  func.func @sum_i32_reduce_R(%arg: tensor<128x96xsi32>) -> tensor<128x1xsi32> {
    %0 = "ttir.sum"(%arg) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<128x96xsi32>) -> tensor<128x1xsi32>
    return %0 : tensor<128x1xsi32>
  }

  // i32 sum on second-to-last dim → reduce_dim C.
  // CHECK-LABEL: func @sum_i32_reduce_C
  // CHECK-NOT: d2m.tile_fill
  // CHECK: d2m.tile_sfpu_reduce_sum{{.+}}d2m<reduce_dim C>
  func.func @sum_i32_reduce_C(%arg: tensor<128x96xsi32>) -> tensor<1x96xsi32> {
    %0 = "ttir.sum"(%arg) <{dim_arg = [-2 : i32], keep_dim = true}> : (tensor<128x96xsi32>) -> tensor<1x96xsi32>
    return %0 : tensor<1x96xsi32>
  }

  // i32 max on last dim → d2m.tile_sfpu_reduce_max.
  // CHECK-LABEL: func @max_i32_reduce_R
  // CHECK-NOT: d2m.tile_fill
  // CHECK: d2m.tile_sfpu_reduce_max{{.+}}d2m<reduce_dim R>
  // CHECK-NOT: d2m.tile_reduce_max
  func.func @max_i32_reduce_R(%arg: tensor<128x96xsi32>) -> tensor<128x1xsi32> {
    %0 = "ttir.max"(%arg) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<128x96xsi32>) -> tensor<128x1xsi32>
    return %0 : tensor<128x1xsi32>
  }

  // i32 min decomposes to bitwise_not → max → bitwise_not inside --ttir-to-d2m
  // (D2MInnerMinDecompositionRewriter uses bitwise_not for integers since
  // -INT_MIN == INT_MIN in two's complement), so we end up with an SFPU max
  // reduce surrounded by tile_bitwise_not.
  // CHECK-LABEL: func @min_i32_reduce_R
  // CHECK: d2m.tile_bitwise_not
  // CHECK: d2m.tile_sfpu_reduce_max{{.+}}d2m<reduce_dim R>
  // CHECK: d2m.tile_bitwise_not
  // CHECK-NOT: d2m.tile_negative
  func.func @min_i32_reduce_R(%arg: tensor<128x96xsi32>) -> tensor<128x1xsi32> {
    %0 = "ttir.min"(%arg) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<128x96xsi32>) -> tensor<128x1xsi32>
    return %0 : tensor<128x1xsi32>
  }

  // 3D i32 sum reducing last dim (R): tensor<32x1x8192xsi32> -> tensor<32x1x1xsi32>
  // CHECK-LABEL: func @sum_i32_3d_reduce_last
  // CHECK-NOT: d2m.tile_fill
  // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel, #reduction]
  // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel", "reduction"]
  // CHECK: d2m.tile_sfpu_reduce_sum{{.+}}d2m<reduce_dim R>
  func.func @sum_i32_3d_reduce_last(%arg0: tensor<32x1x8192xsi32>) -> tensor<32x1x1xsi32> {
    %0 = "ttir.sum"(%arg0) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<32x1x8192xsi32>) -> tensor<32x1x1xsi32>
    return %0 : tensor<32x1x1xsi32>
  }

  // 3D i32 sum reducing both inner dims (RC): tensor<32x128x96xsi32> -> tensor<32x1x1xsi32>
  // CHECK-LABEL: func @sum_i32_3d_reduce_both_inner
  // CHECK-NOT: d2m.tile_fill
  // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #reduction, #reduction]
  // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "reduction", "reduction"]
  // CHECK: d2m.tile_sfpu_reduce_sum{{.+}}d2m<reduce_dim RC>
  func.func @sum_i32_3d_reduce_both_inner(%arg0: tensor<32x128x96xsi32>) -> tensor<32x1x1xsi32> {
    %0 = "ttir.sum"(%arg0) <{dim_arg = [1 : i32, 2 : i32], keep_dim = true}> : (tensor<32x128x96xsi32>) -> tensor<32x1x1xsi32>
    return %0 : tensor<32x1x1xsi32>
  }

  // 3D i32 max reducing second-to-last dim (C).
  // CHECK-LABEL: func @max_i32_3d_reduce_C
  // CHECK-NOT: d2m.tile_fill
  // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #reduction, #parallel]
  // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "reduction", "parallel"]
  // CHECK: d2m.tile_sfpu_reduce_max{{.+}}d2m<reduce_dim C>
  func.func @max_i32_3d_reduce_C(%arg0: tensor<32x8192x1xsi32>) -> tensor<32x1x1xsi32> {
    %0 = "ttir.max"(%arg0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<32x8192x1xsi32>) -> tensor<32x1x1xsi32>
    return %0 : tensor<32x1x1xsi32>
  }
}
