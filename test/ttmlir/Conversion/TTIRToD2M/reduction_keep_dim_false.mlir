// RUN: ttmlir-opt --mlir-print-local-scope --ttcore-register-device --ttir-to-d2m %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @sum_2d_reduce_last_keep_dim_false
  func.func @sum_2d_reduce_last_keep_dim_false(%arg0: tensor<64x128xbf16>) -> tensor<64xbf16> {
    // CHECK: d2m.empty() : tensor<64x1xbf16>
    // CHECK: d2m.generic {{.*}}indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, 0)>, affine_map<(d0, d1) -> (d0, 0)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<reduction>]
    // CHECK: "d2m.tile_reduce_sum"{{.*}}reduce_dim = #d2m<reduce_dim R>
    // CHECK-NOT: "d2m.tile_transpose"
    // CHECK: d2m.to_layout {{.*}} into tensor<64xbf16>
    // CHECK-NOT: d2m.generic
    // CHECK: return
    %0 = "ttir.sum"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<64x128xbf16>) -> tensor<64xbf16>
    return %0 : tensor<64xbf16>
  }

  // CHECK-LABEL: func.func @sum_2d_reduce_first_keep_dim_false
  func.func @sum_2d_reduce_first_keep_dim_false(%arg0: tensor<64x128xbf16>) -> tensor<128xbf16> {
    // CHECK: d2m.generic {{.*}}indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, 0)>, affine_map<(d0, d1) -> (0, d1)>], iterator_types = [#ttcore.iterator_type<reduction>, #ttcore.iterator_type<parallel>]
    // CHECK: "d2m.tile_reduce_sum"{{.*}}reduce_dim = #d2m<reduce_dim C>
    // CHECK-NOT: "d2m.tile_transpose"
    // CHECK: return
    %0 = "ttir.sum"(%arg0) <{dim_arg = [0 : i32], keep_dim = false}> : (tensor<64x128xbf16>) -> tensor<128xbf16>
    return %0 : tensor<128xbf16>
  }

  // CHECK-LABEL: func.func @sum_3d_reduce_last_keep_dim_false
  func.func @sum_3d_reduce_last_keep_dim_false(%arg0: tensor<2x64x128xbf16>) -> tensor<2x64xbf16> {
    // CHECK: d2m.generic {{.*}}indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (0, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>, #ttcore.iterator_type<reduction>]
    // CHECK: "d2m.tile_reduce_sum"{{.*}}reduce_dim = #d2m<reduce_dim R>
    // CHECK-NOT: "d2m.tile_transpose"
    // CHECK: d2m.to_layout {{.*}} into tensor<2x64xbf16>
    // CHECK: return
    %0 = "ttir.sum"(%arg0) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<2x64x128xbf16>) -> tensor<2x64xbf16>
    return %0 : tensor<2x64xbf16>
  }

  // CHECK-LABEL: func.func @sum_4d_reduce_outer_keep_dim_false
  func.func @sum_4d_reduce_outer_keep_dim_false(%arg0: tensor<2x3x64x128xbf16>) -> tensor<2x64x128xbf16> {
    // CHECK: d2m.generic {{.*}}indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<reduction>, #ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>]
    // CHECK: "d2m.tile_add"
    // CHECK: d2m.remote_store
    // CHECK-NOT: d2m.generic
    // CHECK: return
    %0 = "ttir.sum"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<2x3x64x128xbf16>) -> tensor<2x64x128xbf16>
    return %0 : tensor<2x64x128xbf16>
  }

  // CHECK-LABEL: func.func @i32_min_2d_reduce_last_keep_dim_false
  func.func @i32_min_2d_reduce_last_keep_dim_false(%arg0: tensor<128x64xsi32>) -> tensor<128xsi32> {
    // CHECK: "d2m.tile_sfpu_reduce_max"{{.*}}reduce_dim = #d2m<reduce_dim R>
    // CHECK: "d2m.tile_bitwise_not"
    // CHECK-NOT: d2m.view_layout
    // CHECK: d2m.to_layout {{.*}} : tensor<128x1xsi32> into tensor<128xsi32>
    // CHECK-NOT: d2m.view_layout
    // CHECK: return
    %0 = "ttir.min"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<128x64xsi32>) -> tensor<128xsi32>
    return %0 : tensor<128xsi32>
  }

  // CHECK-LABEL: func.func @i32_min_scalar_keep_dim_false
  func.func @i32_min_scalar_keep_dim_false(%arg0: tensor<64x64xsi32>) -> tensor<si32> {
    // CHECK: d2m.to_layout {{.*}} into tensor<si32>
    // CHECK: d2m.to_layout {{.*}} into tensor<1x1x1x1x!ttcore.tile<32x32, si32>
    // CHECK: d2m.generic {{.*}}indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>]
    // CHECK: d2m.remote_load
    // CHECK-SAME: [%block{{.*}}, %block{{.*}}]
    // CHECK: "d2m.tile_bitwise_not"
    // CHECK: return
    %0 = "ttir.min"(%arg0) <{dim_arg = [0 : i32, 1 : i32], keep_dim = false}> : (tensor<64x64xsi32>) -> tensor<si32>
    return %0 : tensor<si32>
  }
}
