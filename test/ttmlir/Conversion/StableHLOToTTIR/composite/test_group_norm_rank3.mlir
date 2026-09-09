// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @gn_rank3(
      %x: tensor<1x512x16384xf32>,
      %w: tensor<512xf32>,
      %b: tensor<512xf32>) -> tensor<1x512x16384xf32> {
    // CHECK-LABEL: func.func @gn_rank3
    // CHECK: %[[X4D:[0-9]+]] = "ttir.reshape"
    // CHECK-SAME: shape = [1 : i32, 512 : i32, 16384 : i32, 1 : i32]
    // CHECK-SAME: (tensor<1x512x16384xf32>{{.*}}) -> tensor<1x512x16384x1xf32>
    // CHECK: %[[GN:[0-9]+]] = "ttir.group_norm"(%[[X4D]]
    // CHECK-SAME: -> tensor<1x512x16384x1xf32>
    // CHECK: "ttir.reshape"(%[[GN]])
    // CHECK-SAME: shape = [1 : i32, 512 : i32, 16384 : i32]
    // CHECK-SAME: (tensor<1x512x16384x1xf32>{{.*}}) -> tensor<1x512x16384xf32>
    %0 = stablehlo.composite "tenstorrent.group_norm" %x, %w, %b {
        composite_attributes = {channel_dim = 1 : i64, epsilon = 9.99999997E-7 : f32, num_groups = 32 : i64},
        decomposition = @gn_impl_rank3
    } : (tensor<1x512x16384xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x16384xf32>
    return %0 : tensor<1x512x16384xf32>
  }

  func.func @gn_rank4(
      %x: tensor<1x480x1x64xbf16>,
      %w: tensor<480xbf16>,
      %b: tensor<480xbf16>) -> tensor<1x480x1x64xbf16> {
    // CHECK-LABEL: func.func @gn_rank4
    // CHECK-NOT: "ttir.reshape"
    // CHECK: "ttir.group_norm"
    %0 = stablehlo.composite "tenstorrent.group_norm" %x, %w, %b {
        composite_attributes = {channel_dim = 1 : i64, epsilon = 9.99999974E-6 : f32, num_groups = 8 : i64},
        decomposition = @gn_impl_rank4
    } : (tensor<1x480x1x64xbf16>, tensor<480xbf16>, tensor<480xbf16>) -> tensor<1x480x1x64xbf16>
    return %0 : tensor<1x480x1x64xbf16>
  }

  func.func private @gn_impl_rank3(
      %arg0: tensor<1x512x16384xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> tensor<1x512x16384xf32> {
    return %arg0 : tensor<1x512x16384xf32>
  }

  func.func private @gn_impl_rank4(
      %arg0: tensor<1x480x1x64xbf16>, %arg1: tensor<480xbf16>, %arg2: tensor<480xbf16>) -> tensor<1x480x1x64xbf16> {
    return %arg0 : tensor<1x480x1x64xbf16>
  }
}
