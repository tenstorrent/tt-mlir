// RUN: ttmlir-opt --ttcore-register-device --d2m-lower-to-layout %s | FileCheck %s

#layout_squeezed_src = #ttcore.metal_layout<logical_shape = 128x1, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#layout_squeezed_dst = #ttcore.metal_layout<logical_shape = 128, dim_alignments = 32, collapsed_intervals = dense<[[0, 0], [0, 1]]> : tensor<2x2xi64>, l1, sharded>

// CHECK-DAG: #[[SRC:.*]] = #ttcore.metal_layout<logical_shape = 128x1,
// CHECK-DAG: #[[SQUEEZED_TRANSFER:.*]] = #ttcore.metal_layout<logical_shape = 128,

func.func @device_keepdim_to_squeezed_host(%arg0: tensor<1x1x128x1xbf16, #layout_squeezed_src>) -> tensor<128xbf16> {
  %0 = d2m.empty() : tensor<128xbf16>

  // CHECK-LABEL: @device_keepdim_to_squeezed_host
  // CHECK: d2m.to_host %arg0, %{{.*}} layout = #[[SQUEEZED_TRANSFER]] : tensor<1x1x128x1xbf16, #[[SRC]]> into tensor<128xbf16> -> tensor<128xbf16>
  %1 = d2m.to_layout %arg0, %0 : tensor<1x1x128x1xbf16, #layout_squeezed_src> into tensor<128xbf16>
    -> tensor<128xbf16>

  return %1 : tensor<128xbf16>
}

func.func @device_keepdim_to_squeezed_device(%arg0: tensor<1x1x4x1x!ttcore.tile<32x32, si32>, #layout_squeezed_src>) -> tensor<1x1x1x4x!ttcore.tile<32x32, si32>, #layout_squeezed_dst> {
  %0 = d2m.empty() : tensor<1x1x1x4x!ttcore.tile<32x32, si32>, #layout_squeezed_dst>

  // CHECK-LABEL: @device_keepdim_to_squeezed_device
  // CHECK: d2m.view_layout
  // CHECK: d2m.generic
  // CHECK: return
  %1 = d2m.to_layout %arg0, %0 : tensor<1x1x4x1x!ttcore.tile<32x32, si32>, #layout_squeezed_src> into tensor<1x1x1x4x!ttcore.tile<32x32, si32>, #layout_squeezed_dst>
    -> tensor<1x1x1x4x!ttcore.tile<32x32, si32>, #layout_squeezed_dst>

  return %1 : tensor<1x1x1x4x!ttcore.tile<32x32, si32>, #layout_squeezed_dst>
}
