// RUN: ttmlir-opt --split-input-file --ttir-to-ttmetal-fe-pipeline --d2m-elementwise-fusion %s | FileCheck %s

module {
  func.func @check_fusion_disabled(%arg1: tensor<32x1xbf16>, %arg2: tensor<1x32xbf16>, %arg3: tensor<1x1xbf16>) -> tensor<32x32xbf16> {
    // CHECK: d2m.generic
    // CHECK: d2m.tile_bcast
    // CHECK: d2m.tile_add
    %1 = "ttir.add"(%arg3, %arg2) : (tensor<1x1xbf16>, tensor<1x32xbf16>) -> tensor<1x32xbf16>
    // CHECK: d2m.generic
    // CHECK: d2m.tile_bcast
    // CHECK: d2m.tile_sub
    %3 = "ttir.subtract"(%1, %arg3) : (tensor<1x32xbf16>, tensor<1x1xbf16>) -> tensor<1x32xbf16>
    // CHECK: d2m.generic
    // CHECK: d2m.tile_bcast
    // CHECK: d2m.tile_add
    %5 = "ttir.add"(%arg1, %arg3) : (tensor<32x1xbf16>, tensor<1x1xbf16>) -> tensor<32x1xbf16>
    // CHECK: d2m.generic
    // CHECK: d2m.tile_bcast
    // CHECK: d2m.tile_sub
    %7 = "ttir.subtract"(%arg3, %5) : (tensor<1x1xbf16>, tensor<32x1xbf16>) -> tensor<32x1xbf16>
    // CHECK: d2m.generic
    // CHECK: d2m.tile_bcast
    // CHECK: d2m.tile_bcast
    // CHECK: d2m.tile_add
    %9 = "ttir.add"(%7, %3) : (tensor<32x1xbf16>, tensor<1x32xbf16>) -> tensor<32x32xbf16>
    return %9 : tensor<32x32xbf16>
  }
}
