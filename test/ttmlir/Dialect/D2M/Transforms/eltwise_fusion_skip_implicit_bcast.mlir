// RUN: ttmlir-opt --split-input-file --ttir-to-ttmetal-fe-pipeline --d2m-elementwise-fusion %s | FileCheck %s

module {
  func.func @check_fusion_disabled(%arg1: tensor<32x1xbf16>, %arg2: tensor<1x32xbf16>, %arg3: tensor<1x1xbf16>) -> tensor<32x32xbf16> {
    // CHECK: d2m.generic
    // CHECK: d2m.tile_bcast
    // CHECK: d2m.tile_add
    %0 = ttir.empty() : tensor<1x32xbf16>
    %1 = "ttir.add"(%arg3, %arg2, %0) : (tensor<1x1xbf16>, tensor<1x32xbf16>, tensor<1x32xbf16>) -> tensor<1x32xbf16>
    // CHECK: d2m.generic
    // CHECK: d2m.tile_bcast
    // CHECK: d2m.tile_sub
    %2 = ttir.empty() : tensor<1x32xbf16>
    %3 = "ttir.subtract"(%1, %arg3, %2) : (tensor<1x32xbf16>, tensor<1x1xbf16>, tensor<1x32xbf16>) -> tensor<1x32xbf16>
    // CHECK: d2m.generic
    // CHECK: d2m.tile_bcast
    // CHECK: d2m.tile_add
    %4 = ttir.empty() : tensor<32x1xbf16>
    %5 = "ttir.add"(%arg1, %arg3, %4) : (tensor<32x1xbf16>, tensor<1x1xbf16>, tensor<32x1xbf16>) -> tensor<32x1xbf16>
    // CHECK: d2m.generic
    // CHECK: d2m.tile_bcast
    // CHECK: d2m.tile_sub
    %6 = ttir.empty() : tensor<32x1xbf16>
    %7 = "ttir.subtract"(%arg3, %5, %6) : (tensor<1x1xbf16>, tensor<32x1xbf16>, tensor<32x1xbf16>) -> tensor<32x1xbf16>
    // CHECK: d2m.generic
    // CHECK: d2m.tile_bcast
    // CHECK: d2m.tile_bcast
    // CHECK: d2m.tile_add
    %8 = ttir.empty() : tensor<32x32xbf16>
    %9 = "ttir.add"(%7, %3, %8) : (tensor<32x1xbf16>, tensor<1x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %9 : tensor<32x32xbf16>
  }
}
