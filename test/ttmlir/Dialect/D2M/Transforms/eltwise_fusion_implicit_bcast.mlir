// RUN: ttmlir-opt --split-input-file --d2m-fe-pipeline="enable-elementwise-fusion=true" %s | FileCheck %s

// All five eltwise ops (4 broadcasted binary + 1 final 2D-broadcast add) must
// collapse into a SINGLE compute d2m.generic. Tilize/untilize boundary generics
// around input/output buffers are unrelated and may still appear; what we
// forbid is any *additional* compute generic between the fused
// tile_bcast/tile_add/tile_sub sequence.
module {
  func.func @bcast_fusion_enabled(%arg1: tensor<32x1xbf16>, %arg2: tensor<1x32xbf16>, %arg3: tensor<1x1xbf16>) -> tensor<32x32xbf16> {
    %1 = "ttir.add"(%arg3, %arg2) : (tensor<1x1xbf16>, tensor<1x32xbf16>) -> tensor<1x32xbf16>
    %3 = "ttir.subtract"(%1, %arg3) : (tensor<1x32xbf16>, tensor<1x1xbf16>) -> tensor<1x32xbf16>
    %5 = "ttir.add"(%arg1, %arg3) : (tensor<32x1xbf16>, tensor<1x1xbf16>) -> tensor<32x1xbf16>
    %7 = "ttir.subtract"(%arg3, %5) : (tensor<1x1xbf16>, tensor<32x1xbf16>) -> tensor<32x1xbf16>
    %9 = "ttir.add"(%7, %3) : (tensor<32x1xbf16>, tensor<1x32xbf16>) -> tensor<32x32xbf16>
    return %9 : tensor<32x32xbf16>
  }
}

// CHECK-LABEL: func.func @bcast_fusion_enabled
// CHECK:           "d2m.tile_bcast"
// CHECK:           "d2m.tile_add"
// CHECK-NOT:       d2m.generic
// CHECK:           "d2m.tile_bcast"
// CHECK:           "d2m.tile_sub"
// CHECK-NOT:       d2m.generic
// CHECK:           "d2m.tile_bcast"
// CHECK:           "d2m.tile_add"
// CHECK-NOT:       d2m.generic
// CHECK:           "d2m.tile_bcast"
// CHECK:           "d2m.tile_sub"
// CHECK-NOT:       d2m.generic
// CHECK:           "d2m.tile_bcast"
// CHECK:           "d2m.tile_bcast"
// CHECK:           "d2m.tile_add"
