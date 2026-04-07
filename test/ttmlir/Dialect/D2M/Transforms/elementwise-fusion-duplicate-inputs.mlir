// RUN: ttmlir-opt --split-input-file --ttir-to-ttmetal-fe-pipeline --d2m-elementwise-fusion %s | FileCheck %s

module {
  func.func @test_duplicate_input_fusion_1(%arg0: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
    // CHECK-LABEL: func @test_duplicate_input_fusion_1
    // CHECK: d2m.generic
    // CHECK: d2m.tile_abs
    // CHECK-NOT: d2m.generic
    // CHECK: d2m.tile_add
    %t = "ttir.abs"(%arg0) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %out = "ttir.add"(%t, %t) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %out : tensor<64x64xbf16>
  }
}

// -----

module {
  func.func @test_duplicate_input_fusion_2(%arg0: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
    // CHECK-LABEL: func @test_duplicate_input_fusion_2
    // CHECK: d2m.generic
    // CHECK: d2m.tile_abs
    // CHECK-NOT: d2m.generic
    // CHECK: d2m.tile_exp
    // CHECK-NOT: d2m.generic
    // CHECK: d2m.tile_add
    %t = "ttir.abs"(%arg0) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %tt = "ttir.exp"(%t) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %out = "ttir.add"(%t, %tt) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %out : tensor<64x64xbf16>
  }
}
