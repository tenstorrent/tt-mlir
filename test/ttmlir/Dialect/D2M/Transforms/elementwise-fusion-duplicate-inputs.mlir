// RUN: ttmlir-opt --split-input-file --ttir-to-ttmetal-fe-pipeline --d2m-elementwise-fusion %s | FileCheck %s

// D2M elementwise fusion when a binary op consumes the same SSA value twice, or both a value and
// a derived value from the same producer chain. Mirrors test_gpt_oss_20b_gate_up.py:
//   test_duplicate_input_fusion:     abs(x) then add(t, t)
//   test_duplicate_input_fusion_2:   abs(x), exp(t), add(t, tt)
// TTIR + FE pipeline (same as eltwise_fusion_skip_implicit_bcast.mlir); plain ttir-to-d2m can
// duplicate layout paths and miss fusion for add(%t,%t).

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
