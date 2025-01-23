// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative test for moreh cumulative sum operation

module attributes {} {
  func.func public @test_moreh_cumsum_neg_dim(%arg0: tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16> {
    // CHECK: error: 'ttir.cumsum' op specified dimension should be between 0 and 3, but got: -2.
    %0 = tensor.empty() : tensor<1x32x128x128xbf16>
    %1 = "ttir.cumsum"(%arg0, %0) <{dim = -2 : i64}> : (tensor<1x32x128x128xbf16>, tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16>
    return %1 : tensor<1x32x128x128xbf16>
  }
}

// -----

module attributes {} {
  func.func public @test_moreh_cumsum_high_dim(%arg0: tensor<1x32x128xbf16>) -> tensor<1x32x128xbf16> {
    // CHECK: error: 'ttir.cumsum' op specified dimension should be between 0 and 2, but got: 4.
    %0 = tensor.empty() : tensor<1x32x128xbf16>
    %1 = "ttir.cumsum"(%arg0, %0) <{dim = 4 : i64}> : (tensor<1x32x128xbf16>, tensor<1x32x128xbf16>) -> tensor<1x32x128xbf16>
    return %1 : tensor<1x32x128xbf16>
  }
}
