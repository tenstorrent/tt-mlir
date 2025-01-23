// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative test for moreh cumulative sum operation

// Verify that the parsing fails if specified dim is negative.
module attributes {} {
  func.func public @test_moreh_cumsum_neg_dim(%arg0: tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16> {
    // CHECK: error: 'ttir.moreh_cumsum' op specified dimension cannot be negative.
    %0 = tensor.empty() : tensor<1x32x128x128xbf16>
    %1 = "ttir.moreh_cumsum"(%arg0, %0) <{dim = -2 : i64}> : (tensor<1x32x128x128xbf16>, tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16>
    return %1 : tensor<1x32x128x128xbf16>
  }
}
