// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for sort operation

module {
  func.func @test_sort(%arg0: tensor<64x128xbf16>) -> (tensor<64x128xbf16>, tensor<64x128xi16>) {
    // CHECK: error: 'ttir.sort' op Dimension out of range (expected to be in range of [-2, 1], but got -3)
    %1, %2 = "ttir.sort"(%arg0) <{dim = -3 : si32}> : (tensor<64x128xbf16>) -> (tensor<64x128xbf16>, tensor<64x128xi16>)
    return %1, %2 : tensor<64x128xbf16>, tensor<64x128xi16>
  }
}

// -----
module {
  func.func @test_sort(%arg0: tensor<64x128xbf16>) -> (tensor<64x128xbf16>) {
    // CHECK: error: 'ttir.sort' op Expected number of outputs = 2 but got 1
    %1 = "ttir.sort"(%arg0) : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %1 : tensor<64x128xbf16>
  }
}
