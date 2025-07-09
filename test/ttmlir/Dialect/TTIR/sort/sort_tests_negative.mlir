// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for sort operation

module {
  func.func @test_sort(%arg0: tensor<64x128xbf16>) -> (tensor<64x128xbf16>, tensor<64x128xi16>) {
    // CHECK: error: 'ttir.sort' op Dimension out of range (expected to be in range of [-2, 1], but got -3)
    %0 = ttir.empty() : tensor<64x128xbf16>
    %1 = ttir.empty() : tensor<64x128xi16>
    %2, %3 = "ttir.sort"(%arg0, %0, %1) <{dim = -3 : si32}> : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xi16>) -> (tensor<64x128xbf16>, tensor<64x128xi16>)
    return %2, %3 : tensor<64x128xbf16>, tensor<64x128xi16>
  }
}

// -----
module {
  func.func @test_sort(%arg0: tensor<64x128xbf16>) -> (tensor<64x128xbf16>) {
    // CHECK: error: 'ttir.sort' op expected 2 results
    %0 = ttir.empty() : tensor<64x128xbf16>
    %1 = "ttir.sort"(%arg0, %0) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %1 : tensor<64x128xbf16>
  }
}

// -----
module {
  func.func @test_sort(%arg0: tensor<64x128xbf16>) -> (tensor<64x128xf16>, tensor<64x128xi16>) {
    // CHECK: error: 'ttir.sort' op Sorted tensor type does not match with input tensor.
    %0 = ttir.empty() : tensor<64x128xf16>
    %1 = ttir.empty() : tensor<64x128xi16>
    %2, %3 = "ttir.sort"(%arg0, %0, %1) <{dim = -1 : si32}> : (tensor<64x128xbf16>, tensor<64x128xf16>, tensor<64x128xi16>) -> (tensor<64x128xf16>, tensor<64x128xi16>)
    return %2, %3 : tensor<64x128xf16>, tensor<64x128xi16>
  }
}

// -----
module {
  func.func @test_sort(%arg0: tensor<64x128xbf16>) -> (tensor<64x128xbf16>, tensor<64xi16>) {
    // CHECK: error: 'ttir.sort' op Indices shape does not match with input tensor shape.
    %0 = ttir.empty() : tensor<64x128xbf16>
    %1 = ttir.empty() : tensor<64xi16>
    %2, %3 = "ttir.sort"(%arg0, %0, %1) <{dim = -1 : si32}> : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64xi16>) -> (tensor<64x128xbf16>, tensor<64xi16>)
    return %2, %3 : tensor<64x128xbf16>, tensor<64xi16>
  }
}
