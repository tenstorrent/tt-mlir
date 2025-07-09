// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for sort operation

module {
  func.func @test_sort(%arg0: tensor<64x128xbf16>) -> (tensor<64x128xbf16>, tensor<64x128xi16>) {
    // CHECK: error: 'ttnn.sort' op Dimension out of range (expected to be in range of [-2, 1], but got -3)
    %1, %2 = "ttnn.sort"(%arg0) <{dim = -3 : si8}> : (tensor<64x128xbf16>) -> (tensor<64x128xbf16>, tensor<64x128xi16>)
    return %1, %2 : tensor<64x128xbf16>, tensor<64x128xi16>
  }
}

// -----
module {
  func.func @test_sort(%arg0: tensor<64x128xbf16>) -> (tensor<64x128xbf16>, tensor<64x128xbf16>) {
    // CHECK: error: 'ttnn.sort' op Expected data type for indices is i16 but got 'bf16'
    %1, %2 = "ttnn.sort"(%arg0) <{dim = -1 : si8}> : (tensor<64x128xbf16>) -> (tensor<64x128xbf16>, tensor<64x128xbf16>)
    return %1, %2 : tensor<64x128xbf16>, tensor<64x128xbf16>
  }
}

// -----
module {
  func.func @test_sort(%arg0: tensor<64x128xbf16>) -> (tensor<64x128xbf16>) {
    // CHECK: error: 'ttnn.sort' op expected 2 results
    %1 = "ttnn.sort"(%arg0) : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %1 : tensor<64x128xbf16>
  }
}

// -----
module {
  func.func @test_sort(%arg0: tensor<64x128xbf16>) -> (tensor<64x128xf16>, tensor<64x128xi16>) {
    // CHECK: error: 'ttnn.sort' op Sorted tensor type does not match with input tensor.
    %1, %2 = "ttnn.sort"(%arg0) <{dim = -1 : si8}> : (tensor<64x128xbf16>) -> (tensor<64x128xf16>, tensor<64x128xi16>)
    return %1, %2 : tensor<64x128xf16>, tensor<64x128xi16>
  }
}

// -----
module {
  func.func @test_sort(%arg0: tensor<64x128xbf16>) -> (tensor<64x128xbf16>, tensor<64xi16>) {
    // CHECK: error: 'ttnn.sort' op Indices shape does not match with input tensor shape.
    %1, %2 = "ttnn.sort"(%arg0) <{dim = -1 : si8}> : (tensor<64x128xbf16>) -> (tensor<64x128xbf16>, tensor<64xi16>)
    return %1, %2 : tensor<64x128xbf16>, tensor<64xi16>
  }
}
