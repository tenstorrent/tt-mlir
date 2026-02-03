// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for topk operation

module {
  func.func @test_topk(%arg0: tensor<64x128xbf16>) -> (tensor<64x8xbf16>, tensor<64x8xi16>) {
    // CHECK: error: 'ttir.topk' op Dimension out of range (expected to be in range of [-2, 1], but got -3)
    %1, %2 = "ttir.topk"(%arg0) <{k = 8 : ui32, dim = -3 : si32}> : (tensor<64x128xbf16>) -> (tensor<64x8xbf16>, tensor<64x8xi16>)
    return %1, %2 : tensor<64x8xbf16>, tensor<64x8xi16>
  }
}

// -----
module {
  func.func @test_topk(%arg0: tensor<64x128xbf16>) -> (tensor<64x8xbf16>, tensor<64x8xi16>) {
    // CHECK: error: 'ttir.topk' op k must be greater than 0
    %1, %2 = "ttir.topk"(%arg0) <{k = 0 : ui32, dim = -1 : si32}> : (tensor<64x128xbf16>) -> (tensor<64x8xbf16>, tensor<64x8xi16>)
    return %1, %2 : tensor<64x8xbf16>, tensor<64x8xi16>
  }
}

// -----
module {
  func.func @test_topk(%arg0: tensor<64x128xbf16>) -> (tensor<64x8xbf16>, tensor<64x8xbf16>) {
    // CHECK: error: 'ttir.topk' op Expected integer data type for indices but got bf16
    %1, %2 = "ttir.topk"(%arg0) <{k = 8 : ui32, dim = -1 : si32}> : (tensor<64x128xbf16>) -> (tensor<64x8xbf16>, tensor<64x8xbf16>)
    return %1, %2 : tensor<64x8xbf16>, tensor<64x8xbf16>
  }
}

// -----
module {
  func.func @test_topk(%arg0: tensor<64x128xbf16>) -> (tensor<64x7xbf16>, tensor<64x7xi16>) {
    // CHECK: error: 'ttir.topk' op Values dimension does not match k
    %1, %2 = "ttir.topk"(%arg0) <{k = 8 : ui32, dim = -1 : si32}> : (tensor<64x128xbf16>) -> (tensor<64x7xbf16>, tensor<64x7xi16>)
    return %1, %2 : tensor<64x7xbf16>, tensor<64x7xi16>
  }
}

// -----
module {
  func.func @test_topk(%arg0: tensor<64x128xbf16>) -> (tensor<64x256xbf16>, tensor<64x256xi16>) {
    // CHECK: error: 'ttir.topk' op k cannot be greater than input dimension size
    %1, %2 = "ttir.topk"(%arg0) <{k = 256 : ui32, dim = -1 : si32}> : (tensor<64x128xbf16>) -> (tensor<64x256xbf16>, tensor<64x256xi16>)
    return %1, %2 : tensor<64x256xbf16>, tensor<64x256xi16>
  }
}
