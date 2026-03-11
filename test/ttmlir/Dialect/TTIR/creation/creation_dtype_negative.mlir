// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

module {
  func.func @zeros_dtype_mismatch() -> tensor<4x4xbf16> {
    // CHECK: error: 'ttir.zeros' op dtype does not match with output tensor type [dtype = 'f32', output tensor type = 'bf16'].
    %0 = "ttir.zeros"() <{shape = array<i32: 4, 4>, dtype = f32}> : () -> tensor<4x4xbf16>
    return %0 : tensor<4x4xbf16>
  }
}

// -----
module {
  func.func @ones_dtype_mismatch() -> tensor<4x4xbf16> {
    // CHECK: error: 'ttir.ones' op dtype does not match with output tensor type [dtype = 'f32', output tensor type = 'bf16'].
    %0 = "ttir.ones"() <{shape = array<i32: 4, 4>, dtype = f32}> : () -> tensor<4x4xbf16>
    return %0 : tensor<4x4xbf16>
  }
}

// -----
module {
  func.func @arange_dtype_mismatch() -> tensor<8xbf16> {
    // CHECK: error: 'ttir.arange' op dtype does not match with output tensor type [dtype = 'f32', output tensor type = 'bf16'].
    %0 = "ttir.arange"() <{start = 0 : si64, end = 8 : si64, step = 1 : si64, arange_dimension = 0 : i64, dtype = f32}> : () -> tensor<8xbf16>
    return %0 : tensor<8xbf16>
  }
}
