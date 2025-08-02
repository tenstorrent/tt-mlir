// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for rand operation

module {
  func.func @test_rand_dtype() -> (tensor<32x32xbf16>) {
    // CHECK: error: 'ttir.rand' op dtype does not match with output tensor type [dtype = 'f32', output tensor type = 'bf16'].
    %0 = "ttir.rand"() <{dtype = f32, size = [32 : i32, 32 : i32]}> : () -> tensor<32x32xbf16>
    return %0 : tensor<32x32xbf16>
  }
}

// -----
module {
  func.func @test_rand_interval() -> tensor<32x32xbf16> {
    // CHECK: error: 'ttir.rand' op 'low' value must be < 'high' value.
    %0 = "ttir.rand"() <{dtype = bf16, low=1.0 : f32, high=1.0 :f32, size = [32 : i32, 32 : i32]}> : () -> tensor<32x32xbf16>
    return %0 : tensor<32x32xbf16>
  }
}

// -----
module {
  func.func @test_rand_size() -> tensor<32x32xbf16> {
    // CHECK: error: 'ttir.rand' op Size argument does not match with output tensor shape. [Size = [64 : i32, 64 : i32], output tensor shape = (32, 32)].
    %0 = "ttir.rand"() <{dtype = bf16, low=0.0 : f32, high=1.0 :f32, size = [64 : i32, 64 : i32]}> : () -> tensor<32x32xbf16>
    return %0 : tensor<32x32xbf16>
  }
}
