// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for rand operation

module {
  func.func @test_rand(%arg0: tensor<64x128xbf16>) -> (tensor<64x128xbf16>, tensor<64x128xi16>) {
    // CHECK: error: 'ttir.rand' op dtype does not match with output tensor type [dtype = 'f32', output tensor type = 'bf16'].
    %0 = "ttir.rand"() <{dtype = f32, size = [32 : i32, 32 : i32]}> : () -> tensor<32x32xbf16>
    return %0 : tensor<32x32xbf16>
  }
}
