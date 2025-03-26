// RUN: ttmlir-opt --convert-ttir-to-linalg %s | FileCheck %s

module attributes {} {
  func.func @test_abs(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CHECK: {{%[0-9]+}} = tensor.empty() : {{tensor<.*>}}
    %0 = ttir.empty() : tensor<64x128xf32>
    // CHECK: {{%[0-9]+}} = linalg.abs ins(%arg0 : {{tensor<.*>}}) outs({{%[0-9]+}} : {{tensor<.*>}}) -> {{tensor<.*>}}
    %1 = "ttir.abs"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: return %1 : {{tensor<.*>}}
    return %1 : tensor<64x128xf32>
  }
}
