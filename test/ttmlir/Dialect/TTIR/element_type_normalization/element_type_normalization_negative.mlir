// RUN: not ttmlir-opt --ttir-element-type-normalization %s 2>&1 | FileCheck %s

module {
  func.func @test_dense_attr_complex() -> tensor<1x2xf32> {
    // CHECK: error: 'ttir.empty' op failed to convert result types
    %0 = ttir.empty() : tensor<1x2xcomplex<f32>>
    %1 = ttir.empty() : tensor<1x2xf32>
    %2 = "ttir.typecast"(%0, %1)  : (tensor<1x2xcomplex<f32>>, tensor<1x2xf32>) -> tensor<1x2xf32>
    return %2: tensor<1x2xf32>
  }
}
