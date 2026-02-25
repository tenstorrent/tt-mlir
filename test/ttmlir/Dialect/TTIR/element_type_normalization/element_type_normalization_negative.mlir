// RUN: not ttmlir-opt --split-input-file --ttir-element-type-normalization %s 2>&1 | FileCheck %s

module {
  func.func @test_dense_attr_complex(%arg0: tensor<1x2xcomplex<f32>>) -> tensor<1x2xcomplex<f32>> {
    // CHECK: error: 'func.func' op Unsupported type.
    return %arg0 : tensor<1x2xcomplex<f32>>
  }
}

// -----
module {
  func.func @test_dense_attr_complex() -> tensor<1x2xf32> {
    // CHECK: error: 'ttir.empty' op Unsupported type.
    %0 = ttir.empty() : tensor<1x2xcomplex<f32>>
    %2 = "ttir.typecast"(%0)  : (tensor<1x2xcomplex<f32>>) -> tensor<1x2xf32>
    return %2: tensor<1x2xf32>
  }
}
