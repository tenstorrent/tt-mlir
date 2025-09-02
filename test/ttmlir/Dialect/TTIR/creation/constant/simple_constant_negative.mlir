// RUN: not ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline %s 2>&1 | FileCheck %s

module {
  func.func @test_dense_attr() -> tensor<1x2xbf16> {
    // CHECK: error: 'ttir.constant' op value attribute must be one of DenseResourceElementsAttr or DenseElementsAttr.
    %0 = "ttir.constant"() <{value = sparse<[[0, 0], [0, 1]], [2.0, 2.0]> : tensor<1x2xbf16>}> : () -> tensor<1x2xbf16>
    return %0 : tensor<1x2xbf16>
  }
}

// -----

module {
  func.func @test_dense_attr_complex() -> tensor<1x2xcomplex<f32>> {
    // CHECK: error: 'ttir.constant' op value attribute must be of int or float type.
    %0 = "ttir.constant"() <{value = dense<[ [(2.0, 0.0), (3.0, 0.0)]]> : tensor<1x2xcomplex<f32>>}> : () -> tensor<1x2xcomplex<f32>>
    return %0 : tensor<1x2xcomplex<f32>>
  }
}
