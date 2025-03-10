// RUN: not ttmlir-opt --ttir-to-ttnn-backend-pipeline %s 2>&1 | FileCheck %s

module attributes {} {
    func.func @test_dense_attr() -> tensor<1x2xbf16> {
        // CHECK: error: 'ttnn.constant' op value attribute must be one of DenseResourceElementsAttr or DenseElementsAttr.
        %0 = "ttir.constant"() <{value = sparse<[[0, 0], [0, 1]], [2.0, 2.0]> : tensor<1x2xbf16>}> : () -> tensor<1x2xbf16>
        return %0 : tensor<1x2xbf16>
  }
}
