// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=1" -o %t %s --mlir-print-local-scope
// RUN: FileCheck %s --input-file=%t

// This test verifies that ttir.argmax properly converts si32 result types to ui32,
// ensuring consistency between tensor element type and layout encoding data type.
// This prevents mismatches where layoutAttr.getScalarElementType() != tensorType.getElementType()

module attributes {} {
  func.func @argmax_example(%arg0: tensor<3x3xi32> {ttcore.argument_type = #ttcore.argument_type<input>}) -> (tensor<3xi32> {}) {
    // CHECK-LABEL: func.func @argmax_example
    // CHECK: "ttnn.argmax"
    // CHECK-SAME: -> tensor<1x1x3xui32
    // CHECK-SAME: memref<1x3xui32
    // CHECK: "ttnn.typecast"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<si32>
    // CHECK-SAME: (tensor<1x1x3xui32{{.*}}>) -> tensor<1x1x3xsi32
    %0 = "ttir.argmax"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<3x3xi32>) -> tensor<3xi32>
    return %0 : tensor<3xi32>
  }
}
