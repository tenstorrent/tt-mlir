// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline --ttnn-emitpy-workarounds -o %t %s
// RUN: FileCheck --input-file=%t %s

// Test file to make sure that the workaround for 4D constant ops is applied correctly.
// The workaround rewrites constant op with non 4D target shape to 4D target shape and
// adds a reshape op after it to get the original shape back.
module{
    func.func @test_constant_4d() -> tensor<1x4xf32> {
        %0 = "ttir.constant"() <{value = dense<[[1., 2., 3., 4.]]> : tensor<1x4xf32>}> : () -> tensor<1x4xf32>
        // CHECK: "ttnn.constant"
        // CHECK-SAME: value = dense<{{\[\[\[\[}}1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]]]]
        // CHECK-SAME: -> tensor<1x1x1x4xf32
        // CHECK: "ttnn.reshape"
        // CHECK-SAME: shape = [1 : i32, 4 : i32]
        // CHECK-SAME: -> tensor<1x4xf32
        return %0 : tensor<1x4xf32>
    }
}
