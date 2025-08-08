// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck --input-file=%t %s

module attributes {} {
  func.func @test_rand() -> tensor<32x32xbf16>{
    // CHECK-LABEL: @test_rand
    // CHECK: "ttnn.rand"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>,
    // CHECK-SAME: high = 1.000000e+00 : f32,
    // CHECK-SAME: layout = #ttnn.layout<tile>,
    // CHECK-SAME: low = 0.000000e+00 : f32,
    // CHECK-SAME: seed = 0 : ui32,
    // CHECK-SAME: size = #ttnn.shape<32x32>
    // CHECK-SAME: -> tensor<32x32xbf16,
    %0 = "ttir.rand"() <{dtype = bf16, size = [32 : i32, 32 : i32]}> : () -> tensor<32x32xbf16>
    return %0 : tensor<32x32xbf16>
  }
}
