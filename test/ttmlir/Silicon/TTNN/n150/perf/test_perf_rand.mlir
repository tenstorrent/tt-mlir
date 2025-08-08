// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module attributes {} {
  func.func @test_rand() -> tensor<32x32xbf16>{
    // CHECK-LABEL: @test_rand
    // CHECK: %[[DEVICE:[0-9]+]] = "ttnn.get_device"()
    // CHECK: %{{[0-9]+}} = "ttnn.rand"(%[[DEVICE]])
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
