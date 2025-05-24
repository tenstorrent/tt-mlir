// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

module {
  func.func @full_float() -> tensor<64x128xbf16> {
    // CHECK: "ttnn.full"
    // CHECK-SAME: fill_value = 3.000000e+00 : f32
    // CHECK-SAME: shape = #ttnn.shape<64x128>
    %0 = "ttir.full"() <{shape = array<i32: 64, 128>, fill_value = 3.0 : f32}> : () -> tensor<64x128xbf16>
    return %0 : tensor<64x128xbf16>
  }

  func.func @full_int() -> tensor<64x128xi32> {
    // CHECK: "ttnn.full"
    // CHECK-SAME: fill_value = 3 : i32
    // CHECK-SAME: shape = #ttnn.shape<64x128>
    %0 = "ttir.full"() <{shape = array<i32: 64, 128>, fill_value = 3 : i32}> : () -> tensor<64x128xi32>
    return %0 : tensor<64x128xi32>
  }
}
