// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module {
  func.func @ones_2d() -> tensor<32x128xbf16> {
    // CHECK: {{.*}} = "ttnn.ones"({{.*}}) {{.*}}
    %0 = "ttir.ones"() <{shape = array<i32:32, 128>, dtype = bf16}> : () -> tensor<32x128xbf16>
    return %0 : tensor<32x128xbf16>
  }

  func.func @ones_3d() -> tensor<32x64x128xbf16> {
    // CHECK: {{.*}} = "ttnn.ones"({{.*}}) {{.*}}
    %0 = "ttir.ones"() <{shape = array<i32:32, 64, 128>, dtype = bf16}> : () -> tensor<32x64x128xbf16>
    return %0 : tensor<32x64x128xbf16>
  }

  func.func @ones_4d_irregular_shapes() -> tensor<13x24x56x42xbf16> {
    // CHECK: {{.*}} = "ttnn.ones"({{.*}}) {{.*}} -> tensor<13x24x56x42xbf16{{.*}}>
    %0 = "ttir.ones"() <{shape = array<i32:13, 24, 56, 42>, dtype = bf16}> : () -> tensor<13x24x56x42xbf16>
    return %0 : tensor<13x24x56x42xbf16>
  }

  func.func @ones_f32() -> tensor<32x64x128xf32> {
    // CHECK: {{.*}} = "ttnn.ones"({{.*}}) {{.*}} -> tensor<32x64x128xf32{{.*}}>
    %0 = "ttir.ones"() <{shape = array<i32:32, 64, 128>, dtype = f32}> : () -> tensor<32x64x128xf32>
    return %0 : tensor<32x64x128xf32>
  }
}
