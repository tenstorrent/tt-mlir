// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module {
  func.func @ozeros_2d() -> tensor<32x128xbf16> {
    // CHECK: {{.*}} = "ttnn.zeros"({{.*}}) {{.*}}
    %0 = "ttir.zeros"() <{shape = array<i32:32, 128>, dtype = i64}> : () -> tensor<32x128xbf16>
    return %0 : tensor<32x128xbf16>
  }

  func.func @zeros_3d() -> tensor<32x64x128xbf16> {
    // CHECK: {{.*}} = "ttnn.zeros"({{.*}}) {{.*}}
    %0 = "ttir.zeros"() <{shape = array<i32:32, 64, 128>, dtype = i64}> : () -> tensor<32x64x128xbf16>
    return %0 : tensor<32x64x128xbf16>
  }

  func.func @zeros_4d_irregular_shapes() -> tensor<13x24x56x42xbf16> {
    // CHECK: {{.*}} = "ttnn.zeros"({{.*}}) {{.*}} -> tensor<13x24x56x42xbf16{{.*}}>
    %0 = "ttir.zeros"() <{shape = array<i32:13, 24, 56, 42>, dtype = i64}> : () -> tensor<13x24x56x42xbf16>
    return %0 : tensor<13x24x56x42xbf16>
  }

  func.func @zeros_f32() -> tensor<32x64x128xf32> {
    // CHECK: {{.*}} = "ttnn.zeros"({{.*}}) {{.*}} -> tensor<32x64x128xf32{{.*}}>
    %0 = "ttir.zeros"() <{shape = array<i32:32, 64, 128>,dtype = i64}> : () -> tensor<32x64x128xf32>
    return %0 : tensor<32x64x128xf32>
  }
}
