// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="system-desc-path=%system_desc_path% collapse-tensors-2d=false" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttmetal-to-flatbuffer -o %t.ttm %t.mlir

func.func @add(%arg0: tensor<3x32x64xf32>, %arg1: tensor<3x32x64xf32>) -> tensor<3x32x64xf32> {
  %0 = ttir.empty() : tensor<3x32x64xf32>

  // CHECK: "ttmetal.create_buffer"() {{.*}} -> memref<1x1x2x3x1x1x!ttcore.tile<32x32, f32>
  // CHECK: "ttmetal.create_buffer"() {{.*}} -> memref<1x1x2x3x32x32xf32

  // CHECK: "ttmetal.enqueue_write_buffer"(%arg0, {{.*}}) : (memref<3x32x64xf32>, memref<1x1x2x3x32x32xf32

  // CHECK: "ttmetal.enqueue_program"
  %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<3x32x64xf32>, tensor<3x32x64xf32>, tensor<3x32x64xf32>) -> tensor<3x32x64xf32>

  // CHECK: "ttmetal.create_buffer"() {{.*}} -> memref<1x1x2x3x1x1x!ttcore.tile<32x32, f32>
  // CHECK: "ttmetal.create_buffer"() {{.*}} -> memref<1x1x2x3x32x32xf32
  // CHECK: "ttmetal.enqueue_write_buffer"(%arg1, {{.*}}) : (memref<3x32x64xf32>, memref<1x1x2x3x32x32xf32

  // CHECK: "ttmetal.enqueue_read_buffer"({{.*}}, {{.*}}) : (memref<1x1x2x3x32x32xf32{{.*}}, memref<3x32x64xf32>)

  // CHECK: "ttmetal.finish"
  return %1 : tensor<3x32x64xf32>
}
