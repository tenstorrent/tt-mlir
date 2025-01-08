// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

func.func @get_dimension_size(%arg0: tensor<13x21x3xf32>) -> tensor<1xi32> {
  %0 = "ttir.get_dimension_size"(%arg0) <{dimension = 1 : i32}> : (tensor<13x21x3xf32>) -> tensor<1xi32>
  // CHECK: [[VAL:%[0-9]+]] = "ttnn.full"(%{{[0-9]+}}) <{fillValue = 2.100000e+01 : f32}> : (!tt.device<#device>) -> tensor<1xi32, {{.*}}>
  return %0 : tensor<1xi32>
  // CHECK: return [[VAL]] : tensor<1xi32, {{.*}}>
}
