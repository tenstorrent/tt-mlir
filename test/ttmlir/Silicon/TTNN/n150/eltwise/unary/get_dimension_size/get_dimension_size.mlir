// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

func.func @get_dimension_size(%arg0: tensor<13x21x1x3xf32>) -> tensor<1xi32> {
  %0 = "ttir.get_dimension_size"(%arg0) <{dimension = 1 : i32}> : (tensor<13x21x1x3xf32>) -> tensor<1xi32>
  // CHECK: [[VAL:%[0-9]+]] = "ttnn.full"(%{{[0-9]+}})
  // CHECK-SAME: fill_value = 21 : i32
  // CHECK-SAME: -> tensor<1xui32
  return %0 : tensor<1xi32>
}
