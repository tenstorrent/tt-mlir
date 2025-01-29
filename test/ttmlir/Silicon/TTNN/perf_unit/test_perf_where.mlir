// REQUIRES: num-chips-1 || num-chips-2
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

func.func @test_where(%arg0: tensor<13x37xbf16>, %arg1: tensor<13x37xbf16>) -> tensor<13x37xbf16> {
  %0 = tensor.empty() : tensor<13x37xbf16>
  %1 = "ttir.eq"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<13x37xbf16>, tensor<13x37xbf16>, tensor<13x37xbf16>) -> tensor<13x37xbf16>
  %2 = tensor.empty() : tensor<13x37xbf16>
  %3 = "ttir.where"(%1, %arg0, %arg1, %2) <{operandSegmentSizes = array<i32: 3, 1>}> : (tensor<13x37xbf16>, tensor<13x37xbf16>, tensor<13x37xbf16>, tensor<13x37xbf16>) -> tensor<13x37xbf16>
  // CHECK: %[[EMPTY:.*]] = "ttnn.empty"{{.*}}
  // CHECK: %[[VAL1:[0-9]+]] = "ttnn.eq"(%{{[0-9]+}}, %{{[0-9]+}}, %[[EMPTY]])
  // CHECK: %{{[0-9]+}} = "ttnn.where"(%[[VAL1]], %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}})
  return %3 : tensor<13x37xbf16>
}
