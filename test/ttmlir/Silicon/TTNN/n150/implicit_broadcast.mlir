// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

func.func @main0(%arg0: tensor<1x16x32xf32>, %arg1: tensor<1x1x32xf32>) -> tensor<1x16x32xf32> {
  // CHECK-NOT: ttnn.repeat
  // CHECK: %{{[0-9]+}} = "ttnn.add"
  %0 = ttir.empty() : tensor<1x16x32xf32>
  %1 = "ttir.broadcast"(%arg1, %0) <{broadcast_dimensions = array<i64: 1, 16, 1>}> : (tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
  %2 = ttir.empty() : tensor<1x16x32xf32>
  %3 = "ttir.add"(%arg0, %1, %2) : (tensor<1x16x32xf32>, tensor<1x16x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
  return %3 : tensor<1x16x32xf32>
}

func.func @main1(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<784x128xf32> {
  // CHECK-NOT: "ttnn.repeat"
  // CHECK: %{{[0-9]+}} = "ttnn.reshape"
  // CHECK: %{{[0-9]+}} = "ttnn.add"
  // CHECK: %{{[0-9]+}} = "ttnn.repeat"
  // CHECK-SAME: repeat_dims = #ttnn.shape<784x1>
  %0 = ttir.empty() : tensor<1x128xf32>
  %1 = "ttir.reshape"(%arg0, %0) <{shape = [1 : i32, 128 : i32]}> : (tensor<128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
  %2 = ttir.empty() : tensor<784x128xf32>
  %3 = "ttir.broadcast"(%1, %2) <{broadcast_dimensions = array<i64: 784, 1>}> : (tensor<1x128xf32>, tensor<784x128xf32>) -> tensor<784x128xf32>
  %4 = ttir.empty() : tensor<1x128xf32>
  %5 = "ttir.reshape"(%arg1, %4) <{shape = [1 : i32, 128 : i32]}> : (tensor<128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
  %6 = ttir.empty() : tensor<784x128xf32>
  %7 = "ttir.broadcast"(%5, %6) <{broadcast_dimensions = array<i64: 784, 1>}> : (tensor<1x128xf32>, tensor<784x128xf32>) -> tensor<784x128xf32>
  %8 = ttir.empty() : tensor<784x128xf32>
  %9 = "ttir.add"(%3, %7, %8) : (tensor<784x128xf32>, tensor<784x128xf32>, tensor<784x128xf32>) -> tensor<784x128xf32>
  return %9 : tensor<784x128xf32>
}
