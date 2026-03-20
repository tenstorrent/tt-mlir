// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-implicit-broadcast-folding-pass=false system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

func.func @main0(%arg0: tensor<1x16x32xf32>, %arg1: tensor<1x1x32xf32>) -> tensor<1x16x32xf32> {
  // CHECK: %{{[0-9]+}} = "ttnn.repeat"(%arg1)
  %1 = "ttir.broadcast"(%arg1) <{broadcast_dimensions = array<i64 : 1, 16, 1>}> : (tensor<1x1x32xf32>) -> tensor<1x16x32xf32>
  %3 = "ttir.multiply"(%arg0, %1) : (tensor<1x16x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
  return %3 : tensor<1x16x32xf32>
}

func.func public @main1(%arg0: tensor<1xf32>, %arg1: tensor<512x512xf32>) -> (tensor<512x512xf32>) {
  // CHECK: %[[ARG0_RM:.*]] = "ttnn.to_layout"(%arg0)
  // CHECK-SAME: layout = #ttnn.layout<row_major>
  // CHECK: %[[RESHAPE_RM:.*]] = "ttnn.reshape"(%[[ARG0_RM]])
  // CHECK: %[[RESHAPE_TILE:.*]] = "ttnn.to_layout"(%[[RESHAPE_RM]])
  // CHECK-SAME: layout = #ttnn.layout<tile>
  // CHECK: %[[REPEAT0:.*]] = "ttnn.repeat"(%[[RESHAPE_TILE]])
  %1 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xf32>) -> tensor<1x1xf32>
  %3 = "ttir.broadcast"(%1) <{broadcast_dimensions = array<i64 : 512, 512>}> : (tensor<1x1xf32>) -> tensor<512x512xf32>
  %5 = "ttir.maximum"(%3, %arg1) : (tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
  return %5 : tensor<512x512xf32>
}

func.func @main2(%arg0: tensor<1x23x40x1xf32>, %arg1: tensor<128xf32>) -> tensor<1x23x40x128xf32> {
  // CHECK: %{{[0-9]+}} = "ttnn.reshape"
  // CHECK: %{{[0-9]+}} = "ttnn.repeat"
  // CHECK-SAME: repeat_dims = #ttnn.shape<1x23x40x1>
  %1 = "ttir.broadcast"(%arg0) <{broadcast_dimensions = array<i64 : 1, 1, 1, 128>}> : (tensor<1x23x40x1xf32>) -> tensor<1x23x40x128xf32>
  %3 = "ttir.reshape"(%arg1) <{shape = [1 : i32, 1 : i32, 1 : i32, 128 : i32]}> : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
  %5 = "ttir.broadcast"(%3) <{broadcast_dimensions = array<i64 : 1, 23, 40, 1>}> : (tensor<1x1x1x128xf32>) -> tensor<1x23x40x128xf32>
  %7 = "ttir.div"(%1, %5) : (tensor<1x23x40x128xf32>, tensor<1x23x40x128xf32>) -> tensor<1x23x40x128xf32>
  return %7 : tensor<1x23x40x128xf32>
}

func.func @main3(%arg0: tensor<6x2xf32>) -> tensor<2400x2xf32> {
  // CHECK: %[[ARG0_RM:.*]] = "ttnn.to_layout"(%arg0)
  // CHECK-SAME: layout = #ttnn.layout<row_major>
  // CHECK: %[[INPUT_RESHAPE:.*]] = "ttnn.reshape"(%[[ARG0_RM]])
  // CHECK: %[[INPUT_TILE:.*]] = "ttnn.to_layout"(%[[INPUT_RESHAPE]])
  // CHECK-SAME: layout = #ttnn.layout<tile>
  // CHECK: %[[REPEAT:.*]] = "ttnn.repeat"(%[[INPUT_TILE]])
  // CHECK-SAME: repeat_dims = #ttnn.shape<400x1x1x1>
  // CHECK: %[[REPEAT_RM:.*]] = "ttnn.to_layout"(%[[REPEAT]])
  // CHECK-SAME: layout = #ttnn.layout<row_major>
  // CHECK: %[[OUTPUT_RESHAPE:.*]] = "ttnn.reshape"(%[[REPEAT_RM]])
  // CHECK: %[[OUTPUT_TILE:.*]] = "ttnn.to_layout"(%[[OUTPUT_RESHAPE]])
  // CHECK-SAME: layout = #ttnn.layout<tile>
  %1 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 6 : i32, 2 : i32]}> : (tensor<6x2xf32>) -> tensor<1x6x2xf32>
  %3 = "ttir.reshape"(%1) <{shape = [1 : i32, 6 : i32, 1 : i32, 2 : i32]}> : (tensor<1x6x2xf32>) -> tensor<1x6x1x2xf32>
  %5 = "ttir.broadcast"(%3) <{broadcast_dimensions = array<i64 : 400, 1, 1, 1>}> : (tensor<1x6x1x2xf32>) -> tensor<400x6x1x2xf32>
  %7 = "ttir.reshape"(%5) <{shape = [2400 : i32, 1 : i32, 2 : i32]}> : (tensor<400x6x1x2xf32>) -> tensor<2400x1x2xf32>
  %9 = "ttir.reshape"(%7) <{shape = [2400 : i32, 2 : i32]}> : (tensor<2400x1x2xf32>) -> tensor<2400x2xf32>
  return %9 : tensor<2400x2xf32>
}
