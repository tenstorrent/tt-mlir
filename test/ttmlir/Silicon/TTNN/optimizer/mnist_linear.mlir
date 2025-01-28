// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=mnist_linear_out.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer mnist_linear_out.mlir > %t.ttnn
module @MNISTLinear attributes {} {
  func.func @forward(%arg0: tensor<1x784xf32> {ttir.name = "input_1"}, %arg1: tensor<784x256xf32> {ttir.name = "l1.weight"}, %arg2: tensor<256xf32> {ttir.name = "l1.bias"}, %arg3: tensor<256x10xf32> {ttir.name = "l2.weight"}, %arg4: tensor<10xf32> {ttir.name = "l2.bias"}) -> (tensor<1x10xf32> {ttir.name = "MNISTLinear.output_softmax_9"}) {
    // CHECK-DAG: #[[LAYOUT_8:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x8x!tt.tile<32x32, f32>, #dram>, <interleaved>>
    // CHECK-DAG: #[[LAYOUT_10:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<8x1x!tt.tile<32x32, f32>, #dram>, <interleaved>>
    // CHECK-DAG: #[[LAYOUT_11:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!tt.tile<32x32, f32>, #dram>, <interleaved>>
    // CHECK-DAG: #[[LAYOUT_12:.*]] = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!tt.tile<32x32, f32>, #dram>, <interleaved>>
    %0 = tensor.empty() : tensor<1x256xf32>
    // CHECK: %{{.*}} = "ttnn.matmul"{{.*}} -> tensor<1x256xf32, #[[LAYOUT_8]]>
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<1x784xf32>, tensor<784x256xf32>, tensor<1x256xf32>) -> tensor<1x256xf32>
    %2 = tensor.empty() : tensor<1x256xf32>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x256xf32, #[[LAYOUT_8]]>
    %3 = "ttir.add"(%1, %arg2, %2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256xf32>, tensor<256xf32>, tensor<1x256xf32>) -> tensor<1x256xf32>
    %4 = tensor.empty() : tensor<1x256xf32>
    // CHECK: %{{.*}} = "ttnn.relu"{{.*}} -> tensor<1x256xf32, #[[LAYOUT_8]]>
    %5 = "ttir.relu"(%3, %4) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<1x256xf32>
    %6 = tensor.empty() : tensor<1x10xf32>
    // CHECK: %{{.*}} = "ttnn.matmul"{{.*}} -> tensor<1x10xf32, #[[LAYOUT_11]]>
    %7 = "ttir.matmul"(%5, %arg3, %6) : (tensor<1x256xf32>, tensor<256x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    %8 = tensor.empty() : tensor<1x10xf32>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x10xf32, #[[LAYOUT_11]]>
    %9 = "ttir.add"(%7, %arg4, %8) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x10xf32>, tensor<10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    %10 = tensor.empty() : tensor<1x10xf32>
    // CHECK: %{{.*}} = "ttnn.softmax"{{.*}} -> tensor<1x10xf32, #[[LAYOUT_11]]>
    %11 = "ttir.softmax"(%9, %10) <{dimension = 1 : si32}> : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    return %11 : tensor<1x10xf32>
  }
}
