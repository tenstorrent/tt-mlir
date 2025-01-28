// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o mnist_linear_out.mlir %s
// RUN: FileCheck %s --input-file=mnist_linear_out.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer mnist_linear_out.mlir > %t.ttnn
#loc2 = loc("mnist_linear.mlir":5:22)
#loc3 = loc("mnist_linear.mlir":5:72)
#loc4 = loc("mnist_linear.mlir":5:126)
#loc5 = loc("mnist_linear.mlir":5:174)
#loc6 = loc("mnist_linear.mlir":5:227)
module @MNISTLinear attributes {} {
  func.func @forward(%arg0: tensor<1x784xf32> {ttir.name = "input_1"} loc("mnist_linear.mlir":10:22), %arg1: tensor<784x256xf32> {ttir.name = "l1.weight"} loc("mnist_linear.mlir":10:72), %arg2: tensor<256xf32> {ttir.name = "l1.bias"} loc("mnist_linear.mlir":10:126), %arg3: tensor<256x10xf32> {ttir.name = "l2.weight"} loc("mnist_linear.mlir":10:174), %arg4: tensor<10xf32> {ttir.name = "l2.bias"} loc("mnist_linear.mlir":10:227)) -> (tensor<1x10xf32> {ttir.name = "MNISTLinear.output_softmax_9"}) {
    // CHECK-DAG: #[[LAYOUT_8:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x8x!tt.tile<32x32, f32>, #dram>, <interleaved>>
    // CHECK-DAG: #[[LAYOUT_10:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<8x1x!tt.tile<32x32, f32>, #dram>, <interleaved>>
    // CHECK-DAG: #[[LAYOUT_11:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!tt.tile<32x32, f32>, #dram>, <interleaved>>
    // CHECK-DAG: #[[LAYOUT_12:.*]] = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!tt.tile<32x32, f32>, #dram>, <interleaved>>
    %0 = tensor.empty() : tensor<1x256xf32> loc(#loc7)
    // CHECK: %{{.*}} = "ttnn.matmul"{{.*}} -> tensor<1x256xf32, #[[LAYOUT_8]]>
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<1x784xf32>, tensor<784x256xf32>, tensor<1x256xf32>) -> tensor<1x256xf32> loc(#loc8)
    %2 = tensor.empty() : tensor<1x256xf32> loc(#loc9)
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x256xf32, #[[LAYOUT_8]]>
    %3 = "ttir.add"(%1, %arg2, %2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256xf32>, tensor<256xf32>, tensor<1x256xf32>) -> tensor<1x256xf32> loc(#loc10)
    %4 = tensor.empty() : tensor<1x256xf32> loc(#loc11)
    // CHECK: %{{.*}} = "ttnn.relu"{{.*}} -> tensor<1x256xf32, #[[LAYOUT_8]]>
    %5 = "ttir.relu"(%3, %4) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<1x256xf32> loc(#loc12)
    %6 = tensor.empty() : tensor<1x10xf32> loc(#loc13)
    // CHECK: %{{.*}} = "ttnn.matmul"{{.*}} -> tensor<1x10xf32, #[[LAYOUT_11]]>
    %7 = "ttir.matmul"(%5, %arg3, %6) : (tensor<1x256xf32>, tensor<256x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32> loc(#loc14)
    %8 = tensor.empty() : tensor<1x10xf32> loc(#loc15)
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x10xf32, #[[LAYOUT_11]]>
    %9 = "ttir.add"(%7, %arg4, %8) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x10xf32>, tensor<10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32> loc(#loc16)
    %10 = tensor.empty() : tensor<1x10xf32> loc(#loc17)
    // CHECK: %{{.*}} = "ttnn.softmax"{{.*}} -> tensor<1x10xf32, #[[LAYOUT_11]]>
    %11 = "ttir.softmax"(%9, %10) <{dimension = 1 : si32}> : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32> loc(#loc18)
    return %11 : tensor<1x10xf32> loc(#loc19)
  } loc(#loc1)
} loc(#loc)
#loc = loc("mnist_linear.mlir":9:1)
#loc1 = loc("mnist_linear.mlir":10:3)
#loc7 = loc("mnist_linear.mlir":15:10)
#loc8 = loc("mnist_linear.mlir":17:10)
#loc9 = loc("mnist_linear.mlir":18:10)
#loc10 = loc("mnist_linear.mlir":20:10)
#loc11 = loc("mnist_linear.mlir":21:10)
#loc12 = loc("mnist_linear.mlir":23:10)
#loc13 = loc("mnist_linear.mlir":24:10)
#loc14 = loc("mnist_linear.mlir":26:10)
#loc15 = loc("mnist_linear.mlir":27:10)
#loc16 = loc("mnist_linear.mlir":29:10)
#loc17 = loc("mnist_linear.mlir":30:11)
#loc18 = loc("mnist_linear.mlir":32:11)
#loc19 = loc("mnist_linear.mlir":33:5)
