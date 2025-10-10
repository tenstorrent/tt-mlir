// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o mnist_linear_out.mlir %s
// RUN: FileCheck %s --input-file=mnist_linear_out.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn mnist_linear_out.mlir
#loc = loc("MNISTLinear":0:0)
module @MNISTLinear attributes {} {
  func.func @forward(%arg0: tensor<1x784xf32> {ttir.name = "input_1"} loc("MNISTLinear":0:0), %arg1: tensor<784x256xf32> {ttir.name = "l1.weight"} loc("MNISTLinear":0:0), %arg2: tensor<256xf32> {ttir.name = "l1.bias"} loc("MNISTLinear":0:0), %arg3: tensor<256x10xf32> {ttir.name = "l2.weight"} loc("MNISTLinear":0:0), %arg4: tensor<10xf32> {ttir.name = "l2.bias"} loc("MNISTLinear":0:0)) -> (tensor<1x10xf32> {ttir.name = "MNISTLinear.output_softmax_9"}) {
    // CHECK-DAG: #[[LAYOUT_8:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x8x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
    // CHECK-DAG: #[[LAYOUT_10:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<8x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
    // CHECK-DAG: #[[LAYOUT_11:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
    // CHECK-DAG: #[[LAYOUT_12:.*]] = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
    %0 = ttir.empty() : tensor<1x256xf32> loc(#loc8)
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<1x784xf32>, tensor<784x256xf32>, tensor<1x256xf32>) -> tensor<1x256xf32> loc(#loc8)
    %2 = ttir.empty() : tensor<1x256xf32> loc(#loc9)
    // CHECK: %{{.*}} = "ttnn.linear"{{.*}} -> tensor<1x256xf32, #[[LAYOUT_8]]>
    %3 = "ttir.add"(%1, %arg2, %2) : (tensor<1x256xf32>, tensor<256xf32>, tensor<1x256xf32>) -> tensor<1x256xf32> loc(#loc9)
    %4 = ttir.empty() : tensor<1x256xf32> loc(#loc10)
    // CHECK: %{{.*}} = "ttnn.relu"{{.*}} -> tensor<1x256xf32, #[[LAYOUT_8]]>
    %5 = "ttir.relu"(%3, %4) : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<1x256xf32> loc(#loc10)
    %6 = ttir.empty() : tensor<1x10xf32> loc(#loc11)
    %7 = "ttir.matmul"(%5, %arg3, %6) : (tensor<1x256xf32>, tensor<256x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32> loc(#loc11)
    %8 = ttir.empty() : tensor<1x10xf32> loc(#loc12)
    // CHECK: %{{.*}} = "ttnn.linear"{{.*}} -> tensor<1x10xf32, #[[LAYOUT_11]]>
    %9 = "ttir.add"(%7, %arg4, %8) : (tensor<1x10xf32>, tensor<10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32> loc(#loc12)
    %10 = ttir.empty() : tensor<1x10xf32> loc(#loc13)
    // CHECK: %{{.*}} = "ttnn.softmax"{{.*}} -> tensor<1x10xf32, #[[LAYOUT_11]]>
    %11 = "ttir.softmax"(%9, %10) <{dimension = 1 : si32}> : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32> loc(#loc13)
    return %11 : tensor<1x10xf32> loc(#loc7)
  } loc(#loc1)
} loc(#loc)
#loc1 = loc("forward":4294967295:23)
#loc2 = loc("forward":4294967295:25)
#loc3 = loc("forward":4294967295:26)
#loc4 = loc("forward":4294967295:28)
#loc5 = loc("forward":4294967295:30)
#loc6 = loc("forward":4294967295:31)
#loc7 = loc(unknown)
#loc8 = loc("matmul_3"(#loc1))
#loc9 = loc("add_4"(#loc2))
#loc10 = loc("relu_5"(#loc3))
#loc11 = loc("matmul_7"(#loc4))
#loc12 = loc("add_8"(#loc5))
#loc13 = loc("softmax_9"(#loc6))
