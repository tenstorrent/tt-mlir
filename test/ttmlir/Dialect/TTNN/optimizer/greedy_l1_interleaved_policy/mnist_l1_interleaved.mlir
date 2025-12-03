// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=true memory-layout-analysis-policy=GreedyL1Interleaved tensor-l1-usage-cap=0.75" -o %t %s
// RUN: FileCheck %s --input-file=%t
#loc = loc("MNISTLinear":4294967295:0)
module @"tt-forge-graph" attributes {} {
  func.func @main(%arg0: tensor<1x784xf32> loc("MNISTLinear":4294967295:0), %arg1: tensor<1x10xf32> loc("MNISTLinear":4294967295:0), %arg2: tensor<256x10xf32> loc("MNISTLinear":4294967295:0), %arg3: tensor<1x256xf32> loc("MNISTLinear":4294967295:0), %arg4: tensor<784x256xf32> loc("MNISTLinear":4294967295:0)) -> tensor<1x10xf32> {
    // CHECK: #[[LAYOUT_L1:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <{{.*}}>, memref<{{.*}}, #l1>, <interleaved>>
    %1 = "ttir.matmul"(%arg0, %arg4) : (tensor<1x784xf32>, tensor<784x256xf32>) -> tensor<1x256xf32> loc(#loc8)
    // CHECK: %{{.*}} = "ttnn.linear"{{.*}} -> tensor<1x256xf32, #[[LAYOUT_L1]]>
    %3 = "ttir.add"(%1, %arg3) : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<1x256xf32> loc(#loc9)
    // CHECK: %{{.*}} = "ttnn.relu"{{.*}} -> tensor<1x256xf32, #[[LAYOUT_L1]]>
    %5 = "ttir.relu"(%3) : (tensor<1x256xf32>) -> tensor<1x256xf32> loc(#loc10)
    %7 = "ttir.matmul"(%5, %arg2) : (tensor<1x256xf32>, tensor<256x10xf32>) -> tensor<1x10xf32> loc(#loc11)
    // CHECK: %{{.*}} = "ttnn.linear"{{.*}} -> tensor<1x10xf32, #[[LAYOUT_L1]]>
    %9 = "ttir.add"(%7, %arg1) : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32> loc(#loc12)
    // CHECK: %{{.*}} = "ttnn.softmax"{{.*}} -> tensor<1x10xf32, #[[LAYOUT_L1]]>
    %11 = "ttir.softmax"(%9) <{dimension = 1 : si32}> : (tensor<1x10xf32>) -> tensor<1x10xf32> loc(#loc13)
    return %11 : tensor<1x10xf32> loc(#loc7)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("MNISTLinear":4294967295:10)
#loc2 = loc("MNISTLinear":4294967295:8)
#loc3 = loc("MNISTLinear":4294967295:6)
#loc4 = loc("MNISTLinear":4294967295:4)
#loc5 = loc("MNISTLinear":4294967295:3)
#loc6 = loc("MNISTLinear":4294967295:2)
#loc7 = loc(unknown)
#loc8 = loc("matmul_1"(#loc1))
#loc9 = loc("add_2"(#loc2))
#loc10 = loc("relu_3"(#loc3))
#loc11 = loc("matmul_5"(#loc4))
#loc12 = loc("add_6"(#loc5))
#loc13 = loc("softmax_7"(#loc6))
