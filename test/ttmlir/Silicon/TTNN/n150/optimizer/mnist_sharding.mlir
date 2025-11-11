// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=true" -o mnist_sharding_ttnn.mlir %s
// RUN: FileCheck %s --input-file=mnist_sharding_ttnn.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn mnist_sharding_ttnn.mlir
// XFAIL: *
// TODO(rpavlovicTT): #https://github.com/tenstorrent/tt-metal/issues/21846 re-enable

#loc = loc("MNISTLinear":4294967295:0)
module @"tt-forge-graph" attributes {} {
  func.func @main(%arg0: tensor<1x784xf32> loc("MNISTLinear":4294967295:0), %arg1: tensor<1x10xf32> loc("MNISTLinear":4294967295:0), %arg2: tensor<256x10xf32> loc("MNISTLinear":4294967295:0), %arg3: tensor<1x256xf32> loc("MNISTLinear":4294967295:0), %arg4: tensor<784x256xf32> loc("MNISTLinear":4294967295:0)) -> tensor<1x10xf32> {
    // CHECK-DAG: #[[LAYOUT_10:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x8, (d0, d1) -> (0, d1 floordiv 8, d1 mod 8)>, memref<1x1x!ttcore.tile<32x32, f32>, #l1_>, <width_sharded>>
    // CHECK-DAG: #[[LAYOUT_11:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1, (d0, d1) -> (0, d1 floordiv 8, d1 mod 8)>, memref<1x1x!ttcore.tile<32x32, f32>, #l1_>, <width_sharded>>
    %0 = ttir.empty() : tensor<1x256xf32> loc(#loc8)
    // TODO(#3242): Revert to sharded layout we are able to query backend for matmul.
    %1 = "ttir.matmul"(%arg0, %arg4, %0) : (tensor<1x784xf32>, tensor<784x256xf32>, tensor<1x256xf32>) -> tensor<1x256xf32> loc(#loc8)
    %2 = ttir.empty() : tensor<1x256xf32> loc(#loc9)
    %3 = "ttir.add"(%1, %arg3, %2) : (tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<1x256xf32> loc(#loc9)
    %4 = ttir.empty() : tensor<1x256xf32> loc(#loc10)
    // CHECK: %{{.*}} = "ttnn.relu"{{.*}} -> tensor<1x256xf32, #[[LAYOUT_10]]>
    %5 = "ttir.relu"(%3, %4) : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<1x256xf32> loc(#loc10)
    %6 = ttir.empty() : tensor<1x10xf32> loc(#loc11)
    // TODO(#3242): Revert to sharded layout we are able to query backend for matmul.
    %7 = "ttir.matmul"(%5, %arg2, %6) : (tensor<1x256xf32>, tensor<256x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32> loc(#loc11)
    %8 = ttir.empty() : tensor<1x10xf32> loc(#loc12)
    %9 = "ttir.add"(%7, %arg1, %8) : (tensor<1x10xf32>, tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32> loc(#loc12)
    %10 = ttir.empty() : tensor<1x10xf32> loc(#loc13)
    %11 = "ttir.softmax"(%9, %10) <{dimension = 1 : si32}> : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32> loc(#loc13)
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
