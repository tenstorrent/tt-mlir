// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true" %s | FileCheck %s
#loc = loc("test_ops.py:17_0_0":0:0)
module attributes {} {
  func.func @main(%arg0: tensor<1x32x32xf32> loc("test_ops.py:17_0_0":0:0), %arg1: tensor<1x32x32xf32> loc("test_ops.py:17_0_0":0:0), %arg2: tensor<1x32x32xf32> loc("test_ops.py:17_0_0":0:0)) -> (tensor<1x32x32xf32>, tensor<1x32x32xf32>) {
    %0 = ttir.empty() : tensor<1x32x32xf32> loc(#loc5)
    // CHECK: %{{.*}} = "ttnn.add"{{.*}}
    %1 = "ttir.add"(%arg1, %arg2, %0) : (tensor<1x32x32xf32>, tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> tensor<1x32x32xf32> loc(#loc5)
    %2 = ttir.empty() : tensor<1x32x32xf32> loc(#loc6)
    %3 = "ttir.add"(%1, %arg0, %2) : (tensor<1x32x32xf32>, tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> tensor<1x32x32xf32> loc(#loc6)
    %4 = ttir.empty() : tensor<1x32x32xf32> loc(#loc7)
    %5 = "ttir.add"(%arg2, %arg1, %4) : (tensor<1x32x32xf32>, tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> tensor<1x32x32xf32> loc(#loc7)
    return %3, %5 : tensor<1x32x32xf32>, tensor<1x32x32xf32> loc(#loc4)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("test_ops.py:17_0_0":0:4)
#loc2 = loc("test_ops.py:17_0_0":0:6)
#loc3 = loc("test_ops.py:17_0_0":0:3)
#loc4 = loc(unknown)
#loc5 = loc("add_1_0"(#loc1))
#loc6 = loc("add_2_0"(#loc2))
#loc7 = loc("add_0"(#loc3))
