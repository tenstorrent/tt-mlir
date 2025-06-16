// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true  insert-memreconfig=add_2=0 override-output-layout=add_1=1x1:dram:interleaved:row_major:f32" %s | FileCheck %s
// XFAIL: *
// TODO(rpavlovicTT): #https://github.com/tenstorrent/tt-metal/issues/21846 re-enable

#loc = loc("test_ops.py:17_0_0":0:0)
module attributes {} {
  func.func @main(%arg0: tensor<1x32x32xf32> loc("test_ops.py:17_0_0":0:0), %arg1: tensor<1x32x32xf32> loc("test_ops.py:17_0_0":0:0), %arg2: tensor<1x32x32xf32> loc("test_ops.py:17_0_0":0:0)) -> tensor<1x32x32xf32> {
    // CHECK: #[[L1_:.*]] = #ttnn.buffer_type<l1>
    // CHECK-DAG: #[[LAYOUT_1:.*]] = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1, (d0, d1) -> (0, d1 floordiv 8, d1 mod 8)>, memref<1x1x!ttcore.tile<32x32, f32>, #l1_>, <width_sharded>>
    // CHECK-DAG: #[[LAYOUT_2:.*]] = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<32x32xf32, #dram>, <interleaved>>
    %0 = ttir.empty() : tensor<1x32x32xf32> loc(#loc5)
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x32x32xf32, #[[LAYOUT_2]]>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<1x32x32xf32>, tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> tensor<1x32x32xf32> loc(#loc5)
    %2 = ttir.empty() : tensor<1x32x32xf32> loc(#loc6)
    // CHECK: %[[IDX:.*]] = "ttnn.to_memory_config"{{.*}} -> tensor<1x32x32xf32, #[[LAYOUT_1]]>
    // CHECK: %{{.*}} = "ttnn.add"(%[[IDX]]{{.*}}
    %3 = "ttir.add"(%1, %arg0, %2) : (tensor<1x32x32xf32>, tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> tensor<1x32x32xf32> loc(#loc6)
    %4 = ttir.empty() : tensor<1x32x32xf32> loc(#loc7)
    %5 = "ttir.relu"(%3, %4) : (tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> tensor<1x32x32xf32> loc(#loc7)
    return %5 : tensor<1x32x32xf32> loc(#loc4)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("test_ops.py:17_0_0":0:4)
#loc2 = loc("test_ops.py:17_0_0":0:6)
#loc3 = loc("test_ops.py:17_0_0":0:3)
#loc4 = loc(unknown)
#loc5 = loc("add_1"(#loc1))
#loc6 = loc("add_2"(#loc2))
#loc7 = loc("relu"(#loc3))
