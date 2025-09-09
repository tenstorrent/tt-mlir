// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=false override-output-layout=add_0=1x1,add_1=l1,add_2=block_sharded,add_3=bf16,add_4=l1:interleaved,add_5=width_sharded:tile,add_6=1x1:dram:interleaved:row_major:bf16,add_7=4x4:l1:interleaved:tile:f32" -o %t %s
// RUN: FileCheck %s --input-file=%t

#loc = loc("test_ops.py:17_0_0":0:0)
module attributes {} {
  func.func @main(%arg0: tensor<1x32x32xf32> loc("test_ops.py:17_0_0":0:0), %arg1: tensor<1x32x32xf32> loc("test_ops.py:17_0_0":0:0), %arg2: tensor<1x32x32xf32> loc("test_ops.py:17_0_0":0:0)) -> (tensor<1x32x32xf32>, tensor<1x32x32xf32>) {
    // CHECK: #[[L1_:.*]] = #ttnn.buffer_type<l1>
    // CHECK: #[[LAYOUT_0:.*]] = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
    // CHECK: #[[LAYOUT_1:.*]] = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <8x8, (d0, d1) -> (0, d0, d1)>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <interleaved>>
    // CHECK: #[[LAYOUT_2:.*]] = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1, (d0, d1) -> (0, d0, d1)>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>>
    // CHECK: #[[LAYOUT_3:.*]] = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
    // CHECK: #[[LAYOUT_4:.*]] = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1, (d0, d1) -> (0, d1 floordiv 8, d1 mod 8)>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <width_sharded>>
    // CHECK: #[[LAYOUT_5:.*]] = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<32x32xbf16, #dram>, <interleaved>>
    // CHECK: #[[LAYOUT_6:.*]] = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <4x4>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <interleaved>>
    %0 = ttir.empty() : tensor<1x32x32xf32> loc(#loc5)
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x32x32xf32, #[[LAYOUT_0]]>
    %1 = "ttir.add"(%arg1, %arg2, %0) : (tensor<1x32x32xf32>, tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> tensor<1x32x32xf32> loc(#loc5)
    %2 = ttir.empty() : tensor<1x32x32xf32> loc(#loc6)
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x32x32xf32, #[[LAYOUT_1]]>
    %3 = "ttir.add"(%1, %arg0, %2) : (tensor<1x32x32xf32>, tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> tensor<1x32x32xf32> loc(#loc6)
    %4 = ttir.empty() : tensor<1x32x32xf32> loc(#loc7)
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x32x32xf32, #[[LAYOUT_2]]>
    %5 = "ttir.add"(%arg2, %arg1, %4) : (tensor<1x32x32xf32>, tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> tensor<1x32x32xf32> loc(#loc7)
    %6 = ttir.empty() : tensor<1x32x32xf32>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x32x32xbf16, #[[LAYOUT_3]]>
    %7 = "ttir.add"(%arg1, %5, %6) : (tensor<1x32x32xf32>, tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> tensor<1x32x32xf32> loc(#loc8)
    %8 = ttir.empty() : tensor<1x32x32xf32>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x32x32xf32, #[[LAYOUT_1]]>
    %9 = "ttir.add"(%arg1, %7, %8) : (tensor<1x32x32xf32>, tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> tensor<1x32x32xf32> loc(#loc9)
    %10 = ttir.empty() : tensor<1x32x32xf32>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x32x32xf32, #[[LAYOUT_4]]>
    %11 = "ttir.add"(%arg1, %9, %10) : (tensor<1x32x32xf32>, tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> tensor<1x32x32xf32> loc(#loc10)
    %12 = ttir.empty() : tensor<1x32x32xf32>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x32x32xbf16, #[[LAYOUT_3]]>
    %13 = "ttir.add"(%arg1, %11, %12) : (tensor<1x32x32xf32>, tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> tensor<1x32x32xf32> loc(#loc11)
    %14 = ttir.empty() : tensor<1x32x32xf32>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x32x32xf32, #[[LAYOUT_6]]>
    %15 = "ttir.add"(%arg1, %13, %14) : (tensor<1x32x32xf32>, tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> tensor<1x32x32xf32> loc(#loc12)
    %16 = ttir.empty() : tensor<1x32x32xf32>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x32x32xf32, #[[LAYOUT_0]]>
    %17 = "ttir.add"(%arg1, %15, %16) : (tensor<1x32x32xf32>, tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> tensor<1x32x32xf32> loc(#loc13)
    // CHECK: return %[[R0:.*]], %[[R1:.*]] : tensor<1x32x32xf32, #[[LAYOUT_1]]>, tensor<1x32x32xf32, #[[LAYOUT_0]]>
    return %3, %17 : tensor<1x32x32xf32>, tensor<1x32x32xf32> loc(#loc4)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("test_ops.py:17_0_0":0:4)
#loc2 = loc("test_ops.py:17_0_0":0:6)
#loc3 = loc("test_ops.py:17_0_0":0:3)
#loc4 = loc(unknown)
#loc5 = loc("add_0"(#loc1))
#loc6 = loc("add_1"(#loc2))
#loc7 = loc("add_2"(#loc4))
#loc8 = loc("add_3"(#loc4))
#loc9 = loc("add_4"(#loc4))
#loc10 = loc("add_5"(#loc4))
#loc11 = loc("add_6"(#loc4))
#loc12 = loc("add_7"(#loc4))
#loc13 = loc("add_8"(#loc4))
