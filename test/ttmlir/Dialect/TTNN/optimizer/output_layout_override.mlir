// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true override-output-layout=add_0=1x1,add_1=l1,add_2=block_sharded,add_3=bf16,add_4=l1:interleaved,add_5=width_sharded:tile,add_6=4x4:dram:interleaved:row_major:bf16,add_7=4x4:l1:interleaved:tile:f32" %s | FileCheck %s
#loc = loc("test_ops.py:17_0_0":0:0)
module attributes {} {
  func.func @main(%arg0: tensor<1x32x32xf32> loc("test_ops.py:17_0_0":0:0), %arg1: tensor<1x32x32xf32> loc("test_ops.py:17_0_0":0:0), %arg2: tensor<1x32x32xf32> loc("test_ops.py:17_0_0":0:0)) -> (tensor<1x32x32xf32>, tensor<1x32x32xf32>) {
    // CHECK: #[[L1_:.*]] = #ttnn.buffer_type<l1>
    // CHECK: #[[LAYOUT_0:.*]] = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<32x32xf32, #system_memory>>
    // CHECK: #[[LAYOUT_1:.*]] = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1, (d0, d1) -> (0, d0, d1)>, memref<1x1x!tt.tile<32x32, f32>, #l1_>, <block_sharded>>
    // CHECK: #[[LAYOUT_2:.*]] = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <8x8, (d0, d1) -> (0, d0, d1)>, memref<1x1x!tt.tile<32x32, f32>, #l1_>, <interleaved>>
    // CHECK: #[[LAYOUT_3:.*]] = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <8x8, (d0, d1) -> (0, d0, d1)>, memref<1x1x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
    // CHECK: #[[LAYOUT_4:.*]] = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1, (d0, d1) -> (0, d1 floordiv 8, d1 mod 8)>, memref<1x1x!tt.tile<32x32, f32>, #l1_>, <width_sharded>>
    // CHECK: #[[LAYOUT_5:.*]] = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <4x4>, memref<8x8xbf16, #dram>, <interleaved>>
    // CHECK: #[[LAYOUT_6:.*]] = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <4x4>, memref<1x1x!tt.tile<32x32, f32>, #l1_>, <interleaved>>
    // CHECK: #[[LAYOUT_7:.*]] = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <8x8, (d0, d1) -> (0, d0, d1)>, memref<1x1x!tt.tile<32x32, f32>, #dram>, <interleaved>>
    %0 = tensor.empty() : tensor<1x32x32xf32> loc(#loc5)
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x32x32xf32, #[[LAYOUT_1]]>
    %1 = "ttir.add"(%arg1, %arg2, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x32xf32>, tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> tensor<1x32x32xf32> loc(#loc5)
    %2 = tensor.empty() : tensor<1x32x32xf32> loc(#loc6)
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x32x32xf32, #[[LAYOUT_2]]>
    %3 = "ttir.add"(%1, %arg0, %2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x32xf32>, tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> tensor<1x32x32xf32> loc(#loc6)
    %4 = tensor.empty() : tensor<1x32x32xf32> loc(#loc7)
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x32x32xf32, #[[LAYOUT_1]]>
    %5 = "ttir.add"(%arg2, %arg1, %4) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x32xf32>, tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> tensor<1x32x32xf32> loc(#loc7)
    %6 = tensor.empty() : tensor<1x32x32xf32>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x32x32xf32, #[[LAYOUT_3]]>
    %7 = "ttir.add"(%arg1, %5, %6) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x32xf32>, tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> tensor<1x32x32xf32> loc(#loc8)
    %8 = tensor.empty() : tensor<1x32x32xf32>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x32x32xf32, #[[LAYOUT_2]]>
    %9 = "ttir.add"(%arg1, %7, %8) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x32xf32>, tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> tensor<1x32x32xf32> loc(#loc9)
    %10 = tensor.empty() : tensor<1x32x32xf32>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x32x32xf32, #[[LAYOUT_4]]>
    %11 = "ttir.add"(%arg1, %9, %10) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x32xf32>, tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> tensor<1x32x32xf32> loc(#loc10)
    %12 = tensor.empty() : tensor<1x32x32xf32>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x32x32xf32, #[[LAYOUT_5]]>
    %13 = "ttir.add"(%arg1, %11, %12) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x32xf32>, tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> tensor<1x32x32xf32> loc(#loc11)
    %14 = tensor.empty() : tensor<1x32x32xf32>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x32x32xf32, #[[LAYOUT_6]]>
    %15 = "ttir.add"(%arg1, %13, %14) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x32xf32>, tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> tensor<1x32x32xf32> loc(#loc12)
    %16 = tensor.empty() : tensor<1x32x32xf32>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x32x32xf32, #[[LAYOUT_7]]>
    %17 = "ttir.add"(%arg1, %15, %16) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x32xf32>, tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> tensor<1x32x32xf32> loc(#loc13)
    // CHECK: return %[[R0:.*]], %[[R1:.*]] : tensor<1x32x32xf32, #[[LAYOUT_0]]>, tensor<1x32x32xf32, #[[LAYOUT_0]]>
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
