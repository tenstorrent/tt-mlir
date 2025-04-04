// RUN: ttmlir-opt --mlir-print-local-scope --ttir-to-ttnn-backend-pipeline="enable-optimizer=true override-output-layout=add_0=1x1,add_1=l1,add_2=block_sharded,add_3=f32,add_4=l1:interleaved,add_5=width_sharded:tile,add_6=4x4:dram:interleaved:row_major:f32,add_7=4x4:l1:interleaved:tile:f32" %s | FileCheck %s
#loc = loc("test_ops.py:17_0_0":0:0)
module attributes {} {
  func.func @main(%arg0: tensor<1x32x32xbf16> loc("test_ops.py:17_0_0":0:0), %arg1: tensor<1x32x32xbf16> loc("test_ops.py:17_0_0":0:0), %arg2: tensor<1x32x32xbf16> loc("test_ops.py:17_0_0":0:0)) -> (tensor<1x32x32xbf16>, tensor<1x32x32xbf16>) {
    %0 = ttir.empty() : tensor<1x32x32xbf16> loc(#loc5)
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x32x32xbf16
    // CHECK-SAME: <1x1, (d0, d1) -> (0, d0, d1)>
    %1 = "ttir.add"(%arg1, %arg2, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x32xbf16>, tensor<1x32x32xbf16>, tensor<1x32x32xbf16>) -> tensor<1x32x32xbf16> loc(#loc5)
    %2 = ttir.empty() : tensor<1x32x32xbf16> loc(#loc6)
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x32x32xbf16
    // CHECK-SAME: #ttnn.buffer_type<l1>
    %3 = "ttir.add"(%1, %arg0, %2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x32xbf16>, tensor<1x32x32xbf16>, tensor<1x32x32xbf16>) -> tensor<1x32x32xbf16> loc(#loc6)
    %4 = ttir.empty() : tensor<1x32x32xbf16> loc(#loc7)
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x32x32xbf16
    // CHECK-SAME: <block_sharded>
    %5 = "ttir.add"(%arg2, %arg1, %4) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x32xbf16>, tensor<1x32x32xbf16>, tensor<1x32x32xbf16>) -> tensor<1x32x32xbf16> loc(#loc7)
    %6 = ttir.empty() : tensor<1x32x32xbf16>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x32x32xf32
    // CHECK-SAME: memref<{{.*}}f32
    %7 = "ttir.add"(%arg1, %5, %6) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x32xbf16>, tensor<1x32x32xbf16>, tensor<1x32x32xbf16>) -> tensor<1x32x32xbf16> loc(#loc8)
    %8 = ttir.empty() : tensor<1x32x32xbf16>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x32x32xbf16
    // CHECK-SAME: #ttnn.buffer_type<l1>>, <interleaved>
    %9 = "ttir.add"(%arg1, %7, %8) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x32xbf16>, tensor<1x32x32xbf16>, tensor<1x32x32xbf16>) -> tensor<1x32x32xbf16> loc(#loc9)
    %10 = ttir.empty() : tensor<1x32x32xbf16>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x32x32xbf16
    // CHECK-SAME: memref<{{.*}}tt.tile{{.*}}, <width_sharded>
    %11 = "ttir.add"(%arg1, %9, %10) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x32xbf16>, tensor<1x32x32xbf16>, tensor<1x32x32xbf16>) -> tensor<1x32x32xbf16> loc(#loc10)
    %12 = ttir.empty() : tensor<1x32x32xbf16>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x32x32xf32
    // CHECK-SAME: <4x4>, memref<8x8xf32, #ttnn.buffer_type<dram>>, <interleaved>>
    %13 = "ttir.add"(%arg1, %11, %12) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x32xbf16>, tensor<1x32x32xbf16>, tensor<1x32x32xbf16>) -> tensor<1x32x32xbf16> loc(#loc11)
    %14 = ttir.empty() : tensor<1x32x32xbf16>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x32x32xf32
    // CHECK-SAME: <4x4>, memref<1x1x!tt.tile<32x32, f32>, #ttnn.buffer_type<l1>>, <interleaved>>
    %15 = "ttir.add"(%arg1, %13, %14) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x32xbf16>, tensor<1x32x32xbf16>, tensor<1x32x32xbf16>) -> tensor<1x32x32xbf16> loc(#loc12)
    %16 = ttir.empty() : tensor<1x32x32xbf16>
    %17 = "ttir.add"(%arg1, %15, %16) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x32xbf16>, tensor<1x32x32xbf16>, tensor<1x32x32xbf16>) -> tensor<1x32x32xbf16> loc(#loc13)
    return %3, %17 : tensor<1x32x32xbf16>, tensor<1x32x32xbf16> loc(#loc4)
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
