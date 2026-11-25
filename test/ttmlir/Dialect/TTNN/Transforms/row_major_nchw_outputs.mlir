// RUN: ttmlir-opt --ttcore-register-device --ttnn-row-major-nchw-outputs --mlir-print-local-scope -o %t %s
// RUN: FileCheck %s --input-file=%t

// An NHWC sharded TILE value returned as NCHW through reshape -> tile permute
// -> to_memory_config is rebuilt as a sharded untilize into row-major L1, a
// reshape view and a row-major permute into the output memory; the function
// result becomes row-major.

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
// [1,1,624,45] TILE height-sharded on 20 cores, shard [32, 64]
#hs = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 640 + d1 * 640 + d2, d3), <20x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,1)>, #ttnn.core_range<(0,2), (3,2)>]>>
#nhwc = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 768 + d1 * 64 + d2, d3), <8x8>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#nchw = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1440 + d1 * 32 + d2, d3), <8x8>, memref<1x2x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#out = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1440 + d1 * 32 + d2, d3), <1x1>, memref<45x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

// Head [1,1,39,7] TILE height-sharded on 2 cores, cut into outputs 0:3, 3:7.
#hs2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 64 + d2, d3), <2x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (1,0)>]>>
#i2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 64 + d2, d3), <8x8>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#nhwc2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 96 + d1 * 32 + d2, d3), <8x8>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#nchw3 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 96 + d1 * 32 + d2, d3), <8x8>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#nchw4 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 32 + d2, d3), <8x8>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#out3 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 96 + d1 * 32 + d2, d3), <1x1>, memref<3x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#out4 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 32 + d2, d3), <1x1>, memref<4x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module {
  // CHECK-LABEL: func.func @sliced_head
  // CHECK-SAME: -> (tensor<1x3x3x13xbf16, #ttnn.ttnn_layout<{{.*}}memref<9x13xbf16, #ttnn.buffer_type<dram>>, <interleaved>>>, tensor<1x4x3x13xbf16, #ttnn.ttnn_layout<{{.*}}memref<12x13xbf16, #ttnn.buffer_type<dram>>, <interleaved>>>, tensor<1x1x39x7xbf16
  func.func @sliced_head(%arg0: tensor<1x1x39x7xbf16, #hs2>) -> (tensor<1x3x3x13xbf16, #out3>, tensor<1x4x3x13xbf16, #out4>, tensor<1x1x39x7xbf16, #hs2>) {
    // CHECK: %[[U:.*]] = "ttnn.to_layout"(%arg0)
    // CHECK: %[[R:.*]] = "ttnn.reshape"(%[[U]])
    // CHECK-SAME: -> tensor<1x3x13x7xbf16
    // CHECK: %[[P:.*]] = "ttnn.permute"(%[[R]])
    // CHECK-SAME: -> tensor<1x7x3x13xbf16
    // CHECK: %[[S0:.*]] = "ttnn.slice_static"(%[[P]]) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 3 : i32, 3 : i32, 13 : i32]
    // CHECK: %[[S1:.*]] = "ttnn.slice_static"(%[[P]]) <{begins = [0 : i32, 3 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 7 : i32, 3 : i32, 13 : i32]
    // CHECK-NOT: "ttnn.to_memory_config"
    // CHECK: %[[INNER:.*]] = "ttnn.relu"(%arg0)
    // CHECK: return %[[S0]], %[[S1]], %[[INNER]]
    %0 = "ttnn.to_memory_config"(%arg0) : (tensor<1x1x39x7xbf16, #hs2>) -> tensor<1x1x39x7xbf16, #i2>
    %1 = "ttnn.slice_static"(%0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1 : i32, 39 : i32, 3 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x39x7xbf16, #i2>) -> tensor<1x1x39x3xbf16, #i2>
    %2 = "ttnn.reshape"(%1) <{shape = [1 : i32, 3 : i32, 13 : i32, 3 : i32]}> : (tensor<1x1x39x3xbf16, #i2>) -> tensor<1x3x13x3xbf16, #nhwc2>
    %3 = "ttnn.permute"(%2) <{pad_value = 0.000000e+00 : f32, permutation = array<i64: 0, 3, 1, 2>}> : (tensor<1x3x13x3xbf16, #nhwc2>) -> tensor<1x3x3x13xbf16, #nchw3>
    %4 = "ttnn.to_memory_config"(%3) : (tensor<1x3x3x13xbf16, #nchw3>) -> tensor<1x3x3x13xbf16, #out3>
    %5 = "ttnn.to_memory_config"(%arg0) : (tensor<1x1x39x7xbf16, #hs2>) -> tensor<1x1x39x7xbf16, #i2>
    %6 = "ttnn.slice_static"(%5) <{begins = [0 : i32, 0 : i32, 0 : i32, 3 : i32], ends = [1 : i32, 1 : i32, 39 : i32, 7 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x39x7xbf16, #i2>) -> tensor<1x1x39x4xbf16, #i2>
    %7 = "ttnn.reshape"(%6) <{shape = [1 : i32, 3 : i32, 13 : i32, 4 : i32]}> : (tensor<1x1x39x4xbf16, #i2>) -> tensor<1x3x13x4xbf16, #nhwc2>
    %8 = "ttnn.permute"(%7) <{pad_value = 0.000000e+00 : f32, permutation = array<i64: 0, 3, 1, 2>}> : (tensor<1x3x13x4xbf16, #nhwc2>) -> tensor<1x4x3x13xbf16, #nchw4>
    %9 = "ttnn.to_memory_config"(%8) : (tensor<1x4x3x13xbf16, #nchw4>) -> tensor<1x4x3x13xbf16, #out4>
    %10 = "ttnn.relu"(%arg0) : (tensor<1x1x39x7xbf16, #hs2>) -> tensor<1x1x39x7xbf16, #hs2>
    return %4, %9, %10 : tensor<1x3x3x13xbf16, #out3>, tensor<1x4x3x13xbf16, #out4>, tensor<1x1x39x7xbf16, #hs2>
  }

  // CHECK-LABEL: func.func @nchw_output
  // CHECK-SAME: -> tensor<1x45x12x52xbf16, #ttnn.ttnn_layout<{{.*}}memref<540x52xbf16, #ttnn.buffer_type<dram>>, <interleaved>>>
  func.func @nchw_output(%arg0: tensor<1x1x624x45xbf16, #hs>) -> tensor<1x45x12x52xbf16, #out> {
    // CHECK: %[[U:.*]] = "ttnn.to_layout"(%arg0)
    // CHECK-SAME: -> tensor<1x1x624x45xbf16, #ttnn.ttnn_layout<{{.*}}memref<{{.*}}xbf16, #ttnn.buffer_type<l1>>, <interleaved>>>
    // CHECK: %[[R:.*]] = "ttnn.reshape"(%[[U]])
    // CHECK-SAME: -> tensor<1x12x52x45xbf16, #ttnn.ttnn_layout<{{.*}}xbf16, #ttnn.buffer_type<l1>>, <interleaved>>>
    // CHECK: %[[P:.*]] = "ttnn.permute"(%[[R]])
    // CHECK-SAME: permutation = array<i64: 0, 3, 1, 2>
    // CHECK-SAME: -> tensor<1x45x12x52xbf16, #ttnn.ttnn_layout<{{.*}}memref<540x52xbf16, #ttnn.buffer_type<dram>>, <interleaved>>>
    // CHECK-NOT: "ttnn.to_memory_config"
    // CHECK: return %[[P]]
    %0 = "ttnn.reshape"(%arg0) <{shape = [1 : i32, 12 : i32, 52 : i32, 45 : i32]}> : (tensor<1x1x624x45xbf16, #hs>) -> tensor<1x12x52x45xbf16, #nhwc>
    %1 = "ttnn.permute"(%0) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<1x12x52x45xbf16, #nhwc>) -> tensor<1x45x12x52xbf16, #nchw>
    %2 = "ttnn.to_memory_config"(%1) : (tensor<1x45x12x52xbf16, #nchw>) -> tensor<1x45x12x52xbf16, #out>
    return %2 : tensor<1x45x12x52xbf16, #out>
  }

  // A permute that feeds anything other than a layout chain to the return is
  // left alone.
  // CHECK-LABEL: func.func @not_an_output
  // CHECK: "ttnn.permute"
  // CHECK-SAME: -> tensor<1x45x12x52xbf16, #ttnn.ttnn_layout<{{.*}}!ttcore.tile
  func.func @not_an_output(%arg0: tensor<1x1x624x45xbf16, #hs>) -> tensor<1x45x12x52xbf16, #nchw> {
    %0 = "ttnn.reshape"(%arg0) <{shape = [1 : i32, 12 : i32, 52 : i32, 45 : i32]}> : (tensor<1x1x624x45xbf16, #hs>) -> tensor<1x12x52x45xbf16, #nhwc>
    %1 = "ttnn.permute"(%0) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<1x12x52x45xbf16, #nhwc>) -> tensor<1x45x12x52xbf16, #nchw>
    %2 = "ttnn.relu"(%1) : (tensor<1x45x12x52xbf16, #nchw>) -> tensor<1x45x12x52xbf16, #nchw>
    return %2 : tensor<1x45x12x52xbf16, #nchw>
  }
}
