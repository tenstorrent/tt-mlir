// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%"  --ttnn-workaround --canonicalize %s | FileCheck %s

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x4x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
func.func public @test_reduce_max(%arg0: tensor<128x32xsi32, #ttnn_layout>) -> tensor<128xsi32, #ttnn_layout1> {
  // CHECK-LABEL: @test_reduce_max
  // CHECK: %[[ARG0:[0-9]+]] = "ttnn.to_layout"(%arg0)
  // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>,
  // CHECK-SAME: tensor<128x32xsi32,
  // CHECK-SAME: -> tensor<128x32xbf16,
  // CHECK: %[[MAX:[0-9]+]] = "ttnn.max"(%[[ARG0]])
  // CHECK-SAME: <{dim_arg = [1 : i32], keep_dim = false}>
  // CHECK-SAME: tensor<128x32xbf16,
  // CHECK-SAME: -> tensor<128xbf16,
  %0 = "ttnn.max"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<128x32xsi32, #ttnn_layout>) -> tensor<128xsi32, #ttnn_layout1>
  // CHECK: %{{[0-9]+}} = "ttnn.to_layout"(%[[MAX]])
  // CHECK-SAME: dtype = #ttcore.supportedDataTypes<si32>,
  // CHECK-SAME: tensor<128xbf16,
  // CHECK-SAME: -> tensor<128xsi32,
  return %0 : tensor<128xsi32, #ttnn_layout1>
}

func.func public @test_reduce_sum(%arg0: tensor<128x10xsi32, #ttnn_layout>) -> tensor<128xsi32, #ttnn_layout1> {
  // CHECK-LABEL: @test_reduce_sum
  // CHECK: %[[ARG0:[0-9]+]] = "ttnn.to_layout"(%arg0)
  // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>,
  // CHECK-SAME: tensor<128x10xsi32,
  // CHECK-SAME: -> tensor<128x10xbf16,
  // CHECK: %[[SUM:[0-9]+]] = "ttnn.sum"(%[[ARG0]])
  // CHECK-SAME: <{dim_arg = [1 : i32], keep_dim = false}>
  // CHECK-SAME: tensor<128x10xbf16,
  // CHECK-SAME: -> tensor<128xbf16,
  %0 = "ttnn.sum"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<128x10xsi32, #ttnn_layout>) -> tensor<128xsi32, #ttnn_layout1>
  // CHECK: %{{[0-9]+}} = "ttnn.to_layout"(%[[SUM]])
  // CHECK-SAME: dtype = #ttcore.supportedDataTypes<si32>,
  // CHECK-SAME: tensor<128xbf16,
  // CHECK-SAME: -> tensor<128xsi32,
  return %0 : tensor<128xsi32, #ttnn_layout1>
}

#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
func.func public @test_reduce_max_requires_pad(%arg0: tensor<128x30xf32, #ttnn_layout2>) -> tensor<128xf32, #ttnn_layout3> {
  // CHECK-LABEL: @test_reduce_max_requires_pad
  // CHECK: "ttnn.pad"
  // CHECK-SAME: padding = array<i32: 0, 0, 0, 2>
  // CHECK-SAME: value = 0xFF800000
  // CHECK: "ttnn.max"
  // CHECK-SAME: tensor<128x32xf32
  // CHECK-SAME: -> tensor<128xf32
  %0 = "ttnn.max"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<128x30xf32, #ttnn_layout2>) -> tensor<128xf32, #ttnn_layout3>
  return %0 : tensor<128xf32, #ttnn_layout3>
}
