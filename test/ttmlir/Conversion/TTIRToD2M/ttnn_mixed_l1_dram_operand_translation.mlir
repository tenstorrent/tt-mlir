// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m="ttnn-mode=true" --d2m-grid-selection --canonicalize -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

#l1 = #ttnn.buffer_type<l1>

// CHECK: #layout = #ttcore.metal_layout<logical_shape = 512x2048, dim_alignments = 32x32, collapsed_intervals
// CHECK-SAME: dram, interleaved, index_map = map(0)>
// CHECK: #layout1 = #ttcore.metal_layout<logical_shape = 512x2048, dim_alignments = 32x32, collapsed_intervals
// CHECK-SAME: l1, sharded, index_map = map(0)>
// CHECK: #layout2 = #ttcore.metal_layout<logical_shape = 512x2048, dim_alignments = 32x32, collapsed_intervals
// CHECK-SAME: dram, interleaved, index_map = (d0, d1, d2, d3) ->
// CHECK: #layout3 = #ttcore.metal_layout<logical_shape = 512x2048, dim_alignments = 32x32, collapsed_intervals
// CHECK-SAME: l1, sharded, index_map = (d0, d1, d2, d3) ->
// CHECK: #layout4 = #ttcore.metal_layout<logical_shape = 128x160, dim_alignments = 32x32, collapsed_intervals
// CHECK-SAME: l1, sharded, index_map = map(0)>
// CHECK: #layout5 = #ttcore.metal_layout<logical_shape = 160x96, dim_alignments = 32x32, collapsed_intervals
// CHECK-SAME: dram, interleaved, index_map = map(0)>
// CHECK: #layout6 = #ttcore.metal_layout<logical_shape = 160x96, dim_alignments = 32x32, collapsed_intervals
// CHECK-SAME: l1, sharded, index_map = map(0)>
// CHECK: #layout7 = #ttcore.metal_layout<logical_shape = 160x96, dim_alignments = 32x32, collapsed_intervals
// CHECK-SAME: dram, interleaved, index_map = (d0, d1, d2, d3) ->
// CHECK: #layout8 = #ttcore.metal_layout<logical_shape = 128x96, dim_alignments = 32x32, collapsed_intervals
// CHECK-SAME: l1, sharded, index_map = map(0)>
// CHECK: #layout9 = #ttcore.metal_layout<logical_shape = 128x160, dim_alignments = 32x32, collapsed_intervals
// CHECK-SAME: dram, interleaved, index_map = map(0)>
// CHECK: #layout10 = #ttcore.metal_layout<logical_shape = 128x160, dim_alignments = 32x32, collapsed_intervals
// CHECK-SAME: dram, interleaved, index_map = (d0, d1, d2, d3) ->

#ttnn_layout_dram = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>
#ttnn_layout_l1 =   #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <2x2>, memref<32x32x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>>

#ttnn_layout_in0_l1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <4x5>, memref<32x32x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>>
#ttnn_layout_in1_l1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <5x3>, memref<32x32x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>>
#ttnn_layout_out_l1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <4x3>, memref<32x32x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>>
#ttnn_layout_in0_dram = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>
#ttnn_layout_in1_dram = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>
#ttnn_layout_out_dram = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>

module {

// CHECK-LABEL: func.func @test_mixed_operands_eltwise_unary_l1_dram
func.func @test_mixed_operands_eltwise_unary_l1_dram(%arg0: tensor<512x2048xbf16, #ttnn_layout_dram>) -> tensor<512x2048xbf16, #ttnn_layout_l1> {

  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 {{.*}} -> tensor<1x1x16x64x!ttcore.tile<32x32, bf16>, #layout>
  // CHECK: %[[STREAM:.*]] = "d2m.stream_layout"(%[[CAST0]], %{{.*}}) : (tensor<1x1x16x64x!ttcore.tile<32x32, bf16>, #layout>, tensor<8x8x2x8x!ttcore.tile<32x32, bf16>, #layout1>) -> tensor<8x8x2x8x!ttcore.tile<32x32, bf16>, #layout2>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %{{.*}} -> tensor<2x2x8x32x!ttcore.tile<32x32, bf16>, #layout1>
  // CHECK: %[[VIEW:.*]] = d2m.view_layout %[[CAST1]] : tensor<2x2x8x32x!ttcore.tile<32x32, bf16>, #layout1> -> tensor<8x8x2x8x!ttcore.tile<32x32, bf16>, #layout3>
  // CHECK: %{{.*}} = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<8x8>
  // CHECK: ins(%[[STREAM]] : tensor<8x8x2x8x!ttcore.tile<32x32, bf16>, #layout2>)
  // CHECK: outs(%[[VIEW]] : tensor<8x8x2x8x!ttcore.tile<32x32, bf16>, #layout3>)
  %1 = "ttir.abs"(%arg0)  : (tensor<512x2048xbf16, #ttnn_layout_dram>) -> (tensor<512x2048xbf16, #ttnn_layout_l1>)

  return %1 : tensor<512x2048xbf16, #ttnn_layout_l1>
}

// CHECK-LABEL: func.func @test_mixed_operands_eltwise_unary_dram_dram
func.func @test_mixed_operands_eltwise_unary_dram_dram(%arg0: tensor<512x2048xbf16, #ttnn_layout_dram>) -> tensor<512x2048xbf16, #ttnn_layout_dram> {

  // CHECK: %[[instream:.*]] = "d2m.stream_layout"(%cast{{.*}}, %{{.*}}) <{remapping = #map{{.*}}}> : (tensor<1x1x16x64x!ttcore.tile<32x32, bf16>, #layout{{.*}}>, tensor<8x8x2x8x!ttcore.tile<32x32, bf16>, #layout{{.*}}>) -> tensor<8x8x2x8x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[outstream:.*]] = "d2m.stream_layout"(%cast{{.*}}, %{{.*}}) <{remapping = #map{{.*}}}> : (tensor<1x1x16x64x!ttcore.tile<32x32, bf16>, #layout{{.*}}>, tensor<8x8x2x8x!ttcore.tile<32x32, bf16>, #layout{{.*}}>) -> tensor<8x8x2x8x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %{{.*}} = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<8x8>
  // CHECK: ins(%[[STREAM0]] : tensor<8x8x2x8x!ttcore.tile<32x32, bf16>, #layout2>)
  // CHECK: outs(%[[STREAM1]] : tensor<8x8x2x8x!ttcore.tile<32x32, bf16>, #layout2>)
  %1 = "ttir.abs"(%arg0)  : (tensor<512x2048xbf16, #ttnn_layout_dram>) -> (tensor<512x2048xbf16, #ttnn_layout_dram>)

  return %1 : tensor<512x2048xbf16, #ttnn_layout_dram>
}

// CHECK-LABEL: func.func @test_mixed_operands_eltwise_binary_l1_dram_dram
func.func @test_mixed_operands_eltwise_binary_l1_dram_dram(
    %arg0: tensor<512x2048xbf16, #ttnn_layout_l1>,
    %arg1: tensor<512x2048xbf16, #ttnn_layout_dram>)
    -> tensor<512x2048xbf16, #ttnn_layout_dram>
{
  // CHECK: %[[L1_CAST:.*]] = ttir.ttnn_metal_layout_cast %arg0 {{.*}} -> tensor<2x2x8x32x!ttcore.tile<32x32, bf16>, #layout1>
  // CHECK: %[[DRAM_CAST:.*]] = ttir.ttnn_metal_layout_cast %arg1 {{.*}} -> tensor<1x1x16x64x!ttcore.tile<32x32, bf16>, #layout>
  // CHECK: %[[STREAM:.*]] = "d2m.stream_layout"(%[[DRAM_CAST]], %{{.*}}) <{remapping = #map{{.*}}}> : (tensor<1x1x16x64x!ttcore.tile<32x32, bf16>, #layout{{.*}}>, tensor<2x2x8x32x!ttcore.tile<32x32, bf16>, #layout{{.*}}>) -> tensor<2x2x8x32x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[OUT_STREAM:.*]] = "d2m.stream_layout"(%{{.*}}, %{{.*}}) <{remapping = #map{{.*}}}> : (tensor<1x1x16x64x!ttcore.tile<32x32, bf16>, #layout{{.*}}>, tensor<2x2x8x32x!ttcore.tile<32x32, bf16>, #layout{{.*}}>) -> tensor<2x2x8x32x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %{{.*}} = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x2>
  // CHECK: ins(%[[L1_CAST]], %[[STREAM]] : tensor<2x2x8x32x!ttcore.tile<32x32, bf16>, #layout{{.*}}>, tensor<2x2x8x32x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  // CHECK: outs(%[[OUT_STREAM]] : tensor<2x2x8x32x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  %1 = "ttir.add"(%arg0,%arg1)  : (tensor<512x2048xbf16, #ttnn_layout_l1>,tensor<512x2048xbf16, #ttnn_layout_dram>) -> (tensor<512x2048xbf16, #ttnn_layout_dram>)
  return %1 : tensor<512x2048xbf16, #ttnn_layout_dram>
}

// CHECK-LABEL: func.func @test_mixed_operands_eltwise_binary_l1_l1_dram
func.func @test_mixed_operands_eltwise_binary_l1_l1_dram(
    %arg0: tensor<512x2048xbf16, #ttnn_layout_l1>,
    %arg1: tensor<512x2048xbf16, #ttnn_layout_l1>)
    -> tensor<512x2048xbf16, #ttnn_layout_dram>
{
  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 {{.*}} -> tensor<2x2x8x32x!ttcore.tile<32x32, bf16>, #layout1>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %arg1 {{.*}} -> tensor<2x2x8x32x!ttcore.tile<32x32, bf16>, #layout1>
  // CHECK: %[[OUT_STREAM:.*]] = "d2m.stream_layout"(%{{.*}}, %{{.*}}) <{remapping = #map{{.*}}}> : (tensor<1x1x16x64x!ttcore.tile<32x32, bf16>, #layout{{.*}}>, tensor<2x2x8x32x!ttcore.tile<32x32, bf16>, #layout{{.*}}>) -> tensor<2x2x8x32x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %{{.*}} = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x2>
  // CHECK: ins(%[[CAST0]], %[[CAST1]] : tensor<2x2x8x32x!ttcore.tile<32x32, bf16>, #layout1>, tensor<2x2x8x32x!ttcore.tile<32x32, bf16>, #layout1>)
  // CHECK: outs(%[[OUT_STREAM]] : tensor<2x2x8x32x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  %1 = "ttir.add"(%arg0,%arg1)  : (tensor<512x2048xbf16, #ttnn_layout_l1>,tensor<512x2048xbf16, #ttnn_layout_l1>) -> (tensor<512x2048xbf16, #ttnn_layout_dram>)
  return %1 : tensor<512x2048xbf16, #ttnn_layout_dram>
}

// CHECK-LABEL: func.func @test_mixed_operands_eltwise_binary_dram_dram_dram
func.func @test_mixed_operands_eltwise_binary_dram_dram_dram(
    %arg0: tensor<512x2048xbf16, #ttnn_layout_dram>,
    %arg1: tensor<512x2048xbf16, #ttnn_layout_dram>)
    -> tensor<512x2048xbf16, #ttnn_layout_dram>
{
  // CHECK: %[[STREAM0:.*]] = "d2m.stream_layout"(%{{.*}}, %{{.*}}) <{remapping = #map{{.*}}}> : (tensor<1x1x16x64x!ttcore.tile<32x32, bf16>, #layout{{.*}}>, tensor<8x8x2x8x!ttcore.tile<32x32, bf16>, #layout{{.*}}>) -> tensor<8x8x2x8x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[STREAM1:.*]] = "d2m.stream_layout"(%{{.*}}, %{{.*}}) <{remapping = #map{{.*}}}> : (tensor<1x1x16x64x!ttcore.tile<32x32, bf16>, #layout{{.*}}>, tensor<8x8x2x8x!ttcore.tile<32x32, bf16>, #layout{{.*}}>) -> tensor<8x8x2x8x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[OUT_STREAM:.*]] = "d2m.stream_layout"(%{{.*}}, %{{.*}}) <{remapping = #map{{.*}}}> : (tensor<1x1x16x64x!ttcore.tile<32x32, bf16>, #layout{{.*}}>, tensor<8x8x2x8x!ttcore.tile<32x32, bf16>, #layout{{.*}}>) -> tensor<8x8x2x8x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %{{.*}} = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<8x8>
  // CHECK: ins(%[[STREAM0]], %[[STREAM1]] : tensor<8x8x2x8x!ttcore.tile<32x32, bf16>, #layout{{.*}}>, tensor<8x8x2x8x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  // CHECK: outs(%[[OUT_STREAM]] : tensor<8x8x2x8x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  %1 = "ttir.add"(%arg0,%arg1)  : (tensor<512x2048xbf16, #ttnn_layout_dram>,tensor<512x2048xbf16, #ttnn_layout_dram>) -> (tensor<512x2048xbf16, #ttnn_layout_dram>)
  return %1 : tensor<512x2048xbf16, #ttnn_layout_dram>
}

// CHECK-LABEL: func.func @test_mixed_operands_eltwise_binary_dram_dram_l1
func.func @test_mixed_operands_eltwise_binary_dram_dram_l1(
    %arg0: tensor<512x2048xbf16, #ttnn_layout_dram>,
    %arg1: tensor<512x2048xbf16, #ttnn_layout_dram>)
    -> tensor<512x2048xbf16, #ttnn_layout_l1>
{
  // CHECK: %[[STREAM0:.*]] = "d2m.stream_layout"(%{{.*}}, %{{.*}}) <{remapping = #map{{.*}}}> : (tensor<1x1x16x64x!ttcore.tile<32x32, bf16>, #layout{{.*}}>, tensor<2x2x8x32x!ttcore.tile<32x32, bf16>, #layout{{.*}}>) -> tensor<2x2x8x32x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[STREAM1:.*]] = "d2m.stream_layout"(%{{.*}}, %{{.*}}) <{remapping = #map{{.*}}}> : (tensor<1x1x16x64x!ttcore.tile<32x32, bf16>, #layout{{.*}}>, tensor<2x2x8x32x!ttcore.tile<32x32, bf16>, #layout{{.*}}>) -> tensor<2x2x8x32x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[OUT_CAST:.*]] = ttir.ttnn_metal_layout_cast %{{.*}} -> tensor<2x2x8x32x!ttcore.tile<32x32, bf16>, #layout1>
  // CHECK: %{{.*}} = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x2>
  // CHECK: ins(%[[STREAM0]], %[[STREAM1]] : tensor<2x2x8x32x!ttcore.tile<32x32, bf16>, #layout{{.*}}>, tensor<2x2x8x32x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  // CHECK: outs(%[[OUT_CAST]] : tensor<2x2x8x32x!ttcore.tile<32x32, bf16>, #layout1>)
  %1 = "ttir.add"(%arg0,%arg1)  : (tensor<512x2048xbf16, #ttnn_layout_dram>,tensor<512x2048xbf16, #ttnn_layout_dram>) -> (tensor<512x2048xbf16, #ttnn_layout_l1>)
  return %1 : tensor<512x2048xbf16, #ttnn_layout_l1>
}

// CHECK-LABEL: func.func @test_mixed_operands_matmul_l1_dram_l1
func.func @test_mixed_operands_matmul_l1_dram_l1(
    %arg0: tensor<128x160xbf16, #ttnn_layout_in0_l1>,
    %arg1: tensor<160x96xbf16, #ttnn_layout_in1_dram>)
        -> tensor<128x96xbf16, #ttnn_layout_out_l1>
{
  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 {{.*}} -> tensor<4x5x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %arg1 {{.*}} -> tensor<1x1x5x3x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[STREAM:.*]] = "d2m.stream_layout"(%[[CAST1]], %{{.*}}) <{remapping = #map{{.*}}}> : (tensor<1x1x5x3x!ttcore.tile<32x32, bf16>, #layout{{.*}}>, tensor<5x3x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>) -> tensor<5x3x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[OUT_CAST:.*]] = ttir.ttnn_metal_layout_cast %{{.*}} -> tensor<4x3x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %{{.*}} = d2m.generic {block_factors = [1, 1, 5], grid = #ttcore.grid<4x3>
  // CHECK: ins(%[[CAST0]], %[[STREAM]] : tensor<4x5x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>, tensor<5x3x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  // CHECK: outs(%[[OUT_CAST]] : tensor<4x3x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  %1 = "ttir.matmul"(%arg0,%arg1)  :
    (   tensor<128x160xbf16, #ttnn_layout_in0_l1>,
        tensor<160x96xbf16, #ttnn_layout_in1_dram>)
    -> (tensor<128x96xbf16, #ttnn_layout_out_l1>)

  return %1 : tensor<128x96xbf16, #ttnn_layout_out_l1>
}

// CHECK-LABEL: func.func @test_mixed_operands_matmul_dram_dram_l1
func.func @test_mixed_operands_matmul_dram_dram_l1(
    %arg0: tensor<128x160xbf16, #ttnn_layout_in0_dram>,
    %arg1: tensor<160x96xbf16, #ttnn_layout_in1_dram>)
        -> tensor<128x96xbf16, #ttnn_layout_out_l1>
{
  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 {{.*}} -> tensor<1x1x4x5x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[STREAM0:.*]] = "d2m.stream_layout"(%[[CAST0]], %{{.*}}) <{remapping = #map{{.*}}}> : (tensor<1x1x4x5x!ttcore.tile<32x32, bf16>, #layout{{.*}}>, tensor<4x5x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>) -> tensor<4x5x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %arg1 {{.*}} -> tensor<1x1x5x3x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[STREAM1:.*]] = "d2m.stream_layout"(%[[CAST1]], %{{.*}}) <{remapping = #map{{.*}}}> : (tensor<1x1x5x3x!ttcore.tile<32x32, bf16>, #layout{{.*}}>, tensor<5x3x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>) -> tensor<5x3x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[OUT_CAST:.*]] = ttir.ttnn_metal_layout_cast %{{.*}} -> tensor<4x3x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %{{.*}} = d2m.generic {block_factors = [1, 1, 5], grid = #ttcore.grid<4x3>
  // CHECK: ins(%[[STREAM0]], %[[STREAM1]] : tensor<4x5x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>, tensor<5x3x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  // CHECK: outs(%[[OUT_CAST]] : tensor<4x3x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  %1 = "ttir.matmul"(%arg0,%arg1)  :
    (   tensor<128x160xbf16, #ttnn_layout_in0_dram>,
        tensor<160x96xbf16, #ttnn_layout_in1_dram>)
    -> (tensor<128x96xbf16, #ttnn_layout_out_l1>)

  return %1 : tensor<128x96xbf16, #ttnn_layout_out_l1>
}

// NOTE: check in0 being dram
// CHECK-LABEL: func.func @test_mixed_operands_matmul_dram_l1_l1
func.func @test_mixed_operands_matmul_dram_l1_l1(
    %arg0: tensor<128x160xbf16, #ttnn_layout_in0_dram>,
    %arg1: tensor<160x96xbf16, #ttnn_layout_in1_l1>)
        -> tensor<128x96xbf16, #ttnn_layout_out_l1>
{
  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 {{.*}} -> tensor<1x1x4x5x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[STREAM0:.*]] = "d2m.stream_layout"(%[[CAST0]], %{{.*}}) <{remapping = #map{{.*}}}> : (tensor<1x1x4x5x!ttcore.tile<32x32, bf16>, #layout{{.*}}>, tensor<4x5x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>) -> tensor<4x5x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %arg1 {{.*}} -> tensor<5x3x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[OUT_CAST:.*]] = ttir.ttnn_metal_layout_cast %{{.*}} -> tensor<4x3x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %{{.*}} = d2m.generic {block_factors = [1, 1, 5], grid = #ttcore.grid<4x3>
  // CHECK: ins(%[[STREAM0]], %[[CAST1]] : tensor<4x5x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>, tensor<5x3x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  // CHECK: outs(%[[OUT_CAST]] : tensor<4x3x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  %1 = "ttir.matmul"(%arg0,%arg1)  :
    (   tensor<128x160xbf16, #ttnn_layout_in0_dram>,
        tensor<160x96xbf16, #ttnn_layout_in1_l1>)
    -> (tensor<128x96xbf16, #ttnn_layout_out_l1>)

  return %1 : tensor<128x96xbf16, #ttnn_layout_out_l1>
}

}
