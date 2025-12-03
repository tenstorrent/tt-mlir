// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m="ttnn-mode=true" --d2m-grid-selection --canonicalize -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

#l1 = #ttnn.buffer_type<l1>

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

  // CHECK: %[[dramstream:.*]] = "d2m.stream_layout"(%cast{{.*}}, %{{.*}})
  // CHECK: %[[l1cast:.*]] = ttir.ttnn_metal_layout_cast {{.*}} -> tensor<2x2x8x32x!ttcore.tile<32x32, bf16>
  // CHECK: %{{.*}} = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x2>
  // CHECK: ins(%[[dramstream]] : tensor<2x2x8x32x!ttcore.tile<32x32, bf16>
  // CHECK: outs(%[[l1cast]] : tensor<2x2x8x32x!ttcore.tile<32x32, bf16>
  %1 = "ttir.abs"(%arg0)  : (tensor<512x2048xbf16, #ttnn_layout_dram>) -> (tensor<512x2048xbf16, #ttnn_layout_l1>)

  return %1 : tensor<512x2048xbf16, #ttnn_layout_l1>
}

// CHECK-LABEL: func.func @test_mixed_operands_eltwise_unary_dram_dram
func.func @test_mixed_operands_eltwise_unary_dram_dram(%arg0: tensor<512x2048xbf16, #ttnn_layout_dram>) -> tensor<512x2048xbf16, #ttnn_layout_dram> {

  // CHECK: %[[instream:.*]] = "d2m.stream_layout"(%cast{{.*}}, %{{.*}}) : (tensor<1x1x16x64x!ttcore.tile<32x32, bf16>, #layout>, tensor<8x8x2x8x!ttcore.tile<32x32, bf16>, #layout{{.*}}>) -> tensor<8x8x2x8x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[outstream:.*]] = "d2m.stream_layout"(%cast{{.*}}, %{{.*}}) : (tensor<1x1x16x64x!ttcore.tile<32x32, bf16>, #layout>, tensor<8x8x2x8x!ttcore.tile<32x32, bf16>, #layout{{.*}}>) -> tensor<8x8x2x8x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %{{.*}} = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<8x8>
  // CHECK:     ins(%[[instream]] : tensor<8x8x2x8x!ttcore.tile<32x32, bf16>,
  // CHECK:     outs(%[[outstream]] : tensor<8x8x2x8x!ttcore.tile<32x32, bf16>,
  %1 = "ttir.abs"(%arg0)  : (tensor<512x2048xbf16, #ttnn_layout_dram>) -> (tensor<512x2048xbf16, #ttnn_layout_dram>)

  return %1 : tensor<512x2048xbf16, #ttnn_layout_dram>
}

// CHECK-LABEL: func.func @test_mixed_operands_eltwise_binary_l1_dram_dram
func.func @test_mixed_operands_eltwise_binary_l1_dram_dram(
    %arg0: tensor<512x2048xbf16, #ttnn_layout_l1>,
    %arg1: tensor<512x2048xbf16, #ttnn_layout_dram>)
    -> tensor<512x2048xbf16, #ttnn_layout_dram>
{
  %1 = "ttir.add"(%arg0,%arg1)  : (tensor<512x2048xbf16, #ttnn_layout_l1>,tensor<512x2048xbf16, #ttnn_layout_dram>) -> (tensor<512x2048xbf16, #ttnn_layout_dram>)
  return %1 : tensor<512x2048xbf16, #ttnn_layout_dram>
}

// CHECK-LABEL: func.func @test_mixed_operands_eltwise_binary_l1_l1_dram
func.func @test_mixed_operands_eltwise_binary_l1_l1_dram(
    %arg0: tensor<512x2048xbf16, #ttnn_layout_l1>,
    %arg1: tensor<512x2048xbf16, #ttnn_layout_l1>)
    -> tensor<512x2048xbf16, #ttnn_layout_dram>
{
  %1 = "ttir.add"(%arg0,%arg1)  : (tensor<512x2048xbf16, #ttnn_layout_l1>,tensor<512x2048xbf16, #ttnn_layout_l1>) -> (tensor<512x2048xbf16, #ttnn_layout_dram>)
  return %1 : tensor<512x2048xbf16, #ttnn_layout_dram>
}

// CHECK-LABEL: func.func @test_mixed_operands_eltwise_binary_dram_dram_dram
func.func @test_mixed_operands_eltwise_binary_dram_dram_dram(
    %arg0: tensor<512x2048xbf16, #ttnn_layout_dram>,
    %arg1: tensor<512x2048xbf16, #ttnn_layout_dram>)
    -> tensor<512x2048xbf16, #ttnn_layout_dram>
{
  %1 = "ttir.add"(%arg0,%arg1)  : (tensor<512x2048xbf16, #ttnn_layout_dram>,tensor<512x2048xbf16, #ttnn_layout_dram>) -> (tensor<512x2048xbf16, #ttnn_layout_dram>)
  return %1 : tensor<512x2048xbf16, #ttnn_layout_dram>
}

// CHECK-LABEL: func.func @test_mixed_operands_eltwise_binary_dram_dram_l1
func.func @test_mixed_operands_eltwise_binary_dram_dram_l1(
    %arg0: tensor<512x2048xbf16, #ttnn_layout_dram>,
    %arg1: tensor<512x2048xbf16, #ttnn_layout_dram>)
    -> tensor<512x2048xbf16, #ttnn_layout_l1>
{
  %1 = "ttir.add"(%arg0,%arg1)  : (tensor<512x2048xbf16, #ttnn_layout_dram>,tensor<512x2048xbf16, #ttnn_layout_dram>) -> (tensor<512x2048xbf16, #ttnn_layout_l1>)
  return %1 : tensor<512x2048xbf16, #ttnn_layout_l1>
}

// CHECK-LABEL: func.func @test_mixed_operands_matmul_l1_dram_l1
func.func @test_mixed_operands_matmul_l1_dram_l1(
    %arg0: tensor<128x160xbf16, #ttnn_layout_in0_l1>,
    %arg1: tensor<160x96xbf16, #ttnn_layout_in1_dram>)
        -> tensor<128x96xbf16, #ttnn_layout_out_l1>
{
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
  %1 = "ttir.matmul"(%arg0,%arg1)  :
    (   tensor<128x160xbf16, #ttnn_layout_in0_dram>,
        tensor<160x96xbf16, #ttnn_layout_in1_l1>)
    -> (tensor<128x96xbf16, #ttnn_layout_out_l1>)

  return %1 : tensor<128x96xbf16, #ttnn_layout_out_l1>
}

}
