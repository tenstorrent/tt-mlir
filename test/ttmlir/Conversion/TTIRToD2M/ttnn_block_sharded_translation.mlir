#l1 = #ttnn.buffer_type<l1>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1, (d0, d1) -> (0, d0, d1)>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<8x8x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x4>, memref<8x2x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <4x1>, memref<2x8x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>>
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 64 + d1, d2), <2x2>, memref<4x2x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>>
#ttnn_layout6 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 64 + d1, d2), <1x1>, memref<8x4x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>>
#ttnn_layout7 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 64 + d1, d2), <4x1>, memref<2x4x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>>
#ttnn_layout8 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 64 + d1, d2), <1x4>, memref<8x1x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>>
#ttnn_layout9 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2, d3), <8x1>, memref<1x8x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>>
#ttnn_layout10 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2, d3), <1x8>, memref<8x1x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>>
#ttnn_layout11 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2, d3), <8x8>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>>
#ttnn_layout12 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2, d3), <1x1>, memref<8x8x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>>
module {
  func.func @test_lower_block_sharded_l1(%arg0: tensor<32x32xf32, #ttnn_layout>, %arg1: tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout> {
    %0 = "ttir.abs"(%arg0, %arg1) : (tensor<32x32xf32, #ttnn_layout>, tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout>
    return %0 : tensor<32x32xf32, #ttnn_layout>
  }
  func.func @test_lower_block_sharded_l1_1(%arg0: tensor<256x256xf32, #ttnn_layout1>, %arg1: tensor<256x256xf32, #ttnn_layout1>) -> tensor<256x256xf32, #ttnn_layout1> {
    %0 = "ttir.abs"(%arg0, %arg1) : (tensor<256x256xf32, #ttnn_layout1>, tensor<256x256xf32, #ttnn_layout1>) -> tensor<256x256xf32, #ttnn_layout1>
    return %0 : tensor<256x256xf32, #ttnn_layout1>
  }
  func.func @test_lower_block_sharded_l1_2(%arg0: tensor<256x256xf32, #ttnn_layout2>, %arg1: tensor<256x256xf32, #ttnn_layout2>) -> tensor<256x256xf32, #ttnn_layout2> {
    %0 = "ttir.abs"(%arg0, %arg1) : (tensor<256x256xf32, #ttnn_layout2>, tensor<256x256xf32, #ttnn_layout2>) -> tensor<256x256xf32, #ttnn_layout2>
    return %0 : tensor<256x256xf32, #ttnn_layout2>
  }
  func.func @test_lower_block_sharded_l1_3(%arg0: tensor<256x256xf32, #ttnn_layout3>, %arg1: tensor<256x256xf32, #ttnn_layout3>) -> tensor<256x256xf32, #ttnn_layout3> {
    %0 = "ttir.abs"(%arg0, %arg1) : (tensor<256x256xf32, #ttnn_layout3>, tensor<256x256xf32, #ttnn_layout3>) -> tensor<256x256xf32, #ttnn_layout3>
    return %0 : tensor<256x256xf32, #ttnn_layout3>
  }
  func.func @test_lower_block_sharded_l1_4(%arg0: tensor<256x256xf32, #ttnn_layout4>, %arg1: tensor<256x256xf32, #ttnn_layout4>) -> tensor<256x256xf32, #ttnn_layout4> {
    %0 = "ttir.abs"(%arg0, %arg1) : (tensor<256x256xf32, #ttnn_layout4>, tensor<256x256xf32, #ttnn_layout4>) -> tensor<256x256xf32, #ttnn_layout4>
    return %0 : tensor<256x256xf32, #ttnn_layout4>
  }
  func.func @test_lower_block_sharded_l1_5(%arg0: tensor<4x64x128xbf16, #ttnn_layout5>, %arg1: tensor<4x64x128xbf16, #ttnn_layout5>) -> tensor<4x64x128xbf16, #ttnn_layout5> {
    %0 = "ttir.abs"(%arg0, %arg1) : (tensor<4x64x128xbf16, #ttnn_layout5>, tensor<4x64x128xbf16, #ttnn_layout5>) -> tensor<4x64x128xbf16, #ttnn_layout5>
    return %0 : tensor<4x64x128xbf16, #ttnn_layout5>
  }
  func.func @test_lower_block_sharded_l1_6(%arg0: tensor<4x64x128xbf16, #ttnn_layout6>, %arg1: tensor<4x64x128xbf16, #ttnn_layout6>) -> tensor<4x64x128xbf16, #ttnn_layout6> {
    %0 = "ttir.abs"(%arg0, %arg1) : (tensor<4x64x128xbf16, #ttnn_layout6>, tensor<4x64x128xbf16, #ttnn_layout6>) -> tensor<4x64x128xbf16, #ttnn_layout6>
    return %0 : tensor<4x64x128xbf16, #ttnn_layout6>
  }
  func.func @test_lower_block_sharded_l1_7(%arg0: tensor<4x64x128xbf16, #ttnn_layout7>, %arg1: tensor<4x64x128xbf16, #ttnn_layout7>) -> tensor<4x64x128xbf16, #ttnn_layout7> {
    %0 = "ttir.abs"(%arg0, %arg1) : (tensor<4x64x128xbf16, #ttnn_layout7>, tensor<4x64x128xbf16, #ttnn_layout7>) -> tensor<4x64x128xbf16, #ttnn_layout7>
    return %0 : tensor<4x64x128xbf16, #ttnn_layout7>
  }
  func.func @test_lower_block_sharded_l1_8(%arg0: tensor<4x64x128xbf16, #ttnn_layout8>, %arg1: tensor<4x64x128xbf16, #ttnn_layout8>) -> tensor<4x64x128xbf16, #ttnn_layout8> {
    %0 = "ttir.abs"(%arg0, %arg1) : (tensor<4x64x128xbf16, #ttnn_layout8>, tensor<4x64x128xbf16, #ttnn_layout8>) -> tensor<4x64x128xbf16, #ttnn_layout8>
    return %0 : tensor<4x64x128xbf16, #ttnn_layout8>
  }
  func.func @test_lower_block_sharded_l1_9(%arg0: tensor<1x8x32x256xbf16, #ttnn_layout9>, %arg1: tensor<1x8x32x256xbf16, #ttnn_layout9>) -> tensor<1x8x32x256xbf16, #ttnn_layout9> {
    %0 = "ttir.abs"(%arg0, %arg1) : (tensor<1x8x32x256xbf16, #ttnn_layout9>, tensor<1x8x32x256xbf16, #ttnn_layout9>) -> tensor<1x8x32x256xbf16, #ttnn_layout9>
    return %0 : tensor<1x8x32x256xbf16, #ttnn_layout9>
  }
  func.func @test_lower_block_sharded_l1_10(%arg0: tensor<1x8x32x256xbf16, #ttnn_layout10>, %arg1: tensor<1x8x32x256xbf16, #ttnn_layout10>) -> tensor<1x8x32x256xbf16, #ttnn_layout10> {
    %0 = "ttir.abs"(%arg0, %arg1) : (tensor<1x8x32x256xbf16, #ttnn_layout10>, tensor<1x8x32x256xbf16, #ttnn_layout10>) -> tensor<1x8x32x256xbf16, #ttnn_layout10>
    return %0 : tensor<1x8x32x256xbf16, #ttnn_layout10>
  }
  func.func @test_lower_block_sharded_l1_11(%arg0: tensor<1x8x32x256xbf16, #ttnn_layout11>, %arg1: tensor<1x8x32x256xbf16, #ttnn_layout11>) -> tensor<1x8x32x256xbf16, #ttnn_layout11> {
    %0 = "ttir.abs"(%arg0, %arg1) : (tensor<1x8x32x256xbf16, #ttnn_layout11>, tensor<1x8x32x256xbf16, #ttnn_layout11>) -> tensor<1x8x32x256xbf16, #ttnn_layout11>
    return %0 : tensor<1x8x32x256xbf16, #ttnn_layout11>
  }
  func.func @test_lower_block_sharded_l1_12(%arg0: tensor<1x8x32x256xbf16, #ttnn_layout12>, %arg1: tensor<1x8x32x256xbf16, #ttnn_layout12>) -> tensor<1x8x32x256xbf16, #ttnn_layout12> {
    %0 = "ttir.abs"(%arg0, %arg1) : (tensor<1x8x32x256xbf16, #ttnn_layout12>, tensor<1x8x32x256xbf16, #ttnn_layout12>) -> tensor<1x8x32x256xbf16, #ttnn_layout12>
    return %0 : tensor<1x8x32x256xbf16, #ttnn_layout12>
  }
}
