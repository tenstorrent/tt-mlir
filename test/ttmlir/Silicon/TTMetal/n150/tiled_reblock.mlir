// RUN: ttmlir-opt --tt-register-device="system-desc-path=%system_desc_path%" --ttir-allocate --convert-ttir-to-ttmetal %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttmetal-to-flatbuffer %t.mlir > %t.ttm
// UNSUPPORTED: true

#l1_ = #tt.memory_space<l1>

#untilized = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<64x128xf32, #l1_>>
#tilized = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<2x4x!tt.tile<32 x 32, f32>, #l1_>>
#tilized2x2 = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <2x2>, memref<1x2x!tt.tile<32 x 32, f32>, #l1_>>
#untilized2x2 = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <2x2>, memref<32x64xf32, #l1_>>
func.func @tilize_reblock_2D(%arg0: tensor<64x128xf32, #untilized>) -> tensor<64x128xf32, #untilized2x2> {
  // CHECK: = "ttmetal.create_buffer"
  %0 = ttir.empty() : tensor<64x128xf32, #tilized>
  // CHECK: = "ttmetal.enqueue_program"
  %1 = "ttir.to_layout"(%arg0, %0) : (tensor<64x128xf32, #untilized>, tensor<64x128xf32, #tilized>) -> tensor<64x128xf32, #tilized>
  // CHECK: = "ttmetal.create_buffer"
  %2 = ttir.empty() : tensor<64x128xf32, #tilized2x2>
  // CHECK: = "ttmetal.enqueue_program"
  %3 = "ttir.to_layout"(%1, %2) : (tensor<64x128xf32, #tilized>, tensor<64x128xf32, #tilized2x2>) -> tensor<64x128xf32, #tilized2x2>
  // CHECK: = "ttmetal.create_buffer"
  %4 = ttir.empty() : tensor<64x128xf32, #untilized2x2>
  // CHECK: = "ttmetal.enqueue_program"
  %5 = "ttir.to_layout"(%3, %4) : (tensor<64x128xf32, #tilized2x2>, tensor<64x128xf32, #untilized2x2>) -> tensor<64x128xf32, #untilized2x2>
  return %5 : tensor<64x128xf32, #untilized2x2>
}


#untilized4D = #tt.metal_layout<(d0, d1, d2, d3) -> (d0 * 192 + d1 * 64 + d2, d3), undef, <1x1>, memref<384x128xf32, #l1_>>
#tilized4D = #tt.metal_layout<(d0, d1, d2, d3) -> (d0 * 192 + d1 * 64 + d2, d3), undef, <1x1>, memref<12x4x!tt.tile<32 x 32, f32>, #l1_>>
#tilized4D_2x2 = #tt.metal_layout<(d0, d1, d2, d3) -> (d0 * 192 + d1 * 64 + d2, d3), undef, <2x2>, memref<6x2x!tt.tile<32 x 32, f32>, #l1_>>
#untilized4D_2x2 = #tt.metal_layout<(d0, d1, d2, d3) -> (d0 * 192 + d1 * 64 + d2, d3), undef, <2x2>, memref<192x64xf32, #l1_>>
func.func @tilize_reblock_4D(%arg0: tensor<2x3x64x128xf32, #untilized4D>) -> tensor<2x3x64x128xf32, #untilized4D_2x2> {
  // CHECK: = "ttmetal.create_buffer"
  %0 = ttir.empty() : tensor<2x3x64x128xf32, #tilized4D>
  // CHECK: = "ttmetal.enqueue_program"
  %1 = "ttir.to_layout"(%arg0, %0) : (tensor<2x3x64x128xf32, #untilized4D>, tensor<2x3x64x128xf32, #tilized4D>) -> tensor<2x3x64x128xf32, #tilized4D>

  // CHECK: = "ttmetal.create_buffer"
  %2 = ttir.empty() : tensor<2x3x64x128xf32, #tilized4D_2x2>
  // CHECK: = "ttmetal.enqueue_program"
  %3 = "ttir.to_layout"(%1, %2) : (tensor<2x3x64x128xf32, #tilized4D>, tensor<2x3x64x128xf32, #tilized4D_2x2>) -> tensor<2x3x64x128xf32, #tilized4D_2x2>

  // CHECK: = "ttmetal.create_buffer"
  %4 = ttir.empty() : tensor<2x3x64x128xf32, #untilized4D_2x2>
  // CHECK: = "ttmetal.enqueue_program"
  %5 = "ttir.to_layout"(%3, %4) : (tensor<2x3x64x128xf32, #tilized4D_2x2>, tensor<2x3x64x128xf32, #untilized4D_2x2>) -> tensor<2x3x64x128xf32, #untilized4D_2x2>

  return %5 : tensor<2x3x64x128xf32, #untilized4D_2x2>
}

#untilized_big = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<96x192xf32, #l1_>>
#tilized_big = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<3x6x!tt.tile<32 x 32, f32>, #l1_>>
#tilized_big_3x2 = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <3x2>, memref<1x3x!tt.tile<32 x 32, f32>, #l1_>>
#tilized_big_3x6 = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <3x6>, memref<1x1x!tt.tile<32 x 32, f32>, #l1_>>
func.func @tilize_reblock_big(%arg0: tensor<96x192xf32, #untilized_big>) -> tensor<96x192xf32, #untilized_big> {
  // move to tilized 1x1
  // CHECK: = "ttmetal.create_buffer"
  %0 = ttir.empty() : tensor<96x192xf32, #tilized_big>
  // CHECK: = "ttmetal.enqueue_program"
  %1 = "ttir.to_layout"(%arg0, %0) : (tensor<96x192xf32, #untilized_big>, tensor<96x192xf32, #tilized_big>) -> tensor<96x192xf32, #tilized_big>

  // move to tilized 2x3
  // CHECK: = "ttmetal.create_buffer"
  %2 = ttir.empty() : tensor<96x192xf32, #tilized_big_3x2>
  // CHECK: = "ttmetal.enqueue_program"
  %3 = "ttir.to_layout"(%1, %2) : (tensor<96x192xf32, #tilized_big>, tensor<96x192xf32, #tilized_big_3x2>) -> tensor<96x192xf32, #tilized_big_3x2>

  // move to tilized 3x3
  // CHECK: = "ttmetal.create_buffer"
  %4 = ttir.empty() : tensor<96x192xf32, #tilized_big_3x6>
  // CHECK: = "ttmetal.enqueue_program"
  %5 = "ttir.to_layout"(%3, %4) : (tensor<96x192xf32, #tilized_big_3x2>, tensor<96x192xf32, #tilized_big_3x6>) -> tensor<96x192xf32, #tilized_big_3x6>

  // move back to tilized 1x1
  // CHECK: = "ttmetal.create_buffer"
  %6 = ttir.empty() : tensor<96x192xf32, #tilized_big>
  // CHECK: = "ttmetal.enqueue_program"
  %7 = "ttir.to_layout"(%5, %6) : (tensor<96x192xf32, #tilized_big_3x6>, tensor<96x192xf32, #tilized_big>) -> tensor<96x192xf32, #tilized_big>

  // untilize
  // CHECK: = "ttmetal.create_buffer"
  %8 = ttir.empty() : tensor<96x192xf32, #untilized_big>
  // CHECK: = "ttmetal.enqueue_program"
  %9 = "ttir.to_layout"(%7, %8) : (tensor<96x192xf32, #tilized_big>, tensor<96x192xf32, #untilized_big>) -> tensor<96x192xf32, #untilized_big>

  return %9 : tensor<96x192xf32, #untilized_big>
}
