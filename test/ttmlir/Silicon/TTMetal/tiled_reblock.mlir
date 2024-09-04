// RUN: ttmlir-opt --ttir-load-system-desc="path=%system_desc_path%" --ttir-implicit-device --ttir-allocate --convert-ttir-to-ttmetal %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttmetal-to-flatbuffer %t.mlir > %t.ttm

#l1_ = #tt.memory_space<l1>

#untilized = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<64x128xf32, #l1_>>
#tilized = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<2x4x!tt.tile<32 x 32, f32>, #l1_>>
#tilized2x2 = #tt.layout<(d0, d1) -> (d0, d1), undef, <2x2>, memref<1x2x!tt.tile<32 x 32, f32>, #l1_>>
#untilized2x2 = #tt.layout<(d0, d1) -> (d0, d1), undef, <2x2>, memref<32x64xf32, #l1_>>
func.func @tilize_reblock_2D(%arg0: tensor<64x128xf32, #untilized>) -> tensor<64x128xf32, #untilized2x2> {
  // CHECK: %[[C:.*]] = "ttmetal.alloc"[[C:.*]]
  %0 = tensor.empty() : tensor<64x128xf32, #tilized>
  // CHECK: %[[C:.*]] = "ttmetal.dispatch"[[C:.*]]
  %1 = "ttir.to_layout"(%arg0, %0) : (tensor<64x128xf32, #untilized>, tensor<64x128xf32, #tilized>) -> tensor<64x128xf32, #tilized>
  // CHECK: %[[C:.*]] = "ttmetal.alloc"[[C:.*]]
  %2 = tensor.empty() : tensor<64x128xf32, #tilized2x2>
  // CHECK: %[[C:.*]] = "ttmetal.dispatch"[[C:.*]]
  %3 = "ttir.to_layout"(%1, %2) : (tensor<64x128xf32, #tilized>, tensor<64x128xf32, #tilized2x2>) -> tensor<64x128xf32, #tilized2x2>
  // CHECK: %[[C:.*]] = "ttmetal.alloc"[[C:.*]]
  %4 = tensor.empty() : tensor<64x128xf32, #untilized2x2>
  // CHECK: %[[C:.*]] = "ttmetal.dispatch"[[C:.*]]
  %5 = "ttir.to_layout"(%3, %4) : (tensor<64x128xf32, #tilized2x2>, tensor<64x128xf32, #untilized2x2>) -> tensor<64x128xf32, #untilized2x2>
  return %5 : tensor<64x128xf32, #untilized2x2>
}


#untilized4D = #tt.layout<(d0, d1, d2, d3) -> (d0 * 192 + d1 * 64 + d2, d3), undef, <1x1>, memref<384x128xf32, #l1_>>
#tilized4D = #tt.layout<(d0, d1, d2, d3) -> (d0 * 192 + d1 * 64 + d2, d3), undef, <1x1>, memref<12x4x!tt.tile<32 x 32, f32>, #l1_>>
#tilized4D_2x2 = #tt.layout<(d0, d1, d2, d3) -> (d0 * 192 + d1 * 64 + d2, d3), undef, <2x2>, memref<6x2x!tt.tile<32 x 32, f32>, #l1_>>
#untilized4D_2x2 = #tt.layout<(d0, d1, d2, d3) -> (d0 * 192 + d1 * 64 + d2, d3), undef, <2x2>, memref<192x64xf32, #l1_>>
func.func @tilize_reblock_4D(%arg0: tensor<2x3x64x128xf32, #untilized4D>) -> tensor<2x3x64x128xf32, #untilized4D_2x2> {
  // CHECK: %[[C:.*]] = "ttmetal.alloc"[[C:.*]]
  %0 = tensor.empty() : tensor<2x3x64x128xf32, #tilized4D>
  // CHECK: %[[C:.*]] = "ttmetal.dispatch"[[C:.*]]
  %1 = "ttir.to_layout"(%arg0, %0) : (tensor<2x3x64x128xf32, #untilized4D>, tensor<2x3x64x128xf32, #tilized4D>) -> tensor<2x3x64x128xf32, #tilized4D>

  // CHECK: %[[C:.*]] = "ttmetal.alloc"[[C:.*]]
  %2 = tensor.empty() : tensor<2x3x64x128xf32, #tilized4D_2x2>
  // CHECK: %[[C:.*]] = "ttmetal.dispatch"[[C:.*]]
  %3 = "ttir.to_layout"(%1, %2) : (tensor<2x3x64x128xf32, #tilized4D>, tensor<2x3x64x128xf32, #tilized4D_2x2>) -> tensor<2x3x64x128xf32, #tilized4D_2x2>

  // CHECK: %[[C:.*]] = "ttmetal.alloc"[[C:.*]]
  %4 = tensor.empty() : tensor<2x3x64x128xf32, #untilized4D_2x2>
  // CHECK: %[[C:.*]] = "ttmetal.dispatch"[[C:.*]]
  %5 = "ttir.to_layout"(%3, %4) : (tensor<2x3x64x128xf32, #tilized4D_2x2>, tensor<2x3x64x128xf32, #untilized4D_2x2>) -> tensor<2x3x64x128xf32, #untilized4D_2x2>

  return %5 : tensor<2x3x64x128xf32, #untilized4D_2x2>
}

#untilized_big = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<96x192xf32, #l1_>>
#tilized_big = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<3x6x!tt.tile<32 x 32, f32>, #l1_>>
#tilized_big_3x2 = #tt.layout<(d0, d1) -> (d0, d1), undef, <3x2>, memref<1x3x!tt.tile<32 x 32, f32>, #l1_>>
#tilized_big_3x6 = #tt.layout<(d0, d1) -> (d0, d1), undef, <3x6>, memref<1x1x!tt.tile<32 x 32, f32>, #l1_>>
func.func @tilize_reblock_big(%arg0: tensor<96x192xf32, #untilized_big>) -> tensor<96x192xf32, #untilized_big> {
  // move to tilized 1x1
  // CHECK: %[[C:.*]] = "ttmetal.alloc"[[C:.*]]
  %0 = tensor.empty() : tensor<96x192xf32, #tilized_big>
  // CHECK: %[[C:.*]] = "ttmetal.dispatch"[[C:.*]]
  %1 = "ttir.to_layout"(%arg0, %0) : (tensor<96x192xf32, #untilized_big>, tensor<96x192xf32, #tilized_big>) -> tensor<96x192xf32, #tilized_big>

  // move to tilized 2x3
  // CHECK: %[[C:.*]] = "ttmetal.alloc"[[C:.*]]
  %2 = tensor.empty() : tensor<96x192xf32, #tilized_big_3x2>
  // CHECK: %[[C:.*]] = "ttmetal.dispatch"[[C:.*]]
  %3 = "ttir.to_layout"(%1, %2) : (tensor<96x192xf32, #tilized_big>, tensor<96x192xf32, #tilized_big_3x2>) -> tensor<96x192xf32, #tilized_big_3x2>

  // move to tilized 3x3
  // CHECK: %[[C:.*]] = "ttmetal.alloc"[[C:.*]]
  %4 = tensor.empty() : tensor<96x192xf32, #tilized_big_3x6>
  // CHECK: %[[C:.*]] = "ttmetal.dispatch"[[C:.*]]
  %5 = "ttir.to_layout"(%3, %4) : (tensor<96x192xf32, #tilized_big_3x2>, tensor<96x192xf32, #tilized_big_3x6>) -> tensor<96x192xf32, #tilized_big_3x6>

  // move back to tilized 1x1
  // CHECK: %[[C:.*]] = "ttmetal.alloc"[[C:.*]]
  %6 = tensor.empty() : tensor<96x192xf32, #tilized_big>
  // CHECK: %[[C:.*]] = "ttmetal.dispatch"[[C:.*]]
  %7 = "ttir.to_layout"(%5, %6) : (tensor<96x192xf32, #tilized_big_3x6>, tensor<96x192xf32, #tilized_big>) -> tensor<96x192xf32, #tilized_big>

  // untilize
  // CHECK: %[[C:.*]] = "ttmetal.alloc"[[C:.*]]
  %8 = tensor.empty() : tensor<96x192xf32, #untilized_big>
  // CHECK: %[[C:.*]] = "ttmetal.dispatch"[[C:.*]]
  %9 = "ttir.to_layout"(%7, %8) : (tensor<96x192xf32, #tilized_big>, tensor<96x192xf32, #untilized_big>) -> tensor<96x192xf32, #untilized_big>

  return %9 : tensor<96x192xf32, #untilized_big>
}
