// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttir-allocate --convert-ttir-to-ttmetal -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttmetal-to-flatbuffer -o %t.ttm %t.mlir
// UNSUPPORTED: true

#l1_ = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>

#layout = #ttcore.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<4x16xf32, #l1_>>
#layout1 = #ttcore.metal_layout<(d0, d1) -> (d0, d1), undef, <2x2>, memref<2x8xf32, #l1_>>
func.func @simple(%arg0: tensor<4x16xf32, #layout>) -> tensor<4x16xf32, #layout1> {
  %0 = ttir.empty() : tensor<4x16xf32, #layout1>
  // CHECK: = "ttmetal.enqueue_program"
  %1 = "ttir.to_layout"(%arg0, %0) : (tensor<4x16xf32, #layout>, tensor<4x16xf32, #layout1>) -> tensor<4x16xf32, #layout1>
  return %1 : tensor<4x16xf32, #layout1>
}

#untilized = #ttcore.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<64x128xf32, #l1_>>
#tilized = #ttcore.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<2x4x!ttcore.tile<32 x 32, f32>, #l1_>>
func.func @tilize(%arg0: tensor<64x128xf32, #untilized>) -> tensor<64x128xf32, #untilized> {
  %0 = ttir.empty() : tensor<64x128xf32, #tilized>
  // CHECK: = "ttmetal.enqueue_program"
  %1 = "ttir.to_layout"(%arg0, %0) : (tensor<64x128xf32, #untilized>, tensor<64x128xf32, #tilized>) -> tensor<64x128xf32, #tilized>
  %2 = ttir.empty() : tensor<64x128xf32, #untilized>
  // CHECK: = "ttmetal.enqueue_program"
  %3 = "ttir.to_layout"(%1, %2) : (tensor<64x128xf32, #tilized>, tensor<64x128xf32, #untilized>) -> tensor<64x128xf32, #untilized>
  return %3 : tensor<64x128xf32, #untilized>
}

#untilized_dram = #ttcore.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<16x64xf32, #dram>>
#untilized_l1 = #ttcore.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<16x64xf32, #l1_>>
#untilized2x2_dram = #ttcore.metal_layout<(d0, d1) -> (d0, d1), undef, <2x2>, memref<8x32xf32, #dram>>
#untilized2x2_l1 = #ttcore.metal_layout<(d0, d1) -> (d0, d1), undef, <2x2>, memref<8x32xf32, #l1_>>
#untilized1x4_l1 = #ttcore.metal_layout<(d0, d1) -> (d0, d1), undef, <1x4>, memref<16x16xf32, #l1_>>
func.func @dram_to_l1(%arg0: tensor<16x64xf32, #untilized_dram>) -> tensor<16x64xf32, #untilized_l1> {
  %0 = ttir.empty() : tensor<16x64xf32, #untilized_l1>
  // CHECK: = "ttmetal.enqueue_program"
  %1 = "ttir.to_layout"(%arg0, %0) : (tensor<16x64xf32, #untilized_dram>, tensor<16x64xf32, #untilized_l1>) -> tensor<16x64xf32, #untilized_l1>
  return %1 : tensor<16x64xf32, #untilized_l1>
}

func.func @l1_to_dram(%arg0: tensor<16x64xf32, #untilized_l1>) -> tensor<16x64xf32, #untilized_dram> {
  %0 = ttir.empty() : tensor<16x64xf32, #untilized_dram>
  // CHECK: = "ttmetal.enqueue_program"
  %1 = "ttir.to_layout"(%arg0, %0) : (tensor<16x64xf32, #untilized_l1>, tensor<16x64xf32, #untilized_dram>) -> tensor<16x64xf32, #untilized_dram>
  return %1 : tensor<16x64xf32, #untilized_dram>
}

func.func @l1dram_reblock0(%arg0: tensor<16x64xf32, #untilized_l1>) -> tensor<16x64xf32, #untilized_l1> {
  %0 = ttir.empty() : tensor<16x64xf32, #untilized2x2_dram>
  // CHECK: = "ttmetal.enqueue_program"
  %1 = "ttir.to_layout"(%arg0, %0) : (tensor<16x64xf32, #untilized_l1>, tensor<16x64xf32, #untilized2x2_dram>) -> tensor<16x64xf32, #untilized2x2_dram>
  %2 = ttir.empty() : tensor<16x64xf32, #untilized1x4_l1>
  // CHECK: = "ttmetal.enqueue_program"
  %3 = "ttir.to_layout"(%1, %2) : (tensor<16x64xf32, #untilized2x2_dram>, tensor<16x64xf32, #untilized1x4_l1>) -> tensor<16x64xf32, #untilized1x4_l1>
  %4 = ttir.empty() : tensor<16x64xf32, #untilized_l1>
  // CHECK: = "ttmetal.enqueue_program"
  %5 = "ttir.to_layout"(%3, %4) : (tensor<16x64xf32, #untilized1x4_l1>, tensor<16x64xf32, #untilized_l1>) -> tensor<16x64xf32, #untilized_l1>
  return %5 : tensor<16x64xf32, #untilized_l1>
}
