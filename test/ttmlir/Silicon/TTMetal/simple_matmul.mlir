// RUN: ttmlir-opt --ttir-load-system-desc="path=%system_desc_path%" --ttir-implicit-device --ttir-allocate --convert-ttir-to-ttmetal %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttmetal-to-flatbuffer %t.mlir > %t.ttm

#l1_ = #tt.memory_space<l1>

#in0 = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <2x8>, memref<32x32xf16, #l1_>>
#in1 = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <8x4>, memref<32x32xf16, #l1_>>
#out0 = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <2x2>, memref<32x64xf16, #l1_>>

func.func @simple_matmul(%arg0: tensor<64x256xf16, #in0>, %arg1: tensor<256x128xf16, #in1>) -> tensor<64x128xf16, #out0> {
  %0 = tensor.empty() : tensor<64x128xf16, #out0>
  // CHECK: %[[C:.*]] = "ttmetal.dispatch"[[C:.*]]
  %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<64x256xf16, #in0>, tensor<256x128xf16, #in1>, tensor<64x128xf16, #out0>) -> tensor<64x128xf16, #out0>
  return %1 : tensor<64x128xf16, #out0>
}