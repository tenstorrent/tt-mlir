// RUN: ttmlir-opt --ttir-load-system-desc="path=%system_desc_path%" --ttir-implicit-device --ttir-allocate --convert-ttir-to-ttmetal %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttmetal-to-flatbuffer %t.mlir > %t.ttm

#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
#l1_ = #tt.memory_space<l1>

#layout = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x3>, memref<2x2x!tt.tile<32 x 32, f32>, #l1_>>

func.func @simple(
  %arg0: tensor<64x192xf32, #layout>,
  %arg1: tensor<64x192xf32, #layout>,
  %arg2: tensor<64x192xf32, #layout>
  ) -> tensor<64x192xf32, #layout> {
  %0 = tensor.empty() : tensor<64x192xf32, #layout>
  // CHECK: %[[C:.*]] = "ttmetal.dispatch"[[C:.*]]
  %1 = "ttir.multiply"(%arg0, %arg1, %0) <{
    operandSegmentSizes = array<i32: 2, 1>,
    operand_constraints = [#any_device, #any_device, #any_device]
  }> : (tensor<64x192xf32, #layout>,
        tensor<64x192xf32, #layout>,
        tensor<64x192xf32, #layout>) -> tensor<64x192xf32, #layout>
  %2 = tensor.empty() : tensor<64x192xf32, #layout>
  // CHECK: %[[C:.*]] = "ttmetal.dispatch"[[C:.*]]
  %3 = "ttir.add"(%1, %arg2, %2) <{
    operandSegmentSizes = array<i32: 2, 1>,
    operand_constraints = [#any_device, #any_device, #any_device]
  }> : (tensor<64x192xf32, #layout>,
        tensor<64x192xf32, #layout>,
        tensor<64x192xf32, #layout>) -> tensor<64x192xf32, #layout>
  return %3 : tensor<64x192xf32, #layout>
}
