// RUN: ttmlir-opt --ttir-load-system-desc="path=%system_desc_path%" --ttir-to-ttmetal-backend-pipeline  %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
#l1_ = #tt.memory_space<l1>
#layout1 = #tt.layout<(d0, d1) -> (d0, d1), undef, <4x4>, memref<128x256xf32, #l1_>>
#layout2 = #tt.layout<(d0, d1) -> (d0, d1), undef, <4x1>, memref<128x1024xf32, #l1_>>
#layout3 = #tt.layout<(d0, d1) -> (d0, d1), undef, <4x1>, memref<128x32xf32, #l1_>>

// TODO: add checks
func.func @reduce(%arg0: tensor<512x1024xf32, #layout1>) -> tensor<512x32xf32, #layout3> {
  %0 = tensor.empty() : tensor<512x32xf32, #layout3>
  %1 = "ttir.sum"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>,
                               dim_arg = [-1: i32],
                               keep_dim = true,
                               operand_constraints = [#any_device, #any_device, #any_device]}> :
    (tensor<512x1024xf32, #layout1>, tensor<512x32xf32, #layout3>) -> tensor<512x32xf32, #layout3>
  return %1 : tensor<512x32xf32, #layout3>
}
