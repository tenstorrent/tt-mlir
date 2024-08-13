// RUN: not ttmlir-opt --ttir-layout --ttnn-open-device --convert-ttir-to-ttnn %s 2>&1 | FileCheck %s
// CHECK: error: 'ttir.multiply' op Operands are not broadcast compatible
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {tt.system_desc = #tt.system_desc<[{arch = <wormhole_b0>, grid = 8x8, l1_size = 1048576, num_dram_channels = 12, dram_channel_size = 1048576, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32}], [0], [<pcie|host_mmio>], [<0, 0, 0, 0>]>} {
  func.func @bcast_one_dim(%arg0: tensor<2x64x128xf32>, %arg1: tensor<4x64x128xf32>) -> tensor<4x64x128xf32> {
    %0 = tensor.empty() : tensor<4x64x128xf32>
    %1 = "ttir.multiply"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<2x64x128xf32>, tensor<4x64x128xf32>, tensor<4x64x128xf32>) -> tensor<4x64x128xf32>
    return %1 : tensor<4x64x128xf32>
  }
}
