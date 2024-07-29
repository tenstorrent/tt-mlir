// RUN: ttmlir-opt --ttir-layout %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {tt.system_desc = #tt.system_desc<[{arch = <wormhole_b0>, grid = 8x8, l1_size = 1048576, num_dram_channels = 12, dram_channel_size = 1048576, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32}], [0], [<pcie|host_mmio>], [<0, 0, 0, 0>]>} {
  func.func @forward(%arg0: tensor<8x64x128xf32>, %arg1: tensor<8x64x128xf32>) -> tensor<8x64x128xf32> {
    %0 = tensor.empty() : tensor<8x64x128xf32>
    // CHECK: %[[C:.*]] = "ttir.to_layout"[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.to_layout"[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.multiply"[[C:.*]]
    %1 = "ttir.multiply"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<8x64x128xf32>, tensor<8x64x128xf32>, tensor<8x64x128xf32>) -> tensor<8x64x128xf32>
    return %1 : tensor<8x64x128xf32>
  }
}
