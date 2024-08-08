// RUN: ttmlir-opt --ttir-layout --ttnn-open-device --convert-ttir-to-ttnn %s | FileCheck %s
#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>
module attributes {tt.system_desc = #tt.system_desc<[{arch = <wormhole_b0>, grid = 8x8, l1_size = 1048576, num_dram_channels = 12, dram_channel_size = 1048576, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32}], [0], [<pcie|host_mmio>], [<0, 0, 0, 0>]>} {
  func.func @forward(%arg0: tensor<8x8xbf16>) -> tensor<8x8xbf16> {
    %0 = tensor.empty() : tensor<8x8xbf16>
    // CHECK: %[[C:.*]] = "ttnn.transpose"[[C:.*]]
    %1 = "ttir.transpose"(%arg0, %0) <{dimension1 = -1 : si32, dimension2 = -2 : si32, operand_constraints = [#any_device_tile, #any_device_tile]}> : (tensor<8x8xbf16>, tensor<8x8xbf16>) -> tensor<8x8xbf16>
    return %1 : tensor<8x8xbf16>
  }
}
