// RUN: not ttmlir-opt --ttir-layout --ttnn-open-device --convert-ttir-to-ttnn %s 2>&1 | FileCheck %s
// CHECK: error: 'ttir.softmax' op Dimension attribute must be within the bounds of the input tensor
#any_device = #tt.operand_constraint<dram|l1|tile|any_device|any_device_tile>
module attributes {tt.system_desc = #tt.system_desc<[{arch = <wormhole_b0>, grid = <8x8>, l1_size = 1048576, num_dram_channels = 12, dram_channel_size = 1048576, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32}], [0], [<pcie|host_mmio>], [<0, 0, 0, 0>]>} {
  func.func @forward(%arg0: tensor<512x1024xbf16>) -> tensor<512x1024xbf16> {
    %0 = tensor.empty() : tensor<512x1024xbf16>
    %1 = "ttir.softmax"(%arg0, %0) <{dimension = -3 : i32, operand_constraints = [#any_device, #any_device]}> : (tensor<512x1024xbf16>, tensor<512x1024xbf16>) -> tensor<512x1024xbf16>
    return %1 : tensor<512x1024xbf16>
  }
}
