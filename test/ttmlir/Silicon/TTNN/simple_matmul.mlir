// RUN: ttmlir-opt --ttir-implicit-device --ttir-layout --ttnn-open-device --convert-ttir-to-ttnn %s  > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>
module attributes {tt.system_desc = #tt.system_desc<[{arch = <wormhole_b0>, grid = 8x8, l1_size = 1048576, num_dram_channels = 12, dram_channel_size = 1048576, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32}], [0], [<pcie|host_mmio>], [<0, 0, 0, 0>]>} {
  func.func @forward(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x96xbf16>) -> tensor<64x96xbf16> {
    %0 = tensor.empty() : tensor<64x96xbf16>
    // CHECK: %[[C:.*]] = "ttnn.matmul"[[C:.*]]
    %1 = "ttir.matmul"(%arg0, %arg1, %0) <{operand_constraints = [#any_device_tile, #any_device_tile, #any_device_tile]}> : (tensor<64x128xbf16>, tensor<128x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16>
    return %1 : tensor<64x96xbf16>
  }
}
