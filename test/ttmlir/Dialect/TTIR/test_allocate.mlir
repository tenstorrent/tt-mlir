// RUN: ttmlir-opt --ttir-load-system-desc --ttir-implicit-device --ttir-allocate %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
#l1_ = #tt.memory_space<l1>
#layout = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<64x128xf32, #l1_>>
module attributes {tt.device = #tt.device<#tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, [0]>, tt.system_desc = #tt.system_desc<[{arch = <wormhole_b0>, grid = 8x8, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32}], [0], [<pcie|host_mmio>], [<0, 0, 0, 0>]>} {
  func.func @forward(%arg0: tensor<64x128xf32, #layout>, %arg1: tensor<64x128xf32, #layout>) -> tensor<64x128xf32, #layout> {
    // CHECK: %[[C:.*]] = "ttir.alloc"[[C:.*]]
    // CHECK-NOT: %[[C:.*]] = tensor.empty() : tensor<64x128xf32>
    %0 = tensor.empty() : tensor<64x128xf32, #layout>
    %1 = "ttir.multiply"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x128xf32, #layout>, tensor<64x128xf32, #layout>, tensor<64x128xf32, #layout>) -> tensor<64x128xf32, #layout>
    return %1 : tensor<64x128xf32, #layout>
  }
}
