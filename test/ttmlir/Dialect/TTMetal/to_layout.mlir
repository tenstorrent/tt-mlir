// RUN: ttmlir-opt --ttir-allocate --convert-ttir-to-ttmetal %s | FileCheck %s
#l1_ = #tt.memory_space<l1>
#layout = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<64x128xf32, #l1_>>
#layout1 = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x4>, memref<64x32xf32, #l1_>>
module attributes {tt.system_desc = #tt.system_desc<[{arch = <wormhole_b0>, grid = 8x8, l1_size = 1048576, num_dram_channels = 12, dram_channel_size = 1048576, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32}], [0], [<pcie|host_mmio>], [<0, 0, 0, 0>]>} {
  func.func @forward(%arg0: tensor<64x128xf32, #layout>) -> tensor<64x128xf32, #layout1> {
    // CHECK: %[[C:.*]] = "ttmetal.alloc"[[C:.*]]
    %0 = tensor.empty() : tensor<64x128xf32, #layout1>
    %1 = "ttir.to_layout"(%arg0, %0) : (tensor<64x128xf32, #layout>, tensor<64x128xf32, #layout1>) -> tensor<64x128xf32, #layout1>
    return %1 : tensor<64x128xf32, #layout1>
  }
}
