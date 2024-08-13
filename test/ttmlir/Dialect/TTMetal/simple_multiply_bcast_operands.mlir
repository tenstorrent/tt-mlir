// RUN: ttmlir-opt --ttir-load-system-desc --ttir-implicit-device --ttir-generic --ttir-layout --ttir-generic-region-operands-to-memref --ttir-allocate --convert-ttir-to-ttmetal %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {tt.system_desc = #tt.system_desc<[{arch = <wormhole_b0>, grid = 8x8, l1_size = 1048576, num_dram_channels = 12, dram_channel_size = 1048576, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32}], [0], [<pcie|host_mmio>], [<0, 0, 0, 0>]>} {
  func.func @forward(%arg0: tensor<17x16x15x14xf32>, %arg1: tensor<15x1xf32>) -> tensor<17x16x15x14xf32> {
    // "ttir.broadcast" %arg1 : (tensor<15x1xf32>) -> tensor<17x16x15x14xf32>

    // CHECK: %[[C:.*]] = "ttmetal.alloc"[[C:.*]]
    // CHECK: %[[C:.*]] = "ttmetal.host_write"[[C:.*]]
    %0 = tensor.empty() : tensor<17x16x15x14xf32>
    // CHECK: %[[C:.*]] = "ttmetal.dispatch"[[C:.*]]
    %1 = "ttir.multiply"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<17x16x15x14xf32>, tensor<15x1xf32>, tensor<17x16x15x14xf32>) -> tensor<17x16x15x14xf32>
    // CHECK: "ttmetal.dealloc"[[C:.*]]
    // CHECK: %[[C:.*]] = "ttmetal.host_read"[[C:.*]]
    return %1 : tensor<17x16x15x14xf32>
  }
}
