// RUN: ttmlir-opt --ttir-generic --ttir-layout --ttir-generic-region-operands-to-memref --ttir-allocate --convert-ttir-to-ttmetal %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {torch.debug_module_name = "_lambda", tt.system_desc = #tt.system_desc<[<#tt.arch<wormhole_b0>, #tt.grid<8x8>>], [0], [<pcie|host_mmio>], [<0, 0, 0, 0>]>} {
  func.func @forward(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CHECK: %[[C:.*]] = "ttmetal.alloc"[[C:.*]]
    // CHECK: %[[C:.*]] = "ttmetal.host_write"[[C:.*]]
    %0 = tensor.empty() : tensor<64x128xf32>
    // CHECK: %[[C:.*]] = "ttmetal.dispatch"[[C:.*]]
    %1 = "ttir.multiply"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: "ttmetal.dealloc"[[C:.*]]
    // CHECK: %[[C:.*]] = "ttmetal.host_read"[[C:.*]]
    return %1 : tensor<64x128xf32>
  }
}

