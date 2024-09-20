// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|tile|any_device|any_device_tile>
module {
  func.func @forward(%arg0: tensor<1x32x512x1024xbf16>) -> tensor<1x1x1x1024xbf16> {
    %0 = tensor.empty() : tensor<1x1x512x1024xbf16>
    %1 = "ttir.mean"(%arg0, %0) <{dim_arg = [-3: i32], keep_dim = true, operand_constraints = [#any_device, #any_device]}> : (tensor<1x32x512x1024xbf16>, tensor<1x1x512x1024xbf16>) -> tensor<1x1x512x1024xbf16>
    %2 = tensor.empty() : tensor<1x1x1x1024xbf16>
    %3 = "ttir.mean"(%1, %2) <{dim_arg = [-2: i32], keep_dim = true, operand_constraints = [#any_device, #any_device]}> : (tensor<1x1x512x1024xbf16>, tensor<1x1x1x1024xbf16>) -> tensor<1x1x1x1024xbf16>

    return %3 : tensor<1x1x1x1024xbf16>
  }
}
