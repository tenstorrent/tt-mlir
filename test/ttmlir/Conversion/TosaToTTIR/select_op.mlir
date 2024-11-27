// RUN: ttmlir-opt --convert-tosa-to-ttir %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {} {
  func.func @test_select(%arg0: tensor<32x128xi1>, %arg1: tensor<32x128xf32>, %arg2: tensor<32x128xf32>) -> tensor<32x128xf32> {
    %0 = tosa.select %arg0, %arg1, %arg2 : (tensor<32x128xi1>, tensor<32x128xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.where"[[C:.*]]
    return %0 : tensor<32x128xf32>
  }
}
