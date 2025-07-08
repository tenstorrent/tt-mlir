// RUN: ttmlir-opt --ttir-erase-inverse-ops="enable-commute-upwards=false" %s | FileCheck %s

module {
    func.func @test_commute_reshape_downwards(%arg0: tensor<32x64xbf16>) -> tensor<1x2048xbf16> {
      // CHECK: %[[EXP:[0-9]+]] = "ttir.exp"(%arg0
      // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%[[EXP]]
      // CHECK: return %[[RESHAPE]]
      %0 = ttir.empty() : tensor<1x2048xbf16>
      %1 = "ttir.reshape"(%arg0, %0) <{shape = [1 : i32, 2048 : i32]}> : (tensor<32x64xbf16>, tensor<1x2048xbf16>) -> tensor<1x2048xbf16>
      %2 = ttir.empty() : tensor<1x2048xbf16>
      %3 = "ttir.exp"(%1, %2) : (tensor<1x2048xbf16>, tensor<1x2048xbf16>) -> tensor<1x2048xbf16>
      return %3 : tensor<1x2048xbf16>
    }
}

module {
    func.func @test_commute_permute_downwards(%arg0: tensor<1x3x224x224xbf16>) -> tensor<1x224x224x3xbf16> {
      // CHECK: %[[EXP:[0-9]+]] = "ttir.exp"(%arg0
      // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"(%[[EXP]]
      // CHECK: return %[[PERMUTE]]
      %0 = ttir.empty() : tensor<1x224x224x3xbf16>
      %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x3x224x224xbf16>, tensor<1x224x224x3xbf16>) -> tensor<1x224x224x3xbf16>
      %2 = ttir.empty() : tensor<1x224x224x3xbf16>
      %3 = "ttir.exp"(%1, %2) : (tensor<1x224x224x3xbf16>, tensor<1x224x224x3xbf16>) -> tensor<1x224x224x3xbf16>
      return %3 : tensor<1x224x224x3xbf16>
    }
}
