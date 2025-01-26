// RUN: ttmlir-opt --canonicalize %s| FileCheck %s
#input = #tt.argument_type<input>
module attributes {} {
  func.func @erase_inverse_ops_example(%arg0: tensor<1x1024xbf16, #input>) -> tensor<1x1024xbf16>
  attributes {inputs = [0: i32, 1: i32], parameters = [0: i32], constants = [0: i32]}
  {
    // CHECK-NOT: %[[C:.*]] = "ttir.reshape"[[C:.*]]
    // CHECK-NOT: %[[C:.*]] = "ttir.permute"[[C:.*]]
    %permute1DPS = tensor.empty() : tensor<1024x1xbf16>
    %permute1 = "ttir.permute"(%arg0, %permute1DPS) <{permutation = array<i64: 1, 0>}> : (tensor<1x1024xbf16, #input>, tensor<1024x1xbf16>) -> tensor<1024x1xbf16>
    %reshape1DPS = tensor.empty() : tensor<32x32xbf16>
    %reshape1 = "ttir.reshape"(%permute1, %reshape1DPS) <{shape = [32: i32, 32: i32]}> : (tensor<1024x1xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %expDPS = tensor.empty() : tensor<32x32xbf16>
    %exp= "ttir.exp"(%reshape1, %expDPS) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %absDPS= tensor.empty() : tensor<32x32xbf16>
    %abs = "ttir.abs"(%exp, %absDPS) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %reshape2DPS = tensor.empty() : tensor<1024x1xbf16>
    %reshape2 = "ttir.reshape"(%abs, %reshape2DPS) <{shape = [1024: i32, 1: i32]}> : (tensor<32x32xbf16>, tensor<1024x1xbf16>) -> tensor<1024x1xbf16>
    %permute2DPS = tensor.empty() : tensor<1x1024xbf16>
    %permute2 = "ttir.permute"(%reshape2, %permute2DPS) <{permutation = array<i64: 1, 0>}> : (tensor<1024x1xbf16>, tensor<1x1024xbf16>) -> tensor<1x1024xbf16>
    return %permute2: tensor<1x1024xbf16>
  }
}
