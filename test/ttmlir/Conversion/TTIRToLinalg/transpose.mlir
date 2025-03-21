// RUN: ttmlir-opt --convert-ttir-to-linalg %s | FileCheck %s

module {

  func.func @transpose(%arg0: tensor<64x16xbf16>) -> tensor<16x64xbf16> {
    %0 = tensor.empty() : tensor<16x64xbf16>
    // CHECK: %transposed = linalg.transpose ins(%arg0 : tensor<64x16xbf16>) outs(%0 : tensor<16x64xbf16>) permutation = [1, 0]
    %1 = "ttir.transpose"(%arg0, %0) <{dim0 = 1 : si32, dim1 = 0 : si32}> : (tensor<64x16xbf16>, tensor<16x64xbf16>) -> tensor<16x64xbf16>
    return %1 : tensor<16x64xbf16>
  }

  func.func @store_transposed(%arg0: tensor<8x16xbf16>, %arg1: tensor<16x8xbf16>) -> tensor<16x8xbf16> {
    // CHECK: {{%[_a-z0-9]+}} = linalg.transpose ins(%arg0 : tensor<8x16xbf16>) outs(%arg1 : tensor<16x8xbf16>) permutation = [1, 0]
    %result = "ttir.transpose"(%arg0, %arg1) <{dim0 = 0 : si32, dim1 = 1 : si32}> : (tensor<8x16xbf16>, tensor<16x8xbf16>) -> tensor<16x8xbf16>
    return %result : tensor<16x8xbf16>
  }

  func.func @transpose_negative_dim(%arg0: tensor<32x64x128xbf16>) -> tensor<32x128x64xbf16> {
    %0 = tensor.empty() : tensor<32x128x64xbf16>
    // CHECK: {{%[_a-z0-9]+}} = linalg.transpose ins(%arg0 : tensor<32x64x128xbf16>) outs(%0 : tensor<32x128x64xbf16>) permutation = [0, 2, 1]
    %1 = "ttir.transpose"(%arg0, %0) <{dim0 = -1 : si32, dim1 = -2 : si32}> : (tensor<32x64x128xbf16>, tensor<32x128x64xbf16>) -> tensor<32x128x64xbf16>
    return %1 : tensor<32x128x64xbf16>
  }

  func.func @higher_dims(%arg0: tensor<1x16x32x64xbf16>) -> tensor<1x32x64x16xbf16> {
    %0 = tensor.empty() : tensor<1x64x32x16xbf16>
    // CHECK: {{%[_a-z0-9]+}} = linalg.transpose ins(%arg0 : tensor<1x16x32x64xbf16>) outs(%0 : tensor<1x64x32x16xbf16>) permutation = [0, 3, 2, 1]
    %1 = "ttir.transpose"(%arg0, %0) <{dim0 = 3 : si32, dim1 = 1 : si32}> : (tensor<1x16x32x64xbf16>, tensor<1x64x32x16xbf16>) -> tensor<1x64x32x16xbf16>
    %2 = tensor.empty() : tensor<1x32x64x16xbf16>
    // CHECK: {{%[_a-z0-9]+}} = linalg.transpose ins(%transposed : tensor<1x64x32x16xbf16>) outs(%1 : tensor<1x32x64x16xbf16>) permutation = [0, 2, 1, 3]
    %3 = "ttir.transpose"(%1, %2) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x64x32x16xbf16>, tensor<1x32x64x16xbf16>) -> tensor<1x32x64x16xbf16>
    return %3 : tensor<1x32x64x16xbf16>
  }


}
