// REQUIRES: opmodel
// RUN: not ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 enable-d2m-fusing-pass=true" --mlir-print-local-scope %s 2>&1 | FileCheck %s
//
// Minimal reproducer for the gpt-oss D2M + optimizer issue.
//
// The add + multiply chain gets fused into a D2M subgraph. The 1D concat
// stays as a TTNN op in the main function. When the D2M pipeline runs
// TTIRRankNormalization, it expands 1D tensor types to 2D (prepending a
// size-1 dimension) on the surviving ttnn.concat without updating its dim
// attribute, causing the concat verifier to fail.

// THIS TEST FAILS WITHOUT THE CHANGE IN RANKNORMALIZATION.CPP BUT SUCCEEDS WITH THE CHANGE.

module {
  func.func @concat_with_d2m(
      %arg0: tensor<64x64xbf16>,
      %arg1: tensor<64x64xbf16>,
      %arg2: tensor<64x64xbf16>,
      %arg3: tensor<32xbf16>,
      %arg4: tensor<32xbf16>) -> (tensor<64x64xbf16>, tensor<64xbf16>) {
    %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %1 = "ttir.add"(%0, %arg2) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %2 = "ttir.multiply"(%1, %arg0) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %3 = "ttir.matmul"(%2, %arg0) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %4 = "ttir.concat"(%arg3, %arg4) <{dim = 0 : si32}> : (tensor<32xbf16>, tensor<32xbf16>) -> tensor<64xbf16>
    return %3, %4 : tensor<64x64xbf16>, tensor<64xbf16>
  }
}

// CHECK: 'ttnn.concat' op Output tensor dimension 0 must be equal to the sum of input tensor dimensions at the same axis (2), but got 1
