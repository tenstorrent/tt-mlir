// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 enable-d2m-fusing-pass=true" --mlir-print-local-scope -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// Regression test for the D2M + optimizer rank normalization issue (#7735).
//
// The add + multiply chain gets fused into a D2M subgraph. The 1D concat
// stays as a TTNN op in the main function. TTIRRankNormalization must NOT
// touch the main function (which has only TTNN ops after D2M fusing), so
// the 1D concat survives without its dim attribute being invalidated.

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

// CHECK: func.func @concat_with_d2m

// D2M subgraph compiled into generic ops (add + multiply).
// CHECK: "ttnn.generic"

// Concat must survive unchanged.
// CHECK: "ttnn.concat"

// No stray casts or leftover D2M ops.
// CHECK-NOT: unrealized_conversion_cast
// CHECK-NOT: ttnn.d2m_subgraph
