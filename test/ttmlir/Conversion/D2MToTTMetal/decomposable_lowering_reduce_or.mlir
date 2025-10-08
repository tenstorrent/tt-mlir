// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// UNSUPPORTED: true
// Full lowering requires:
//   Sum: https://github.com/tenstorrent/tt-mlir/issues/3018

module {
  func.func @test_reduce_or_2d(%arg0: tensor<32x32xbf16>) -> tensor<32x1xbf16> {
    %0 = ttir.empty() : tensor<32x1xbf16>
    // CHECK-NOT: ttir.reduce_or
    // CHECK: add_tiles_init
    // CHECK: add_tiles
    %1 = "ttir.reduce_or"(%arg0, %0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<32x32xbf16>, tensor<32x1xbf16>) -> tensor<32x1xbf16>
    return %1 : tensor<32x1xbf16>
  }

  func.func @test_reduce_or_3d(%arg0: tensor<32x32x32xbf16>) -> tensor<32x32x1xbf16> {
    %0 = ttir.empty() : tensor<32x32x1xbf16>
    // CHECK-NOT: ttir.reduce_or
    // CHECK: add_tiles_init
    // CHECK: add_tiles
    %1 = "ttir.reduce_or"(%arg0, %0) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<32x32x32xbf16>, tensor<32x32x1xbf16>) -> tensor<32x32x1xbf16>
    return %1 : tensor<32x32x1xbf16>
  }
}
