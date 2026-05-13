// RUN: ttmlir-opt --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

module attributes {} {
  // Test for folding slice of concat when taking the entire first input
  func.func @slice_concat_first_input(%t1: tensor<8x128x1x1x4xbf16>,
                                      %t2: tensor<8x128x1x1x4xbf16>,
                                      %t3: tensor<8x128x1x1x4xbf16>,
                                      %t4: tensor<8x128x1x1x4xbf16>) -> tensor<8x128x1x1x4xbf16> {
    // CHECK-NOT: ttir.concat
    // CHECK-NOT: ttir.slice_static
    // CHECK: return %arg0 : tensor<8x128x1x1x4xbf16>
    %concat = "ttir.concat"(%t1, %t2, %t3, %t4) <{dim = 3 : si32}> : (tensor<8x128x1x1x4xbf16>, tensor<8x128x1x1x4xbf16>, tensor<8x128x1x1x4xbf16>, tensor<8x128x1x1x4xbf16>) -> tensor<8x128x1x4x4xbf16>
    %slice = "ttir.slice_static"(%concat) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [8 : i32, 128 : i32, 1 : i32, 1 : i32, 4 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<8x128x1x4x4xbf16>) -> tensor<8x128x1x1x4xbf16>
    return %slice : tensor<8x128x1x1x4xbf16>
  }

  // Test for folding slice of concat when taking the entire second input
  func.func @slice_concat_second_input(%t1: tensor<8x128x1x2x4xbf16>,
                                       %t2: tensor<8x128x1x2x4xbf16>,
                                       %t3: tensor<8x128x1x2x4xbf16>) -> tensor<8x128x1x2x4xbf16> {
    // CHECK-NOT: ttir.concat
    // CHECK-NOT: ttir.slice_static
    // CHECK: return %arg1 : tensor<8x128x1x2x4xbf16>
    %concat = "ttir.concat"(%t1, %t2, %t3) <{dim = 3 : si32}> : (tensor<8x128x1x2x4xbf16>, tensor<8x128x1x2x4xbf16>, tensor<8x128x1x2x4xbf16>) -> tensor<8x128x1x6x4xbf16>
    %slice = "ttir.slice_static"(%concat) <{begins = [0 : i32, 0 : i32, 0 : i32, 2 : i32, 0 : i32], ends = [8 : i32, 128 : i32, 1 : i32, 4 : i32, 4 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<8x128x1x6x4xbf16>) -> tensor<8x128x1x2x4xbf16>
    return %slice : tensor<8x128x1x2x4xbf16>
  }

  // Test for negative case: slice that doesn't match an entire input should NOT fold
  func.func @slice_concat_partial_input(%t1: tensor<8x128x1x2x4xbf16>,
                                        %t2: tensor<8x128x1x2x4xbf16>) -> tensor<8x128x1x1x4xbf16> {
    // CHECK: ttir.concat
    // CHECK: ttir.slice_static
    %concat = "ttir.concat"(%t1, %t2) <{dim = 3 : si32}> : (tensor<8x128x1x2x4xbf16>, tensor<8x128x1x2x4xbf16>) -> tensor<8x128x1x4x4xbf16>
    %slice = "ttir.slice_static"(%concat) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [8 : i32, 128 : i32, 1 : i32, 1 : i32, 4 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<8x128x1x4x4xbf16>) -> tensor<8x128x1x1x4xbf16>
    return %slice : tensor<8x128x1x1x4xbf16>
  }

  // Test for folding slice of concat when taking the entire first input but with negative concat dim
  func.func @slice_concat_first_input_neg_dim(%t1: tensor<8x128x1x1x4xbf16>,
                                      %t2: tensor<8x128x1x1x4xbf16>,
                                      %t3: tensor<8x128x1x1x4xbf16>,
                                      %t4: tensor<8x128x1x1x4xbf16>) -> tensor<8x128x1x1x4xbf16> {
    // CHECK-NOT: ttir.concat
    // CHECK-NOT: ttir.slice_static
    // CHECK: return %arg0 : tensor<8x128x1x1x4xbf16>
    %concat = "ttir.concat"(%t1, %t2, %t3, %t4) <{dim = -2 : si32}> : (tensor<8x128x1x1x4xbf16>, tensor<8x128x1x1x4xbf16>, tensor<8x128x1x1x4xbf16>, tensor<8x128x1x1x4xbf16>) -> tensor<8x128x1x4x4xbf16>
    %slice = "ttir.slice_static"(%concat) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [8 : i32, 128 : i32, 1 : i32, 1 : i32, 4 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<8x128x1x4x4xbf16>) -> tensor<8x128x1x1x4xbf16>
    return %slice : tensor<8x128x1x1x4xbf16>
  }
}
