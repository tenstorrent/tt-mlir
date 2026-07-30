// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-materialize-view-returns -o %t %s
// RUN: FileCheck %s --input-file=%t
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline -o %t.ttmetal %s

module {

  // ---- dim=1 (last dim) ----

  // 2D sort along dim 1 on a 32x64 input (2 tiles wide, already a power of two).
  // The lowering must:
  //   1. Transpose tiles (dim=1 operates on rows, SortBlockOp on columns).
  //   2. Run sort_block with the padded reduction extent.
  //   3. Extract results with tile_transpose (untranspose).
  // CHECK-LABEL: func @sort_dim1_pow2
  func.func @sort_dim1_pow2(%arg0: tensor<32x64xf32>) -> (tensor<32x64xf32>, tensor<32x64xsi32>) {
    // The pre-transpose is a generic op wrapping tile_transpose.
    // CHECK: d2m.generic
    // CHECK: d2m.tile_transpose

    // The index buffer is built by arange + row broadcast.
    // CHECK: d2m.arange_block
    // CHECK: d2m.tile_bcast

    // The sort generic op contains sort_block.
    // CHECK: d2m.generic
    // CHECK: d2m.sort_block
    // CHECK-SAME: descending = false
    // CHECK-SAME: dim = 1
    // CHECK-SAME: num_elements = 64

    // Extract values and indices using tile_transpose to undo the
    // pre-transpose.
    // CHECK: d2m.generic
    // CHECK: d2m.tile_transpose
    // CHECK: d2m.generic
    // CHECK: d2m.tile_transpose
    %values, %indices = "ttir.sort"(%arg0) <{dim = -1 : si32, descending = false, stable = false}> : (tensor<32x64xf32>) -> (tensor<32x64xf32>, tensor<32x64xsi32>)
    return %values, %indices : tensor<32x64xf32>, tensor<32x64xsi32>
  }

  // Descending flips the mask fill so padding still sorts to the tail.
  // CHECK-LABEL: func @sort_dim1_descending
  func.func @sort_dim1_descending(%arg0: tensor<32x256xf32>) -> (tensor<32x256xf32>, tensor<32x256xsi32>) {
    // CHECK: d2m.sort_block
    // CHECK-SAME: descending = true
    // CHECK-SAME: num_elements = 256
    %values, %indices = "ttir.sort"(%arg0) <{dim = -1 : si32, descending = true, stable = false}> : (tensor<32x256xf32>) -> (tensor<32x256xf32>, tensor<32x256xsi32>)
    return %values, %indices : tensor<32x256xf32>, tensor<32x256xsi32>
  }

  // A non-power-of-two tile count (Wt=3) must be padded up to 4 tiles, so
  // num_elements becomes 128 while the logical result stays 96 wide.
  // CHECK-LABEL: func @sort_dim1_nonpow2
  func.func @sort_dim1_nonpow2(%arg0: tensor<32x96xf32>) -> (tensor<32x96xf32>, tensor<32x96xsi32>) {
    // The pad tail is masked so it sorts behind every real element.
    // CHECK: d2m.mask
    // CHECK: d2m.sort_block
    // CHECK-SAME: num_elements = 128
    %values, %indices = "ttir.sort"(%arg0) <{dim = -1 : si32, descending = false, stable = false}> : (tensor<32x96xf32>) -> (tensor<32x96xf32>, tensor<32x96xsi32>)
    return %values, %indices : tensor<32x96xf32>, tensor<32x96xsi32>
  }

  // A single-tile reduction dim is padded up to the two tiles that
  // topk_local_sort spans.
  // CHECK-LABEL: func @sort_dim1_single_tile
  func.func @sort_dim1_single_tile(%arg0: tensor<32x32xf32>) -> (tensor<32x32xf32>, tensor<32x32xsi32>) {
    // CHECK: d2m.mask
    // CHECK: d2m.sort_block
    // CHECK-SAME: num_elements = 64
    %values, %indices = "ttir.sort"(%arg0) <{dim = -1 : si32, descending = false, stable = false}> : (tensor<32x32xf32>) -> (tensor<32x32xf32>, tensor<32x32xsi32>)
    return %values, %indices : tensor<32x32xf32>, tensor<32x32xsi32>
  }

  // ---- dim=0 ----

  // dim=0 needs no value pre-transpose; the index buffer is built shape-swapped
  // and reoriented by a grid+tile transpose instead.
  // CHECK-LABEL: func @sort_dim0_pow2
  func.func @sort_dim0_pow2(%arg0: tensor<64x32xf32>) -> (tensor<64x32xf32>, tensor<64x32xsi32>) {
    // CHECK: d2m.arange_block
    // CHECK: d2m.generic
    // CHECK: d2m.sort_block
    // CHECK-SAME: dim = 0
    // CHECK-SAME: num_elements = 64
    %values, %indices = "ttir.sort"(%arg0) <{dim = 0 : si32, descending = false, stable = false}> : (tensor<64x32xf32>) -> (tensor<64x32xf32>, tensor<64x32xsi32>)
    return %values, %indices : tensor<64x32xf32>, tensor<64x32xsi32>
  }

  // CHECK-LABEL: func @sort_dim0_nonpow2
  func.func @sort_dim0_nonpow2(%arg0: tensor<96x32xf32>) -> (tensor<96x32xf32>, tensor<96x32xsi32>) {
    // CHECK: d2m.mask
    // CHECK: d2m.sort_block
    // CHECK-SAME: dim = 0
    // CHECK-SAME: num_elements = 128
    %values, %indices = "ttir.sort"(%arg0) <{dim = 0 : si32, descending = false, stable = false}> : (tensor<96x32xf32>) -> (tensor<96x32xf32>, tensor<96x32xsi32>)
    return %values, %indices : tensor<96x32xf32>, tensor<96x32xsi32>
  }

  // ---- data-parallel multicore ----

  // The non-target dim spans far more tiles than one core's budget, so it is
  // sliced across cores. Rows are independent, so there is no cross-core merge
  // and hence no composite view.
  // CHECK-LABEL: func @sort_dim1_data_parallel
  func.func @sort_dim1_data_parallel(%arg0: tensor<2048x128xf32>) -> (tensor<2048x128xf32>, tensor<2048x128xsi32>) {
    // CHECK-NOT: d2m.composite_view
    // CHECK: d2m.sort_block
    // CHECK-SAME: dim = 1
    // CHECK-SAME: num_elements = 128
    %values, %indices = "ttir.sort"(%arg0) <{dim = -1 : si32, descending = false, stable = false}> : (tensor<2048x128xf32>) -> (tensor<2048x128xf32>, tensor<2048x128xsi32>)
    return %values, %indices : tensor<2048x128xf32>, tensor<2048x128xsi32>
  }

  // CHECK-LABEL: func @sort_dim0_data_parallel
  func.func @sort_dim0_data_parallel(%arg0: tensor<128x2048xf32>) -> (tensor<128x2048xf32>, tensor<128x2048xsi32>) {
    // CHECK-NOT: d2m.composite_view
    // CHECK: d2m.sort_block
    // CHECK-SAME: dim = 0
    // CHECK-SAME: num_elements = 128
    %values, %indices = "ttir.sort"(%arg0) <{dim = 0 : si32, descending = false, stable = false}> : (tensor<128x2048xf32>) -> (tensor<128x2048xf32>, tensor<128x2048xsi32>)
    return %values, %indices : tensor<128x2048xf32>, tensor<128x2048xsi32>
  }
}
