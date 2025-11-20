// RUN: ttmlir-opt --split-input-file %s | FileCheck %s
//
// Test that d2m.generic with pure tensor semantics can have multiple regions
// (e.g., datamovement and compute threads).

#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// CHECK-LABEL: func.func @pure_tensor_multiple_regions
func.func @pure_tensor_multiple_regions(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = d2m.empty() : tensor<64x128xf32>
  // CHECK: d2m.generic
  // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>]
  // CHECK: ^datamovement0
  // CHECK: ^datamovement1
  // CHECK: ^compute0
  %1 = d2m.generic {
    block_factors = [],
    grid = #ttcore.grid<1x1>,
    indexing_maps = [],
    iterator_types = [],
    threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>]
  }
  ins(%arg0, %arg1 : tensor<64x128xf32>, tensor<64x128xf32>)
  outs(%0 : tensor<64x128xf32>) {
  ^datamovement0(%cb_in0: !d2m.cb<tensor<64x128xf32>>, %cb_in1: !d2m.cb<tensor<64x128xf32>>, %cb_out: !d2m.cb<tensor<64x128xf32>>, %sem0: !d2m.semaphore, %sem1: !d2m.semaphore, %sem2: !d2m.semaphore, %sem3: !d2m.semaphore):
    // Simple datamovement: wait on input, reserve output, copy data
    %in0 = d2m.wait %cb_in0 : !d2m.cb<tensor<64x128xf32>> -> tensor<64x128xf32>
    %out = d2m.reserve %cb_out : !d2m.cb<tensor<64x128xf32>> -> tensor<64x128xf32>
    // Simple copy operation
    %c0 = arith.constant 0 : index
    %val = tensor.extract %in0[%c0, %c0] : tensor<64x128xf32>
    %result = tensor.insert %val into %out[%c0, %c0] : tensor<64x128xf32>
    d2m.yield %result : (tensor<64x128xf32>)
  }, {
  ^datamovement1(%cb_in0_2: !d2m.cb<tensor<64x128xf32>>, %cb_in1_2: !d2m.cb<tensor<64x128xf32>>, %cb_out_2: !d2m.cb<tensor<64x128xf32>>, %sem0_2: !d2m.semaphore, %sem1_2: !d2m.semaphore, %sem2_2: !d2m.semaphore, %sem3_2: !d2m.semaphore):
    // Another datamovement region: wait on second input
    %in1 = d2m.wait %cb_in1_2 : !d2m.cb<tensor<64x128xf32>> -> tensor<64x128xf32>
    %out_2 = d2m.reserve %cb_out_2 : !d2m.cb<tensor<64x128xf32>> -> tensor<64x128xf32>
    // Simple copy operation
    %c0_2 = arith.constant 0 : index
    %val_2 = tensor.extract %in1[%c0_2, %c0_2] : tensor<64x128xf32>
    %result_2 = tensor.insert %val_2 into %out_2[%c0_2, %c0_2] : tensor<64x128xf32>
    d2m.yield %result_2 : (tensor<64x128xf32>)
  }, {
  ^compute0(%cb_in0_3: !d2m.cb<tensor<64x128xf32>>, %cb_in1_3: !d2m.cb<tensor<64x128xf32>>, %cb_out_3: !d2m.cb<tensor<64x128xf32>>, %sem0_3: !d2m.semaphore, %sem1_3: !d2m.semaphore, %sem2_3: !d2m.semaphore, %sem3_3: !d2m.semaphore):
    // Compute region: add the two inputs
    %in0_3 = d2m.wait %cb_in0_3 : !d2m.cb<tensor<64x128xf32>> -> tensor<64x128xf32>
    %in1_3 = d2m.wait %cb_in1_3 : !d2m.cb<tensor<64x128xf32>> -> tensor<64x128xf32>
    %out_3 = d2m.reserve %cb_out_3 : !d2m.cb<tensor<64x128xf32>> -> tensor<64x128xf32>
    %c0_3 = arith.constant 0 : index
    %val0 = tensor.extract %in0_3[%c0_3, %c0_3] : tensor<64x128xf32>
    %val1 = tensor.extract %in1_3[%c0_3, %c0_3] : tensor<64x128xf32>
    %sum = arith.addf %val0, %val1 : f32
    %result_3 = tensor.insert %sum into %out_3[%c0_3, %c0_3] : tensor<64x128xf32>
    d2m.yield %result_3 : (tensor<64x128xf32>)
  } : tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}
