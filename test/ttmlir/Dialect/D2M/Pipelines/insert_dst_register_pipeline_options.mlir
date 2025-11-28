// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="system-desc-path=%system_desc_path% dst-allocation-strategy=legacy" %s | FileCheck %s
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="system-desc-path=%system_desc_path% dst-allocation-strategy=graph-coloring-greedy" %s | FileCheck %s
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="system-desc-path=%system_desc_path% dst-allocation-strategy=graph-coloring-cb" %s | FileCheck %s

// Test: Verify that the pipeline accepts all three DST allocation strategy options
// and successfully lowers to TTMetal IR. All strategies should produce valid TTMetal output.

// CHECK-LABEL: func.func @forward
// CHECK: ttmetal.create_buffer
// CHECK: ttmetal.enqueue_program

func.func @forward(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}
