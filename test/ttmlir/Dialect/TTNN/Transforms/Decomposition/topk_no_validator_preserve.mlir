// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// Regression test for tt-mlir issue #8575.
// At opt_level=0 (the default), the TTNN decomposition pass has no OpModel
// validator available, so ttnn.topk must be preserved as-is. Decomposing it
// back to ttnn.sort + ttnn.slice_static would regenerate the sort with
// stable=false and a null memory_config, which differs from the original IR
// and breaks downstream paths such as trace capture.

module {
  func.func @test_topk_preserved_at_opt_level_0(%arg0: tensor<32x32768xf32>) -> (tensor<32x32xf32>, tensor<32x32xi64>) {
    // CHECK-LABEL: func.func @test_topk_preserved_at_opt_level_0
    // CHECK: "ttnn.topk"
    // CHECK-NOT: "ttnn.sort"
    %values, %indices = "ttir.topk"(%arg0) <{dim = -1 : i32, k = 32 : i32, largest = true, sorted = true}> : (tensor<32x32768xf32>) -> (tensor<32x32xf32>, tensor<32x32xi64>)
    return %values, %indices : tensor<32x32xf32>, tensor<32x32xi64>
  }
}
