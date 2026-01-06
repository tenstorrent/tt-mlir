// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

#loc = loc("test":0:0)

module attributes {} {
  func.func @forward(%arg0: tensor<512x1024xbf16>, %arg1: tensor<1024x2048xbf16>, %arg2: tensor<512x2048xbf16>) -> tensor<512x2048xbf16> {
    // CHECK: "ttnn.matmul"
    // CHECK-SAME: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast
    // CHECK-SAME: compute_with_storage_grid_size
    // CHECK-SAME: in0_block_w
    // CHECK-SAME: out_subblock_h
    // CHECK-SAME: out_subblock_w
    // CHECK-SAME: per_core_m
    // CHECK-SAME: per_core_n
    %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<512x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<512x2048xbf16> loc(#loc1)

    // Consumer op to form an L1 chain for sharding
    %1 = "ttir.multiply"(%0, %arg2) : (tensor<512x2048xbf16>, tensor<512x2048xbf16>) -> tensor<512x2048xbf16>
    return %1 : tensor<512x2048xbf16>
  }
}

#loc1 = loc("matmul_out"(#loc))
