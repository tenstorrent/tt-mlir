// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true optimization-level=2" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

// This test verifies that the optimizer rejects 3-way merges that would cause
// L1 fragmentation. The GeGLU pattern has two large matmul outputs (278KB each)
// that feed into a multiply. If 3-way merged, the multiply output gets squeezed
// to a low address and may overlap with the CB region of the subsequent matmul.
//
// Pattern:
//   %rms_norm
//       │
//       ├──► %matmul_gelu (544x16384, 278KB/core) ──┐
//       │                                           │ (3-way merge attempt)
//       └──► %matmul_up (544x16384, 278KB/core) ────┼──► %multiply
//                                                   │
//                                                   └──► %matmul_down (high CB)
//
// The fragmentation check should reject this 3-way merge because:
// 1. Both matmul outputs = 278KB + 278KB = 556KB would push multiply output low
// 2. The subsequent matmul_down has high CB requirements (~693KB)
// 3. The multiply output at low address would overlap with CB region
//
// After rejection, at least one matmul output should go to DRAM (not L1 sharded).

module attributes {} {
  func.func @threeway_merge_fragmentation(
    %input: tensor<544x2048xbf16>,
    %gate_weight: tensor<16384x2048xbf16>,
    %up_weight: tensor<16384x2048xbf16>,
    %down_weight: tensor<2048x16384xbf16>
  ) -> tensor<544x2048xbf16> {

    // Two parallel matmuls - would be 3-way merged into multiply
    // Each output is 544x16384 = 278,528 bytes/core when L1 width-sharded
    %matmul_gelu = "ttir.matmul"(%input, %gate_weight) <{transpose_a = false, transpose_b = true}>
      : (tensor<544x2048xbf16>, tensor<16384x2048xbf16>)
      -> tensor<544x16384xbf16>

    %matmul_up = "ttir.matmul"(%input, %up_weight) <{transpose_a = false, transpose_b = true}>
      : (tensor<544x2048xbf16>, tensor<16384x2048xbf16>)
      -> tensor<544x16384xbf16>

    // Multiply - join point for potential 3-way merge
    %mul = "ttir.multiply"(%matmul_gelu, %matmul_up)
      : (tensor<544x16384xbf16>, tensor<544x16384xbf16>)
      -> tensor<544x16384xbf16>

    // Down projection - has high CB requirements that would overlap with
    // multiply output if it were at a low address
    %matmul_down = "ttir.matmul"(%mul, %down_weight) <{transpose_a = false, transpose_b = true}>
      : (tensor<544x16384xbf16>, tensor<2048x16384xbf16>)
      -> tensor<544x2048xbf16>

    return %matmul_down : tensor<544x2048xbf16>
  }
}

// Verify the 3-way merge was rejected by checking that the first matmul output
// is spilled to DRAM before the multiply. This prevents the L1 fragmentation
// that would cause OOM at the down-projection matmul.
//
// Expected pattern:
//   %0 = ttnn.matmul (first matmul, output in L1)
//   %1 = ttnn.to_memory_config(%0) -> DRAM (spill!)
//   %2 = ttnn.matmul (second matmul, output in L1)
//   %3 = ttnn.multiply(%1, %2) (one input from DRAM, one from L1)
//
// CHECK: ttnn.matmul
// CHECK: ttnn.to_memory_config
// CHECK-SAME: #ttnn.memory_config<#dram
// CHECK: ttnn.matmul
// CHECK: ttnn.multiply
