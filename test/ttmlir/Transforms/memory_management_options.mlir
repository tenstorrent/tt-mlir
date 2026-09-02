// Diagnostic options on --ttnn-memory-management. Both default to current
// behaviour; they exist so a non-converging graph can be characterised and
// bisected without a rebuild.

// Default: every aggressive pattern kind is registered, so PermuteRowMajorAdjusting fires.
// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-memory-management -o %t.default %s
// RUN: FileCheck %s --input-file=%t.default --check-prefix=DEFAULT

// aggressive-pattern-mask=11 clears 0x4, leaving PermuteRowMajorAdjusting out
// of the pattern set while the other three kinds stay enabled.
// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-memory-management="aggressive-pattern-mask=11" -o %t.masked %s
// RUN: FileCheck %s --input-file=%t.masked --check-prefix=MASKED

// max-iterations=1 lets the driver run a single sweep. That sweep rewrites the
// permute, so it ends with changes still pending and the pass reports
// non-convergence instead of failing mute.
// RUN: not ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-memory-management="max-iterations=1" -o %t.noconv %s 2>&1 | FileCheck %s --check-prefix=NOCONV

// NOCONV: ttnn-memory-management: pattern application did not converge within 1 iterations
// NOCONV-SAME: max-iterations

#dram = #ttnn.buffer_type<dram>
#layout_1x67M_tile = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2097152x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_67Mx1_tile = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2097152x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_8192x8192_tile = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<256x256x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

module {
  // DEFAULT-LABEL: func.func @permute_reshape_row_major_gated
  // DEFAULT: %[[RM_IN:.*]] = "ttnn.to_tensor_spec"(%arg0)
  // DEFAULT: %[[PERM:.*]] = "ttnn.permute"(%[[RM_IN]])
  // DEFAULT-SAME: -> tensor<67108864x1xf32
  // DEFAULT: "ttnn.reshape"(%[[PERM]]) <{shape = [8192 : i32, 8192 : i32]}>

  // MASKED-LABEL: func.func @permute_reshape_row_major_gated
  // MASKED: "ttnn.permute"(%arg0)
  // MASKED-NOT: "ttnn.to_tensor_spec"
  func.func @permute_reshape_row_major_gated(%arg0: tensor<1x67108864xf32, #layout_1x67M_tile>) -> tensor<8192x8192xf32, #layout_8192x8192_tile> {
    %0 = "ttnn.permute"(%arg0) <{permutation = array<i64: 1, 0>}> : (tensor<1x67108864xf32, #layout_1x67M_tile>) -> tensor<67108864x1xf32, #layout_67Mx1_tile>
    %1 = "ttnn.reshape"(%0) <{shape = [8192 : i32, 8192 : i32]}> : (tensor<67108864x1xf32, #layout_67Mx1_tile>) -> tensor<8192x8192xf32, #layout_8192x8192_tile>
    return %1 : tensor<8192x8192xf32, #layout_8192x8192_tile>
  }
}
