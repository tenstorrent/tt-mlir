// REQUIRES: opmodel
// XFAIL: *
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=1 enable-greedy-optimizer=false" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// RUN: ttrt run %t.ttnn

// Silicon smoke for the Phase 2/5 vertical slice: lower a single Conv3dOp
// through the optimizer-enabled pipeline, attach a Conv3dConfig (Phase 5
// scoring), and execute on n150 hardware.
//
// Currently XFAIL: tt-metal's conv3d kernel rejects the optimizer-emitted IR
// with "Layout mismatch, expected TILE, got ROW_MAJOR" at runtime, even
// when the chosen config matches the tt-metal default (c_in_block=32, all
// other blocks=1). The MLIR IR appears valid — weight tensor is in TILE
// layout (memref<108x1x!ttcore.tile<32x32, bf16>, #dram>) and input is
// row-major as required by Conv3dOp's operand workaround. The no-opt path
// runs successfully on the same shape, so the issue is in how
// optimizer-attached attributes interact with tt-metal's Conv3d execution
// path. Needs joint debugging with tt-metal. Remove XFAIL once resolved.
module {
  func.func @conv3d_silicon_smoke(
      %arg0: tensor<1x8x28x28x128xbf16>,
      %arg1: tensor<32x128x3x3x3xbf16>)
      -> tensor<1x6x26x26x32xbf16> {
    %0 = "ttir.conv3d"(%arg0, %arg1) <{
        stride = array<i32: 1, 1, 1>,
        padding = array<i32: 0, 0, 0>,
        groups = 1 : i32,
        padding_mode = "zeros"
      }> : (tensor<1x8x28x28x128xbf16>, tensor<32x128x3x3x3xbf16>)
        -> tensor<1x6x26x26x32xbf16>
    return %0 : tensor<1x6x26x26x32xbf16>
  }
}
