// RUN: ttmlir-opt --ttir-spatial-row-group-packing-opt -o %t %s
// RUN: FileCheck %s --input-file=%t \
// RUN:   --implicit-check-not="arange_dimension = 2" \
// RUN:   --implicit-check-not="arange_dimension = 3" \
// RUN:   --implicit-check-not="xi64>" \
// RUN:   --implicit-check-not="xi1>"

// ElementTypeNormalization runs BEFORE createTTNNPipelineTTIRPasses, so types
// this pass introduces are never normalized. Emitting an unsupported type here
// (i64, or i1 from a comparison) leaves the tensor element type raw while
// TTNNLayout assigns a normalized scalar type, tripping the
// ScalarDataTypeAnalysis layout/element-type consistency assertion. Assert the
// pass output is already normalized, i.e. a fixed point of normalization:
//
// RUN: ttmlir-opt --ttir-spatial-row-group-packing-opt \
// RUN:   --ttir-element-type-normalization -o %t.norm %s
// RUN: diff %t %t.norm

// TTIRSpatialRowGroupPackingOpt builds a K*K diagonal identity mask from an
// arange/arange/eq triple. ArangeOpConversionPattern (TTIRToTTNN) only lowers
// ttir.arange to ttnn.arange when arange_dimension is the final dimension, so
// the pass must emit that canonical form: a rank-1 arange (dim 0 == rank-1)
// reshaped and broadcast into the row/col index grids.
//
// A rank-4 arange on dim 2 is illegal here. It previously escaped detection in
// the flatbuffer pipeline, where the const-eval'd weight subgraph is hoisted to
// the CPU module and never reaches TTIRToTTNN, but aborted the EmitC pipeline,
// which keeps const-eval subgraphs on device (enableCPUHoistedConstEval=false,
// issue #6100).

module {
  // IC=3 is coprime to TILE_WIDTH=32, so packingFactor(3) == 32 == K, and
  // H=64 is divisible by K. 1x1 pointwise conv sandwiched between the
  // NCHW->NHWC / NHWC->NCHW permute pair the pass matches on.
  func.func @yuv_adapter_pack(%act: tensor<1x3x64x64xbf16>, %weight: tensor<16x3x1x1xbf16>) -> tensor<1x16x64x64xbf16> {
    %nhwc = "ttir.permute"(%act) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x3x64x64xbf16>) -> tensor<1x64x64x3xbf16>
    %conv = "ttir.conv2d"(%nhwc, %weight) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> : (tensor<1x64x64x3xbf16>, tensor<16x3x1x1xbf16>) -> tensor<1x64x64x16xbf16>
    %nchw = "ttir.permute"(%conv) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<1x64x64x16xbf16>) -> tensor<1x16x64x64xbf16>
    return %nchw : tensor<1x16x64x64xbf16>
  }
}

// The pass must fire (proving the negative checks above are meaningful rather
// than vacuously true on an unmodified input) and emit only a rank-1 arange.
// CHECK-LABEL: func.func @yuv_adapter_pack
// Canonical last-dim arange, in the already-normalized si32 index type.
// CHECK: "ttir.arange"
// CHECK-SAME: arange_dimension = 0
// CHECK-SAME: -> tensor<32xsi32>
// The diagonal mask is produced directly in bf16 (no i1, so no bool typecast).
// CHECK: "ttir.eq"
// CHECK-SAME: -> tensor<1x1x32x32xbf16>
// CHECK: "ttir.linear"
