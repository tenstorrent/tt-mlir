// RUN: ttmlir-opt %s --split-input-file --ttir-to-ttmetal-pipeline="enable-elementwise-fusion=true" | FileCheck %s

// Regression tests for tenstorrent/tt-mlir#5968. Both modules below were the
// historical failure scenarios that motivated tagging `D2M_TileBcastOp` with
// `D2M_SkipOpEltwiseFusionTrait`. The trait has since been removed; these
// tests pin the relaxed behaviour by lowering each reproducer all the way to
// emitc with `enable-elementwise-fusion=true`.
//
// The checks intentionally verify (a) that compilation completes, (b) that no
// `unrealized_conversion_cast` survives the D2MToTTKernel pass (Type 2), and
// (c) that the fused compute kernel actually emits a `tile_bcast`-style
// `unary_bcast` together with an `add_binary_tile`/`binary_dest_reuse_tiles`
// add, ruling out the DST-overflow scenario (Type 1).

// CHECK-LABEL: func.func @bcast_all_cases_type1
// CHECK-NOT:   unrealized_conversion_cast
// CHECK:       emitc.call_opaque "unary_bcast"
// CHECK:       emitc.call_opaque "add_binary_tile"
// CHECK-NOT:   unrealized_conversion_cast
module {
  func.func @bcast_all_cases_type1(%arg1: tensor<32x1xbf16>, %arg2: tensor<1x32xbf16>, %arg3: tensor<1x1xbf16>) -> tensor<32x32xbf16> {
    %1 = "ttir.add"(%arg3, %arg2) : (tensor<1x1xbf16>, tensor<1x32xbf16>) -> tensor<1x32xbf16>
    %3 = "ttir.add"(%arg1, %arg3) : (tensor<32x1xbf16>, tensor<1x1xbf16>) -> tensor<32x1xbf16>
    %5 = "ttir.add"(%3, %1) : (tensor<32x1xbf16>, tensor<1x32xbf16>) -> tensor<32x32xbf16>
    return %5 : tensor<32x32xbf16>
  }
}

// -----

// CHECK-LABEL: func.func @bcast_all_cases_type2
// CHECK-NOT:   unrealized_conversion_cast
// CHECK:       emitc.call_opaque "unary_bcast"
// CHECK:       emitc.call_opaque "add_binary_tile"
// CHECK-NOT:   unrealized_conversion_cast
module {
  func.func @bcast_all_cases_type2(%arg1: tensor<32x1xbf16>, %arg2: tensor<1x32xbf16>, %arg3: tensor<1x1xbf16>) -> tensor<32x32xbf16> {
    %1 = "ttir.add"(%arg3, %arg2) : (tensor<1x1xbf16>, tensor<1x32xbf16>) -> tensor<1x32xbf16>
    %3 = "ttir.add"(%1, %arg3) : (tensor<1x32xbf16>, tensor<1x1xbf16>) -> tensor<1x32xbf16>
    %5 = "ttir.add"(%arg1, %arg3) : (tensor<32x1xbf16>, tensor<1x1xbf16>) -> tensor<32x1xbf16>
    %7 = "ttir.add"(%arg3, %5) : (tensor<1x1xbf16>, tensor<32x1xbf16>) -> tensor<32x1xbf16>
    %9 = "ttir.add"(%7, %3) : (tensor<32x1xbf16>, tensor<1x32xbf16>) -> tensor<32x32xbf16>
    return %9 : tensor<32x32xbf16>
  }
}
