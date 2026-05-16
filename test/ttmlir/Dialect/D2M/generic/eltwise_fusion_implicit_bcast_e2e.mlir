// RUN: ttmlir-opt %s --split-input-file --ttir-to-ttmetal-pipeline="enable-elementwise-fusion=true" | FileCheck %s

// Lower the eltwise + implicit-bcast chain to TTMetal/EmitC and verify that
// the broadcast prologue (`unary_bcast`) and the consumer add (`add_binary_tile`)
// are emitted from the SAME compute kernel — i.e. the fusion didn't split
// the chain at the bcast boundary.

// CHECK-LABEL: func.func @bcast_all_cases_type1
// CHECK:       emitc.call_opaque "unary_bcast"
// CHECK-NOT:   func.func
// CHECK:       emitc.call_opaque "add_binary_tile"
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
// CHECK:       emitc.call_opaque "unary_bcast"
// CHECK-NOT:   func.func
// CHECK:       emitc.call_opaque "add_binary_tile"
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
