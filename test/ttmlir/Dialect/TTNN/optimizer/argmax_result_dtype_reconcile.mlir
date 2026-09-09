// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=1" -o %t1 %s --mlir-print-local-scope
// RUN: FileCheck %s --input-file=%t1
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2" -o %t2 %s --mlir-print-local-scope
// RUN: FileCheck %s --input-file=%t2

// tt-metal's argmax really returns uint32 where the frontend declares si32, and
// at optimization-level >= 1 no operand workaround declares that output
// contract (ArgMaxRuleBook owns the input only). The greedy optimizer adopts
// the op-model's real ui32 result, so it must also reconcile it back to the
// declared si32 -- otherwise the ui32 escapes into the program's signature and
// the runtime, handed that tensor as the next program's input, is asked for a
// device-resident row-major pure-dtype cast it cannot perform. See #9115.
// Workarounds stay enabled here: this is the shipping configuration.
//
// The reconciling cast is a direct ttnn.typecast rather than a ttnn.to_layout:
// only the data type needs reconciling, and unlike a to_layout - which
// OperationValidationAndFallback skips and TTNNDecomposeLayouts has to expand
// later - a typecast is validated against the op-model right where it is
// inserted.

module {
  func.func @add_argmax(%a: tensor<64x128xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}, %b: tensor<64x128xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<64xi32> {
    // The declared si32 result element type survives into the signature.
    // CHECK-LABEL: func.func @add_argmax
    // CHECK-SAME: -> tensor<64xsi32
    // argmax keeps the op-model's real ui32 result...
    // CHECK: "ttnn.argmax"
    // CHECK-SAME: -> tensor<64xui32
    // ...and a direct typecast reconciles it back to si32 for its consumers.
    // The CHECK-NOT pins the cast to be a typecast: no to_layout stands
    // between argmax and it.
    // CHECK-NOT: ttnn.to_layout
    // CHECK: "ttnn.typecast"
    // CHECK-SAME: tensor<64xui32
    // CHECK-SAME: -> tensor<64xsi32
    // CHECK: return
    // CHECK-SAME: tensor<64xsi32
    %0 = "ttir.add"(%a, %b) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    %1 = "ttir.argmax"(%0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<64x128xbf16>) -> tensor<64xi32>
    return %1 : tensor<64xi32>
  }
}
