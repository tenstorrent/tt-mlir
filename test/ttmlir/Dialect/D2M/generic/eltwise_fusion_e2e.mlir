// RUN: ttmlir-opt %s --split-input-file --ttir-to-ttmetal-pipeline="enable-affine-loop-fusion-and-scalar-replacement=false" | FileCheck %s

// Tests are mainly checking that the operation sequence over the main FUSED compute kernel
// are correct when lowered all the way down to emitc and that the compiler doesn't fail.

// Check for basic structure of the lowered IR
// CHECK: #l1 = #ttcore.memory_space<l1>
// CHECK: #system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}]
// CHECK: module {
// CHECK:   ttcore.device_module {
// CHECK:     builtin.module attributes {ttcore.system_desc = #system_desc} {
// CHECK:       ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8

// Check the main function signature has been transformed
// CHECK-LABEL: func.func @cosh(%arg0: memref<128x128xbf16>, %arg1: memref<128x128xbf16>) -> memref<128x128xbf16>

// Check that the function has been lowered to TTMetal operations
// CHECK: "ttmetal.create_buffer"()
// CHECK: "ttmetal.enqueue_write_buffer"
// CHECK: "ttmetal.enqueue_program"

// Check for the fused elementwise operations in compute_kernel7
// This kernel should contain all the fused operations
// CHECK: func.func private @compute_kernel7()
// CHECK-SAME: ttkernel.thread = #ttkernel.thread<compute>

// Verify the sequence of operations that should be fused together
// This represents: cosh(x) = 0.5 * (exp(x) + exp(-x)) * y
// First exp(x)
// CHECK: emitc.call_opaque "exp_tile"(%{{[0-9]+}}) : (!emitc.size_t) -> ()
// Then neg(x)
// CHECK: emitc.call_opaque "negative_tile"(%{{[0-9]+}}) : (!emitc.size_t) -> ()
// Then exp(-x)
// CHECK: emitc.call_opaque "exp_tile"(%{{[0-9]+}}) : (!emitc.size_t) -> ()
// Then div exp(x) / exp(-x)
// CHECK: emitc.call_opaque "div_binary_tile"(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}) : (!emitc.size_t, !emitc.size_t, !emitc.size_t) -> ()
// Finally pow by y (arg1)
// CHECK: emitc.call_opaque "power_binary_tile"(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}) : (!emitc.size_t, !emitc.size_t, !emitc.size_t) -> ()

module {
  func.func @cosh(%arg0: tensor<128x128xbf16>, %arg1: tensor<128x128xbf16>) -> tensor<128x128xbf16> {
    %1 = "ttir.neg"(%arg0) : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
    %3 = "ttir.exp"(%1) : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
    %5 = "ttir.exp"(%arg0) : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
    %7 = "ttir.div"(%5, %3) : (tensor<128x128xbf16>, tensor<128x128xbf16>) -> tensor<128x128xbf16>
    %9 = "ttir.pow"(%7, %arg1) : (tensor<128x128xbf16>, tensor<128x128xbf16>) -> tensor<128x128xbf16>
    return %9 : tensor<128x128xbf16>
  }
}
