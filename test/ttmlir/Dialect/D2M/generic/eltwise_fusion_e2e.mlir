// RUN: ttmlir-opt %s --split-input-file --ttir-to-ttmetal-pipeline | FileCheck %s

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

// Check for the fused elementwise operations in compute_kernel11
// This kernel should contain all the fused operations
// CHECK: func.func private @compute_kernel11()
// CHECK-SAME: ttkernel.thread = #ttkernel.thread<compute>

// Verify the sequence of operations that should be fused together
// This represents: cosh(x) = 0.5 * (exp(x) + exp(-x)) * y
// First exp(x)
// CHECK: emitc.call_opaque "exp_tile"(%{{[0-9]+}}) : (!emitc.size_t) -> ()
// Then neg(x)
// CHECK: emitc.call_opaque "negative_tile"(%{{[0-9]+}}) : (!emitc.size_t) -> ()
// Then exp(-x)
// CHECK: emitc.call_opaque "exp_tile"(%{{[0-9]+}}) : (!emitc.size_t) -> ()
// Then add exp(x) + exp(-x)
// CHECK: emitc.call_opaque "add_binary_tile"(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}) : (!emitc.size_t, !emitc.size_t, !emitc.size_t) -> ()
// Finally multiply by y (arg1)
// CHECK: emitc.call_opaque "mul_binary_tile"(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}) : (!emitc.size_t, !emitc.size_t, !emitc.size_t) -> ()

module {
  func.func @cosh(%arg0: tensor<128x128xbf16>, %arg1: tensor<128x128xbf16>) -> tensor<128x128xbf16> {
    %1 = "ttir.neg"(%arg0) : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
    %3 = "ttir.exp"(%1) : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
    %5 = "ttir.exp"(%arg0) : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
    %7 = "ttir.add"(%5, %3) : (tensor<128x128xbf16>, tensor<128x128xbf16>) -> tensor<128x128xbf16>
    %9 = "ttir.multiply"(%7, %arg1) : (tensor<128x128xbf16>, tensor<128x128xbf16>) -> tensor<128x128xbf16>
    return %9 : tensor<128x128xbf16>
  }
}

// -----

// CHECK-LABEL: func.func @digamma(%arg0: memref<128x128xf32>) -> memref<128x128xf32>

// Verify the large fused kernel (compute_kernel77) exists with all operations
// This kernel should contain the complex digamma computation
// CHECK: func.func private @compute_kernel77()
// CHECK-SAME: ttkernel.thread = #ttkernel.thread<compute>
// CHECK: emitc.call_opaque "sub_binary_tile"
// CHECK: emitc.call_opaque "add_binary_tile"
// CHECK: emitc.call_opaque "sub_binary_tile"
// CHECK: emitc.call_opaque "add_binary_tile"
// CHECK: emitc.call_opaque "sub_binary_tile"
// CHECK: emitc.call_opaque "add_binary_tile"
// CHECK: emitc.call_opaque "mul_binary_tile"
// CHECK: emitc.call_opaque "sub_binary_tile"
// CHECK: emitc.call_opaque "log_tile"
// CHECK: emitc.call_opaque "sub_binary_tile"

module {
  func.func @digamma(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = "ttir.constant"() <{value = dense<5.000000e-01> : tensor<128x128xf32>}> : () -> tensor<128x128xf32>
    %1 = "ttir.constant"() <{value = dense<0.0833333358> : tensor<128x128xf32>}> : () -> tensor<128x128xf32>
    %2 = "ttir.constant"() <{value = dense<0.00833333377> : tensor<128x128xf32>}> : () -> tensor<128x128xf32>
    %3 = "ttir.constant"() <{value = dense<0.0039682542> : tensor<128x128xf32>}> : () -> tensor<128x128xf32>
    %4 = "ttir.constant"() <{value = dense<0.00416666688> : tensor<128x128xf32>}> : () -> tensor<128x128xf32>
    %5 = "ttir.constant"() <{value = dense<0.0075757578> : tensor<128x128xf32>}> : () -> tensor<128x128xf32>
    %6 = "ttir.constant"() <{value = dense<0.0210927967> : tensor<128x128xf32>}> : () -> tensor<128x128xf32>
    %7 = "ttir.constant"() <{value = dense<0.0833333358> : tensor<128x128xf32>}> : () -> tensor<128x128xf32>
    %8 = "ttir.reciprocal"(%arg0) : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %9 = "ttir.multiply"(%8, %0) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %10 = "ttir.multiply"(%8, %8) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %11 = "ttir.multiply"(%10, %1) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %12 = "ttir.subtract"(%9, %11) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %13 = "ttir.multiply"(%10, %10) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %14 = "ttir.multiply"(%13, %2) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %15 = "ttir.add"(%12, %14) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %16 = "ttir.multiply"(%13, %10) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %17 = "ttir.multiply"(%16, %3) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %18 = "ttir.subtract"(%15, %17) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %19 = "ttir.multiply"(%16, %10) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %20 = "ttir.multiply"(%19, %4) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %21 = "ttir.add"(%18, %20) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %22 = "ttir.multiply"(%19, %10) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %23 = "ttir.multiply"(%22, %5) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %24 = "ttir.subtract"(%21, %23) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %25 = "ttir.multiply"(%22, %10) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %26 = "ttir.multiply"(%25, %6) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %27 = "ttir.add"(%24, %26) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %28 = "ttir.multiply"(%25, %10) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %29 = "ttir.multiply"(%28, %7) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %30 = "ttir.subtract"(%27, %29) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %31 = "ttir.log"(%arg0) : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %32 = "ttir.subtract"(%31, %30) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    return %32 : tensor<128x128xf32>
  }
}
