// RUN: ttmlir-opt %s --ttir-to-ttmetal-pipeline="dst-allocation-strategy=legacy" | FileCheck %s --check-prefixes=COMMON,LEGACY
// RUN: ttmlir-opt %s --ttir-to-ttmetal-pipeline="dst-allocation-strategy=graph-coloring-greedy" | FileCheck %s --check-prefixes=COMMON,GREEDY
// RUN: ttmlir-opt %s --ttir-to-ttmetal-pipeline="dst-allocation-strategy=graph-coloring-cb" | FileCheck %s --check-prefixes=COMMON,CB

// Check for basic structure of the lowered IR
// COMMON: #l1 = #ttcore.memory_space<l1>
// COMMON: #system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}]
// COMMON: module {
// COMMON:   ttcore.device_module {
// COMMON:     builtin.module attributes {ttcore.system_desc = #system_desc} {
// COMMON:       ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8

// Check the main function signature has been transformed
// COMMON-LABEL: func.func @cosh(%arg0: memref<128x128xbf16>, %arg1: memref<128x128xbf16>) -> memref<128x128xbf16>

// Check that the function has been lowered to TTMetal operations
// COMMON: "ttmetal.create_buffer"()
// COMMON: "ttmetal.enqueue_write_buffer"
// COMMON: "ttmetal.enqueue_program"

// Check for the fused elementwise operations in compute_kernel11
// This kernel should contain all the fused operations
// COMMON: func.func private @compute_kernel11()
// COMMON-SAME: ttkernel.thread = #ttkernel.thread<compute>

// Verify the sequence of operations that should be fused together
// This represents: cosh(x) = 0.5 * (exp(x) + exp(-x)) * y
// First neg(x)
// COMMON: emitc.call_opaque "negative_tile"(%{{[0-9]+}}) : (!emitc.size_t) -> ()
// Then exp(-x)
// COMMON: emitc.call_opaque "exp_tile"(%{{[0-9]+}}) : (!emitc.size_t) -> ()
// Then exp(x)
// COMMON: emitc.call_opaque "exp_tile"(%{{[0-9]+}}) : (!emitc.size_t) -> ()
// Then add exp(x) + exp(-x)
// COMMON: emitc.call_opaque "add_binary_tile"(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}) : (!emitc.size_t, !emitc.size_t, !emitc.size_t) -> ()
// Finally multiply by y (arg1)
// COMMON: emitc.call_opaque "mul_binary_tile"(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}) : (!emitc.size_t, !emitc.size_t, !emitc.size_t) -> ()

module {
  func.func @cosh(%arg0: tensor<128x128xbf16>, %arg1: tensor<128x128xbf16>) -> tensor<128x128xbf16> {
    %0 = ttir.empty() : tensor<128x128xbf16>
    %1 = "ttir.neg"(%arg0, %0) : (tensor<128x128xbf16>, tensor<128x128xbf16>) -> tensor<128x128xbf16>
    %2 = ttir.empty() : tensor<128x128xbf16>
    %3 = "ttir.exp"(%1, %2) : (tensor<128x128xbf16>, tensor<128x128xbf16>) -> tensor<128x128xbf16>
    %4 = ttir.empty() : tensor<128x128xbf16>
    %5 = "ttir.exp"(%arg0, %4) : (tensor<128x128xbf16>, tensor<128x128xbf16>) -> tensor<128x128xbf16>
    %6 = ttir.empty() : tensor<128x128xbf16>
    %7 = "ttir.add"(%5, %3, %6) : (tensor<128x128xbf16>, tensor<128x128xbf16>, tensor<128x128xbf16>) -> tensor<128x128xbf16>
    %8 = ttir.empty() : tensor<128x128xbf16>
    %9 = "ttir.multiply"(%7, %arg1, %8) : (tensor<128x128xbf16>, tensor<128x128xbf16>, tensor<128x128xbf16>) -> tensor<128x128xbf16>
    return %9 : tensor<128x128xbf16>
  }
}
