// RUN: ttmlir-opt --ttnn-deallocate -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verifies that the TTNN deallocate pass does NOT insert a deallocation
// for any value that `ttnn.tt_lang_op` consumes as an `"out"`-roled
// operand, and that this holds regardless of operand ordering inside
// `arg_roles`.
//
// The op publishes `MemoryEffects::Read` / `MemoryEffects::Write` on
// each `OpOperand &` via the `MemoryEffectOpInterface` (see
// `TtLangOp::getEffects` in lib/Dialect/TTNN/IR/TTNNOps.cpp). The
// dealloc pass queries that effect generically
// (`isWrittenByLastUser`) rather than special-casing the op by name:
// any op that publishes a `Write` effect on `value` is treated as
// aliasing it, so we leave the lifetime to the standard return-value
// path.
//
// The fixture uses `arg_roles = "in,out,in"` (interleaved order) on
// purpose: it pins the contract that operand ordering inside
// `arg_roles` is free-form and the role-to-OpOperand binding flows
// through the side-effect interface, not through a positional
// convention.

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#host = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32xf32, #system_memory>>
#dev = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

module {
  // CHECK-LABEL: func.func @tt_lang_kernel_interleaved
  // The kernel result is what the function returns; we capture its SSA
  // name and use it to scope the negative checks below.
  // CHECK: %[[RES:[0-9]+]] = ttnn.tt_lang_op(%[[A:[0-9]+]], %[[OUT:[0-9]+]], %[[B:[0-9]+]])
  // CHECK-SAME: arg_roles = "in,out,in"
  // Both "in"-roled operands are dead at the call site and get
  // deallocated (the dealloc pass places these right after the op,
  // since that is their last use).
  // CHECK-DAG: "ttnn.deallocate"(%[[A]])
  // CHECK-DAG: "ttnn.deallocate"(%[[B]])
  // The "out"-roled operand (in the middle of arg_roles!) aliases the
  // result that the function returns; the dealloc pass MUST NOT emit a
  // deallocate for it anywhere in the function.
  // CHECK-NOT: "ttnn.deallocate"(%[[OUT]])
  // CHECK: return %[[RES]]
  func.func @tt_lang_kernel_interleaved(
      %arg0: tensor<32x32xf32, #host>,
      %arg1: tensor<32x32xf32, #host>,
      %arg2: tensor<32x32xf32, #host>) -> tensor<32x32xf32, #dev> {
    %device = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %a = "ttnn.to_device"(%arg0, %device) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x32xf32, #host>, !ttnn.device) -> tensor<32x32xf32, #dev>
    %out = "ttnn.to_device"(%arg1, %device) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x32xf32, #host>, !ttnn.device) -> tensor<32x32xf32, #dev>
    %b = "ttnn.to_device"(%arg2, %device) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x32xf32, #host>, !ttnn.device) -> tensor<32x32xf32, #dev>
    // Interleaved arg_roles: in / out / in, NOT canonical [in, in, out].
    %0 = "ttnn.tt_lang_op"(%a, %out, %b) <{
      kernel_id = "test.interleaved::v1",
      version_tag = "1.0",
      arg_roles = "in,out,in",
      shard_spec = ""
    }> : (tensor<32x32xf32, #dev>, tensor<32x32xf32, #dev>, tensor<32x32xf32, #dev>) -> tensor<32x32xf32, #dev>
    return %0 : tensor<32x32xf32, #dev>
  }
}
