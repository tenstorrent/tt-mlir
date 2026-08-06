// RUN: ttmlir-opt --const-eval-hoist-transform %s | FileCheck %s

// ttnn.zeros_buffer carries TTCore_NonCacheableTrait, so unlike the other
// standalone creation ops (see cacheable-creation-ops.mlir) it must never be
// hoisted into a const-eval subgraph. Hoisting it would put it behind
// ttcore.load_cached, which returns the same cached buffer on every call --
// exactly what a freshly allocated KV cache must not do.

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module {
  // A neighbouring ttnn.full in the same function *is* hoisted, proving the
  // trait is what does the work here rather than something incidental.

  // CHECK-LABEL: func.func private @not_hoisted_const_eval_0
  // CHECK: %[[DEVICE:.*]] = "ttnn.get_device"()
  // CHECK: %[[FULL:.*]] = "ttnn.full"(%[[DEVICE]])
  // CHECK-SAME: fill_value = 7.000000e+00 : f32
  // CHECK: return %[[FULL]]
  // CHECK-NOT: ttnn.zeros_buffer

  // CHECK-LABEL: func.func @not_hoisted(
  func.func @not_hoisted(%arg0: tensor<32x32xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<32x32xbf16, #ttnn_layout> {
    %device = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // The full op is hoisted...
    // CHECK: %[[CACHED:.*]] = ttcore.load_cached(@not_hoisted_const_eval_0, [])
    %full = "ttnn.full"(%device) <{fill_value = 7.000000e+00 : f32, layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #ttnn_layout>
    // ...but zeros_buffer stays right here in the forward function.
    // Note: the hoist pass clones ttnn.get_device into the const-eval function
    // (TTCore_DuplicateConstEvalTrait), so the device value here is a fresh one.
    // CHECK: %[[CACHE:.*]] = "ttnn.zeros_buffer"
    // CHECK-SAME: shape = #ttnn.shape<32x32>
    %cache = "ttnn.zeros_buffer"(%device) <{layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #ttnn_layout>
    // CHECK: %[[ADD:.*]] = "ttnn.add"(%arg0, %[[CACHED]])
    %add = "ttnn.add"(%arg0, %full) : (tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout>
    // CHECK: %[[RESULT:.*]] = "ttnn.add"(%[[ADD]], %[[CACHE]])
    %result = "ttnn.add"(%add, %cache) : (tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout>
    return %result : tensor<32x32xbf16, #ttnn_layout>
  }

  // Two identical zeros_buffer ops must both survive as separate ops and
  // neither may be hoisted.

  // CHECK-LABEL: func.func @two_caches(
  func.func @two_caches() -> (tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>) {
    %device = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK-NOT: ttcore.load_cached
    // CHECK: %[[K:.*]] = "ttnn.zeros_buffer"
    // CHECK: %[[V:.*]] = "ttnn.zeros_buffer"
    // CHECK: return %[[K]], %[[V]]
    %k = "ttnn.zeros_buffer"(%device) <{layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #ttnn_layout>
    %v = "ttnn.zeros_buffer"(%device) <{layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #ttnn_layout>
    return %k, %v : tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>
  }
}
