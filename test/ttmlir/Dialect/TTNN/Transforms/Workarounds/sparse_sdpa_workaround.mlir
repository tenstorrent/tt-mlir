// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --ttnn-workaround --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// Also run at optimization-level 1 (the level used by the optimizer pipeline
// and by composite-promotion OpModel validation). At this level the operand
// workaround only fires for ops in the allow-list, so this run guards that
// sparse_sdpa is registered there (otherwise the layout/dtype coercions would
// be skipped and every op-model query for the op would fail).
// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --ttnn-workaround="ttnn-optimization-level=1" --canonicalize -o %t1 %s
// RUN: FileCheck %s --input-file=%t1

// The tt-metal sparse_sdpa kernel reads q/kv/indices (and drains the output)
// through row-major paged accessors, so all four tensors must be ROW_MAJOR and
// DRAM interleaved, q/kv/out must be bf16, and indices must be uint32
// (sparse_sdpa_device_operation.cpp). The operand workaround inserts the
// to_layout ops that enforce that.
//
// The workaround-produced layouts are aliased at the top of the module. A
// ROW_MAJOR layout prints as a plain scalar memref (e.g. memref<1024x64xbf16>)
// whereas the tiled default prints as memref<32x2x!ttcore.tile<32x32, bf16>>,
// so matching the scalar form is what pins ROW_MAJOR.
// CHECK-DAG: #[[Q_RM:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<1024x64xbf16, #dram>, <interleaved>>
// CHECK-DAG: #[[KV_RM:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<64x64xbf16, #dram>, <interleaved>>
// CHECK-DAG: #[[IDX_RM:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<32x32xui32, #dram>, <interleaved>>
// CHECK-DAG: #[[OUT_RM:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<1024x32xbf16, #dram>, <interleaved>>

module {
  // f32 q/kv and si32 indices: every operand and the output needs a cast, on
  // top of the ROW_MAJOR coercion.
  func.func public @sparse_sdpa_f32_si32(%q: tensor<1x32x32x64xf32>, %kv: tensor<1x1x64x64xf32>, %idx: tensor<1x1x32x32xsi32>) -> tensor<1x32x32x32xf32> {
    // CHECK: func.func public @sparse_sdpa_f32_si32
    // CHECK-DAG: "ttnn.to_tensor_spec"(%arg0)
    // CHECK-DAG: "ttnn.to_tensor_spec"(%arg1)
    // CHECK-DAG: "ttnn.to_tensor_spec"(%arg2)
    // CHECK: %[[OUT:.*]] = "ttnn.sparse_sdpa"(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}})
    // CHECK-SAME: (tensor<1x32x32x64xbf16, #[[Q_RM]]>, tensor<1x1x64x64xbf16, #[[KV_RM]]>, tensor<1x1x32x32xui32, #[[IDX_RM]]>) -> tensor<1x32x32x32xbf16, #[[OUT_RM]]>
    // CHECK: "ttnn.to_tensor_spec"(%[[OUT]])
    // CHECK-SAME: -> tensor<1x32x32x32xf32
    %0 = "ttnn.sparse_sdpa"(%q, %kv, %idx) <{v_dim = 32 : ui32, k_chunk_size = 32 : ui32}> : (tensor<1x32x32x64xf32>, tensor<1x1x64x64xf32>, tensor<1x1x32x32xsi32>) -> tensor<1x32x32x32xf32>
    return %0 : tensor<1x32x32x32xf32>
  }

  // Already bf16 / uint32: the dtypes stay put, but the tiled default layout
  // must still be coerced to ROW_MAJOR on all operands and the output.
  func.func public @sparse_sdpa_row_major_coercion(%q: tensor<1x32x32x64xbf16>, %kv: tensor<1x1x64x64xbf16>, %idx: tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16> {
    // CHECK: func.func public @sparse_sdpa_row_major_coercion
    // CHECK: "ttnn.sparse_sdpa"(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}})
    // CHECK-SAME: (tensor<1x32x32x64xbf16, #[[Q_RM]]>, tensor<1x1x64x64xbf16, #[[KV_RM]]>, tensor<1x1x32x32xui32, #[[IDX_RM]]>) -> tensor<1x32x32x32xbf16, #[[OUT_RM]]>
    %0 = "ttnn.sparse_sdpa"(%q, %kv, %idx) <{v_dim = 32 : ui32, k_chunk_size = 32 : ui32}> : (tensor<1x32x32x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16>
    return %0 : tensor<1x32x32x32xbf16>
  }
}
