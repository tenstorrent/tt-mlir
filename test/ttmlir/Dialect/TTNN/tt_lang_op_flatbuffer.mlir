// RUN: ttmlir-opt --ttcore-register-device --ttnn-lower-tt-lang-to-generic -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// Smoke-tests the `--ttnn-lower-tt-lang-to-generic` pass and the
// downstream `ttnn.generic` flatbuffer emission. The pass rewrites a
// resolved `ttnn.tt_lang_op` (carrying its compiled `kernel_artifact`
// JSON) into a `ttnn.generic` op with an inline-source `#ttnn.program`
// descriptor; flatbuffer emission then reuses the generic-kernel path
// with no tt-lang-specific handling. This is a HOST-SIDE test only: it
// registers a mock device, FileChecks the lowered MLIR, and confirms the
// flatbuffer translates without diagnostics. It is deliberately NOT
// under `test/ttmlir/Silicon/` because the `kernel_artifact` below uses
// stub kernel bodies (plain comments, no `kernel_main`) that cannot be
// JIT-built on device -- the `ttrt run Silicon` and chisel jobs scan the
// Silicon tree and would try to execute every emitted flatbuffer. Real
// on-device coverage for the tt-lang path lives in tt-xla's
// `tests/torch/ops/test_tt_lang_kernel_e2e.py`, which compiles a real
// tt-lang kernel end to end.
//
// `kernel_artifact` is a synthetic JSON payload produced offline (mirrors
// the shape that tt-xla's tt_torch.tt_lang._serialize_compiled_operation
// produces). Decoded, it is::
//
//   {
//     "format_version": 1,
//     "kernels": [
//       {"thread_type": "compute",
//        "cpp_source": "// compute kernel stub",
//        "tensor_indices": [0, 1, 2],
//        "kernel_config": {"type": "ComputeKernelConfig",
//                          "math_fidelity": "HiFi4",
//                          "fp32_dest_acc_en": false,
//                          "dst_full_sync_en": false,
//                          "bfp8_pack_precise": false,
//                          "math_approx_mode": false}},
//       {"thread_type": "noc",
//        "cpp_source": "// reader kernel stub",
//        "tensor_indices": [0, 1],
//        "kernel_config": {"type": "ReaderKernelConfig"}},
//       {"thread_type": "noc",
//        "cpp_source": "// writer kernel stub",
//        "tensor_indices": [2],
//        "kernel_config": {"type": "WriterKernelConfig"}}],
//     "core_range":  {"start": [0, 0], "end": [0, 0]},
//     "cb_configs":  [{"buffer_index": 0, "data_format": "Float32",
//                      "page_size": 4096, "total_size": 8192,
//                      "num_tiles": 2, "block_count": 2}, ...x3],
//     "num_tensors": 3,
//     "num_pipe_nets": 0
//   }
//
// The pass is expected to:
//   - Parse the JSON payload (no diagnostic errors).
//   - Build a `#ttnn.program` with one inline-source kernel attr per
//     kernels[*] entry (source_compute_kernel / source_read_kernel /
//     source_write_kernel), each carrying its C++ `source` inline plus
//     its compile-time / common-runtime args.
//   - Emit one `#ttnn.kernel_cb` per cb_configs[*] entry.
//   - Replace the `ttnn.tt_lang_op` with a `ttnn.generic` op and rewire
//     the result to the tied "out" operand.
//
// We only FileCheck the MLIR side here (the flatbuffer is a binary blob
// that lit can't easily inspect inline). If emission throws/errors the
// third RUN line will fail.

#dram = #ttnn.buffer_type<dram>

#dram_layout = #ttnn.ttnn_layout<
  (d0, d1) -> (d0, d1),
  <1x1>,
  memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>
>

// CHECK: module attributes {ttcore.system_desc = #system_desc}
// `ttnn.tt_lang_op` is DPS-style: every operand carries an `arg_roles`
// token, and `out`-roled operands also surface as op results (they are
// pre-allocated output buffers). So a 2-input + 1-output kernel takes 3
// operands and produces 1 result; arg_roles = "in,in,out".
module {
  // CHECK-LABEL: func.func @tt_lang_kernel
  func.func @tt_lang_kernel(%arg0: tensor<32x32xf32, #dram_layout>,
                            %arg1: tensor<32x32xf32, #dram_layout>,
                            %arg2: tensor<32x32xf32, #dram_layout>)
      -> tensor<32x32xf32, #dram_layout>
      attributes {tt.function_type = "forward_device"} {
    // The tt_lang_op is gone; lowering produced a ttnn.generic carrying an
    // inline-source program, and the result was rewired to the "out"
    // operand (%arg2).
    // CHECK-NOT: ttnn.tt_lang_op
    // CHECK: "ttnn.generic"(%arg0, %arg1, %arg2)
    // CHECK-SAME: #ttnn.source_compute_kernel<source = "// compute kernel stub"
    // CHECK-SAME: #ttnn.source_read_kernel<source = "// reader kernel stub"
    // CHECK-SAME: #ttnn.source_write_kernel<source = "// writer kernel stub"
    // CHECK: return %arg2
    %0 = "ttnn.tt_lang_op"(%arg0, %arg1, %arg2) <{
      kernel_id = "test.add::v1",
      version_tag = "1.0",
      arg_roles = "in,in,out",
      shard_spec = "",
      kernel_artifact = "{\"format_version\": 1, \"kernels\": [{\"thread_type\": \"compute\", \"cpp_source\": \"// compute kernel stub\", \"tensor_indices\": [0, 1, 2], \"kernel_config\": {\"type\": \"ComputeKernelConfig\", \"math_fidelity\": \"HiFi4\", \"fp32_dest_acc_en\": false, \"dst_full_sync_en\": false, \"bfp8_pack_precise\": false, \"math_approx_mode\": false}}, {\"thread_type\": \"noc\", \"cpp_source\": \"// reader kernel stub\", \"tensor_indices\": [0, 1], \"kernel_config\": {\"type\": \"ReaderKernelConfig\"}}, {\"thread_type\": \"noc\", \"cpp_source\": \"// writer kernel stub\", \"tensor_indices\": [2], \"kernel_config\": {\"type\": \"WriterKernelConfig\"}}], \"core_range\": {\"start\": [0, 0], \"end\": [0, 0]}, \"cb_configs\": [{\"buffer_index\": 0, \"data_format\": \"Float32\", \"page_size\": 4096, \"total_size\": 8192, \"num_tiles\": 2, \"block_count\": 2}, {\"buffer_index\": 1, \"data_format\": \"Float32\", \"page_size\": 4096, \"total_size\": 8192, \"num_tiles\": 2, \"block_count\": 2}, {\"buffer_index\": 2, \"data_format\": \"Float32\", \"page_size\": 4096, \"total_size\": 8192, \"num_tiles\": 2, \"block_count\": 2}], \"num_tensors\": 3, \"num_pipe_nets\": 0}"
    }> : (tensor<32x32xf32, #dram_layout>, tensor<32x32xf32, #dram_layout>,
          tensor<32x32xf32, #dram_layout>)
        -> (tensor<32x32xf32, #dram_layout>)
    return %0 : tensor<32x32xf32, #dram_layout>
  }
}
