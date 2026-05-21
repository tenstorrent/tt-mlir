// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// Smoke-tests the `ttnn.tt_lang_op` -> `GenericOp` flatbuffer emitter
// added in TTNNToFlatbuffer.cpp.
//
// `kernel_artifact` is a synthetic JSON payload produced offline (mirrors
// the shape that tt-xla's tt_torch.tt_lang._serialize_compiled_kernel
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
// The emitter is expected to:
//   - Parse the JSON payload (no diagnostic errors).
//   - Build a flatbuffer ProgramDescriptor with one KernelDescriptor per
//     kernels[*] entry, populated with the matching KernelConfig and
//     each kernel's compile_time_args / common_runtime_args.
//   - Emit one KernelCBDescriptor per cb_configs[*] entry.
//   - Use the existing GenericOp flatbuffer record as the carrier.
//
// We only FileCheck the MLIR side here (the flatbuffer is a binary blob
// that lit can't easily inspect inline). If the emitter throws/errors
// the second RUN line will fail.

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
    // CHECK: ttnn.tt_lang_op
    // CHECK-SAME: arg_roles = "in,in,out"
    // CHECK-SAME: kernel_id = "test.add::v1"
    // CHECK-SAME: version_tag = "1.0"
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
