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
    // CHECK-SAME: kernel_id = "test.add::v1"
    // CHECK-SAME: version_tag = "1.0"
    // CHECK-SAME: arg_roles = "in,in,out"
    %0 = "ttnn.tt_lang_op"(%arg0, %arg1, %arg2) <{
      kernel_id = "test.add::v1",
      version_tag = "1.0",
      arg_roles = "in,in,out",
      shard_spec = "",
      kernel_artifact = array<i8:
        123, 34, 102, 111, 114, 109, 97, 116, 95, 118, 101, 114, 115, 105, 111,
        110, 34, 58, 32, 49, 44, 32, 34, 107, 101, 114, 110, 101, 108, 115,
        34, 58, 32, 91, 123, 34, 116, 104, 114, 101, 97, 100, 95, 116, 121,
        112, 101, 34, 58, 32, 34, 99, 111, 109, 112, 117, 116, 101, 34, 44,
        32, 34, 99, 112, 112, 95, 115, 111, 117, 114, 99, 101, 34, 58, 32,
        34, 47, 47, 32, 99, 111, 109, 112, 117, 116, 101, 32, 107, 101, 114,
        110, 101, 108, 32, 115, 116, 117, 98, 34, 44, 32, 34, 116, 101, 110,
        115, 111, 114, 95, 105, 110, 100, 105, 99, 101, 115, 34, 58, 32, 91,
        48, 44, 32, 49, 44, 32, 50, 93, 44, 32, 34, 107, 101, 114, 110,
        101, 108, 95, 99, 111, 110, 102, 105, 103, 34, 58, 32, 123, 34, 116,
        121, 112, 101, 34, 58, 32, 34, 67, 111, 109, 112, 117, 116, 101, 75,
        101, 114, 110, 101, 108, 67, 111, 110, 102, 105, 103, 34, 44, 32, 34,
        109, 97, 116, 104, 95, 102, 105, 100, 101, 108, 105, 116, 121, 34, 58,
        32, 34, 72, 105, 70, 105, 52, 34, 44, 32, 34, 102, 112, 51, 50,
        95, 100, 101, 115, 116, 95, 97, 99, 99, 95, 101, 110, 34, 58, 32,
        102, 97, 108, 115, 101, 44, 32, 34, 100, 115, 116, 95, 102, 117, 108,
        108, 95, 115, 121, 110, 99, 95, 101, 110, 34, 58, 32, 102, 97, 108,
        115, 101, 44, 32, 34, 98, 102, 112, 56, 95, 112, 97, 99, 107, 95,
        112, 114, 101, 99, 105, 115, 101, 34, 58, 32, 102, 97, 108, 115, 101,
        44, 32, 34, 109, 97, 116, 104, 95, 97, 112, 112, 114, 111, 120, 95,
        109, 111, 100, 101, 34, 58, 32, 102, 97, 108, 115, 101, 125, 125, 44,
        32, 123, 34, 116, 104, 114, 101, 97, 100, 95, 116, 121, 112, 101, 34,
        58, 32, 34, 110, 111, 99, 34, 44, 32, 34, 99, 112, 112, 95, 115,
        111, 117, 114, 99, 101, 34, 58, 32, 34, 47, 47, 32, 114, 101, 97,
        100, 101, 114, 32, 107, 101, 114, 110, 101, 108, 32, 115, 116, 117, 98,
        34, 44, 32, 34, 116, 101, 110, 115, 111, 114, 95, 105, 110, 100, 105,
        99, 101, 115, 34, 58, 32, 91, 48, 44, 32, 49, 93, 44, 32, 34,
        107, 101, 114, 110, 101, 108, 95, 99, 111, 110, 102, 105, 103, 34, 58,
        32, 123, 34, 116, 121, 112, 101, 34, 58, 32, 34, 82, 101, 97, 100,
        101, 114, 75, 101, 114, 110, 101, 108, 67, 111, 110, 102, 105, 103, 34,
        125, 125, 44, 32, 123, 34, 116, 104, 114, 101, 97, 100, 95, 116, 121,
        112, 101, 34, 58, 32, 34, 110, 111, 99, 34, 44, 32, 34, 99, 112,
        112, 95, 115, 111, 117, 114, 99, 101, 34, 58, 32, 34, 47, 47, 32,
        119, 114, 105, 116, 101, 114, 32, 107, 101, 114, 110, 101, 108, 32, 115,
        116, 117, 98, 34, 44, 32, 34, 116, 101, 110, 115, 111, 114, 95, 105,
        110, 100, 105, 99, 101, 115, 34, 58, 32, 91, 50, 93, 44, 32, 34,
        107, 101, 114, 110, 101, 108, 95, 99, 111, 110, 102, 105, 103, 34, 58,
        32, 123, 34, 116, 121, 112, 101, 34, 58, 32, 34, 87, 114, 105, 116,
        101, 114, 75, 101, 114, 110, 101, 108, 67, 111, 110, 102, 105, 103, 34,
        125, 125, 93, 44, 32, 34, 99, 111, 114, 101, 95, 114, 97, 110, 103,
        101, 34, 58, 32, 123, 34, 115, 116, 97, 114, 116, 34, 58, 32, 91,
        48, 44, 32, 48, 93, 44, 32, 34, 101, 110, 100, 34, 58, 32, 91,
        48, 44, 32, 48, 93, 125, 44, 32, 34, 99, 98, 95, 99, 111, 110,
        102, 105, 103, 115, 34, 58, 32, 91, 123, 34, 98, 117, 102, 102, 101,
        114, 95, 105, 110, 100, 101, 120, 34, 58, 32, 48, 44, 32, 34, 100,
        97, 116, 97, 95, 102, 111, 114, 109, 97, 116, 34, 58, 32, 34, 70,
        108, 111, 97, 116, 51, 50, 34, 44, 32, 34, 112, 97, 103, 101, 95,
        115, 105, 122, 101, 34, 58, 32, 52, 48, 57, 54, 44, 32, 34, 116,
        111, 116, 97, 108, 95, 115, 105, 122, 101, 34, 58, 32, 56, 49, 57,
        50, 44, 32, 34, 110, 117, 109, 95, 116, 105, 108, 101, 115, 34, 58,
        32, 50, 44, 32, 34, 98, 108, 111, 99, 107, 95, 99, 111, 117, 110,
        116, 34, 58, 32, 50, 125, 44, 32, 123, 34, 98, 117, 102, 102, 101,
        114, 95, 105, 110, 100, 101, 120, 34, 58, 32, 49, 44, 32, 34, 100,
        97, 116, 97, 95, 102, 111, 114, 109, 97, 116, 34, 58, 32, 34, 70,
        108, 111, 97, 116, 51, 50, 34, 44, 32, 34, 112, 97, 103, 101, 95,
        115, 105, 122, 101, 34, 58, 32, 52, 48, 57, 54, 44, 32, 34, 116,
        111, 116, 97, 108, 95, 115, 105, 122, 101, 34, 58, 32, 56, 49, 57,
        50, 44, 32, 34, 110, 117, 109, 95, 116, 105, 108, 101, 115, 34, 58,
        32, 50, 44, 32, 34, 98, 108, 111, 99, 107, 95, 99, 111, 117, 110,
        116, 34, 58, 32, 50, 125, 44, 32, 123, 34, 98, 117, 102, 102, 101,
        114, 95, 105, 110, 100, 101, 120, 34, 58, 32, 50, 44, 32, 34, 100,
        97, 116, 97, 95, 102, 111, 114, 109, 97, 116, 34, 58, 32, 34, 70,
        108, 111, 97, 116, 51, 50, 34, 44, 32, 34, 112, 97, 103, 101, 95,
        115, 105, 122, 101, 34, 58, 32, 52, 48, 57, 54, 44, 32, 34, 116,
        111, 116, 97, 108, 95, 115, 105, 122, 101, 34, 58, 32, 56, 49, 57,
        50, 44, 32, 34, 110, 117, 109, 95, 116, 105, 108, 101, 115, 34, 58,
        32, 50, 44, 32, 34, 98, 108, 111, 99, 107, 95, 99, 111, 117, 110,
        116, 34, 58, 32, 50, 125, 93, 44, 32, 34, 110, 117, 109, 95, 116,
        101, 110, 115, 111, 114, 115, 34, 58, 32, 51, 44, 32, 34, 110, 117,
        109, 95, 112, 105, 112, 101, 95, 110, 101, 116, 115, 34, 58, 32, 48,
        125>
    }> : (tensor<32x32xf32, #dram_layout>, tensor<32x32xf32, #dram_layout>,
          tensor<32x32xf32, #dram_layout>)
        -> (tensor<32x32xf32, #dram_layout>)
    return %0 : tensor<32x32xf32, #dram_layout>
  }
}
