// RUN: ttmlir-opt --ttcore-register-device --ttnn-lower-tt-lang-to-generic -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// Covers the multi-output side of `--ttnn-lower-tt-lang-to-generic`, which
// `tt_lang_op_flatbuffer.mlir` cannot reach because it only has one result.
//
// `ttnn.generic` writes each output in place, so two results backed by the
// same buffer clobber each other and only the last write survives -- both
// then read back identical, with nothing in the IR to flag it. Identical
// per-result `ttir.empty` inits are side-effect free, so CSE folds them onto
// a single SSA value before this pass runs; the pass has to hand the
// duplicated destinations their own buffers again.
//
// Rebuilding is deliberately limited to the duplicates: an init that is
// already unique belongs to the caller and must keep being written in place,
// which is what the DPS interface promises.
//
// Host-side only, and no flatbuffer RUN line, because the `kernel_artifact`
// below carries stub kernel bodies. Real on-device coverage lives in tt-xla's
// `tests/torch/ops/test_tt_lang_kernel_e2e.py`.

#dram = #ttnn.buffer_type<dram>

#dram_layout = #ttnn.ttnn_layout<
  (d0, d1) -> (d0, d1),
  <1x1>,
  memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>
>

module {
  // Both "out" operands are the same SSA value, i.e. the post-CSE shape the
  // frontend's two identical `ttir.empty` placeholders arrive in. The first
  // keeps %arg2; the second must get a buffer of its own, and result 1 must
  // follow it.
  // CHECK-LABEL: func.func @tt_lang_deduped_outs
  func.func @tt_lang_deduped_outs(%arg0: tensor<32x32xf32, #dram_layout>,
                                  %arg1: tensor<32x32xf32, #dram_layout>,
                                  %arg2: tensor<32x32xf32, #dram_layout>)
      -> (tensor<32x32xf32, #dram_layout>, tensor<32x32xf32, #dram_layout>)
      attributes {tt.function_type = "forward_device"} {
    // CHECK-NOT: ttnn.tt_lang_op
    // CHECK: %[[DUP:[0-9a-zA-Z_]+]] = "ttnn.empty"
    // CHECK: "ttnn.generic"(%arg0, %arg1, %arg2, %[[DUP]])
    // CHECK-SAME: #ttnn.source_compute_kernel<source = "// compute kernel stub"
    // CHECK: return %arg2, %[[DUP]]
    %0, %1 = "ttnn.tt_lang_op"(%arg0, %arg1, %arg2, %arg2) <{
      kernel_id = "test.dual::v1",
      version_tag = "1.0",
      arg_roles = "in,in,out,out",
      shard_spec = "",
      kernel_artifact = "{\"format_version\": 1, \"kernels\": [{\"thread_type\": \"compute\", \"cpp_source\": \"// compute kernel stub\", \"tensor_indices\": [0, 1, 2, 3], \"kernel_config\": {\"type\": \"ComputeKernelConfig\", \"math_fidelity\": \"HiFi4\", \"fp32_dest_acc_en\": false, \"dst_full_sync_en\": false, \"bfp8_pack_precise\": false, \"math_approx_mode\": false}}, {\"thread_type\": \"noc\", \"cpp_source\": \"// reader kernel stub\", \"tensor_indices\": [0, 1], \"kernel_config\": {\"type\": \"ReaderKernelConfig\"}}, {\"thread_type\": \"noc\", \"cpp_source\": \"// writer kernel stub\", \"tensor_indices\": [2, 3], \"kernel_config\": {\"type\": \"WriterKernelConfig\"}}], \"core_range\": {\"start\": [0, 0], \"end\": [0, 0]}, \"cb_configs\": [{\"buffer_index\": 0, \"data_format\": \"Float32\", \"page_size\": 4096, \"total_size\": 8192, \"num_tiles\": 2, \"block_count\": 2}, {\"buffer_index\": 1, \"data_format\": \"Float32\", \"page_size\": 4096, \"total_size\": 8192, \"num_tiles\": 2, \"block_count\": 2}, {\"buffer_index\": 2, \"data_format\": \"Float32\", \"page_size\": 4096, \"total_size\": 8192, \"num_tiles\": 2, \"block_count\": 2}, {\"buffer_index\": 3, \"data_format\": \"Float32\", \"page_size\": 4096, \"total_size\": 8192, \"num_tiles\": 2, \"block_count\": 2}], \"num_tensors\": 4, \"num_pipe_nets\": 0}"
    }> : (tensor<32x32xf32, #dram_layout>, tensor<32x32xf32, #dram_layout>,
          tensor<32x32xf32, #dram_layout>, tensor<32x32xf32, #dram_layout>)
        -> (tensor<32x32xf32, #dram_layout>, tensor<32x32xf32, #dram_layout>)
    return %0, %1 : tensor<32x32xf32, #dram_layout>, tensor<32x32xf32, #dram_layout>
  }

  // The two "out" operands are already distinct, so both are the caller's to
  // keep: the generic must write straight into %arg2 and %arg3 with no
  // substitute buffer allocated.
  // CHECK-LABEL: func.func @tt_lang_distinct_outs
  func.func @tt_lang_distinct_outs(%arg0: tensor<32x32xf32, #dram_layout>,
                                   %arg1: tensor<32x32xf32, #dram_layout>,
                                   %arg2: tensor<32x32xf32, #dram_layout>,
                                   %arg3: tensor<32x32xf32, #dram_layout>)
      -> (tensor<32x32xf32, #dram_layout>, tensor<32x32xf32, #dram_layout>)
      attributes {tt.function_type = "forward_device"} {
    // CHECK-NOT: ttnn.tt_lang_op
    // CHECK-NOT: ttnn.empty
    // CHECK: "ttnn.generic"(%arg0, %arg1, %arg2, %arg3)
    // CHECK: return %arg2, %arg3
    %0, %1 = "ttnn.tt_lang_op"(%arg0, %arg1, %arg2, %arg3) <{
      kernel_id = "test.dual::v1",
      version_tag = "1.0",
      arg_roles = "in,in,out,out",
      shard_spec = "",
      kernel_artifact = "{\"format_version\": 1, \"kernels\": [{\"thread_type\": \"compute\", \"cpp_source\": \"// compute kernel stub\", \"tensor_indices\": [0, 1, 2, 3], \"kernel_config\": {\"type\": \"ComputeKernelConfig\", \"math_fidelity\": \"HiFi4\", \"fp32_dest_acc_en\": false, \"dst_full_sync_en\": false, \"bfp8_pack_precise\": false, \"math_approx_mode\": false}}, {\"thread_type\": \"noc\", \"cpp_source\": \"// reader kernel stub\", \"tensor_indices\": [0, 1], \"kernel_config\": {\"type\": \"ReaderKernelConfig\"}}, {\"thread_type\": \"noc\", \"cpp_source\": \"// writer kernel stub\", \"tensor_indices\": [2, 3], \"kernel_config\": {\"type\": \"WriterKernelConfig\"}}], \"core_range\": {\"start\": [0, 0], \"end\": [0, 0]}, \"cb_configs\": [{\"buffer_index\": 0, \"data_format\": \"Float32\", \"page_size\": 4096, \"total_size\": 8192, \"num_tiles\": 2, \"block_count\": 2}, {\"buffer_index\": 1, \"data_format\": \"Float32\", \"page_size\": 4096, \"total_size\": 8192, \"num_tiles\": 2, \"block_count\": 2}, {\"buffer_index\": 2, \"data_format\": \"Float32\", \"page_size\": 4096, \"total_size\": 8192, \"num_tiles\": 2, \"block_count\": 2}, {\"buffer_index\": 3, \"data_format\": \"Float32\", \"page_size\": 4096, \"total_size\": 8192, \"num_tiles\": 2, \"block_count\": 2}], \"num_tensors\": 4, \"num_pipe_nets\": 0}"
    }> : (tensor<32x32xf32, #dram_layout>, tensor<32x32xf32, #dram_layout>,
          tensor<32x32xf32, #dram_layout>, tensor<32x32xf32, #dram_layout>)
        -> (tensor<32x32xf32, #dram_layout>, tensor<32x32xf32, #dram_layout>)
    return %0, %1 : tensor<32x32xf32, #dram_layout>, tensor<32x32xf32, #dram_layout>
  }

  // CSE folds identical `ttir.empty` inits across the whole module, so two
  // chained kernels with same-shaped outputs reach this pass sharing one
  // destination for all four "out" slots. Only the very first slot may keep it:
  // the second kernel writing there would overwrite the first kernel's result,
  // which it is also reading as its input.
  // CHECK-LABEL: func.func @tt_lang_chained_shared_out
  func.func @tt_lang_chained_shared_out(%arg0: tensor<32x32xf32, #dram_layout>,
                                        %arg1: tensor<32x32xf32, #dram_layout>,
                                        %arg2: tensor<32x32xf32, #dram_layout>)
      -> (tensor<32x32xf32, #dram_layout>, tensor<32x32xf32, #dram_layout>,
          tensor<32x32xf32, #dram_layout>, tensor<32x32xf32, #dram_layout>)
      attributes {tt.function_type = "forward_device"} {
    // CHECK-NOT: ttnn.tt_lang_op
    // CHECK: %[[E1:[0-9a-zA-Z_]+]] = "ttnn.empty"
    // CHECK: "ttnn.generic"(%arg0, %arg1, %arg2, %[[E1]])
    // CHECK: %[[E2:[0-9a-zA-Z_]+]] = "ttnn.empty"
    // CHECK: %[[E3:[0-9a-zA-Z_]+]] = "ttnn.empty"
    // The second kernel reads %arg2 and must write somewhere else entirely.
    // CHECK: "ttnn.generic"(%arg2, %arg1, %[[E2]], %[[E3]])
    // CHECK: return %arg2, %[[E1]], %[[E2]], %[[E3]]
    %0, %1 = "ttnn.tt_lang_op"(%arg0, %arg1, %arg2, %arg2) <{
      kernel_id = "test.dual::v1",
      version_tag = "1.0",
      arg_roles = "in,in,out,out",
      shard_spec = "",
      kernel_artifact = "{\"format_version\": 1, \"kernels\": [{\"thread_type\": \"compute\", \"cpp_source\": \"// compute kernel stub\", \"tensor_indices\": [0, 1, 2, 3], \"kernel_config\": {\"type\": \"ComputeKernelConfig\", \"math_fidelity\": \"HiFi4\", \"fp32_dest_acc_en\": false, \"dst_full_sync_en\": false, \"bfp8_pack_precise\": false, \"math_approx_mode\": false}}, {\"thread_type\": \"noc\", \"cpp_source\": \"// reader kernel stub\", \"tensor_indices\": [0, 1], \"kernel_config\": {\"type\": \"ReaderKernelConfig\"}}, {\"thread_type\": \"noc\", \"cpp_source\": \"// writer kernel stub\", \"tensor_indices\": [2, 3], \"kernel_config\": {\"type\": \"WriterKernelConfig\"}}], \"core_range\": {\"start\": [0, 0], \"end\": [0, 0]}, \"cb_configs\": [{\"buffer_index\": 0, \"data_format\": \"Float32\", \"page_size\": 4096, \"total_size\": 8192, \"num_tiles\": 2, \"block_count\": 2}, {\"buffer_index\": 1, \"data_format\": \"Float32\", \"page_size\": 4096, \"total_size\": 8192, \"num_tiles\": 2, \"block_count\": 2}, {\"buffer_index\": 2, \"data_format\": \"Float32\", \"page_size\": 4096, \"total_size\": 8192, \"num_tiles\": 2, \"block_count\": 2}, {\"buffer_index\": 3, \"data_format\": \"Float32\", \"page_size\": 4096, \"total_size\": 8192, \"num_tiles\": 2, \"block_count\": 2}], \"num_tensors\": 4, \"num_pipe_nets\": 0}"
    }> : (tensor<32x32xf32, #dram_layout>, tensor<32x32xf32, #dram_layout>,
          tensor<32x32xf32, #dram_layout>, tensor<32x32xf32, #dram_layout>)
        -> (tensor<32x32xf32, #dram_layout>, tensor<32x32xf32, #dram_layout>)
    %2, %3 = "ttnn.tt_lang_op"(%0, %arg1, %arg2, %arg2) <{
      kernel_id = "test.dual::v1",
      version_tag = "1.0",
      arg_roles = "in,in,out,out",
      shard_spec = "",
      kernel_artifact = "{\"format_version\": 1, \"kernels\": [{\"thread_type\": \"compute\", \"cpp_source\": \"// compute kernel stub\", \"tensor_indices\": [0, 1, 2, 3], \"kernel_config\": {\"type\": \"ComputeKernelConfig\", \"math_fidelity\": \"HiFi4\", \"fp32_dest_acc_en\": false, \"dst_full_sync_en\": false, \"bfp8_pack_precise\": false, \"math_approx_mode\": false}}, {\"thread_type\": \"noc\", \"cpp_source\": \"// reader kernel stub\", \"tensor_indices\": [0, 1], \"kernel_config\": {\"type\": \"ReaderKernelConfig\"}}, {\"thread_type\": \"noc\", \"cpp_source\": \"// writer kernel stub\", \"tensor_indices\": [2, 3], \"kernel_config\": {\"type\": \"WriterKernelConfig\"}}], \"core_range\": {\"start\": [0, 0], \"end\": [0, 0]}, \"cb_configs\": [{\"buffer_index\": 0, \"data_format\": \"Float32\", \"page_size\": 4096, \"total_size\": 8192, \"num_tiles\": 2, \"block_count\": 2}, {\"buffer_index\": 1, \"data_format\": \"Float32\", \"page_size\": 4096, \"total_size\": 8192, \"num_tiles\": 2, \"block_count\": 2}, {\"buffer_index\": 2, \"data_format\": \"Float32\", \"page_size\": 4096, \"total_size\": 8192, \"num_tiles\": 2, \"block_count\": 2}, {\"buffer_index\": 3, \"data_format\": \"Float32\", \"page_size\": 4096, \"total_size\": 8192, \"num_tiles\": 2, \"block_count\": 2}], \"num_tensors\": 4, \"num_pipe_nets\": 0}"
    }> : (tensor<32x32xf32, #dram_layout>, tensor<32x32xf32, #dram_layout>,
          tensor<32x32xf32, #dram_layout>, tensor<32x32xf32, #dram_layout>)
        -> (tensor<32x32xf32, #dram_layout>, tensor<32x32xf32, #dram_layout>)
    return %0, %1, %2, %3 : tensor<32x32xf32, #dram_layout>, tensor<32x32xf32, #dram_layout>,
                            tensor<32x32xf32, #dram_layout>, tensor<32x32xf32, #dram_layout>
  }
}
