// RUN: ttmlir-opt --ttcore-register-device --ttnn-lower-tt-lang-to-generic -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// Host-side coverage for PipeNet resource plumbing in
// `--ttnn-lower-tt-lang-to-generic`: when the artifact requests non-zero
// `pipe_sram_scratch_bytes`, `num_pipe_global_semaphores`, and
// `num_pipe_sync_semaphores`, the pass must
//   * allocate an L1 height-sharded scratch tensor and append it to
//     `ttnn.generic` io_tensors,
//   * insert `ttnn.create_global_semaphore` ops into `additional_args`,
//   * emit program-local `KernelSemaphoreAttr`s, and
//   * append address-of-scratch + global-semaphore markers to every
//     kernel's `common_rt_args`.
//
// Stub kernel bodies (plain comments) -- not Silicon-runnable. Real
// on-device PipeNet coverage lives in tt-xla.

#dram = #ttnn.buffer_type<dram>

#dram_layout = #ttnn.ttnn_layout<
  (d0, d1) -> (d0, d1),
  <1x1>,
  memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>
>

// CHECK: module attributes {ttcore.system_desc = #system_desc}
module {
  // CHECK-LABEL: func.func @tt_lang_kernel_with_pipe_resources
  func.func @tt_lang_kernel_with_pipe_resources(
      %arg0: tensor<32x32xf32, #dram_layout>,
      %arg1: tensor<32x32xf32, #dram_layout>,
      %arg2: tensor<32x32xf32, #dram_layout>)
      -> tensor<32x32xf32, #dram_layout>
      attributes {tt.function_type = "forward_device"} {
    // Scratch tensor allocation + one global semaphore, then the generic.
    // CHECK-NOT: ttnn.tt_lang_op
    // CHECK: %[[DEVICE:.+]] = "ttnn.get_device"
    // CHECK: %[[SCRATCH:.+]] = "ttnn.empty"(%[[DEVICE]])
    // CHECK: %[[SEM:.+]] = "ttnn.create_global_semaphore"(%[[DEVICE]])
    // CHECK-SAME: initial_value = 0 : ui32
    // io_tensors = [%arg0, %arg1, %arg2, scratch]; additional_args = [sem]
    // CHECK: "ttnn.generic"(%arg0, %arg1, %arg2, %[[SCRATCH]], %[[SEM]])
    // Scratch address (io index 3) and global-semaphore marker (argRefs
    // index 4) appear in common_rt_args before the program-local
    // semaphores list further down the same attribute.
    // CHECK-SAME: #ttnn.kernel_arg_address_of_tensor<3>
    // CHECK-SAME: #ttnn.kernel_arg_global_semaphore<4>
    // Program-local sync semaphores from num_pipe_sync_semaphores=2.
    // CHECK-SAME: semaphores = [<id = 0, core_type = worker,
    // CHECK-SAME: <id = 1, core_type = worker,
    // CHECK: return %arg2
    %0 = "ttnn.tt_lang_op"(%arg0, %arg1, %arg2) <{
      kernel_id = "test.add_pipe::v1",
      version_tag = "1.0",
      arg_roles = "in,in,out",
      shard_spec = "",
      kernel_artifact = "{\"format_version\": 2, \"kernels\": [{\"thread_type\": \"compute\", \"cpp_source\": \"// compute kernel stub\", \"tensor_indices\": [0, 1, 2], \"kernel_config\": {\"type\": \"ComputeKernelConfig\", \"math_fidelity\": \"HiFi4\", \"fp32_dest_acc_en\": false, \"dst_full_sync_en\": false, \"bfp8_pack_precise\": false, \"math_approx_mode\": false}}, {\"thread_type\": \"noc\", \"cpp_source\": \"// reader kernel stub\", \"tensor_indices\": [0, 1], \"kernel_config\": {\"type\": \"ReaderKernelConfig\"}}, {\"thread_type\": \"noc\", \"cpp_source\": \"// writer kernel stub\", \"tensor_indices\": [2], \"kernel_config\": {\"type\": \"WriterKernelConfig\"}}], \"core_range\": {\"start\": [0, 0], \"end\": [0, 0]}, \"cb_configs\": [{\"buffer_index\": 0, \"data_format\": \"Float32\", \"page_size\": 4096, \"total_size\": 8192, \"num_tiles\": 2, \"block_count\": 2}, {\"buffer_index\": 1, \"data_format\": \"Float32\", \"page_size\": 4096, \"total_size\": 8192, \"num_tiles\": 2, \"block_count\": 2}, {\"buffer_index\": 2, \"data_format\": \"Float32\", \"page_size\": 4096, \"total_size\": 8192, \"num_tiles\": 2, \"block_count\": 2}], \"num_tensors\": 3, \"pipe_sram_scratch_bytes\": 128, \"num_pipe_global_semaphores\": 1, \"num_pipe_sync_semaphores\": 2}"
    }> : (tensor<32x32xf32, #dram_layout>, tensor<32x32xf32, #dram_layout>,
          tensor<32x32xf32, #dram_layout>)
        -> (tensor<32x32xf32, #dram_layout>)
    return %0 : tensor<32x32xf32, #dram_layout>
  }
}
