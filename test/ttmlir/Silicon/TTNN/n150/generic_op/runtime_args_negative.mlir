// RUN: not ttmlir-opt %s 2>&1 | FileCheck %s

// Test that duplicate CoreCoords in rt_args should fail verification.
#core = #ttnn.core_coord<0, 0>
#core_range = #ttnn.core_range<(0,0), (0,0)>
#core_ranges = #ttnn.core_range_set<[#core_range]>

#arg0 = #ttnn.kernel_arg_address_of_tensor<0>
#arg1 = #ttnn.kernel_arg_address_of_tensor<1>

// CHECK: error: Duplicate CoreCoord (0, 0) in runtime arguments
#write_kernel = #ttnn.write_kernel<
  symbol_ref = @write_kernel,
  core_ranges = #core_ranges,
  ct_args = [],
  common_rt_args = [],
  rt_args = [
    #ttnn.core_runtime_args<core_coord = #core, args = [#arg0]>,
    #ttnn.core_runtime_args<core_coord = #core, args = [#arg1]>
  ]>
