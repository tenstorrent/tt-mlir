// RUN: ttmlir-opt --ttnn-set-compute-kernel-config="fp32-dest-acc-en=true" %s | FileCheck %s
//
// Reductions must receive the compute-kernel config too, exactly like
// matmul/conv. This per-op behaviour is what lets the pipeline re-run this pass
// after the optimizer to restore fp32_dest_acc_en on reductions: the optimizer
// / canonicalizer can rebuild a ttnn.sum via a builder that defaults
// compute_config to null, dropping the fp32_dest_acc_en stamped before it.
// Without fp32 dest accumulation, large reductions (e.g. the cross-entropy
// vocab-axis sum / token count) accumulate in low precision on device and
// return garbage.
//
// The post-optimizer re-stamp's "only-unconfigured" mode (skip ops that already
// carry a config) is not a CLI option - it is reachable only from the pipeline
// via createTTNNSetComputeKernelConfigRestamp - so it is exercised by pipeline
// tests, not here. This pins the pass's per-op contract.

// A reduction with no compute_config receives fp32_dest_acc_en (+ the
// pass-default math_fidelity).
// CHECK-LABEL: func @sum_gets_fp32_dest_acc
func.func @sum_gets_fp32_dest_acc(%arg0: tensor<512x1024xbf16>) -> tensor<512x1xbf16> {
  // CHECK: "ttnn.sum"
  // CHECK-SAME: compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>
  %0 = "ttnn.sum"(%arg0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<512x1024xbf16>) -> tensor<512x1xbf16>
  return %0 : tensor<512x1xbf16>
}

// An explicit fp32_dest_acc_en=false already on the op must be preserved: the
// pass only fills knobs the op does not already set. The full emitted attr is
// pinned (not just the substring the input already contains), so a no-op or a
// mangled sibling field - e.g. failing to fill math_fidelity - would fail.
// CHECK-LABEL: func @sum_preserves_existing_false
func.func @sum_preserves_existing_false(%arg0: tensor<512x1024xbf16>) -> tensor<512x1xbf16> {
  // CHECK: "ttnn.sum"
  // CHECK-SAME: compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = false>
  %0 = "ttnn.sum"(%arg0) <{compute_config = #ttnn.device_compute_kernel_config<fp32_dest_acc_en = false>, dim_arg = [1 : i32], keep_dim = true}> : (tensor<512x1024xbf16>) -> tensor<512x1xbf16>
  return %0 : tensor<512x1xbf16>
}
