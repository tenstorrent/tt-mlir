// RUN: ttmlir-opt --ttnn-set-compute-kernel-config="math-fidelity=undefined fp32-dest-acc-en=false" %s | FileCheck %s

// The forward lm_head / vocab projection: [.., hidden] x [hidden, vocab].
// Inner (contraction) dim = hidden (small), output dim = vocab (> threshold).
// A vocab-sized *output* dim must force fp32 dest-acc + packer_l1_acc on the
// bf16 matmul so greedy argmax over these logits is reproducible (tt-xla #5520).

// CHECK-LABEL: func @test_large_n_lm_head_matmul
func.func @test_large_n_lm_head_matmul(%arg0: tensor<1x128xbf16>, %arg1: tensor<128x50001xbf16>) -> tensor<1x50001xbf16> {
  // CHECK: "ttnn.matmul"
  // CHECK-SAME: compute_config = #ttnn.device_compute_kernel_config<fp32_dest_acc_en = true, packer_l1_acc = true>
  %0 = "ttnn.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<1x128xbf16>, tensor<128x50001xbf16>) -> tensor<1x50001xbf16>
  return %0 : tensor<1x50001xbf16>
}

// A small output dim (typical hidden/intermediate matmul) must be left alone:
// no fp32 dest-acc, no packer_l1_acc.

// CHECK-LABEL: func @test_small_n_matmul
func.func @test_small_n_matmul(%arg0: tensor<1x128xbf16>, %arg1: tensor<128x256xbf16>) -> tensor<1x256xbf16> {
  // CHECK: "ttnn.matmul"
  // CHECK-NOT: fp32_dest_acc_en = true
  // CHECK-NOT: packer_l1_acc = true
  %0 = "ttnn.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<1x128xbf16>, tensor<128x256xbf16>) -> tensor<1x256xbf16>
  return %0 : tensor<1x256xbf16>
}
