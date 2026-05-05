// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t %s
// RUN: FileCheck %s --input-file=%t --check-prefix=CHECK-WITHOUT-ENV
// RUN: TTMLIR_ENABLE_DISK_CACHE=1 ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t2 %s
// RUN: FileCheck %s --input-file=%t2 --check-prefix=CHECK-WITH-ENV

// Test for --ttnn-insert-disk-cache-ops pass.
// The pass is gated behind TTMLIR_ENABLE_DISK_CACHE environment variable.
// Without the env var, no disk cache ops should be inserted.
// With the env var set, disk cache ops should be inserted for each tensor argument.

module {
  // CHECK-WITHOUT-ENV-LABEL: func.func @forward
  // CHECK-WITHOUT-ENV-NOT: ttcore.get_or_insert_into_disk_cache

  // CHECK-WITH-ENV-LABEL: func.func @forward
  // CHECK-WITH-ENV: ttcore.get_or_insert_into_disk_cache
  // CHECK-WITH-ENV: ttcore.get_or_insert_into_disk_cache
  func.func @forward(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<32x32xbf16> {
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %0 : tensor<32x32xbf16>
  }
}
