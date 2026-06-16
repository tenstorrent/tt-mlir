// RUN: ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline="mesh-shape=1,2" -o %t %s
// RUN: FileCheck %s --check-prefix=ENABLED --input-file=%t
// RUN: ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline="mesh-shape=1,2 enable-all-reduce-workaround=false" -o %t2 %s
// RUN: FileCheck %s --check-prefix=DISABLED --input-file=%t2

// Tests for the all_reduce decomposition workaround toggle
// (enable-all-reduce-workaround / ttnn-enable-all-reduce-workaround).
//
// When enabled (default), TTNNAllReduceWorkarounds decomposes ttir.all_reduce
// into reduce_scatter + all_gather. When disabled, ttir.all_reduce lowers to
// ttnn.all_reduce and is left untouched.

module attributes {} {
  // ENABLED-LABEL: all_reduce_workaround_toggle
  // DISABLED-LABEL: all_reduce_workaround_toggle
  func.func @all_reduce_workaround_toggle(%arg0: tensor<256x256xbf16>) -> tensor<256x256xbf16> {
    %0 = "ttir.all_reduce"(%arg0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<256x256xbf16>) -> tensor<256x256xbf16>
    // Default: workaround enabled, all_reduce is decomposed.
    // ENABLED: "ttnn.reduce_scatter"
    // ENABLED: "ttnn.all_gather"
    // ENABLED-NOT: "ttnn.all_reduce"

    // Workaround disabled: all_reduce is preserved.
    // DISABLED: "ttnn.all_reduce"
    // DISABLED-NOT: "ttnn.reduce_scatter"
    // DISABLED-NOT: "ttnn.all_gather"
    return %0 : tensor<256x256xbf16>
  }
}
