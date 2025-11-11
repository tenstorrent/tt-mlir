// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t %s
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" --ttnn-adjust-deallocs -o %t2 %s
//
// Test for --ttnn-adjust-deallocs pass.
// The test runs the --ttir-to-ttnn-backend-pipeline twice, and follows up with --ttnn-adjust-deallocs for the second run.
// The first run is expected to have 4 "ttnn.deallocate" occurrences, while the second one expects 2 of them. Lines 3 and 4 check for this.

module {
  func.func @matmul_with_bias(%arg0: tensor<784x1096xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<1096x784xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<784x784xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<784x784xbf16> {
    %0 = ttir.empty() : tensor<784x784xbf16>
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<784x1096xbf16>, tensor<1096x784xbf16>, tensor<784x784xbf16>) -> tensor<784x784xbf16>
    %2 = ttir.empty() : tensor<784x784xbf16>
    %3 = "ttir.add"(%1, %arg2, %2) : (tensor<784x784xbf16>, tensor<784x784xbf16>, tensor<784x784xbf16>) -> tensor<784x784xbf16>
    return %3 : tensor<784x784xbf16>
  }
}
