// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t %s
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" --ttnn-adjust-deallocs -o %t2 %s
// RUN: [ "$(cat %t | grep -c ttnn.deallocate)" -eq 5 ]
// RUN: [ "$(cat %t2 | grep -c ttnn.deallocate)" -eq 3 ]
//
// Test for --ttnn-adjust-deallocs pass.
// The test runs the --ttir-to-ttnn-backend-pipeline twice, and follows up with --ttnn-adjust-deallocs for the second run.
// The first run is expected to have 5 "ttnn.deallocate" occurrences, while the second one expects 3 of them. Lines 3 and 4 check for this.

module {
  func.func @matmul_with_bias(%arg0: tensor<784x1096xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<1096x784xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<784x784xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<784x784xbf16> {
    %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<784x1096xbf16>, tensor<1096x784xbf16>) -> tensor<784x784xbf16>
    %1 = "ttir.add"(%0, %arg2) : (tensor<784x784xbf16>, tensor<784x784xbf16>) -> tensor<784x784xbf16>
    return %1 : tensor<784x784xbf16>
  }
}
