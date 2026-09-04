// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-scalarize-const-tensors "--d2m-dataflow-planning=dump-plan=true" %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=PLAN
// RUN: ttmlir-opt "--d2m-fe-pipeline=enable-dataflow-planning=true" --mlir-print-ir-after=d2m-dataflow-planning %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=PIPELINE
// RUN: not --crash ttmlir-opt "--d2m-fe-pipeline=enable-dataflow-planning=true ttnn-mode=true" %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=TTNN-REJECTED
// RUN: ttmlir-opt --split-input-file "--d2m-dataflow-planning=dump-plan=true" %S/../../../Conversion/D2MToTTMetal/spatial_fabric_config_scope.mlir -o /dev/null 2>&1 | FileCheck %s --allow-empty --check-prefix=SPATIAL

#any_device = #ttcore.device<workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>

module attributes {ttcore.device = #any_device} {
  // PLAN: d2m-dataflow-plan function=@planner_skeleton scope=0 strategy=temporal-fallback generics=1
  // PLAN-NEXT: d2m-dataflow-plan function=@planner_second_function scope=0 strategy=temporal-fallback generics=1
  // PIPELINE: IR Dump After D2MDataflowPlanning (d2m-dataflow-planning)
  // PIPELINE: func.func @planner_skeleton
  // PIPELINE: d2m.generic {{.*}}grid = #ttcore.grid<1x1>
  // TTNN-REJECTED: LLVM ERROR: D2M dataflow planning currently supports only the TTMetal path
  // SPATIAL-NOT: d2m-dataflow-plan
  func.func @planner_skeleton(%arg0: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
    %0 = "ttir.relu"(%arg0) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %0 : tensor<64x64xbf16>
  }

  func.func @planner_second_function(%arg0: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
    %0 = "ttir.exp"(%arg0) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %0 : tensor<64x64xbf16>
  }
}
