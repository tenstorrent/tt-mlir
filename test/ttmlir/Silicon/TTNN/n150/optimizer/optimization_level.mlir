// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=0" %s -o %t_level0.mlir
// RUN: FileCheck %s --input-file=%t_level0.mlir --check-prefix=CHECKLEVEL0
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=1" %s -o %t_level1.mlir
// RUN: FileCheck %s --input-file=%t_level1.mlir --check-prefix=CHECKLEVEL1
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=2" %s -o %t_level2.mlir
// RUN: FileCheck %s --input-file=%t_level2.mlir --check-prefix=CHECKLEVEL2
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=2 memory-layout-analysis-enabled=false" %s -o %t_override.mlir
// RUN: FileCheck %s --input-file=%t_override.mlir --check-prefix=CHECKOVERRIDE

// Test optimization_level option with conv2d to verify:
// - Level 0: No optimizer (DRAM interleaved, no prepare_conv2d_weights)
// - Level 1: Optimizer enabled (DRAM interleaved, WITH prepare_conv2d_weights)
// - Level 2: Optimizer + memory layout analysis (sharded, WITH prepare_conv2d_weights)
// - Override: Level 2 with explicit memory-layout-analysis-enabled=false

func.func @conv2d_opt_levels(%arg0: tensor<16x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<16x32x32x64xbf16> {
    // CHECKLEVEL0-DAG: #[[LAYOUT:.*]] = #ttnn.ttnn_layout{{.*}}#dram{{.*}}interleaved{{.*}}
    // CHECKLEVEL0-NOT: ttnn.prepare_conv2d_weights
    // CHECKLEVEL0: "ttnn.conv2d"{{.*}}-> {{.*}}#[[LAYOUT]]{{.*}}

    // CHECKLEVEL1-DAG: #[[LAYOUT:.*]] = #ttnn.ttnn_layout{{.*}}#dram{{.*}}interleaved{{.*}}
    // CHECKLEVEL1: "ttnn.prepare_conv2d_weights"
    // CHECKLEVEL1: "ttnn.conv2d"{{.*}}-> {{.*}}#[[LAYOUT]]{{.*}}
    // CHECKLEVEL1-NOT: sharded

    // CHECKLEVEL2-DAG: #[[LAYOUT:.*]] = #ttnn.ttnn_layout{{.*}}sharded{{.*}}
    // CHECKLEVEL2: "ttnn.prepare_conv2d_weights"
    // CHECKLEVEL2: = "ttnn.conv2d"{{.*}}#[[LAYOUT]]{{.*}} -> {{.*}}#[[LAYOUT]]{{.*}}

    // CHECKOVERRIDE-DAG: #[[LAYOUT:.*]] = #ttnn.ttnn_layout{{.*}}#dram{{.*}}interleaved{{.*}}
    // CHECKOVERRIDE: "ttnn.prepare_conv2d_weights"
    // CHECKOVERRIDE: "ttnn.conv2d"{{.*}}-> {{.*}}#[[LAYOUT]]{{.*}}
    // CHECKOVERRIDE-NOT: sharded

    %0 = ttir.empty() : tensor<16x32x32x64xbf16>
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 1: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<16x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<16x32x32x64xbf16>) -> tensor<16x32x32x64xbf16>
    return %1 : tensor<16x32x32x64xbf16>
}
