// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=true" %s -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

func.func @simple_fork_join_schedule_test(%arg0: tensor<4x32x32x64xbf16>, %arg1: tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16> {
    // Initial operation - creates fork point
    %0 = ttir.empty() : tensor<4x32x32x64xbf16>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>

    // Branch 1: relu → exp → log
    %2 = ttir.empty() : tensor<4x32x32x64xbf16>
    %3 = "ttir.relu"(%1, %2) : (tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>

    %4 = ttir.empty() : tensor<4x32x32x64xbf16>
    %5 = "ttir.exp"(%3, %4) : (tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>

    %6 = ttir.empty() : tensor<4x32x32x64xbf16>
    %7 = "ttir.log"(%5, %6) : (tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>

    // Branch 2: sigmoid → neg
    %8 = ttir.empty() : tensor<4x32x32x64xbf16>
    %9 = "ttir.sigmoid"(%1, %8) : (tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>

    %10 = ttir.empty() : tensor<4x32x32x64xbf16>
    %11 = "ttir.neg"(%9, %10) : (tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>

    // Join point
    %12 = ttir.empty() : tensor<4x32x32x64xbf16>
    %13 = "ttir.add"(%7, %11, %12) : (tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>

    return %13 : tensor<4x32x32x64xbf16>
}

// Initial add must come first
// CHECK: %{{.*}} = "ttnn.add"

// Test non-interleaving of branches by using CHECK-NOT to prevent opposite branch operations
// Branch 1 executes first, because it is defined first

// CHECK: %{{.*}} = "ttnn.relu"
// CHECK-NOT: ttnn.sigmoid
// CHECK-NOT: ttnn.neg
// CHECK: %{{.*}} = "ttnn.exp"
// CHECK-NOT: ttnn.sigmoid
// CHECK-NOT: ttnn.neg
// CHECK: %{{.*}} = "ttnn.log"

// Now Branch 2 can start (after Branch 1 completes)
// CHECK: %{{.*}} = "ttnn.sigmoid"
// CHECK: %{{.*}} = "ttnn.neg"

// Join in final add
// CHECK: %{{.*}} = "ttnn.add"
