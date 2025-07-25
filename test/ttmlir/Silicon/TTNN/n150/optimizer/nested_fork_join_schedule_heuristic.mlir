// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=true" %s -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

func.func @complex_nested_fork_join_schedule_test(%arg0: tensor<4x32x32x64xbf16>, %arg1: tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16> {
    // Initial operation - creates the first fork point
    %0 = ttir.empty() : tensor<4x32x32x64xbf16>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>

    // ========== MAJOR BRANCH A: Complex nested processing ==========

    // Branch A.1: relu → exp → (nested fork) → log
    %2 = ttir.empty() : tensor<4x32x32x64xbf16>
    %3 = "ttir.relu"(%1, %2) : (tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>

    %4 = ttir.empty() : tensor<4x32x32x64xbf16>
    %5 = "ttir.exp"(%3, %4) : (tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>

    // NESTED FORK within Branch A.1: exp result goes to TWO sub-branches

    // Sub-branch A.1.a: sin → cos → tanh
    %6 = ttir.empty() : tensor<4x32x32x64xbf16>
    %7 = "ttir.sin"(%5, %6) : (tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>

    %8 = ttir.empty() : tensor<4x32x32x64xbf16>
    %9 = "ttir.cos"(%7, %8) : (tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>

    %10 = ttir.empty() : tensor<4x32x32x64xbf16>
    %11 = "ttir.tanh"(%9, %10) : (tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>

    // Sub-branch A.1.b: abs → sqrt → reciprocal
    %12 = ttir.empty() : tensor<4x32x32x64xbf16>
    %13 = "ttir.abs"(%5, %12) : (tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>

    %14 = ttir.empty() : tensor<4x32x32x64xbf16>
    %15 = "ttir.sqrt"(%13, %14) : (tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>

    %16 = ttir.empty() : tensor<4x32x32x64xbf16>
    %17 = "ttir.reciprocal"(%15, %16) : (tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>

    // NESTED JOIN: Combine the two sub-branches with multiply
    %18 = ttir.empty() : tensor<4x32x32x64xbf16>
    %19 = "ttir.multiply"(%11, %17, %18) : (tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>

    // Continue Branch A.1: log of the nested result
    %20 = ttir.empty() : tensor<4x32x32x64xbf16>
    %21 = "ttir.log"(%19, %20) : (tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>

    // Branch A.2: sigmoid → (another nested fork)
    %22 = ttir.empty() : tensor<4x32x32x64xbf16>
    %23 = "ttir.sigmoid"(%1, %22) : (tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>

    // ANOTHER NESTED FORK from sigmoid result

    // Sub-branch A.2.a: neg → cbrt
    %24 = ttir.empty() : tensor<4x32x32x64xbf16>
    %25 = "ttir.neg"(%23, %24) : (tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>

    %26 = ttir.empty() : tensor<4x32x32x64xbf16>
    %27 = "ttir.cbrt"(%25, %26) : (tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>

    // Sub-branch A.2.b: Just a simple copy (gelu)
    %28 = ttir.empty() : tensor<4x32x32x64xbf16>
    %29 = "ttir.gelu"(%23, %28) : (tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>

    // NESTED JOIN A.2: subtract
    %30 = ttir.empty() : tensor<4x32x32x64xbf16>
    %31 = "ttir.subtract"(%27, %29, %30) : (tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>

    // MAJOR JOIN A: Combine Branch A.1 and A.2
    %32 = ttir.empty() : tensor<4x32x32x64xbf16>
    %33 = "ttir.add"(%21, %31, %32) : (tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>

    // ========== MAJOR BRANCH B: Simpler but still forked ==========

    // Branch B.1: Simple chain
    %34 = ttir.empty() : tensor<4x32x32x64xbf16>
    %35 = "ttir.floor"(%1, %34) : (tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>

    %36 = ttir.empty() : tensor<4x32x32x64xbf16>
    %37 = "ttir.ceil"(%35, %36) : (tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>

    // Branch B.2: Another simple chain
    %38 = ttir.empty() : tensor<4x32x32x64xbf16>
    %39 = "ttir.rsqrt"(%1, %38) : (tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>

    %40 = ttir.empty() : tensor<4x32x32x64xbf16>
    %41 = "ttir.log"(%39, %40) : (tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>

    // Branch B.3: Triple nested madness!
    %42 = ttir.empty() : tensor<4x32x32x64xbf16>
    %43 = "ttir.log1p"(%1, %42) : (tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>

    // Triple nested fork from log1p
    // Sub-sub-branch B.3.a.i
    %44 = ttir.empty() : tensor<4x32x32x64xbf16>
    %45 = "ttir.sign"(%43, %44) : (tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>

    // Sub-sub-branch B.3.a.ii
    %46 = ttir.empty() : tensor<4x32x32x64xbf16>
    %47 = "ttir.logical_not"(%43, %46) : (tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>

    // Triple nested join
    %48 = ttir.empty() : tensor<4x32x32x64xbf16>
    %49 = "ttir.maximum"(%45, %47, %48) : (tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>

    // MAJOR JOIN B: Combine all B branches (3-way join!)
    %50 = ttir.empty() : tensor<4x32x32x64xbf16>
    %51 = "ttir.add"(%37, %41, %50) : (tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>

    %52 = ttir.empty() : tensor<4x32x32x64xbf16>
    %53 = "ttir.multiply"(%51, %49, %52) : (tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>

    // ========== FINAL MEGA JOIN: Combine Major Branches A and B ==========
    %54 = ttir.empty() : tensor<4x32x32x64xbf16>
    %55 = "ttir.subtract"(%33, %53, %54) : (tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>

    return %55 : tensor<4x32x32x64xbf16>
}

// Initial operation
// CHECK: %{{.*}} = "ttnn.add"

// Branch A.1 start
// CHECK: %{{.*}} = "ttnn.relu"
// CHECK: %{{.*}} = "ttnn.exp"

// Nested sub-branches within A.1 must complete before any Branch B operations
// Sub-branch A.1.a: sin → cos → tanh
// CHECK: %{{.*}} = "ttnn.sin"
// CHECK: %{{.*}} = "ttnn.cos"
// CHECK: %{{.*}} = "ttnn.tanh"

// Sub-branch A.1.b: abs → sqrt → reciprocal
// CHECK: %{{.*}} = "ttnn.abs"
// CHECK: %{{.*}} = "ttnn.sqrt"
// CHECK: %{{.*}} = "ttnn.reciprocal"

// Nested join and continuation of A.1
// CHECK: %{{.*}} = "ttnn.multiply"
// CHECK: %{{.*}} = "ttnn.log"

// Branch A.2 with its nested structure
// CHECK: %{{.*}} = "ttnn.sigmoid"

// NOW A.2.b (gelu) will happen before A.2.a (neg → cbrt)
// should be because of hasBlockedSuccessor heuristic
// CHECK: %{{.*}} = "ttnn.gelu"
// CHECK: %{{.*}} = "ttnn.neg"
// CHECK: %{{.*}} = "ttnn.cbrt"

// CHECK: %{{.*}} = "ttnn.subtract"

// Major join A (no Branch B operations yet)
// CHECK-NOT: ttnn.floor
// CHECK-NOT: ttnn.ceil
// CHECK-NOT: ttnn.rsqrt
// CHECK-NOT: ttnn.log
// CHECK-NOT: ttnn.log1p
// CHECK-NOT: ttnn.sign
// CHECK-NOT: ttnn.logical_not
// CHECK-NOT: ttnn.maximum
// CHECK: %{{.*}} = "ttnn.add"

// NOW Major Branch B can start (after entire Branch A completes)
// CHECK: %{{.*}} = "ttnn.floor"
// CHECK: %{{.*}} = "ttnn.ceil"
// CHECK: %{{.*}} = "ttnn.rsqrt"
// CHECK: %{{.*}} = "ttnn.log"

// NOW join B.1 and B.2 will happen before B.3 starts
// should be because of hasBlockedSuccessor heuristic
// CHECK: %{{.*}} = "ttnn.add"

// CHECK: %{{.*}} = "ttnn.log1p"
// CHECK: %{{.*}} = "ttnn.sign"
// CHECK: %{{.*}} = "ttnn.logical_not"
// CHECK: %{{.*}} = "ttnn.maximum"

// Major joins B
// CHECK: %{{.*}} = "ttnn.multiply"

// Final mega join
// CHECK: %{{.*}} = "ttnn.subtract"
