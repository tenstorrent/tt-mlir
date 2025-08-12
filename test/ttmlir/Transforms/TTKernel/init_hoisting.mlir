// RUN: ttmlir-opt --loop-invariant-code-motion %s | FileCheck %s

// CHECK-LABEL func.func @hoist_single_resource_basic
func.func @hoist_single_resource_basic() attributes {} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 10 : i32
  %c2_i32 = arith.constant 1 : i32
  %c0_i32_0 = arith.constant 0 : i32
  // CHECK: ttkernel.exp_tile_init() : () -> ()
  scf.for %arg0 = %c0_i32 to %c1_i32 step %c2_i32  : i32 {
    scf.for %arg1 = %c0_i32 to %c1_i32 step %c2_i32  : i32 {
      scf.for %arg2 = %c0_i32 to %c1_i32 step %c2_i32  : i32 {
        ttkernel.exp_tile_init() : () -> ()
        ttkernel.exp_tile(%c0_i32_0) : (i32) -> ()
      }
    }
  }
  return
}

// CHECK-LABEL func.func @hoist_single_resource_basic_conflict
func.func @hoist_single_resource_basic_conflict() attributes {} {
  %0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<memref<1x4x1x1x!ttcore.tile<32x32, bf16>>>
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 10 : i32
  %c2_i32 = arith.constant 1 : i32
  %c0_i32_0 = arith.constant 0 : i32
  %c0_i32_1 = arith.constant 0 : i32
  %c0_i32_2 = arith.constant 0 : i32
  
  scf.for %arg0 = %c0_i32 to %c1_i32 step %c2_i32  : i32 {
    scf.for %arg1 = %c0_i32 to %c1_i32 step %c2_i32  : i32 {
      scf.for %arg2 = %c0_i32 to %c1_i32 step %c2_i32  : i32 {
        // CHECK: ttkernel.copy_tile_init(%0) : (!ttkernel.cb<memref<1x4x1x1x!ttcore.tile<32x32, bf16>>>) -> ()
        ttkernel.copy_tile_init(%0) : (!ttkernel.cb<memref<1x4x1x1x!ttcore.tile<32x32, bf16>>>) -> ()
        ttkernel.copy_tile(%0, %c0_i32_0, %c0_i32_1) : (!ttkernel.cb<memref<1x4x1x1x!ttcore.tile<32x32, bf16>>>, i32, i32) -> ()
        // CHECK: ttkernel.exp_tile_init() : () -> ()
        ttkernel.exp_tile_init() : () -> ()
        ttkernel.exp_tile(%c0_i32_2) : (i32) -> ()
      }
    }
  }
  return
}

// CHECK-LABEL func.func @hoist_single_resource_if_region
func.func @hoist_single_resource_if_region() attributes {} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 10 : i32
  %c2_i32 = arith.constant 1 : i32
  %c3_i32 = arith.constant 5 : i32

  scf.for %arg0 = %c0_i32 to %c1_i32 step %c2_i32  : i32 {
    scf.for %arg1 = %c0_i32 to %c1_i32 step %c2_i32  : i32 {
      scf.for %arg2 = %c0_i32 to %c1_i32 step %c2_i32  : i32 {
        %1 = arith.cmpi slt, %arg0, %c3_i32 : i32

        scf.if %1 {
          // CHECK: ttkernel.exp_tile_init() : () -> ()
          ttkernel.exp_tile_init() : () -> ()
          %c0_i32_0 = arith.constant 0 : i32
          ttkernel.exp_tile(%c0_i32_0) : (i32) -> ()
        }
      }
    }
  }
  return
}

// CHECK-LABEL func.func @hoist_single_resource_for_inside_if_region
func.func @hoist_single_resource_for_inside_if_region() attributes {} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 10 : i32
  %c2_i32 = arith.constant 1 : i32
  %c3_i32 = arith.constant 5 : i32

  scf.for %arg0 = %c0_i32 to %c1_i32 step %c2_i32  : i32 {
    %1 = arith.cmpi slt, %arg0, %c3_i32 : i32

    scf.if %1 {
      %c0_i32_0 = arith.constant 0 : i32
      // CHECK: ttkernel.exp_tile_init() : () -> ()
      scf.for %arg1 = %c0_i32 to %c1_i32 step %c2_i32  : i32 { 
        scf.for %arg2 = %c0_i32 to %c1_i32 step %c2_i32  : i32 { 
          ttkernel.exp_tile_init() : () -> ()
          ttkernel.exp_tile(%c0_i32_0) : (i32) -> ()
        }
      }
    }
  }
  return
}

// CHECK-LABEL func.func @hoist_single_resource_comprehensive
func.func @hoist_single_resource_comprehensive() attributes {} {
  %0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<memref<1x4x1x1x!ttcore.tile<32x32, bf16>>>
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 10 : i32
  %c2_i32 = arith.constant 1 : i32
  %c0_i32_2 = arith.constant 0 : i32
  %c3_i32 = arith.constant 5 : i32

  scf.for %arg0 = %c0_i32 to %c1_i32 step %c2_i32  : i32 {
    // CHECK: ttkernel.exp_tile_init() : () -> ()
    scf.for %arg1 = %c0_i32 to %c1_i32 step %c2_i32  : i32 {
      scf.for %arg2 = %c0_i32 to %c1_i32 step %c2_i32  : i32 {
        ttkernel.exp_tile_init() : () -> ()
        ttkernel.exp_tile(%c0_i32_2) : (i32) -> ()
      }
    }

    %1 = arith.cmpi slt, %arg0, %c3_i32 : i32
    scf.if %1 {
      %c0_i32_0 = arith.constant 0 : i32
      %c0_i32_1 = arith.constant 0 : i32
      // CHECK: ttkernel.copy_tile_init(%0) : (!ttkernel.cb<memref<1x4x1x1x!ttcore.tile<32x32, bf16>>>) -> ()
      scf.for %arg3 = %c0_i32 to %c1_i32 step %c2_i32  : i32 {
        scf.for %arg4 = %c0_i32 to %c1_i32 step %c2_i32  : i32 {
          ttkernel.copy_tile_init(%0) : (!ttkernel.cb<memref<1x4x1x1x!ttcore.tile<32x32, bf16>>>) -> ()
          ttkernel.copy_tile(%0, %c0_i32_0, %c0_i32_1) : (!ttkernel.cb<memref<1x4x1x1x!ttcore.tile<32x32, bf16>>>, i32, i32) -> ()
        }
      }
    }
  }
  return
}

// CHECK-LABEL func.func @hoist_multi_resource_comprehensive
func.func @hoist_multi_resource_comprehensive() attributes {} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 10 : i32
  %c2_i32 = arith.constant 1 : i32
  %c0_i32_2 = arith.constant 0 : i32
  %c3_i32 = arith.constant 5 : i32 

  scf.for %arg0 = %c0_i32 to %c1_i32 step %c2_i32  : i32 { 
    // CHECK: ttkernel.CD_test_init() : () -> () 
    // CHECK: ttkernel.EF_test_init() : () -> ()
    scf.for %arg1 = %c0_i32 to %c1_i32 step %c2_i32  : i32 {
      scf.for %arg2 = %c0_i32 to %c1_i32 step %c2_i32  : i32 { 
        ttkernel.CD_test_init() : () -> ()
        ttkernel.CD_test() : () -> ()
        ttkernel.EF_test_init() : () -> ()
        ttkernel.EF_test() : () -> ()
      }
    }

    %1 = arith.cmpi slt, %arg0, %c3_i32 : i32
    scf.if %1 {
      %c0_i32_0 = arith.constant 0 : i32
      %c0_i32_1 = arith.constant 0 : i32
      
      // CHECK: ttkernel.B_test_init() : () -> ()
      scf.for %arg3 = %c0_i32 to %c1_i32 step %c2_i32  : i32 {
        // CHECK: ttkernel.A_test_init() : () -> () 
        scf.for %arg4 = %c0_i32 to %c1_i32 step %c2_i32  : i32 {
          ttkernel.A_test_init() : () -> ()
          ttkernel.A_test() : () -> ()
        }

        scf.for %arg5 = %c0_i32 to %c1_i32 step %c2_i32  : i32 {
          ttkernel.B_test_init() : () -> ()
          ttkernel.B_test() : () -> ()
        }

        // CHECK: ttkernel.AC_test_init() : () -> ()
        scf.for %arg6 = %c0_i32 to %c1_i32 step %c2_i32  : i32 {
          ttkernel.AC_test_init() : () -> ()
          ttkernel.AC_test() : () -> ()
        }
      }
    } 
    else {
      // CHECK: ttkernel.F_test_init() : () -> ()
      ttkernel.F_test_init() : () -> ()
      ttkernel.F_test() : () -> ()
    }
  }
  return
}
