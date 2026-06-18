// RUN: ttmlir-opt --ttkernel-dedup-inits -o %t %s
// RUN: FileCheck %s --input-file=%t

!cb0 = !ttkernel.cb<8, !ttcore.tile<32x32, f32>>
!cb1 = !ttkernel.cb<8, !ttcore.tile<32x32, f32>>
!cb2 = !ttkernel.cb<8, !ttcore.tile<32x32, f32>>

module {
  // R1: a run of identical adjacent inits collapses to one. The duplicate
  // compute_kernel_hw_startup / init_sfpu calls the per-op lowering hoists to
  // the top of the kernel are exactly this shape.
  // CHECK-LABEL: func.func @dedup_adjacent_inits
  func.func @dedup_adjacent_inits() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %in = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0
    %scale = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> !cb1
    %out = "ttkernel.get_compile_time_arg_val"() <{arg_index = 2 : i32}> : () -> !cb2
    // CHECK: ttkernel.compute_kernel_hw_startup
    // CHECK-NOT: ttkernel.compute_kernel_hw_startup
    "ttkernel.compute_kernel_hw_startup"(%in, %scale, %out) : (!cb0, !cb1, !cb2) -> ()
    "ttkernel.compute_kernel_hw_startup"(%in, %scale, %out) : (!cb0, !cb1, !cb2) -> ()
    "ttkernel.compute_kernel_hw_startup"(%in, %scale, %out) : (!cb0, !cb1, !cb2) -> ()
    // CHECK: ttkernel.init_sfpu
    // CHECK-NOT: ttkernel.init_sfpu
    "ttkernel.init_sfpu"(%in, %out) : (!cb0, !cb2) -> ()
    "ttkernel.init_sfpu"(%in, %out) : (!cb0, !cb2) -> ()
    // CHECK: return
    return
  }

  // R2: same-config reduces emitted as init;tile;uninit;init;tile;uninit;...
  // collapse to init-once / reduce-many / uninit-once.
  // CHECK-LABEL: func.func @coalesce_reduce
  func.func @coalesce_reduce() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %in = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0
    %scale = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> !cb1
    %out = "ttkernel.get_compile_time_arg_val"() <{arg_index = 2 : i32}> : () -> !cb2
    %t0 = arith.constant 0 : i32
    %t1 = arith.constant 1 : i32
    %t2 = arith.constant 2 : i32
    %dst = arith.constant 0 : i32
    // CHECK: ttkernel.reduce_init
    // CHECK-NOT: ttkernel.reduce_init
    // CHECK-COUNT-3: ttkernel.reduce_tile
    // CHECK: ttkernel.reduce_uninit
    // CHECK-NOT: ttkernel.reduce_uninit
    // CHECK-NOT: ttkernel.reduce_init
    "ttkernel.reduce_init"(%in, %scale, %out) <{reduce_dim = #ttkernel.reduce_dim<reduce_dim_row>, reduce_type = #ttkernel.reduce_type<reduce_max>}> : (!cb0, !cb1, !cb2) -> ()
    "ttkernel.reduce_tile"(%in, %scale, %t0, %t0, %dst) <{reduce_dim = #ttkernel.reduce_dim<reduce_dim_row>, reduce_type = #ttkernel.reduce_type<reduce_max>}> : (!cb0, !cb1, i32, i32, i32) -> ()
    "ttkernel.reduce_uninit"() : () -> ()
    "ttkernel.reduce_init"(%in, %scale, %out) <{reduce_dim = #ttkernel.reduce_dim<reduce_dim_row>, reduce_type = #ttkernel.reduce_type<reduce_max>}> : (!cb0, !cb1, !cb2) -> ()
    "ttkernel.reduce_tile"(%in, %scale, %t1, %t0, %dst) <{reduce_dim = #ttkernel.reduce_dim<reduce_dim_row>, reduce_type = #ttkernel.reduce_type<reduce_max>}> : (!cb0, !cb1, i32, i32, i32) -> ()
    "ttkernel.reduce_uninit"() : () -> ()
    "ttkernel.reduce_init"(%in, %scale, %out) <{reduce_dim = #ttkernel.reduce_dim<reduce_dim_row>, reduce_type = #ttkernel.reduce_type<reduce_max>}> : (!cb0, !cb1, !cb2) -> ()
    "ttkernel.reduce_tile"(%in, %scale, %t2, %t0, %dst) <{reduce_dim = #ttkernel.reduce_dim<reduce_dim_row>, reduce_type = #ttkernel.reduce_type<reduce_max>}> : (!cb0, !cb1, i32, i32, i32) -> ()
    "ttkernel.reduce_uninit"() : () -> ()
    return
  }

  // R2 with interspersed index math: the lowering emits a reduce_init /
  // reduce_tile / reduce_uninit triplet per tile, with arith index ops between
  // them (the real softmax shape). Index math is transparent to the reduce
  // config, so the run still collapses to init-once / reduce-many / uninit-once.
  // CHECK-LABEL: func.func @coalesce_reduce_with_arith
  func.func @coalesce_reduce_with_arith() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %in = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0
    %scale = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> !cb1
    %out = "ttkernel.get_compile_time_arg_val"() <{arg_index = 2 : i32}> : () -> !cb2
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %dst = arith.constant 0 : i32
    // CHECK: ttkernel.reduce_init
    // CHECK-NOT: ttkernel.reduce_init
    // CHECK-COUNT-3: ttkernel.reduce_tile
    // CHECK: ttkernel.reduce_uninit
    // CHECK-NOT: ttkernel.reduce_uninit
    // CHECK-NOT: ttkernel.reduce_init
    "ttkernel.reduce_init"(%in, %scale, %out) <{reduce_dim = #ttkernel.reduce_dim<reduce_dim_row>, reduce_type = #ttkernel.reduce_type<reduce_max>}> : (!cb0, !cb1, !cb2) -> ()
    "ttkernel.reduce_tile"(%in, %scale, %c0, %c0, %dst) <{reduce_dim = #ttkernel.reduce_dim<reduce_dim_row>, reduce_type = #ttkernel.reduce_type<reduce_max>}> : (!cb0, !cb1, i32, i32, i32) -> ()
    "ttkernel.reduce_uninit"() : () -> ()
    %i1 = arith.addi %c1, %c1 : i32
    "ttkernel.reduce_init"(%in, %scale, %out) <{reduce_dim = #ttkernel.reduce_dim<reduce_dim_row>, reduce_type = #ttkernel.reduce_type<reduce_max>}> : (!cb0, !cb1, !cb2) -> ()
    "ttkernel.reduce_tile"(%in, %scale, %i1, %c0, %dst) <{reduce_dim = #ttkernel.reduce_dim<reduce_dim_row>, reduce_type = #ttkernel.reduce_type<reduce_max>}> : (!cb0, !cb1, i32, i32, i32) -> ()
    "ttkernel.reduce_uninit"() : () -> ()
    %i2 = arith.addi %c2, %c1 : i32
    "ttkernel.reduce_init"(%in, %scale, %out) <{reduce_dim = #ttkernel.reduce_dim<reduce_dim_row>, reduce_type = #ttkernel.reduce_type<reduce_max>}> : (!cb0, !cb1, !cb2) -> ()
    "ttkernel.reduce_tile"(%in, %scale, %i2, %c0, %dst) <{reduce_dim = #ttkernel.reduce_dim<reduce_dim_row>, reduce_type = #ttkernel.reduce_type<reduce_max>}> : (!cb0, !cb1, i32, i32, i32) -> ()
    "ttkernel.reduce_uninit"() : () -> ()
    "ttkernel.tile_regs_commit"() : () -> ()
    return
  }

  // Negative: a reduce_uninit that is followed by a reduce_tile (before any
  // matching reinit) is load-bearing - that tile ran under the post-uninit
  // reset config - so the uninit must NOT be coalesced away even though a
  // same-config reduce_init follows later. Nothing is erased here.
  // CHECK-LABEL: func.func @keep_uninit_before_tile
  func.func @keep_uninit_before_tile() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %in = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0
    %scale = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> !cb1
    %out = "ttkernel.get_compile_time_arg_val"() <{arg_index = 2 : i32}> : () -> !cb2
    %t0 = arith.constant 0 : i32
    %dst = arith.constant 0 : i32
    // Full sequence is preserved, in order (the middle uninit is load-bearing).
    // CHECK: ttkernel.reduce_init
    // CHECK: ttkernel.reduce_tile
    // CHECK: ttkernel.reduce_uninit
    // CHECK: ttkernel.reduce_tile
    // CHECK: ttkernel.reduce_init
    // CHECK: ttkernel.reduce_tile
    // CHECK: ttkernel.reduce_uninit
    "ttkernel.reduce_init"(%in, %scale, %out) <{reduce_dim = #ttkernel.reduce_dim<reduce_dim_row>, reduce_type = #ttkernel.reduce_type<reduce_max>}> : (!cb0, !cb1, !cb2) -> ()
    "ttkernel.reduce_tile"(%in, %scale, %t0, %t0, %dst) <{reduce_dim = #ttkernel.reduce_dim<reduce_dim_row>, reduce_type = #ttkernel.reduce_type<reduce_max>}> : (!cb0, !cb1, i32, i32, i32) -> ()
    "ttkernel.reduce_uninit"() : () -> ()
    "ttkernel.reduce_tile"(%in, %scale, %t0, %t0, %dst) <{reduce_dim = #ttkernel.reduce_dim<reduce_dim_row>, reduce_type = #ttkernel.reduce_type<reduce_max>}> : (!cb0, !cb1, i32, i32, i32) -> ()
    "ttkernel.reduce_init"(%in, %scale, %out) <{reduce_dim = #ttkernel.reduce_dim<reduce_dim_row>, reduce_type = #ttkernel.reduce_type<reduce_max>}> : (!cb0, !cb1, !cb2) -> ()
    "ttkernel.reduce_tile"(%in, %scale, %t0, %t0, %dst) <{reduce_dim = #ttkernel.reduce_dim<reduce_dim_row>, reduce_type = #ttkernel.reduce_type<reduce_max>}> : (!cb0, !cb1, i32, i32, i32) -> ()
    "ttkernel.reduce_uninit"() : () -> ()
    "ttkernel.tile_regs_commit"() : () -> ()
    return
  }

  // Negative: distinct adjacent inits must be kept, and a reduce of a DIFFERENT
  // config across the uninit must NOT be coalesced.
  // CHECK-LABEL: func.func @keep_distinct
  func.func @keep_distinct() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %in = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0
    %scale = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> !cb1
    %out = "ttkernel.get_compile_time_arg_val"() <{arg_index = 2 : i32}> : () -> !cb2
    %t0 = arith.constant 0 : i32
    %dst = arith.constant 0 : i32
    // Two different inits, adjacent: both survive.
    // CHECK: ttkernel.compute_kernel_hw_startup
    // CHECK: ttkernel.init_sfpu
    "ttkernel.compute_kernel_hw_startup"(%in, %scale, %out) : (!cb0, !cb1, !cb2) -> ()
    "ttkernel.init_sfpu"(%in, %out) : (!cb0, !cb2) -> ()
    // MAX then SUM across the uninit: distinct config, not coalesced -> both
    // reduce_inits and both reduce_uninits survive, in order.
    // CHECK: ttkernel.reduce_init
    // CHECK: ttkernel.reduce_uninit
    // CHECK: ttkernel.reduce_init
    // CHECK: ttkernel.reduce_uninit
    "ttkernel.reduce_init"(%in, %scale, %out) <{reduce_dim = #ttkernel.reduce_dim<reduce_dim_row>, reduce_type = #ttkernel.reduce_type<reduce_max>}> : (!cb0, !cb1, !cb2) -> ()
    "ttkernel.reduce_tile"(%in, %scale, %t0, %t0, %dst) <{reduce_dim = #ttkernel.reduce_dim<reduce_dim_row>, reduce_type = #ttkernel.reduce_type<reduce_max>}> : (!cb0, !cb1, i32, i32, i32) -> ()
    "ttkernel.reduce_uninit"() : () -> ()
    "ttkernel.reduce_init"(%in, %scale, %out) <{reduce_dim = #ttkernel.reduce_dim<reduce_dim_row>, reduce_type = #ttkernel.reduce_type<reduce_sum>}> : (!cb0, !cb1, !cb2) -> ()
    "ttkernel.reduce_tile"(%in, %scale, %t0, %t0, %dst) <{reduce_dim = #ttkernel.reduce_dim<reduce_dim_row>, reduce_type = #ttkernel.reduce_type<reduce_sum>}> : (!cb0, !cb1, i32, i32, i32) -> ()
    "ttkernel.reduce_uninit"() : () -> ()
    return
  }
}
