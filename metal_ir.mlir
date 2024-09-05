#l1_ = #tt.memory_space<l1>
#layout = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<64x128xf32, #l1_>>
#layout1 = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<2x4x!tt.tile<32x32, f32>, #l1_>>
module attributes {tt.device = #tt.device<workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1)[s0, s1] -> (0, d0 floordiv s0, d1 floordiv s1, (d0 mod s0) * s1 + d1 mod s1), dramMap = (d0, d1)[s0, s1] -> (0, 0, ((((d1 floordiv s1) * 8 + d0 floordiv s0) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 8192) mod 12, (((d1 floordiv s1) * 8 + d0 floordiv s0) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 98304 + (((d1 floordiv s1) * 8 + d0 floordiv s0) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) mod 8192), chipIds = [0]>, tt.system_desc = #tt.system_desc<[{arch = <wormhole_b0>, grid = 8x8, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 107360, erisc_l1_unreserved_base = 107360, dram_unreserved_base = 32, dram_unreserved_end = 1073097664, physical_cores = {worker = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 6), (1, 7), (1, 8), (1, 9), (2, 1), (2, 2), (2, 3), (2, 4), (2, 6), (2, 7), (2, 8), (2, 9), (3, 1), (3, 2), (3, 3), (3, 4), (3, 6), (3, 7), (3, 8), (3, 9), (4, 1), (4, 2), (4, 3), (4, 4), (4, 6), (4, 7), (4, 8), (4, 9), (5, 1), (5, 2), (5, 3), (5, 4), (5, 6), (5, 7), (5, 8), (5, 9), (7, 1), (7, 2), (7, 3), (7, 4), (7, 6), (7, 7), (7, 8), (7, 9), (8, 1), (8, 2), (8, 3), (8, 4), (8, 6), (8, 7), (8, 8), (8, 9), (9, 1), (9, 2), (9, 3), (9, 4), (9, 6), (9, 7), (9, 8), (9, 9)] dram = [(1, 0), (1, 5), (2, 5), (3, 5), (5, 0), (5, 5), (7, 0), (7, 5), (8, 5), (9, 5), (11, 0), (11, 5)] eth_inactive = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 6), (0, 7), (0, 8), (0, 9), (6, 2), (6, 3), (6, 6), (6, 7), (6, 8)]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>], supported_tile_sizes = [(4 x 16), (16 x 16), (32 x 16), (4 x 32), (16 x 32), (32 x 32)]}], [0], [3 : i32], [<0, 0, 0, 0>]>} {
  func.func @multiply(%arg0: tensor<64x128xf32, #layout>, %arg1: tensor<64x128xf32, #layout>) -> tensor<64x128xf32, #layout> attributes {tt.arg_alloc = [#tt.arg_alloc<107360, 32768, l1>, #tt.arg_alloc<140128, 32768, l1>]} {
    %0 = "ttmetal.alloc"() <{address = 172896 : i64, memory_space = #l1_, size = 32768 : i64}> : () -> tensor<64x128xf32, #layout1>
    %1 = "ttmetal.dispatch"(%arg0, %0) <{core_ranges = [#ttmetal.core_range<0x0, 1x1>], operandSegmentSizes = array<i32: 1, 1>, threadTypes = [#ttkernel.thread<tensix>]}> ({
    ^bb0(%arg2: !ttkernel.cb<cb_in0, 107360, memref<64x128xf32, #l1_>, 4096, 1>, %arg3: !ttkernel.cb<cb_out0, 172896, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>):
      %c4_i32 = arith.constant 4 : i32
      "ttkernel.tilize_init"(%arg2, %c4_i32, %arg3) : (!ttkernel.cb<cb_in0, 107360, memref<64x128xf32, #l1_>, 4096, 1>, i32, !ttkernel.cb<cb_out0, 172896, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>) -> ()
      "ttkernel.tilize_block"(%arg2, %c4_i32, %arg3) : (!ttkernel.cb<cb_in0, 107360, memref<64x128xf32, #l1_>, 4096, 1>, i32, !ttkernel.cb<cb_out0, 172896, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>) -> ()
      "ttkernel.cb_pop_front"(%arg2, %c4_i32) : (!ttkernel.cb<cb_in0, 107360, memref<64x128xf32, #l1_>, 4096, 1>, i32) -> ()
      "ttkernel.cb_push_back"(%arg3, %c4_i32) : (!ttkernel.cb<cb_out0, 172896, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32) -> ()
      "ttkernel.tilize_block"(%arg2, %c4_i32, %arg3) : (!ttkernel.cb<cb_in0, 107360, memref<64x128xf32, #l1_>, 4096, 1>, i32, !ttkernel.cb<cb_out0, 172896, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>) -> ()
      "ttkernel.cb_pop_front"(%arg2, %c4_i32) : (!ttkernel.cb<cb_in0, 107360, memref<64x128xf32, #l1_>, 4096, 1>, i32) -> ()
      "ttkernel.cb_push_back"(%arg3, %c4_i32) : (!ttkernel.cb<cb_out0, 172896, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32) -> ()
      "ttkernel.return"() : () -> ()
    }) : (tensor<64x128xf32, #layout>, tensor<64x128xf32, #layout1>) -> tensor<64x128xf32, #layout1>
    %2 = "ttmetal.alloc"() <{address = 205664 : i64, memory_space = #l1_, size = 32768 : i64}> : () -> tensor<64x128xf32, #layout1>
    %3 = "ttmetal.dispatch"(%arg1, %2) <{core_ranges = [#ttmetal.core_range<0x0, 1x1>], operandSegmentSizes = array<i32: 1, 1>, threadTypes = [#ttkernel.thread<tensix>]}> ({
    ^bb0(%arg2: !ttkernel.cb<cb_in0, 140128, memref<64x128xf32, #l1_>, 4096, 1>, %arg3: !ttkernel.cb<cb_out0, 205664, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>):
      %c4_i32 = arith.constant 4 : i32
      "ttkernel.tilize_init"(%arg2, %c4_i32, %arg3) : (!ttkernel.cb<cb_in0, 140128, memref<64x128xf32, #l1_>, 4096, 1>, i32, !ttkernel.cb<cb_out0, 205664, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>) -> ()
      "ttkernel.tilize_block"(%arg2, %c4_i32, %arg3) : (!ttkernel.cb<cb_in0, 140128, memref<64x128xf32, #l1_>, 4096, 1>, i32, !ttkernel.cb<cb_out0, 205664, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>) -> ()
      "ttkernel.cb_pop_front"(%arg2, %c4_i32) : (!ttkernel.cb<cb_in0, 140128, memref<64x128xf32, #l1_>, 4096, 1>, i32) -> ()
      "ttkernel.cb_push_back"(%arg3, %c4_i32) : (!ttkernel.cb<cb_out0, 205664, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32) -> ()
      "ttkernel.tilize_block"(%arg2, %c4_i32, %arg3) : (!ttkernel.cb<cb_in0, 140128, memref<64x128xf32, #l1_>, 4096, 1>, i32, !ttkernel.cb<cb_out0, 205664, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>) -> ()
      "ttkernel.cb_pop_front"(%arg2, %c4_i32) : (!ttkernel.cb<cb_in0, 140128, memref<64x128xf32, #l1_>, 4096, 1>, i32) -> ()
      "ttkernel.cb_push_back"(%arg3, %c4_i32) : (!ttkernel.cb<cb_out0, 205664, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32) -> ()
      "ttkernel.return"() : () -> ()
    }) : (tensor<64x128xf32, #layout>, tensor<64x128xf32, #layout1>) -> tensor<64x128xf32, #layout1>
    %4 = "ttmetal.alloc"() <{address = 238432 : i64, memory_space = #l1_, size = 32768 : i64}> : () -> tensor<64x128xf32, #layout1>
    %5 = "ttmetal.dispatch"(%1, %3, %4) <{core_ranges = [#ttmetal.core_range<0x0, 1x1>], operandSegmentSizes = array<i32: 2, 1>, threadTypes = [#ttkernel.thread<tensix>]}> ({
    ^bb0(%arg2: !ttkernel.cb<cb_in0, 172896, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, %arg3: !ttkernel.cb<cb_in1, 205664, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, %arg4: !ttkernel.cb<cb_out0, 238432, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>):
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %c0_i32 = arith.constant 0 : i32
      "ttkernel.binary_op_init_common"(%arg2, %arg3, %arg4) : (!ttkernel.cb<cb_in0, 172896, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, !ttkernel.cb<cb_in1, 205664, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, !ttkernel.cb<cb_out0, 238432, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>) -> ()
      "ttkernel.mul_tiles_init"(%arg2, %arg3) : (!ttkernel.cb<cb_in0, 172896, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, !ttkernel.cb<cb_in1, 205664, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>) -> ()
      %8 = scf.for %arg5 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg6 = %c0_i32) -> (i32)  : i32 {
        %9 = scf.for %arg7 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg8 = %arg6) -> (i32)  : i32 {
          "ttkernel.tile_regs_acquire"() : () -> ()
          "ttkernel.mul_tiles"(%arg2, %arg3, %arg8, %arg8, %c0_i32) : (!ttkernel.cb<cb_in0, 172896, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, !ttkernel.cb<cb_in1, 205664, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32, i32, i32) -> ()
          "ttkernel.tile_regs_commit"() : () -> ()
          "ttkernel.tile_regs_wait"() : () -> ()
          "ttkernel.pack_tile"(%c0_i32, %arg4, %arg8) : (i32, !ttkernel.cb<cb_out0, 238432, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32) -> ()
          "ttkernel.tile_regs_release"() : () -> ()
          %10 = arith.addi %arg8, %c1_i32 : i32
          scf.yield %10 : i32
        }
        scf.yield %9 : i32
      }
      "ttkernel.return"() : () -> ()
    }) : (tensor<64x128xf32, #layout1>, tensor<64x128xf32, #layout1>, tensor<64x128xf32, #layout1>) -> tensor<64x128xf32, #layout1>
    "ttmetal.dealloc"(%2) : (tensor<64x128xf32, #layout1>) -> ()
    "ttmetal.dealloc"(%0) : (tensor<64x128xf32, #layout1>) -> ()
    %6 = "ttmetal.alloc"() <{address = 271200 : i64, memory_space = #l1_, size = 32768 : i64}> : () -> tensor<64x128xf32, #layout>
    %7 = "ttmetal.dispatch"(%5, %6) <{core_ranges = [#ttmetal.core_range<0x0, 1x1>], operandSegmentSizes = array<i32: 1, 1>, threadTypes = [#ttkernel.thread<tensix>]}> ({
    ^bb0(%arg2: !ttkernel.cb<cb_in0, 238432, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, %arg3: !ttkernel.cb<cb_out0, 271200, memref<64x128xf32, #l1_>, 4096, 1>):
      %c4_i32 = arith.constant 4 : i32
      "ttkernel.untilize_init"(%arg2, %arg3) : (!ttkernel.cb<cb_in0, 238432, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, !ttkernel.cb<cb_out0, 271200, memref<64x128xf32, #l1_>, 4096, 1>) -> ()
      "ttkernel.untilize_block"(%arg2, %c4_i32, %arg3) : (!ttkernel.cb<cb_in0, 238432, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32, !ttkernel.cb<cb_out0, 271200, memref<64x128xf32, #l1_>, 4096, 1>) -> ()
      "ttkernel.cb_pop_front"(%arg2, %c4_i32) : (!ttkernel.cb<cb_in0, 238432, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32) -> ()
      "ttkernel.cb_push_back"(%arg3, %c4_i32) : (!ttkernel.cb<cb_out0, 271200, memref<64x128xf32, #l1_>, 4096, 1>, i32) -> ()
      "ttkernel.untilize_block"(%arg2, %c4_i32, %arg3) : (!ttkernel.cb<cb_in0, 238432, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32, !ttkernel.cb<cb_out0, 271200, memref<64x128xf32, #l1_>, 4096, 1>) -> ()
      "ttkernel.cb_pop_front"(%arg2, %c4_i32) : (!ttkernel.cb<cb_in0, 238432, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32) -> ()
      "ttkernel.cb_push_back"(%arg3, %c4_i32) : (!ttkernel.cb<cb_out0, 271200, memref<64x128xf32, #l1_>, 4096, 1>, i32) -> ()
      "ttkernel.return"() : () -> ()
    }) : (tensor<64x128xf32, #layout1>, tensor<64x128xf32, #layout>) -> tensor<64x128xf32, #layout>
    "ttmetal.dealloc"(%4) : (tensor<64x128xf32, #layout1>) -> ()
    return %7 : tensor<64x128xf32, #layout>
  }
  func.func @add(%arg0: tensor<64x128xf32, #layout>, %arg1: tensor<64x128xf32, #layout>) -> tensor<64x128xf32, #layout> attributes {tt.arg_alloc = [#tt.arg_alloc<107360, 32768, l1>, #tt.arg_alloc<140128, 32768, l1>]} {
    %0 = "ttmetal.alloc"() <{address = 172896 : i64, memory_space = #l1_, size = 32768 : i64}> : () -> tensor<64x128xf32, #layout1>
    %1 = "ttmetal.dispatch"(%arg0, %0) <{core_ranges = [#ttmetal.core_range<0x0, 1x1>], operandSegmentSizes = array<i32: 1, 1>, threadTypes = [#ttkernel.thread<tensix>]}> ({
    ^bb0(%arg2: !ttkernel.cb<cb_in0, 107360, memref<64x128xf32, #l1_>, 4096, 1>, %arg3: !ttkernel.cb<cb_out0, 172896, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>):
      %c4_i32 = arith.constant 4 : i32
      "ttkernel.tilize_init"(%arg2, %c4_i32, %arg3) : (!ttkernel.cb<cb_in0, 107360, memref<64x128xf32, #l1_>, 4096, 1>, i32, !ttkernel.cb<cb_out0, 172896, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>) -> ()
      "ttkernel.tilize_block"(%arg2, %c4_i32, %arg3) : (!ttkernel.cb<cb_in0, 107360, memref<64x128xf32, #l1_>, 4096, 1>, i32, !ttkernel.cb<cb_out0, 172896, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>) -> ()
      "ttkernel.cb_pop_front"(%arg2, %c4_i32) : (!ttkernel.cb<cb_in0, 107360, memref<64x128xf32, #l1_>, 4096, 1>, i32) -> ()
      "ttkernel.cb_push_back"(%arg3, %c4_i32) : (!ttkernel.cb<cb_out0, 172896, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32) -> ()
      "ttkernel.tilize_block"(%arg2, %c4_i32, %arg3) : (!ttkernel.cb<cb_in0, 107360, memref<64x128xf32, #l1_>, 4096, 1>, i32, !ttkernel.cb<cb_out0, 172896, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>) -> ()
      "ttkernel.cb_pop_front"(%arg2, %c4_i32) : (!ttkernel.cb<cb_in0, 107360, memref<64x128xf32, #l1_>, 4096, 1>, i32) -> ()
      "ttkernel.cb_push_back"(%arg3, %c4_i32) : (!ttkernel.cb<cb_out0, 172896, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32) -> ()
      "ttkernel.return"() : () -> ()
    }) : (tensor<64x128xf32, #layout>, tensor<64x128xf32, #layout1>) -> tensor<64x128xf32, #layout1>
    %2 = "ttmetal.alloc"() <{address = 205664 : i64, memory_space = #l1_, size = 32768 : i64}> : () -> tensor<64x128xf32, #layout1>
    %3 = "ttmetal.dispatch"(%arg1, %2) <{core_ranges = [#ttmetal.core_range<0x0, 1x1>], operandSegmentSizes = array<i32: 1, 1>, threadTypes = [#ttkernel.thread<tensix>]}> ({
    ^bb0(%arg2: !ttkernel.cb<cb_in0, 140128, memref<64x128xf32, #l1_>, 4096, 1>, %arg3: !ttkernel.cb<cb_out0, 205664, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>):
      %c4_i32 = arith.constant 4 : i32
      "ttkernel.tilize_init"(%arg2, %c4_i32, %arg3) : (!ttkernel.cb<cb_in0, 140128, memref<64x128xf32, #l1_>, 4096, 1>, i32, !ttkernel.cb<cb_out0, 205664, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>) -> ()
      "ttkernel.tilize_block"(%arg2, %c4_i32, %arg3) : (!ttkernel.cb<cb_in0, 140128, memref<64x128xf32, #l1_>, 4096, 1>, i32, !ttkernel.cb<cb_out0, 205664, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>) -> ()
      "ttkernel.cb_pop_front"(%arg2, %c4_i32) : (!ttkernel.cb<cb_in0, 140128, memref<64x128xf32, #l1_>, 4096, 1>, i32) -> ()
      "ttkernel.cb_push_back"(%arg3, %c4_i32) : (!ttkernel.cb<cb_out0, 205664, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32) -> ()
      "ttkernel.tilize_block"(%arg2, %c4_i32, %arg3) : (!ttkernel.cb<cb_in0, 140128, memref<64x128xf32, #l1_>, 4096, 1>, i32, !ttkernel.cb<cb_out0, 205664, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>) -> ()
      "ttkernel.cb_pop_front"(%arg2, %c4_i32) : (!ttkernel.cb<cb_in0, 140128, memref<64x128xf32, #l1_>, 4096, 1>, i32) -> ()
      "ttkernel.cb_push_back"(%arg3, %c4_i32) : (!ttkernel.cb<cb_out0, 205664, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32) -> ()
      "ttkernel.return"() : () -> ()
    }) : (tensor<64x128xf32, #layout>, tensor<64x128xf32, #layout1>) -> tensor<64x128xf32, #layout1>
    %4 = "ttmetal.alloc"() <{address = 238432 : i64, memory_space = #l1_, size = 32768 : i64}> : () -> tensor<64x128xf32, #layout1>
    %5 = "ttmetal.dispatch"(%1, %3, %4) <{core_ranges = [#ttmetal.core_range<0x0, 1x1>], operandSegmentSizes = array<i32: 2, 1>, threadTypes = [#ttkernel.thread<tensix>]}> ({
    ^bb0(%arg2: !ttkernel.cb<cb_in0, 172896, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, %arg3: !ttkernel.cb<cb_in1, 205664, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, %arg4: !ttkernel.cb<cb_out0, 238432, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>):
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %c0_i32 = arith.constant 0 : i32
      "ttkernel.binary_op_init_common"(%arg2, %arg3, %arg4) : (!ttkernel.cb<cb_in0, 172896, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, !ttkernel.cb<cb_in1, 205664, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, !ttkernel.cb<cb_out0, 238432, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>) -> ()
      "ttkernel.add_tiles_init"(%arg2, %arg3) : (!ttkernel.cb<cb_in0, 172896, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, !ttkernel.cb<cb_in1, 205664, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>) -> ()
      %8 = scf.for %arg5 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg6 = %c0_i32) -> (i32)  : i32 {
        %9 = scf.for %arg7 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg8 = %arg6) -> (i32)  : i32 {
          "ttkernel.tile_regs_acquire"() : () -> ()
          "ttkernel.add_tiles"(%arg2, %arg3, %arg8, %arg8, %c0_i32) : (!ttkernel.cb<cb_in0, 172896, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, !ttkernel.cb<cb_in1, 205664, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32, i32, i32) -> ()
          "ttkernel.tile_regs_commit"() : () -> ()
          "ttkernel.tile_regs_wait"() : () -> ()
          "ttkernel.pack_tile"(%c0_i32, %arg4, %arg8) : (i32, !ttkernel.cb<cb_out0, 238432, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32) -> ()
          "ttkernel.tile_regs_release"() : () -> ()
          %10 = arith.addi %arg8, %c1_i32 : i32
          scf.yield %10 : i32
        }
        scf.yield %9 : i32
      }
      "ttkernel.return"() : () -> ()
    }) : (tensor<64x128xf32, #layout1>, tensor<64x128xf32, #layout1>, tensor<64x128xf32, #layout1>) -> tensor<64x128xf32, #layout1>
    "ttmetal.dealloc"(%2) : (tensor<64x128xf32, #layout1>) -> ()
    "ttmetal.dealloc"(%0) : (tensor<64x128xf32, #layout1>) -> ()
    %6 = "ttmetal.alloc"() <{address = 271200 : i64, memory_space = #l1_, size = 32768 : i64}> : () -> tensor<64x128xf32, #layout>
    %7 = "ttmetal.dispatch"(%5, %6) <{core_ranges = [#ttmetal.core_range<0x0, 1x1>], operandSegmentSizes = array<i32: 1, 1>, threadTypes = [#ttkernel.thread<tensix>]}> ({
    ^bb0(%arg2: !ttkernel.cb<cb_in0, 238432, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, %arg3: !ttkernel.cb<cb_out0, 271200, memref<64x128xf32, #l1_>, 4096, 1>):
      %c4_i32 = arith.constant 4 : i32
      "ttkernel.untilize_init"(%arg2, %arg3) : (!ttkernel.cb<cb_in0, 238432, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, !ttkernel.cb<cb_out0, 271200, memref<64x128xf32, #l1_>, 4096, 1>) -> ()
      "ttkernel.untilize_block"(%arg2, %c4_i32, %arg3) : (!ttkernel.cb<cb_in0, 238432, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32, !ttkernel.cb<cb_out0, 271200, memref<64x128xf32, #l1_>, 4096, 1>) -> ()
      "ttkernel.cb_pop_front"(%arg2, %c4_i32) : (!ttkernel.cb<cb_in0, 238432, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32) -> ()
      "ttkernel.cb_push_back"(%arg3, %c4_i32) : (!ttkernel.cb<cb_out0, 271200, memref<64x128xf32, #l1_>, 4096, 1>, i32) -> ()
      "ttkernel.untilize_block"(%arg2, %c4_i32, %arg3) : (!ttkernel.cb<cb_in0, 238432, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32, !ttkernel.cb<cb_out0, 271200, memref<64x128xf32, #l1_>, 4096, 1>) -> ()
      "ttkernel.cb_pop_front"(%arg2, %c4_i32) : (!ttkernel.cb<cb_in0, 238432, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32) -> ()
      "ttkernel.cb_push_back"(%arg3, %c4_i32) : (!ttkernel.cb<cb_out0, 271200, memref<64x128xf32, #l1_>, 4096, 1>, i32) -> ()
      "ttkernel.return"() : () -> ()
    }) : (tensor<64x128xf32, #layout1>, tensor<64x128xf32, #layout>) -> tensor<64x128xf32, #layout>
    "ttmetal.dealloc"(%4) : (tensor<64x128xf32, #layout1>) -> ()
    return %7 : tensor<64x128xf32, #layout>
  }
}

