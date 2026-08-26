#dram = #ttnn.buffer_type<dram>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 105536, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 2968640, dram_unreserved_end = 1073108608, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 0), (4, 0), (1, 2), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 4), (4, 4), (3, 6)], dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 0), (4, 0), (1, 2), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 4), (4, 4), (3, 6)]}], [0], [1 : i32], [ 0x0x0x0]>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4008x64x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x64x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<16x64x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<256x64x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x256x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout5 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x64x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout6 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout7 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x512xsi32, #dram>, <interleaved>>
#ttnn_layout8 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 511 + d1, d2), <1x1>, memref<511x128256xbf16, #dram>, <interleaved>>
#ttnn_layout9 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 511 + d1, d2), <1x1>, memref<511x1xf32, #dram>, <interleaved>>
#ttnn_layout10 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout11 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x16x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout12 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 512 + d1, d2), <1x1>, memref<16x4008x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout13 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 512 + d1, d2), <1x1>, memref<16x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout14 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 512 + d1, d2), <1x1>, memref<16x64x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout15 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 512 + d1, d2), <1x1>, memref<16x64x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout16 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 512 + d2, d3), <1x1>, memref<16x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout17 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 512 + d2, d3), <1x1>, memref<512x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout18 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <1x1>, memref<64x16x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout19 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 512 + d2, d3), <1x1>, memref<512x16x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout20 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 512 + d1, d2), <1x1>, memref<16x256x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout21 = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 32 + d1 * 32 + d2 * 32 + d3, d4), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout22 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout23 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout24 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout25 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout26 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x512xui32, #dram>, <interleaved>>
#ttnn_layout27 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128256x2048xbf16, #dram>, <interleaved>>
#ttnn_layout28 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x16x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout29 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout30 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x16x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout31 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 512 + d2, d3), <1x1>, memref<16x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout32 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x16x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout33 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 512 + d2, d3), <1x1>, memref<16x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout34 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout35 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x16x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout36 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout37 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x16x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
#ttnn_layout38 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x16x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
#ttnn_layout39 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x16x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout40 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 512 + d1, d2), <1x1>, memref<16x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout41 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 512 + d1, d2), <1x1>, memref<16x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout42 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<96x64x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout43 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 512 + d1, d2), <1x1>, memref<16x96x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout44 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 512 + d1, d2), <1x1>, memref<16x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout45 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 32 + d2, d3), <1x1>, memref<512x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout46 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 512 + d2, d3), <1x1>, memref<512x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout47 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 4096 + d1 * 512 + d2, d3), <1x1>, memref<128x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout48 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 512 + d2, d3), <1x1>, memref<512x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout49 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 4096 + d1 * 512 + d2, d3), <1x1>, memref<128x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout50 = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 4096 + d1 * 512 + d2 * 512 + d3, d4), <1x1>, memref<128x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout51 = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 4096 + d1 * 512 + d2 * 512 + d3, d4), <1x1>, memref<128x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout52 = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 16384 + d1 * 2048 + d2 * 512 + d3, d4), <1x1>, memref<512x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout53 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 512 + d2, d3), <1x1>, memref<16x16x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout54 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 512 + d1, d2), <1x1>, memref<16x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module @llama_3_2_1b_512_fwd attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  ttcore.device_module {
    builtin.module @llama_3_2_1b_512_fwd attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>, ttcore.system_desc = #system_desc} {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = 1x1, chipIds = [0], meshTopology = [linear, linear]>
      func.func @main(%arg0: tensor<128256x2048xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128256x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.embed_tokens.weight"}, %arg1: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.0.self_attn.q_proj.weight"}, %arg2: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.0.self_attn.k_proj.weight"}, %arg3: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.0.self_attn.v_proj.weight"}, %arg4: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.0.self_attn.o_proj.weight"}, %arg5: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.0.mlp.gate_proj.weight"}, %arg6: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.0.mlp.up_proj.weight"}, %arg7: tensor<2048x8192xbf16, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.0.mlp.down_proj.weight"}, %arg8: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.0.input_layernorm.weight"}, %arg9: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.0.post_attention_layernorm.weight"}, %arg10: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.1.self_attn.q_proj.weight"}, %arg11: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.1.self_attn.k_proj.weight"}, %arg12: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.1.self_attn.v_proj.weight"}, %arg13: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.1.self_attn.o_proj.weight"}, %arg14: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.1.mlp.gate_proj.weight"}, %arg15: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.1.mlp.up_proj.weight"}, %arg16: tensor<2048x8192xbf16, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.1.mlp.down_proj.weight"}, %arg17: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.1.input_layernorm.weight"}, %arg18: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.1.post_attention_layernorm.weight"}, %arg19: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.2.self_attn.q_proj.weight"}, %arg20: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.2.self_attn.k_proj.weight"}, %arg21: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.2.self_attn.v_proj.weight"}, %arg22: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.2.self_attn.o_proj.weight"}, %arg23: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.2.mlp.gate_proj.weight"}, %arg24: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.2.mlp.up_proj.weight"}, %arg25: tensor<2048x8192xbf16, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.2.mlp.down_proj.weight"}, %arg26: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.2.input_layernorm.weight"}, %arg27: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.2.post_attention_layernorm.weight"}, %arg28: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.3.self_attn.q_proj.weight"}, %arg29: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.3.self_attn.k_proj.weight"}, %arg30: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.3.self_attn.v_proj.weight"}, %arg31: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.3.self_attn.o_proj.weight"}, %arg32: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.3.mlp.gate_proj.weight"}, %arg33: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.3.mlp.up_proj.weight"}, %arg34: tensor<2048x8192xbf16, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.3.mlp.down_proj.weight"}, %arg35: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.3.input_layernorm.weight"}, %arg36: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.3.post_attention_layernorm.weight"}, %arg37: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.4.self_attn.q_proj.weight"}, %arg38: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.4.self_attn.k_proj.weight"}, %arg39: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.4.self_attn.v_proj.weight"}, %arg40: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.4.self_attn.o_proj.weight"}, %arg41: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.4.mlp.gate_proj.weight"}, %arg42: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.4.mlp.up_proj.weight"}, %arg43: tensor<2048x8192xbf16, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.4.mlp.down_proj.weight"}, %arg44: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.4.input_layernorm.weight"}, %arg45: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.4.post_attention_layernorm.weight"}, %arg46: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.5.self_attn.q_proj.weight"}, %arg47: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.5.self_attn.k_proj.weight"}, %arg48: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.5.self_attn.v_proj.weight"}, %arg49: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.5.self_attn.o_proj.weight"}, %arg50: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.5.mlp.gate_proj.weight"}, %arg51: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.5.mlp.up_proj.weight"}, %arg52: tensor<2048x8192xbf16, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.5.mlp.down_proj.weight"}, %arg53: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.5.input_layernorm.weight"}, %arg54: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.5.post_attention_layernorm.weight"}, %arg55: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.6.self_attn.q_proj.weight"}, %arg56: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.6.self_attn.k_proj.weight"}, %arg57: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.6.self_attn.v_proj.weight"}, %arg58: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.6.self_attn.o_proj.weight"}, %arg59: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.6.mlp.gate_proj.weight"}, %arg60: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.6.mlp.up_proj.weight"}, %arg61: tensor<2048x8192xbf16, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.6.mlp.down_proj.weight"}, %arg62: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.6.input_layernorm.weight"}, %arg63: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.6.post_attention_layernorm.weight"}, %arg64: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.7.self_attn.q_proj.weight"}, %arg65: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.7.self_attn.k_proj.weight"}, %arg66: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.7.self_attn.v_proj.weight"}, %arg67: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.7.self_attn.o_proj.weight"}, %arg68: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.7.mlp.gate_proj.weight"}, %arg69: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.7.mlp.up_proj.weight"}, %arg70: tensor<2048x8192xbf16, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.7.mlp.down_proj.weight"}, %arg71: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.7.input_layernorm.weight"}, %arg72: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.7.post_attention_layernorm.weight"}, %arg73: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.8.self_attn.q_proj.weight"}, %arg74: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.8.self_attn.k_proj.weight"}, %arg75: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.8.self_attn.v_proj.weight"}, %arg76: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.8.self_attn.o_proj.weight"}, %arg77: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.8.mlp.gate_proj.weight"}, %arg78: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.8.mlp.up_proj.weight"}, %arg79: tensor<2048x8192xbf16, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.8.mlp.down_proj.weight"}, %arg80: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.8.input_layernorm.weight"}, %arg81: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.8.post_attention_layernorm.weight"}, %arg82: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.9.self_attn.q_proj.weight"}, %arg83: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.9.self_attn.k_proj.weight"}, %arg84: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.9.self_attn.v_proj.weight"}, %arg85: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.9.self_attn.o_proj.weight"}, %arg86: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.9.mlp.gate_proj.weight"}, %arg87: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.9.mlp.up_proj.weight"}, %arg88: tensor<2048x8192xbf16, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.9.mlp.down_proj.weight"}, %arg89: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.9.input_layernorm.weight"}, %arg90: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.9.post_attention_layernorm.weight"}, %arg91: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.10.self_attn.q_proj.weight"}, %arg92: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.10.self_attn.k_proj.weight"}, %arg93: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.10.self_attn.v_proj.weight"}, %arg94: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.10.self_attn.o_proj.weight"}, %arg95: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.10.mlp.gate_proj.weight"}, %arg96: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.10.mlp.up_proj.weight"}, %arg97: tensor<2048x8192xbf16, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.10.mlp.down_proj.weight"}, %arg98: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.10.input_layernorm.weight"}, %arg99: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.10.post_attention_layernorm.weight"}, %arg100: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.11.self_attn.q_proj.weight"}, %arg101: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.11.self_attn.k_proj.weight"}, %arg102: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.11.self_attn.v_proj.weight"}, %arg103: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.11.self_attn.o_proj.weight"}, %arg104: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.11.mlp.gate_proj.weight"}, %arg105: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.11.mlp.up_proj.weight"}, %arg106: tensor<2048x8192xbf16, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.11.mlp.down_proj.weight"}, %arg107: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.11.input_layernorm.weight"}, %arg108: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.11.post_attention_layernorm.weight"}, %arg109: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.12.self_attn.q_proj.weight"}, %arg110: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.12.self_attn.k_proj.weight"}, %arg111: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.12.self_attn.v_proj.weight"}, %arg112: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.12.self_attn.o_proj.weight"}, %arg113: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.12.mlp.gate_proj.weight"}, %arg114: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.12.mlp.up_proj.weight"}, %arg115: tensor<2048x8192xbf16, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.12.mlp.down_proj.weight"}, %arg116: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.12.input_layernorm.weight"}, %arg117: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.12.post_attention_layernorm.weight"}, %arg118: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.13.self_attn.q_proj.weight"}, %arg119: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.13.self_attn.k_proj.weight"}, %arg120: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.13.self_attn.v_proj.weight"}, %arg121: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.13.self_attn.o_proj.weight"}, %arg122: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.13.mlp.gate_proj.weight"}, %arg123: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.13.mlp.up_proj.weight"}, %arg124: tensor<2048x8192xbf16, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.13.mlp.down_proj.weight"}, %arg125: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.13.input_layernorm.weight"}, %arg126: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.13.post_attention_layernorm.weight"}, %arg127: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.14.self_attn.q_proj.weight"}, %arg128: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.14.self_attn.k_proj.weight"}, %arg129: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.14.self_attn.v_proj.weight"}, %arg130: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.14.self_attn.o_proj.weight"}, %arg131: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.14.mlp.gate_proj.weight"}, %arg132: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.14.mlp.up_proj.weight"}, %arg133: tensor<2048x8192xbf16, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.14.mlp.down_proj.weight"}, %arg134: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.14.input_layernorm.weight"}, %arg135: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.14.post_attention_layernorm.weight"}, %arg136: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.15.self_attn.q_proj.weight"}, %arg137: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.15.self_attn.k_proj.weight"}, %arg138: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.15.self_attn.v_proj.weight"}, %arg139: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.15.self_attn.o_proj.weight"}, %arg140: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.15.mlp.gate_proj.weight"}, %arg141: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.15.mlp.up_proj.weight"}, %arg142: tensor<2048x8192xbf16, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.15.mlp.down_proj.weight"}, %arg143: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.15.input_layernorm.weight"}, %arg144: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.15.post_attention_layernorm.weight"}, %arg145: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.norm.weight"}, %arg146: tensor<32xf32, #ttnn_layout6> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<32xf32>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.rotary_emb.inv_freq"}, %arg147: tensor<1x512xsi32, #ttnn_layout7> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x512xi64>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "input_ids"}, %arg148: tensor<1x512xsi32, #ttnn_layout7> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x512xi64>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "attention_mask"}, %arg149: tensor<1x511x128256xbf16, #ttnn_layout8> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x511x128256xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "labels_one_hot"}, %arg150: tensor<1x511x1xf32, #ttnn_layout9> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x511x1xf32>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "loss_weight"}) -> (tensor<1x1x1xf32, #ttnn_layout10>, tensor<128256x2048xbf16, #ttnn_layout>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<1x512xsi32, #ttnn_layout11>, tensor<1x511x128256xbf16, #ttnn_layout12>, tensor<1x511x1xf32, #ttnn_layout13>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x511x128256xbf16, #ttnn_layout12>, tensor<1x511x1xf32, #ttnn_layout13>, tensor<1x511x1xf32, #ttnn_layout13>) attributes {tt.function_type = "forward_device"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
        %1 = "ttnn.full"(%0) <{fill_value = 0.353553385 : f32, shape = #ttnn.shape<1x1x1x1x1>}> : (!ttnn.device) -> tensor<1x1x1x1x1xf32, #ttnn_layout21>
        %2 = "ttnn.full"(%0) <{fill_value = 512 : i32, shape = #ttnn.shape<1x1x1x1>}> : (!ttnn.device) -> tensor<1x1x1x1xsi32, #ttnn_layout22>
        %3 = "ttnn.full"(%0) <{fill_value = 0.353553385 : f32, shape = #ttnn.shape<1x1x1x1>}> : (!ttnn.device) -> tensor<1x1x1x1xf32, #ttnn_layout23>
        %4 = "ttnn.full"(%0) <{fill_value = 0xFF800000 : f32, shape = #ttnn.shape<1x1x1x1>}> : (!ttnn.device) -> tensor<1x1x1x1xbf16, #ttnn_layout24>
        %5 = "ttnn.zeros"(%0) <{shape = #ttnn.shape<1x1x1x1>}> : (!ttnn.device) -> tensor<1x1x1x1xbf16, #ttnn_layout24>
        %6 = "ttnn.full"(%0) <{fill_value = 9.99999974E-6 : f32, shape = #ttnn.shape<1x1x1>}> : (!ttnn.device) -> tensor<1x1x1xf32, #ttnn_layout10>
        %7 = "ttnn.ones"(%0) <{shape = #ttnn.shape<1x1x1>}> : (!ttnn.device) -> tensor<1x1x1xf32, #ttnn_layout10>
        %8 = "ttnn.zeros"(%0) <{shape = #ttnn.shape<1x1x1x1>}> : (!ttnn.device) -> tensor<1x1x1x1xsi32, #ttnn_layout22>
        %9 = "ttnn.ones"(%0) <{shape = #ttnn.shape<1x1x1x1>}> : (!ttnn.device) -> tensor<1x1x1x1xsi32, #ttnn_layout22>
        %10 = "ttnn.zeros"(%0) <{shape = #ttnn.shape<1>}> : (!ttnn.device) -> tensor<1xsi32, #ttnn_layout25>
        %11 = "ttnn.typecast"(%arg147) : (tensor<1x512xsi32, #ttnn_layout7>) -> tensor<1x512xui32, #ttnn_layout26>
        %12 = "ttnn.to_layout"(%arg0) : (tensor<128256x2048xbf16, #ttnn_layout>) -> tensor<128256x2048xbf16, #ttnn_layout27>
        %13 = "ttnn.embedding"(%11, %12) : (tensor<1x512xui32, #ttnn_layout26>, tensor<128256x2048xbf16, #ttnn_layout27>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%12) <{force = false}> : (tensor<128256x2048xbf16, #ttnn_layout27>) -> ()
        "ttnn.deallocate"(%11) <{force = false}> : (tensor<1x512xui32, #ttnn_layout26>) -> ()
        %14 = "ttnn.arange"(%0) <{end = 512 : si64, start = 0 : si64, step = 1 : si64}> : (!ttnn.device) -> tensor<512xsi32, #ttnn_layout28>
        %15 = "ttnn.add"(%14, %10) : (tensor<512xsi32, #ttnn_layout28>, tensor<1xsi32, #ttnn_layout25>) -> tensor<512xsi32, #ttnn_layout28>
        "ttnn.deallocate"(%14) <{force = false}> : (tensor<512xsi32, #ttnn_layout28>) -> ()
        "ttnn.deallocate"(%10) <{force = false}> : (tensor<1xsi32, #ttnn_layout25>) -> ()
        %16 = "ttnn.to_layout"(%arg148) : (tensor<1x512xsi32, #ttnn_layout7>) -> tensor<1x512xsi32, #ttnn_layout11>
        "ttnn.deallocate"(%arg148) <{force = false}> : (tensor<1x512xsi32, #ttnn_layout7>) -> ()
        %17 = "ttnn.typecast"(%16) : (tensor<1x512xsi32, #ttnn_layout11>) -> tensor<1x512xbf16, #ttnn_layout29>
        "ttnn.deallocate"(%16) <{force = false}> : (tensor<1x512xsi32, #ttnn_layout11>) -> ()
        %18 = "ttnn.arange"(%0) <{end = 1 : si64, start = 0 : si64, step = 1 : si64}> : (!ttnn.device) -> tensor<1xsi32, #ttnn_layout25>
        %19 = "ttnn.reshape"(%18) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1xsi32, #ttnn_layout25>) -> tensor<1x1x1x1xsi32, #ttnn_layout22>
        "ttnn.deallocate"(%18) <{force = false}> : (tensor<1xsi32, #ttnn_layout25>) -> ()
        %20 = "ttnn.reshape"(%15) <{shape = [1 : i32, 1 : i32, 512 : i32]}> : (tensor<512xsi32, #ttnn_layout28>) -> tensor<1x1x512xsi32, #ttnn_layout30>
        %21 = "ttnn.reshape"(%15) <{shape = [1 : i32, 1 : i32, 512 : i32, 1 : i32]}> : (tensor<512xsi32, #ttnn_layout28>) -> tensor<1x1x512x1xsi32, #ttnn_layout31>
        %22 = "ttnn.reshape"(%15) <{shape = [1 : i32, 1 : i32, 1 : i32, 512 : i32]}> : (tensor<512xsi32, #ttnn_layout28>) -> tensor<1x1x1x512xsi32, #ttnn_layout32>
        "ttnn.deallocate"(%15) <{force = false}> : (tensor<512xsi32, #ttnn_layout28>) -> ()
        %23 = "ttnn.ge"(%21, %22) : (tensor<1x1x512x1xsi32, #ttnn_layout31>, tensor<1x1x1x512xsi32, #ttnn_layout32>) -> tensor<1x1x512x512xbf16, #ttnn_layout33>
        "ttnn.deallocate"(%21) <{force = false}> : (tensor<1x1x512x1xsi32, #ttnn_layout31>) -> ()
        %24 = "ttnn.add"(%19, %9) : (tensor<1x1x1x1xsi32, #ttnn_layout22>, tensor<1x1x1x1xsi32, #ttnn_layout22>) -> tensor<1x1x1x1xsi32, #ttnn_layout22>
        "ttnn.deallocate"(%9) <{force = false}> : (tensor<1x1x1x1xsi32, #ttnn_layout22>) -> ()
        %25 = "ttnn.gt"(%8, %19) : (tensor<1x1x1x1xsi32, #ttnn_layout22>, tensor<1x1x1x1xsi32, #ttnn_layout22>) -> tensor<1x1x1x1xbf16, #ttnn_layout24>
        %26 = "ttnn.typecast"(%25) : (tensor<1x1x1x1xbf16, #ttnn_layout24>) -> tensor<1x1x1x1xf32, #ttnn_layout23>
        "ttnn.deallocate"(%25) <{force = false}> : (tensor<1x1x1x1xbf16, #ttnn_layout24>) -> ()
        %27 = "ttnn.typecast"(%24) : (tensor<1x1x1x1xsi32, #ttnn_layout22>) -> tensor<1x1x1x1xf32, #ttnn_layout23>
        "ttnn.deallocate"(%24) <{force = false}> : (tensor<1x1x1x1xsi32, #ttnn_layout22>) -> ()
        %28 = "ttnn.typecast"(%19) : (tensor<1x1x1x1xsi32, #ttnn_layout22>) -> tensor<1x1x1x1xf32, #ttnn_layout23>
        "ttnn.deallocate"(%19) <{force = false}> : (tensor<1x1x1x1xsi32, #ttnn_layout22>) -> ()
        %29 = "ttnn.where"(%26, %27, %28) : (tensor<1x1x1x1xf32, #ttnn_layout23>, tensor<1x1x1x1xf32, #ttnn_layout23>, tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x1x1x1xf32, #ttnn_layout23>
        "ttnn.deallocate"(%28) <{force = false}> : (tensor<1x1x1x1xf32, #ttnn_layout23>) -> ()
        "ttnn.deallocate"(%27) <{force = false}> : (tensor<1x1x1x1xf32, #ttnn_layout23>) -> ()
        "ttnn.deallocate"(%26) <{force = false}> : (tensor<1x1x1x1xf32, #ttnn_layout23>) -> ()
        %30 = "ttnn.typecast"(%29) : (tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x1x1x1xsi32, #ttnn_layout22>
        "ttnn.deallocate"(%29) <{force = false}> : (tensor<1x1x1x1xf32, #ttnn_layout23>) -> ()
        %31 = "ttnn.multiply"(%30, %2) : (tensor<1x1x1x1xsi32, #ttnn_layout22>, tensor<1x1x1x1xsi32, #ttnn_layout22>) -> tensor<1x1x1x1xsi32, #ttnn_layout22>
        "ttnn.deallocate"(%30) <{force = false}> : (tensor<1x1x1x1xsi32, #ttnn_layout22>) -> ()
        %32 = "ttnn.reshape"(%31) <{shape = [1 : i32]}> : (tensor<1x1x1x1xsi32, #ttnn_layout22>) -> tensor<1xsi32, #ttnn_layout25>
        "ttnn.deallocate"(%31) <{force = false}> : (tensor<1x1x1x1xsi32, #ttnn_layout22>) -> ()
        %33 = "ttnn.add"(%22, %2) : (tensor<1x1x1x512xsi32, #ttnn_layout32>, tensor<1x1x1x1xsi32, #ttnn_layout22>) -> tensor<1x1x1x512xsi32, #ttnn_layout32>
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<1x1x1x1xsi32, #ttnn_layout22>) -> ()
        %34 = "ttnn.gt"(%8, %22) : (tensor<1x1x1x1xsi32, #ttnn_layout22>, tensor<1x1x1x512xsi32, #ttnn_layout32>) -> tensor<1x1x1x512xbf16, #ttnn_layout34>
        "ttnn.deallocate"(%8) <{force = false}> : (tensor<1x1x1x1xsi32, #ttnn_layout22>) -> ()
        %35 = "ttnn.typecast"(%34) : (tensor<1x1x1x512xbf16, #ttnn_layout34>) -> tensor<1x1x1x512xf32, #ttnn_layout35>
        "ttnn.deallocate"(%34) <{force = false}> : (tensor<1x1x1x512xbf16, #ttnn_layout34>) -> ()
        %36 = "ttnn.typecast"(%33) : (tensor<1x1x1x512xsi32, #ttnn_layout32>) -> tensor<1x1x1x512xf32, #ttnn_layout35>
        "ttnn.deallocate"(%33) <{force = false}> : (tensor<1x1x1x512xsi32, #ttnn_layout32>) -> ()
        %37 = "ttnn.typecast"(%22) : (tensor<1x1x1x512xsi32, #ttnn_layout32>) -> tensor<1x1x1x512xf32, #ttnn_layout35>
        "ttnn.deallocate"(%22) <{force = false}> : (tensor<1x1x1x512xsi32, #ttnn_layout32>) -> ()
        %38 = "ttnn.where"(%35, %36, %37) : (tensor<1x1x1x512xf32, #ttnn_layout35>, tensor<1x1x1x512xf32, #ttnn_layout35>, tensor<1x1x1x512xf32, #ttnn_layout35>) -> tensor<1x1x1x512xf32, #ttnn_layout35>
        "ttnn.deallocate"(%37) <{force = false}> : (tensor<1x1x1x512xf32, #ttnn_layout35>) -> ()
        "ttnn.deallocate"(%36) <{force = false}> : (tensor<1x1x1x512xf32, #ttnn_layout35>) -> ()
        "ttnn.deallocate"(%35) <{force = false}> : (tensor<1x1x1x512xf32, #ttnn_layout35>) -> ()
        %39 = "ttnn.typecast"(%38) : (tensor<1x1x1x512xf32, #ttnn_layout35>) -> tensor<1x1x1x512xsi32, #ttnn_layout32>
        "ttnn.deallocate"(%38) <{force = false}> : (tensor<1x1x1x512xf32, #ttnn_layout35>) -> ()
        %40 = "ttnn.reshape"(%39) <{shape = [512 : i32]}> : (tensor<1x1x1x512xsi32, #ttnn_layout32>) -> tensor<512xsi32, #ttnn_layout28>
        "ttnn.deallocate"(%39) <{force = false}> : (tensor<1x1x1x512xsi32, #ttnn_layout32>) -> ()
        %41 = "ttnn.add"(%32, %40) : (tensor<1xsi32, #ttnn_layout25>, tensor<512xsi32, #ttnn_layout28>) -> tensor<512xsi32, #ttnn_layout28>
        "ttnn.deallocate"(%40) <{force = false}> : (tensor<512xsi32, #ttnn_layout28>) -> ()
        "ttnn.deallocate"(%32) <{force = false}> : (tensor<1xsi32, #ttnn_layout25>) -> ()
        %42 = "ttnn.full"(%0) <{fill_value = 0 : i32, shape = #ttnn.shape<512>}> : (!ttnn.device) -> tensor<512xsi32, #ttnn_layout28>
        %43 = "ttnn.lt"(%41, %42) : (tensor<512xsi32, #ttnn_layout28>, tensor<512xsi32, #ttnn_layout28>) -> tensor<512xbf16, #ttnn_layout36>
        %44 = "ttnn.maximum"(%41, %42) : (tensor<512xsi32, #ttnn_layout28>, tensor<512xsi32, #ttnn_layout28>) -> tensor<512xsi32, #ttnn_layout28>
        "ttnn.deallocate"(%42) <{force = false}> : (tensor<512xsi32, #ttnn_layout28>) -> ()
        "ttnn.deallocate"(%41) <{force = false}> : (tensor<512xsi32, #ttnn_layout28>) -> ()
        %45 = "ttnn.typecast"(%44) : (tensor<512xsi32, #ttnn_layout28>) -> tensor<512xui32, #ttnn_layout37>
        "ttnn.deallocate"(%44) <{force = false}> : (tensor<512xsi32, #ttnn_layout28>) -> ()
        %46 = "ttnn.reshape"(%45) <{shape = [1 : i32, 512 : i32]}> : (tensor<512xui32, #ttnn_layout37>) -> tensor<1x512xui32, #ttnn_layout38>
        "ttnn.deallocate"(%45) <{force = false}> : (tensor<512xui32, #ttnn_layout37>) -> ()
        %47 = "ttnn.gather"(%17, %46) <{dim = 1 : si32}> : (tensor<1x512xbf16, #ttnn_layout29>, tensor<1x512xui32, #ttnn_layout38>) -> tensor<1x512xbf16, #ttnn_layout29>
        "ttnn.deallocate"(%46) <{force = false}> : (tensor<1x512xui32, #ttnn_layout38>) -> ()
        "ttnn.deallocate"(%17) <{force = false}> : (tensor<1x512xbf16, #ttnn_layout29>) -> ()
        %48 = "ttnn.reshape"(%47) <{shape = [512 : i32]}> : (tensor<1x512xbf16, #ttnn_layout29>) -> tensor<512xbf16, #ttnn_layout36>
        "ttnn.deallocate"(%47) <{force = false}> : (tensor<1x512xbf16, #ttnn_layout29>) -> ()
        %49 = "ttnn.full"(%0) <{fill_value = 0x7FC00000 : f32, shape = #ttnn.shape<512>}> : (!ttnn.device) -> tensor<512xbf16, #ttnn_layout36>
        %50 = "ttnn.where"(%43, %49, %48) : (tensor<512xbf16, #ttnn_layout36>, tensor<512xbf16, #ttnn_layout36>, tensor<512xbf16, #ttnn_layout36>) -> tensor<512xbf16, #ttnn_layout36>
        "ttnn.deallocate"(%49) <{force = false}> : (tensor<512xbf16, #ttnn_layout36>) -> ()
        "ttnn.deallocate"(%48) <{force = false}> : (tensor<512xbf16, #ttnn_layout36>) -> ()
        "ttnn.deallocate"(%43) <{force = false}> : (tensor<512xbf16, #ttnn_layout36>) -> ()
        %51 = "ttnn.reshape"(%50) <{shape = [1 : i32, 1 : i32, 1 : i32, 512 : i32]}> : (tensor<512xbf16, #ttnn_layout36>) -> tensor<1x1x1x512xbf16, #ttnn_layout34>
        "ttnn.deallocate"(%50) <{force = false}> : (tensor<512xbf16, #ttnn_layout36>) -> ()
        %52 = "ttnn.logical_and"(%23, %51) : (tensor<1x1x512x512xbf16, #ttnn_layout33>, tensor<1x1x1x512xbf16, #ttnn_layout34>) -> tensor<1x1x512x512xbf16, #ttnn_layout33>
        "ttnn.deallocate"(%51) <{force = false}> : (tensor<1x1x1x512xbf16, #ttnn_layout34>) -> ()
        "ttnn.deallocate"(%23) <{force = false}> : (tensor<1x1x512x512xbf16, #ttnn_layout33>) -> ()
        %53 = "ttnn.reshape"(%arg146) <{shape = [1 : i32, 32 : i32, 1 : i32]}> : (tensor<32xf32, #ttnn_layout6>) -> tensor<1x32x1xf32, #ttnn_layout10>
        "ttnn.deallocate"(%arg146) <{force = false}> : (tensor<32xf32, #ttnn_layout6>) -> ()
        %54 = "ttnn.typecast"(%20) : (tensor<1x1x512xsi32, #ttnn_layout30>) -> tensor<1x1x512xf32, #ttnn_layout39>
        "ttnn.deallocate"(%20) <{force = false}> : (tensor<1x1x512xsi32, #ttnn_layout30>) -> ()
        %55 = "ttnn.matmul"(%53, %54) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x32x1xf32, #ttnn_layout10>, tensor<1x1x512xf32, #ttnn_layout39>) -> tensor<1x32x512xf32, #ttnn_layout39>
        "ttnn.deallocate"(%54) <{force = false}> : (tensor<1x1x512xf32, #ttnn_layout39>) -> ()
        "ttnn.deallocate"(%53) <{force = false}> : (tensor<1x32x1xf32, #ttnn_layout10>) -> ()
        %56 = "ttnn.permute"(%55) <{permutation = array<i64: 0, 2, 1>}> : (tensor<1x32x512xf32, #ttnn_layout39>) -> tensor<1x512x32xf32, #ttnn_layout13>
        "ttnn.deallocate"(%55) <{force = false}> : (tensor<1x32x512xf32, #ttnn_layout39>) -> ()
        %57 = "ttnn.concat"(%56, %56) <{dim = 2 : si32}> : (tensor<1x512x32xf32, #ttnn_layout13>, tensor<1x512x32xf32, #ttnn_layout13>) -> tensor<1x512x64xf32, #ttnn_layout40>
        "ttnn.deallocate"(%56) <{force = false}> : (tensor<1x512x32xf32, #ttnn_layout13>) -> ()
        %58 = "ttnn.cos"(%57) : (tensor<1x512x64xf32, #ttnn_layout40>) -> tensor<1x512x64xf32, #ttnn_layout40>
        %59 = "ttnn.multiply"(%58, %7) : (tensor<1x512x64xf32, #ttnn_layout40>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x64xf32, #ttnn_layout40>
        "ttnn.deallocate"(%58) <{force = false}> : (tensor<1x512x64xf32, #ttnn_layout40>) -> ()
        %60 = "ttnn.sin"(%57) : (tensor<1x512x64xf32, #ttnn_layout40>) -> tensor<1x512x64xf32, #ttnn_layout40>
        "ttnn.deallocate"(%57) <{force = false}> : (tensor<1x512x64xf32, #ttnn_layout40>) -> ()
        %61 = "ttnn.multiply"(%60, %7) : (tensor<1x512x64xf32, #ttnn_layout40>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x64xf32, #ttnn_layout40>
        "ttnn.deallocate"(%60) <{force = false}> : (tensor<1x512x64xf32, #ttnn_layout40>) -> ()
        "ttnn.deallocate"(%7) <{force = false}> : (tensor<1x1x1xf32, #ttnn_layout10>) -> ()
        %62 = "ttnn.typecast"(%59) : (tensor<1x512x64xf32, #ttnn_layout40>) -> tensor<1x512x64xbf16, #ttnn_layout41>
        "ttnn.deallocate"(%59) <{force = false}> : (tensor<1x512x64xf32, #ttnn_layout40>) -> ()
        %63 = "ttnn.typecast"(%61) : (tensor<1x512x64xf32, #ttnn_layout40>) -> tensor<1x512x64xbf16, #ttnn_layout41>
        "ttnn.deallocate"(%61) <{force = false}> : (tensor<1x512x64xf32, #ttnn_layout40>) -> ()
        %64 = "ttnn.typecast"(%13) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %65 = "ttnn.pow_scalar"(%64) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %66 = "ttnn.mean"(%65) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%65) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %67 = "ttnn.add"(%66, %6) : (tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%66) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %68 = "ttnn.rsqrt"(%67) : (tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%67) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %69 = "ttnn.multiply"(%64, %68) : (tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %70 = "ttnn.typecast"(%69) : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%69) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %71 = "ttnn.rms_norm"(%13, %arg8) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %72 = "ttnn.concat"(%arg1, %arg2, %arg3) <{dim = 0 : si32}> : (tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>) -> tensor<3072x2048xbf16, #ttnn_layout42>
        %73 = "ttnn.matmul"(%71, %72) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<3072x2048xbf16, #ttnn_layout42>) -> tensor<1x512x3072xbf16, #ttnn_layout43>
        "ttnn.deallocate"(%72) <{force = false}> : (tensor<3072x2048xbf16, #ttnn_layout42>) -> ()
        %74 = "ttnn.slice_static"(%73) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 512 : i32, 2048 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %75 = "ttnn.slice_static"(%73) <{begins = [0 : i32, 0 : i32, 2048 : i32], ends = [1 : i32, 512 : i32, 2560 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x512xbf16, #ttnn_layout44>
        %76 = "ttnn.slice_static"(%73) <{begins = [0 : i32, 0 : i32, 2560 : i32], ends = [1 : i32, 512 : i32, 3072 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x512xbf16, #ttnn_layout44>
        "ttnn.deallocate"(%73) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> ()
        %77 = "ttnn.reshape"(%74) <{shape = [1 : i32, 512 : i32, 32 : i32, 64 : i32]}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x32x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%74) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %78 = "ttnn.permute"(%77) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%77) <{force = false}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> ()
        %79 = "ttnn.reshape"(%75) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> tensor<1x512x8x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%75) <{force = false}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> ()
        %80 = "ttnn.permute"(%79) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%79) <{force = false}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> ()
        %81 = "ttnn.reshape"(%76) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> tensor<1x512x8x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%76) <{force = false}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> ()
        %82 = "ttnn.permute"(%81) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%81) <{force = false}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> ()
        %83 = "ttnn.reshape"(%62) <{shape = [1 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x512x64xbf16, #ttnn_layout41>) -> tensor<1x1x512x64xbf16, #ttnn_layout16>
        "ttnn.deallocate"(%62) <{force = false}> : (tensor<1x512x64xbf16, #ttnn_layout41>) -> ()
        %84 = "ttnn.reshape"(%63) <{shape = [1 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x512x64xbf16, #ttnn_layout41>) -> tensor<1x1x512x64xbf16, #ttnn_layout16>
        "ttnn.deallocate"(%63) <{force = false}> : (tensor<1x512x64xbf16, #ttnn_layout41>) -> ()
        %85 = "ttnn.slice_static"(%78) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        %86 = "ttnn.slice_static"(%78) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        %87 = "ttnn.neg"(%86) : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        "ttnn.deallocate"(%86) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        %88 = "ttnn.concat"(%87, %85) <{dim = 3 : si32}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>, tensor<1x32x512x32xbf16, #ttnn_layout48>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%87) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        "ttnn.deallocate"(%85) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        %89 = "ttnn.multiply"(%78, %83) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%78) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %90 = "ttnn.multiply"(%88, %84) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%88) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %91 = "ttnn.add"(%89, %90) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%90) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        "ttnn.deallocate"(%89) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %92 = "ttnn.slice_static"(%80) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        %93 = "ttnn.slice_static"(%80) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        %94 = "ttnn.neg"(%93) : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        "ttnn.deallocate"(%93) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        %95 = "ttnn.concat"(%94, %92) <{dim = 3 : si32}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>, tensor<1x8x512x32xbf16, #ttnn_layout49>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%94) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        "ttnn.deallocate"(%92) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        %96 = "ttnn.multiply"(%80, %83) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%80) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %97 = "ttnn.multiply"(%95, %84) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%95) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %98 = "ttnn.add"(%96, %97) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%97) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        "ttnn.deallocate"(%96) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %99 = "ttnn.reshape"(%98) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout50>
        "ttnn.deallocate"(%98) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %100 = "ttnn.reshape"(%82) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout50>
        "ttnn.deallocate"(%82) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %101 = "ttnn.where"(%52, %5, %4) : (tensor<1x1x512x512xbf16, #ttnn_layout33>, tensor<1x1x1x1xbf16, #ttnn_layout24>, tensor<1x1x1x1xbf16, #ttnn_layout24>) -> tensor<1x1x512x512xbf16, #ttnn_layout33>
        "ttnn.deallocate"(%52) <{force = false}> : (tensor<1x1x512x512xbf16, #ttnn_layout33>) -> ()
        "ttnn.deallocate"(%5) <{force = false}> : (tensor<1x1x1x1xbf16, #ttnn_layout24>) -> ()
        "ttnn.deallocate"(%4) <{force = false}> : (tensor<1x1x1x1xbf16, #ttnn_layout24>) -> ()
        %102 = "ttnn.typecast"(%91) : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%91) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %103 = "ttnn.typecast"(%99) : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%99) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> ()
        %104 = "ttnn.typecast"(%100) : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%100) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> ()
        %105 = "ttnn.repeat"(%104) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> tensor<1x8x4x512x64xf32, #ttnn_layout52>
        "ttnn.deallocate"(%104) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %106 = "ttnn.reshape"(%105) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%105) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> ()
        %107 = "ttnn.multiply"(%102, %3) : (tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%102) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %108 = "ttnn.multiply"(%103, %1) : (tensor<1x8x1x512x64xf32, #ttnn_layout51>, tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%103) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %109 = "ttnn.repeat"(%108) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> tensor<1x8x4x512x64xf32, #ttnn_layout52>
        "ttnn.deallocate"(%108) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %110 = "ttnn.reshape"(%109) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%109) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> ()
        %111 = "ttnn.permute"(%110) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x64x512xf32, #ttnn_layout18>
        %112 = "ttnn.typecast"(%101) : (tensor<1x1x512x512xbf16, #ttnn_layout33>) -> tensor<1x1x512x512xf32, #ttnn_layout53>
        "ttnn.deallocate"(%101) <{force = false}> : (tensor<1x1x512x512xbf16, #ttnn_layout33>) -> ()
        %113 = "ttnn.matmul"(%107, %110) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%110) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %114 = "ttnn.add"(%113, %112) : (tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x1x512x512xf32, #ttnn_layout53>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%113) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %115 = "ttnn.softmax"(%114) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%114) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %116 = "ttnn.matmul"(%115, %106) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        %117 = "ttnn.typecast"(%116) : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%116) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %118 = "ttnn.permute"(%117) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x512x32x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%117) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %119 = "ttnn.reshape"(%118) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%118) <{force = false}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> ()
        %120 = "ttnn.matmul"(%119, %arg4) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048x2048xbf16, #ttnn_layout1>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %121 = "ttnn.add"(%120, %13) : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%120) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        "ttnn.deallocate"(%13) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %122 = "ttnn.typecast"(%121) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %123 = "ttnn.pow_scalar"(%122) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %124 = "ttnn.mean"(%123) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%123) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %125 = "ttnn.add"(%124, %6) : (tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%124) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %126 = "ttnn.rsqrt"(%125) : (tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%125) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %127 = "ttnn.multiply"(%122, %126) : (tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %128 = "ttnn.typecast"(%127) : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%127) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %129 = "ttnn.rms_norm"(%121, %arg9) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %130 = "ttnn.matmul"(%129, %arg5) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %131 = "ttnn.silu"(%130) : (tensor<1x512x8192xbf16, #ttnn_layout20>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %132 = "ttnn.matmul"(%129, %arg6) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %133 = "ttnn.multiply"(%131, %132) : (tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %134 = "ttnn.matmul"(%133, %arg7) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<2048x8192xbf16, #ttnn_layout4>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %135 = "ttnn.add"(%134, %121) : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%134) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        "ttnn.deallocate"(%121) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %136 = "ttnn.typecast"(%135) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %137 = "ttnn.pow_scalar"(%136) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %138 = "ttnn.mean"(%137) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%137) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %139 = "ttnn.add"(%138, %6) : (tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%138) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %140 = "ttnn.rsqrt"(%139) : (tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%139) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %141 = "ttnn.multiply"(%136, %140) : (tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %142 = "ttnn.typecast"(%141) : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%141) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %143 = "ttnn.rms_norm"(%135, %arg17) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %144 = "ttnn.concat"(%arg10, %arg11, %arg12) <{dim = 0 : si32}> : (tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>) -> tensor<3072x2048xbf16, #ttnn_layout42>
        %145 = "ttnn.matmul"(%143, %144) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<3072x2048xbf16, #ttnn_layout42>) -> tensor<1x512x3072xbf16, #ttnn_layout43>
        "ttnn.deallocate"(%144) <{force = false}> : (tensor<3072x2048xbf16, #ttnn_layout42>) -> ()
        %146 = "ttnn.slice_static"(%145) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 512 : i32, 2048 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %147 = "ttnn.slice_static"(%145) <{begins = [0 : i32, 0 : i32, 2048 : i32], ends = [1 : i32, 512 : i32, 2560 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x512xbf16, #ttnn_layout44>
        %148 = "ttnn.slice_static"(%145) <{begins = [0 : i32, 0 : i32, 2560 : i32], ends = [1 : i32, 512 : i32, 3072 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x512xbf16, #ttnn_layout44>
        "ttnn.deallocate"(%145) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> ()
        %149 = "ttnn.reshape"(%146) <{shape = [1 : i32, 512 : i32, 32 : i32, 64 : i32]}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x32x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%146) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %150 = "ttnn.permute"(%149) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%149) <{force = false}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> ()
        %151 = "ttnn.reshape"(%147) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> tensor<1x512x8x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%147) <{force = false}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> ()
        %152 = "ttnn.permute"(%151) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%151) <{force = false}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> ()
        %153 = "ttnn.reshape"(%148) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> tensor<1x512x8x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%148) <{force = false}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> ()
        %154 = "ttnn.permute"(%153) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%153) <{force = false}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> ()
        %155 = "ttnn.slice_static"(%150) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        %156 = "ttnn.slice_static"(%150) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        %157 = "ttnn.neg"(%156) : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        "ttnn.deallocate"(%156) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        %158 = "ttnn.concat"(%157, %155) <{dim = 3 : si32}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>, tensor<1x32x512x32xbf16, #ttnn_layout48>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%157) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        "ttnn.deallocate"(%155) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        %159 = "ttnn.multiply"(%150, %83) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%150) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %160 = "ttnn.multiply"(%158, %84) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%158) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %161 = "ttnn.add"(%159, %160) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%160) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        "ttnn.deallocate"(%159) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %162 = "ttnn.slice_static"(%152) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        %163 = "ttnn.slice_static"(%152) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        %164 = "ttnn.neg"(%163) : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        "ttnn.deallocate"(%163) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        %165 = "ttnn.concat"(%164, %162) <{dim = 3 : si32}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>, tensor<1x8x512x32xbf16, #ttnn_layout49>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%164) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        "ttnn.deallocate"(%162) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        %166 = "ttnn.multiply"(%152, %83) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%152) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %167 = "ttnn.multiply"(%165, %84) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%165) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %168 = "ttnn.add"(%166, %167) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%167) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        "ttnn.deallocate"(%166) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %169 = "ttnn.reshape"(%168) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout50>
        "ttnn.deallocate"(%168) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %170 = "ttnn.reshape"(%154) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout50>
        "ttnn.deallocate"(%154) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %171 = "ttnn.typecast"(%161) : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%161) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %172 = "ttnn.typecast"(%169) : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%169) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> ()
        %173 = "ttnn.typecast"(%170) : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%170) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> ()
        %174 = "ttnn.repeat"(%173) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> tensor<1x8x4x512x64xf32, #ttnn_layout52>
        "ttnn.deallocate"(%173) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %175 = "ttnn.reshape"(%174) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%174) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> ()
        %176 = "ttnn.multiply"(%171, %3) : (tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%171) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %177 = "ttnn.multiply"(%172, %1) : (tensor<1x8x1x512x64xf32, #ttnn_layout51>, tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%172) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %178 = "ttnn.repeat"(%177) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> tensor<1x8x4x512x64xf32, #ttnn_layout52>
        "ttnn.deallocate"(%177) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %179 = "ttnn.reshape"(%178) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%178) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> ()
        %180 = "ttnn.permute"(%179) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x64x512xf32, #ttnn_layout18>
        %181 = "ttnn.matmul"(%176, %179) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%179) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %182 = "ttnn.add"(%181, %112) : (tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x1x512x512xf32, #ttnn_layout53>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%181) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %183 = "ttnn.softmax"(%182) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%182) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %184 = "ttnn.matmul"(%183, %175) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        %185 = "ttnn.typecast"(%184) : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%184) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %186 = "ttnn.permute"(%185) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x512x32x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%185) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %187 = "ttnn.reshape"(%186) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%186) <{force = false}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> ()
        %188 = "ttnn.matmul"(%187, %arg13) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048x2048xbf16, #ttnn_layout1>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %189 = "ttnn.add"(%188, %135) : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%188) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        "ttnn.deallocate"(%135) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %190 = "ttnn.typecast"(%189) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %191 = "ttnn.pow_scalar"(%190) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %192 = "ttnn.mean"(%191) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%191) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %193 = "ttnn.add"(%192, %6) : (tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%192) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %194 = "ttnn.rsqrt"(%193) : (tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%193) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %195 = "ttnn.multiply"(%190, %194) : (tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %196 = "ttnn.typecast"(%195) : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%195) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %197 = "ttnn.rms_norm"(%189, %arg18) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %198 = "ttnn.matmul"(%197, %arg14) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %199 = "ttnn.silu"(%198) : (tensor<1x512x8192xbf16, #ttnn_layout20>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %200 = "ttnn.matmul"(%197, %arg15) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %201 = "ttnn.multiply"(%199, %200) : (tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %202 = "ttnn.matmul"(%201, %arg16) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<2048x8192xbf16, #ttnn_layout4>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %203 = "ttnn.add"(%202, %189) : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%202) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        "ttnn.deallocate"(%189) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %204 = "ttnn.typecast"(%203) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %205 = "ttnn.pow_scalar"(%204) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %206 = "ttnn.mean"(%205) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%205) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %207 = "ttnn.add"(%206, %6) : (tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%206) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %208 = "ttnn.rsqrt"(%207) : (tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%207) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %209 = "ttnn.multiply"(%204, %208) : (tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %210 = "ttnn.typecast"(%209) : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%209) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %211 = "ttnn.rms_norm"(%203, %arg26) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %212 = "ttnn.concat"(%arg19, %arg20, %arg21) <{dim = 0 : si32}> : (tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>) -> tensor<3072x2048xbf16, #ttnn_layout42>
        %213 = "ttnn.matmul"(%211, %212) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<3072x2048xbf16, #ttnn_layout42>) -> tensor<1x512x3072xbf16, #ttnn_layout43>
        "ttnn.deallocate"(%212) <{force = false}> : (tensor<3072x2048xbf16, #ttnn_layout42>) -> ()
        %214 = "ttnn.slice_static"(%213) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 512 : i32, 2048 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %215 = "ttnn.slice_static"(%213) <{begins = [0 : i32, 0 : i32, 2048 : i32], ends = [1 : i32, 512 : i32, 2560 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x512xbf16, #ttnn_layout44>
        %216 = "ttnn.slice_static"(%213) <{begins = [0 : i32, 0 : i32, 2560 : i32], ends = [1 : i32, 512 : i32, 3072 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x512xbf16, #ttnn_layout44>
        "ttnn.deallocate"(%213) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> ()
        %217 = "ttnn.reshape"(%214) <{shape = [1 : i32, 512 : i32, 32 : i32, 64 : i32]}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x32x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%214) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %218 = "ttnn.permute"(%217) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%217) <{force = false}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> ()
        %219 = "ttnn.reshape"(%215) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> tensor<1x512x8x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%215) <{force = false}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> ()
        %220 = "ttnn.permute"(%219) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%219) <{force = false}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> ()
        %221 = "ttnn.reshape"(%216) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> tensor<1x512x8x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%216) <{force = false}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> ()
        %222 = "ttnn.permute"(%221) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%221) <{force = false}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> ()
        %223 = "ttnn.slice_static"(%218) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        %224 = "ttnn.slice_static"(%218) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        %225 = "ttnn.neg"(%224) : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        "ttnn.deallocate"(%224) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        %226 = "ttnn.concat"(%225, %223) <{dim = 3 : si32}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>, tensor<1x32x512x32xbf16, #ttnn_layout48>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%225) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        "ttnn.deallocate"(%223) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        %227 = "ttnn.multiply"(%218, %83) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%218) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %228 = "ttnn.multiply"(%226, %84) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%226) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %229 = "ttnn.add"(%227, %228) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%228) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        "ttnn.deallocate"(%227) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %230 = "ttnn.slice_static"(%220) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        %231 = "ttnn.slice_static"(%220) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        %232 = "ttnn.neg"(%231) : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        "ttnn.deallocate"(%231) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        %233 = "ttnn.concat"(%232, %230) <{dim = 3 : si32}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>, tensor<1x8x512x32xbf16, #ttnn_layout49>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%232) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        "ttnn.deallocate"(%230) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        %234 = "ttnn.multiply"(%220, %83) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%220) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %235 = "ttnn.multiply"(%233, %84) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%233) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %236 = "ttnn.add"(%234, %235) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%235) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        "ttnn.deallocate"(%234) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %237 = "ttnn.reshape"(%236) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout50>
        "ttnn.deallocate"(%236) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %238 = "ttnn.reshape"(%222) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout50>
        "ttnn.deallocate"(%222) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %239 = "ttnn.typecast"(%229) : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%229) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %240 = "ttnn.typecast"(%237) : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%237) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> ()
        %241 = "ttnn.typecast"(%238) : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%238) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> ()
        %242 = "ttnn.repeat"(%241) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> tensor<1x8x4x512x64xf32, #ttnn_layout52>
        "ttnn.deallocate"(%241) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %243 = "ttnn.reshape"(%242) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%242) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> ()
        %244 = "ttnn.multiply"(%239, %3) : (tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%239) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %245 = "ttnn.multiply"(%240, %1) : (tensor<1x8x1x512x64xf32, #ttnn_layout51>, tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%240) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %246 = "ttnn.repeat"(%245) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> tensor<1x8x4x512x64xf32, #ttnn_layout52>
        "ttnn.deallocate"(%245) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %247 = "ttnn.reshape"(%246) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%246) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> ()
        %248 = "ttnn.permute"(%247) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x64x512xf32, #ttnn_layout18>
        %249 = "ttnn.matmul"(%244, %247) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%247) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %250 = "ttnn.add"(%249, %112) : (tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x1x512x512xf32, #ttnn_layout53>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%249) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %251 = "ttnn.softmax"(%250) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%250) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %252 = "ttnn.matmul"(%251, %243) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        %253 = "ttnn.typecast"(%252) : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%252) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %254 = "ttnn.permute"(%253) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x512x32x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%253) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %255 = "ttnn.reshape"(%254) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%254) <{force = false}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> ()
        %256 = "ttnn.matmul"(%255, %arg22) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048x2048xbf16, #ttnn_layout1>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %257 = "ttnn.add"(%256, %203) : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%256) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        "ttnn.deallocate"(%203) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %258 = "ttnn.typecast"(%257) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %259 = "ttnn.pow_scalar"(%258) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %260 = "ttnn.mean"(%259) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%259) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %261 = "ttnn.add"(%260, %6) : (tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%260) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %262 = "ttnn.rsqrt"(%261) : (tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%261) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %263 = "ttnn.multiply"(%258, %262) : (tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %264 = "ttnn.typecast"(%263) : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%263) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %265 = "ttnn.rms_norm"(%257, %arg27) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %266 = "ttnn.matmul"(%265, %arg23) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %267 = "ttnn.silu"(%266) : (tensor<1x512x8192xbf16, #ttnn_layout20>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %268 = "ttnn.matmul"(%265, %arg24) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %269 = "ttnn.multiply"(%267, %268) : (tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %270 = "ttnn.matmul"(%269, %arg25) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<2048x8192xbf16, #ttnn_layout4>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %271 = "ttnn.add"(%270, %257) : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%270) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        "ttnn.deallocate"(%257) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %272 = "ttnn.typecast"(%271) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %273 = "ttnn.pow_scalar"(%272) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %274 = "ttnn.mean"(%273) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%273) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %275 = "ttnn.add"(%274, %6) : (tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%274) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %276 = "ttnn.rsqrt"(%275) : (tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%275) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %277 = "ttnn.multiply"(%272, %276) : (tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %278 = "ttnn.typecast"(%277) : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%277) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %279 = "ttnn.rms_norm"(%271, %arg35) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %280 = "ttnn.concat"(%arg28, %arg29, %arg30) <{dim = 0 : si32}> : (tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>) -> tensor<3072x2048xbf16, #ttnn_layout42>
        %281 = "ttnn.matmul"(%279, %280) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<3072x2048xbf16, #ttnn_layout42>) -> tensor<1x512x3072xbf16, #ttnn_layout43>
        "ttnn.deallocate"(%280) <{force = false}> : (tensor<3072x2048xbf16, #ttnn_layout42>) -> ()
        %282 = "ttnn.slice_static"(%281) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 512 : i32, 2048 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %283 = "ttnn.slice_static"(%281) <{begins = [0 : i32, 0 : i32, 2048 : i32], ends = [1 : i32, 512 : i32, 2560 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x512xbf16, #ttnn_layout44>
        %284 = "ttnn.slice_static"(%281) <{begins = [0 : i32, 0 : i32, 2560 : i32], ends = [1 : i32, 512 : i32, 3072 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x512xbf16, #ttnn_layout44>
        "ttnn.deallocate"(%281) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> ()
        %285 = "ttnn.reshape"(%282) <{shape = [1 : i32, 512 : i32, 32 : i32, 64 : i32]}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x32x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%282) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %286 = "ttnn.permute"(%285) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%285) <{force = false}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> ()
        %287 = "ttnn.reshape"(%283) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> tensor<1x512x8x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%283) <{force = false}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> ()
        %288 = "ttnn.permute"(%287) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%287) <{force = false}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> ()
        %289 = "ttnn.reshape"(%284) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> tensor<1x512x8x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%284) <{force = false}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> ()
        %290 = "ttnn.permute"(%289) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%289) <{force = false}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> ()
        %291 = "ttnn.slice_static"(%286) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        %292 = "ttnn.slice_static"(%286) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        %293 = "ttnn.neg"(%292) : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        "ttnn.deallocate"(%292) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        %294 = "ttnn.concat"(%293, %291) <{dim = 3 : si32}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>, tensor<1x32x512x32xbf16, #ttnn_layout48>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%293) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        "ttnn.deallocate"(%291) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        %295 = "ttnn.multiply"(%286, %83) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%286) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %296 = "ttnn.multiply"(%294, %84) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%294) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %297 = "ttnn.add"(%295, %296) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%296) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        "ttnn.deallocate"(%295) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %298 = "ttnn.slice_static"(%288) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        %299 = "ttnn.slice_static"(%288) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        %300 = "ttnn.neg"(%299) : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        "ttnn.deallocate"(%299) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        %301 = "ttnn.concat"(%300, %298) <{dim = 3 : si32}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>, tensor<1x8x512x32xbf16, #ttnn_layout49>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%300) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        "ttnn.deallocate"(%298) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        %302 = "ttnn.multiply"(%288, %83) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%288) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %303 = "ttnn.multiply"(%301, %84) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%301) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %304 = "ttnn.add"(%302, %303) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%303) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        "ttnn.deallocate"(%302) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %305 = "ttnn.reshape"(%304) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout50>
        "ttnn.deallocate"(%304) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %306 = "ttnn.reshape"(%290) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout50>
        "ttnn.deallocate"(%290) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %307 = "ttnn.typecast"(%297) : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%297) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %308 = "ttnn.typecast"(%305) : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%305) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> ()
        %309 = "ttnn.typecast"(%306) : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%306) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> ()
        %310 = "ttnn.repeat"(%309) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> tensor<1x8x4x512x64xf32, #ttnn_layout52>
        "ttnn.deallocate"(%309) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %311 = "ttnn.reshape"(%310) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%310) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> ()
        %312 = "ttnn.multiply"(%307, %3) : (tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%307) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %313 = "ttnn.multiply"(%308, %1) : (tensor<1x8x1x512x64xf32, #ttnn_layout51>, tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%308) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %314 = "ttnn.repeat"(%313) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> tensor<1x8x4x512x64xf32, #ttnn_layout52>
        "ttnn.deallocate"(%313) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %315 = "ttnn.reshape"(%314) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%314) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> ()
        %316 = "ttnn.permute"(%315) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x64x512xf32, #ttnn_layout18>
        %317 = "ttnn.matmul"(%312, %315) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%315) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %318 = "ttnn.add"(%317, %112) : (tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x1x512x512xf32, #ttnn_layout53>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%317) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %319 = "ttnn.softmax"(%318) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%318) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %320 = "ttnn.matmul"(%319, %311) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        %321 = "ttnn.typecast"(%320) : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%320) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %322 = "ttnn.permute"(%321) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x512x32x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%321) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %323 = "ttnn.reshape"(%322) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%322) <{force = false}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> ()
        %324 = "ttnn.matmul"(%323, %arg31) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048x2048xbf16, #ttnn_layout1>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %325 = "ttnn.add"(%324, %271) : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%324) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        "ttnn.deallocate"(%271) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %326 = "ttnn.typecast"(%325) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %327 = "ttnn.pow_scalar"(%326) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %328 = "ttnn.mean"(%327) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%327) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %329 = "ttnn.add"(%328, %6) : (tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%328) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %330 = "ttnn.rsqrt"(%329) : (tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%329) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %331 = "ttnn.multiply"(%326, %330) : (tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %332 = "ttnn.typecast"(%331) : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%331) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %333 = "ttnn.rms_norm"(%325, %arg36) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %334 = "ttnn.matmul"(%333, %arg32) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %335 = "ttnn.silu"(%334) : (tensor<1x512x8192xbf16, #ttnn_layout20>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %336 = "ttnn.matmul"(%333, %arg33) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %337 = "ttnn.multiply"(%335, %336) : (tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %338 = "ttnn.matmul"(%337, %arg34) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<2048x8192xbf16, #ttnn_layout4>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %339 = "ttnn.add"(%338, %325) : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%338) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        "ttnn.deallocate"(%325) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %340 = "ttnn.typecast"(%339) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %341 = "ttnn.pow_scalar"(%340) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %342 = "ttnn.mean"(%341) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%341) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %343 = "ttnn.add"(%342, %6) : (tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%342) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %344 = "ttnn.rsqrt"(%343) : (tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%343) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %345 = "ttnn.multiply"(%340, %344) : (tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %346 = "ttnn.typecast"(%345) : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%345) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %347 = "ttnn.rms_norm"(%339, %arg44) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %348 = "ttnn.concat"(%arg37, %arg38, %arg39) <{dim = 0 : si32}> : (tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>) -> tensor<3072x2048xbf16, #ttnn_layout42>
        %349 = "ttnn.matmul"(%347, %348) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<3072x2048xbf16, #ttnn_layout42>) -> tensor<1x512x3072xbf16, #ttnn_layout43>
        "ttnn.deallocate"(%348) <{force = false}> : (tensor<3072x2048xbf16, #ttnn_layout42>) -> ()
        %350 = "ttnn.slice_static"(%349) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 512 : i32, 2048 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %351 = "ttnn.slice_static"(%349) <{begins = [0 : i32, 0 : i32, 2048 : i32], ends = [1 : i32, 512 : i32, 2560 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x512xbf16, #ttnn_layout44>
        %352 = "ttnn.slice_static"(%349) <{begins = [0 : i32, 0 : i32, 2560 : i32], ends = [1 : i32, 512 : i32, 3072 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x512xbf16, #ttnn_layout44>
        "ttnn.deallocate"(%349) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> ()
        %353 = "ttnn.reshape"(%350) <{shape = [1 : i32, 512 : i32, 32 : i32, 64 : i32]}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x32x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%350) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %354 = "ttnn.permute"(%353) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%353) <{force = false}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> ()
        %355 = "ttnn.reshape"(%351) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> tensor<1x512x8x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%351) <{force = false}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> ()
        %356 = "ttnn.permute"(%355) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%355) <{force = false}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> ()
        %357 = "ttnn.reshape"(%352) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> tensor<1x512x8x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%352) <{force = false}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> ()
        %358 = "ttnn.permute"(%357) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%357) <{force = false}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> ()
        %359 = "ttnn.slice_static"(%354) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        %360 = "ttnn.slice_static"(%354) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        %361 = "ttnn.neg"(%360) : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        "ttnn.deallocate"(%360) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        %362 = "ttnn.concat"(%361, %359) <{dim = 3 : si32}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>, tensor<1x32x512x32xbf16, #ttnn_layout48>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%361) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        "ttnn.deallocate"(%359) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        %363 = "ttnn.multiply"(%354, %83) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%354) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %364 = "ttnn.multiply"(%362, %84) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%362) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %365 = "ttnn.add"(%363, %364) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%364) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        "ttnn.deallocate"(%363) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %366 = "ttnn.slice_static"(%356) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        %367 = "ttnn.slice_static"(%356) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        %368 = "ttnn.neg"(%367) : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        "ttnn.deallocate"(%367) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        %369 = "ttnn.concat"(%368, %366) <{dim = 3 : si32}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>, tensor<1x8x512x32xbf16, #ttnn_layout49>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%368) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        "ttnn.deallocate"(%366) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        %370 = "ttnn.multiply"(%356, %83) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%356) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %371 = "ttnn.multiply"(%369, %84) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%369) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %372 = "ttnn.add"(%370, %371) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%371) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        "ttnn.deallocate"(%370) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %373 = "ttnn.reshape"(%372) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout50>
        "ttnn.deallocate"(%372) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %374 = "ttnn.reshape"(%358) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout50>
        "ttnn.deallocate"(%358) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %375 = "ttnn.typecast"(%365) : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%365) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %376 = "ttnn.typecast"(%373) : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%373) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> ()
        %377 = "ttnn.typecast"(%374) : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%374) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> ()
        %378 = "ttnn.repeat"(%377) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> tensor<1x8x4x512x64xf32, #ttnn_layout52>
        "ttnn.deallocate"(%377) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %379 = "ttnn.reshape"(%378) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%378) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> ()
        %380 = "ttnn.multiply"(%375, %3) : (tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%375) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %381 = "ttnn.multiply"(%376, %1) : (tensor<1x8x1x512x64xf32, #ttnn_layout51>, tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%376) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %382 = "ttnn.repeat"(%381) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> tensor<1x8x4x512x64xf32, #ttnn_layout52>
        "ttnn.deallocate"(%381) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %383 = "ttnn.reshape"(%382) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%382) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> ()
        %384 = "ttnn.permute"(%383) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x64x512xf32, #ttnn_layout18>
        %385 = "ttnn.matmul"(%380, %383) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%383) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %386 = "ttnn.add"(%385, %112) : (tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x1x512x512xf32, #ttnn_layout53>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%385) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %387 = "ttnn.softmax"(%386) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%386) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %388 = "ttnn.matmul"(%387, %379) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        %389 = "ttnn.typecast"(%388) : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%388) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %390 = "ttnn.permute"(%389) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x512x32x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%389) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %391 = "ttnn.reshape"(%390) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%390) <{force = false}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> ()
        %392 = "ttnn.matmul"(%391, %arg40) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048x2048xbf16, #ttnn_layout1>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %393 = "ttnn.add"(%392, %339) : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%392) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        "ttnn.deallocate"(%339) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %394 = "ttnn.typecast"(%393) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %395 = "ttnn.pow_scalar"(%394) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %396 = "ttnn.mean"(%395) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%395) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %397 = "ttnn.add"(%396, %6) : (tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%396) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %398 = "ttnn.rsqrt"(%397) : (tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%397) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %399 = "ttnn.multiply"(%394, %398) : (tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %400 = "ttnn.typecast"(%399) : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%399) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %401 = "ttnn.rms_norm"(%393, %arg45) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %402 = "ttnn.matmul"(%401, %arg41) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %403 = "ttnn.silu"(%402) : (tensor<1x512x8192xbf16, #ttnn_layout20>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %404 = "ttnn.matmul"(%401, %arg42) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %405 = "ttnn.multiply"(%403, %404) : (tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %406 = "ttnn.matmul"(%405, %arg43) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<2048x8192xbf16, #ttnn_layout4>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %407 = "ttnn.add"(%406, %393) : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%406) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        "ttnn.deallocate"(%393) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %408 = "ttnn.typecast"(%407) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %409 = "ttnn.pow_scalar"(%408) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %410 = "ttnn.mean"(%409) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%409) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %411 = "ttnn.add"(%410, %6) : (tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%410) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %412 = "ttnn.rsqrt"(%411) : (tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%411) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %413 = "ttnn.multiply"(%408, %412) : (tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %414 = "ttnn.typecast"(%413) : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%413) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %415 = "ttnn.rms_norm"(%407, %arg53) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %416 = "ttnn.concat"(%arg46, %arg47, %arg48) <{dim = 0 : si32}> : (tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>) -> tensor<3072x2048xbf16, #ttnn_layout42>
        %417 = "ttnn.matmul"(%415, %416) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<3072x2048xbf16, #ttnn_layout42>) -> tensor<1x512x3072xbf16, #ttnn_layout43>
        "ttnn.deallocate"(%416) <{force = false}> : (tensor<3072x2048xbf16, #ttnn_layout42>) -> ()
        %418 = "ttnn.slice_static"(%417) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 512 : i32, 2048 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %419 = "ttnn.slice_static"(%417) <{begins = [0 : i32, 0 : i32, 2048 : i32], ends = [1 : i32, 512 : i32, 2560 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x512xbf16, #ttnn_layout44>
        %420 = "ttnn.slice_static"(%417) <{begins = [0 : i32, 0 : i32, 2560 : i32], ends = [1 : i32, 512 : i32, 3072 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x512xbf16, #ttnn_layout44>
        "ttnn.deallocate"(%417) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> ()
        %421 = "ttnn.reshape"(%418) <{shape = [1 : i32, 512 : i32, 32 : i32, 64 : i32]}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x32x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%418) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %422 = "ttnn.permute"(%421) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%421) <{force = false}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> ()
        %423 = "ttnn.reshape"(%419) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> tensor<1x512x8x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%419) <{force = false}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> ()
        %424 = "ttnn.permute"(%423) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%423) <{force = false}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> ()
        %425 = "ttnn.reshape"(%420) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> tensor<1x512x8x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%420) <{force = false}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> ()
        %426 = "ttnn.permute"(%425) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%425) <{force = false}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> ()
        %427 = "ttnn.slice_static"(%422) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        %428 = "ttnn.slice_static"(%422) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        %429 = "ttnn.neg"(%428) : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        "ttnn.deallocate"(%428) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        %430 = "ttnn.concat"(%429, %427) <{dim = 3 : si32}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>, tensor<1x32x512x32xbf16, #ttnn_layout48>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%429) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        "ttnn.deallocate"(%427) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        %431 = "ttnn.multiply"(%422, %83) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%422) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %432 = "ttnn.multiply"(%430, %84) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%430) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %433 = "ttnn.add"(%431, %432) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%432) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        "ttnn.deallocate"(%431) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %434 = "ttnn.slice_static"(%424) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        %435 = "ttnn.slice_static"(%424) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        %436 = "ttnn.neg"(%435) : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        "ttnn.deallocate"(%435) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        %437 = "ttnn.concat"(%436, %434) <{dim = 3 : si32}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>, tensor<1x8x512x32xbf16, #ttnn_layout49>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%436) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        "ttnn.deallocate"(%434) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        %438 = "ttnn.multiply"(%424, %83) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%424) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %439 = "ttnn.multiply"(%437, %84) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%437) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %440 = "ttnn.add"(%438, %439) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%439) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        "ttnn.deallocate"(%438) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %441 = "ttnn.reshape"(%440) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout50>
        "ttnn.deallocate"(%440) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %442 = "ttnn.reshape"(%426) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout50>
        "ttnn.deallocate"(%426) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %443 = "ttnn.typecast"(%433) : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%433) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %444 = "ttnn.typecast"(%441) : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%441) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> ()
        %445 = "ttnn.typecast"(%442) : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%442) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> ()
        %446 = "ttnn.repeat"(%445) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> tensor<1x8x4x512x64xf32, #ttnn_layout52>
        "ttnn.deallocate"(%445) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %447 = "ttnn.reshape"(%446) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%446) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> ()
        %448 = "ttnn.multiply"(%443, %3) : (tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%443) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %449 = "ttnn.multiply"(%444, %1) : (tensor<1x8x1x512x64xf32, #ttnn_layout51>, tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%444) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %450 = "ttnn.repeat"(%449) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> tensor<1x8x4x512x64xf32, #ttnn_layout52>
        "ttnn.deallocate"(%449) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %451 = "ttnn.reshape"(%450) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%450) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> ()
        %452 = "ttnn.permute"(%451) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x64x512xf32, #ttnn_layout18>
        %453 = "ttnn.matmul"(%448, %451) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%451) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %454 = "ttnn.add"(%453, %112) : (tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x1x512x512xf32, #ttnn_layout53>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%453) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %455 = "ttnn.softmax"(%454) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%454) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %456 = "ttnn.matmul"(%455, %447) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        %457 = "ttnn.typecast"(%456) : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%456) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %458 = "ttnn.permute"(%457) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x512x32x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%457) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %459 = "ttnn.reshape"(%458) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%458) <{force = false}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> ()
        %460 = "ttnn.matmul"(%459, %arg49) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048x2048xbf16, #ttnn_layout1>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %461 = "ttnn.add"(%460, %407) : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%460) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        "ttnn.deallocate"(%407) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %462 = "ttnn.typecast"(%461) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %463 = "ttnn.pow_scalar"(%462) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %464 = "ttnn.mean"(%463) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%463) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %465 = "ttnn.add"(%464, %6) : (tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%464) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %466 = "ttnn.rsqrt"(%465) : (tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%465) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %467 = "ttnn.multiply"(%462, %466) : (tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %468 = "ttnn.typecast"(%467) : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%467) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %469 = "ttnn.rms_norm"(%461, %arg54) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %470 = "ttnn.matmul"(%469, %arg50) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %471 = "ttnn.silu"(%470) : (tensor<1x512x8192xbf16, #ttnn_layout20>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %472 = "ttnn.matmul"(%469, %arg51) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %473 = "ttnn.multiply"(%471, %472) : (tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %474 = "ttnn.matmul"(%473, %arg52) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<2048x8192xbf16, #ttnn_layout4>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %475 = "ttnn.add"(%474, %461) : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%474) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        "ttnn.deallocate"(%461) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %476 = "ttnn.typecast"(%475) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %477 = "ttnn.pow_scalar"(%476) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %478 = "ttnn.mean"(%477) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%477) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %479 = "ttnn.add"(%478, %6) : (tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%478) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %480 = "ttnn.rsqrt"(%479) : (tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%479) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %481 = "ttnn.multiply"(%476, %480) : (tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %482 = "ttnn.typecast"(%481) : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%481) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %483 = "ttnn.rms_norm"(%475, %arg62) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %484 = "ttnn.concat"(%arg55, %arg56, %arg57) <{dim = 0 : si32}> : (tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>) -> tensor<3072x2048xbf16, #ttnn_layout42>
        %485 = "ttnn.matmul"(%483, %484) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<3072x2048xbf16, #ttnn_layout42>) -> tensor<1x512x3072xbf16, #ttnn_layout43>
        "ttnn.deallocate"(%484) <{force = false}> : (tensor<3072x2048xbf16, #ttnn_layout42>) -> ()
        %486 = "ttnn.slice_static"(%485) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 512 : i32, 2048 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %487 = "ttnn.slice_static"(%485) <{begins = [0 : i32, 0 : i32, 2048 : i32], ends = [1 : i32, 512 : i32, 2560 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x512xbf16, #ttnn_layout44>
        %488 = "ttnn.slice_static"(%485) <{begins = [0 : i32, 0 : i32, 2560 : i32], ends = [1 : i32, 512 : i32, 3072 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x512xbf16, #ttnn_layout44>
        "ttnn.deallocate"(%485) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> ()
        %489 = "ttnn.reshape"(%486) <{shape = [1 : i32, 512 : i32, 32 : i32, 64 : i32]}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x32x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%486) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %490 = "ttnn.permute"(%489) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%489) <{force = false}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> ()
        %491 = "ttnn.reshape"(%487) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> tensor<1x512x8x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%487) <{force = false}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> ()
        %492 = "ttnn.permute"(%491) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%491) <{force = false}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> ()
        %493 = "ttnn.reshape"(%488) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> tensor<1x512x8x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%488) <{force = false}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> ()
        %494 = "ttnn.permute"(%493) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%493) <{force = false}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> ()
        %495 = "ttnn.slice_static"(%490) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        %496 = "ttnn.slice_static"(%490) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        %497 = "ttnn.neg"(%496) : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        "ttnn.deallocate"(%496) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        %498 = "ttnn.concat"(%497, %495) <{dim = 3 : si32}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>, tensor<1x32x512x32xbf16, #ttnn_layout48>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%497) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        "ttnn.deallocate"(%495) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        %499 = "ttnn.multiply"(%490, %83) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%490) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %500 = "ttnn.multiply"(%498, %84) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%498) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %501 = "ttnn.add"(%499, %500) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%500) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        "ttnn.deallocate"(%499) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %502 = "ttnn.slice_static"(%492) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        %503 = "ttnn.slice_static"(%492) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        %504 = "ttnn.neg"(%503) : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        "ttnn.deallocate"(%503) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        %505 = "ttnn.concat"(%504, %502) <{dim = 3 : si32}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>, tensor<1x8x512x32xbf16, #ttnn_layout49>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%504) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        "ttnn.deallocate"(%502) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        %506 = "ttnn.multiply"(%492, %83) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%492) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %507 = "ttnn.multiply"(%505, %84) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%505) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %508 = "ttnn.add"(%506, %507) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%507) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        "ttnn.deallocate"(%506) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %509 = "ttnn.reshape"(%508) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout50>
        "ttnn.deallocate"(%508) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %510 = "ttnn.reshape"(%494) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout50>
        "ttnn.deallocate"(%494) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %511 = "ttnn.typecast"(%501) : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%501) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %512 = "ttnn.typecast"(%509) : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%509) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> ()
        %513 = "ttnn.typecast"(%510) : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%510) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> ()
        %514 = "ttnn.repeat"(%513) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> tensor<1x8x4x512x64xf32, #ttnn_layout52>
        "ttnn.deallocate"(%513) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %515 = "ttnn.reshape"(%514) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%514) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> ()
        %516 = "ttnn.multiply"(%511, %3) : (tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%511) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %517 = "ttnn.multiply"(%512, %1) : (tensor<1x8x1x512x64xf32, #ttnn_layout51>, tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%512) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %518 = "ttnn.repeat"(%517) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> tensor<1x8x4x512x64xf32, #ttnn_layout52>
        "ttnn.deallocate"(%517) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %519 = "ttnn.reshape"(%518) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%518) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> ()
        %520 = "ttnn.permute"(%519) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x64x512xf32, #ttnn_layout18>
        %521 = "ttnn.matmul"(%516, %519) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%519) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %522 = "ttnn.add"(%521, %112) : (tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x1x512x512xf32, #ttnn_layout53>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%521) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %523 = "ttnn.softmax"(%522) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%522) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %524 = "ttnn.matmul"(%523, %515) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        %525 = "ttnn.typecast"(%524) : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%524) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %526 = "ttnn.permute"(%525) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x512x32x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%525) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %527 = "ttnn.reshape"(%526) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%526) <{force = false}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> ()
        %528 = "ttnn.matmul"(%527, %arg58) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048x2048xbf16, #ttnn_layout1>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %529 = "ttnn.add"(%528, %475) : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%528) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        "ttnn.deallocate"(%475) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %530 = "ttnn.typecast"(%529) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %531 = "ttnn.pow_scalar"(%530) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %532 = "ttnn.mean"(%531) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%531) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %533 = "ttnn.add"(%532, %6) : (tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%532) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %534 = "ttnn.rsqrt"(%533) : (tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%533) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %535 = "ttnn.multiply"(%530, %534) : (tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %536 = "ttnn.typecast"(%535) : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%535) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %537 = "ttnn.rms_norm"(%529, %arg63) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %538 = "ttnn.matmul"(%537, %arg59) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %539 = "ttnn.silu"(%538) : (tensor<1x512x8192xbf16, #ttnn_layout20>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %540 = "ttnn.matmul"(%537, %arg60) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %541 = "ttnn.multiply"(%539, %540) : (tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %542 = "ttnn.matmul"(%541, %arg61) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<2048x8192xbf16, #ttnn_layout4>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %543 = "ttnn.add"(%542, %529) : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%542) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        "ttnn.deallocate"(%529) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %544 = "ttnn.typecast"(%543) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %545 = "ttnn.pow_scalar"(%544) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %546 = "ttnn.mean"(%545) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%545) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %547 = "ttnn.add"(%546, %6) : (tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%546) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %548 = "ttnn.rsqrt"(%547) : (tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%547) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %549 = "ttnn.multiply"(%544, %548) : (tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %550 = "ttnn.typecast"(%549) : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%549) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %551 = "ttnn.rms_norm"(%543, %arg71) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %552 = "ttnn.concat"(%arg64, %arg65, %arg66) <{dim = 0 : si32}> : (tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>) -> tensor<3072x2048xbf16, #ttnn_layout42>
        %553 = "ttnn.matmul"(%551, %552) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<3072x2048xbf16, #ttnn_layout42>) -> tensor<1x512x3072xbf16, #ttnn_layout43>
        "ttnn.deallocate"(%552) <{force = false}> : (tensor<3072x2048xbf16, #ttnn_layout42>) -> ()
        %554 = "ttnn.slice_static"(%553) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 512 : i32, 2048 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %555 = "ttnn.slice_static"(%553) <{begins = [0 : i32, 0 : i32, 2048 : i32], ends = [1 : i32, 512 : i32, 2560 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x512xbf16, #ttnn_layout44>
        %556 = "ttnn.slice_static"(%553) <{begins = [0 : i32, 0 : i32, 2560 : i32], ends = [1 : i32, 512 : i32, 3072 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x512xbf16, #ttnn_layout44>
        "ttnn.deallocate"(%553) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> ()
        %557 = "ttnn.reshape"(%554) <{shape = [1 : i32, 512 : i32, 32 : i32, 64 : i32]}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x32x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%554) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %558 = "ttnn.permute"(%557) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%557) <{force = false}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> ()
        %559 = "ttnn.reshape"(%555) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> tensor<1x512x8x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%555) <{force = false}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> ()
        %560 = "ttnn.permute"(%559) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%559) <{force = false}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> ()
        %561 = "ttnn.reshape"(%556) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> tensor<1x512x8x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%556) <{force = false}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> ()
        %562 = "ttnn.permute"(%561) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%561) <{force = false}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> ()
        %563 = "ttnn.slice_static"(%558) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        %564 = "ttnn.slice_static"(%558) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        %565 = "ttnn.neg"(%564) : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        "ttnn.deallocate"(%564) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        %566 = "ttnn.concat"(%565, %563) <{dim = 3 : si32}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>, tensor<1x32x512x32xbf16, #ttnn_layout48>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%565) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        "ttnn.deallocate"(%563) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        %567 = "ttnn.multiply"(%558, %83) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%558) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %568 = "ttnn.multiply"(%566, %84) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%566) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %569 = "ttnn.add"(%567, %568) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%568) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        "ttnn.deallocate"(%567) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %570 = "ttnn.slice_static"(%560) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        %571 = "ttnn.slice_static"(%560) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        %572 = "ttnn.neg"(%571) : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        "ttnn.deallocate"(%571) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        %573 = "ttnn.concat"(%572, %570) <{dim = 3 : si32}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>, tensor<1x8x512x32xbf16, #ttnn_layout49>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%572) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        "ttnn.deallocate"(%570) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        %574 = "ttnn.multiply"(%560, %83) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%560) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %575 = "ttnn.multiply"(%573, %84) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%573) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %576 = "ttnn.add"(%574, %575) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%575) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        "ttnn.deallocate"(%574) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %577 = "ttnn.reshape"(%576) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout50>
        "ttnn.deallocate"(%576) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %578 = "ttnn.reshape"(%562) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout50>
        "ttnn.deallocate"(%562) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %579 = "ttnn.typecast"(%569) : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%569) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %580 = "ttnn.typecast"(%577) : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%577) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> ()
        %581 = "ttnn.typecast"(%578) : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%578) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> ()
        %582 = "ttnn.repeat"(%581) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> tensor<1x8x4x512x64xf32, #ttnn_layout52>
        "ttnn.deallocate"(%581) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %583 = "ttnn.reshape"(%582) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%582) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> ()
        %584 = "ttnn.multiply"(%579, %3) : (tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%579) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %585 = "ttnn.multiply"(%580, %1) : (tensor<1x8x1x512x64xf32, #ttnn_layout51>, tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%580) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %586 = "ttnn.repeat"(%585) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> tensor<1x8x4x512x64xf32, #ttnn_layout52>
        "ttnn.deallocate"(%585) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %587 = "ttnn.reshape"(%586) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%586) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> ()
        %588 = "ttnn.permute"(%587) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x64x512xf32, #ttnn_layout18>
        %589 = "ttnn.matmul"(%584, %587) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%587) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %590 = "ttnn.add"(%589, %112) : (tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x1x512x512xf32, #ttnn_layout53>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%589) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %591 = "ttnn.softmax"(%590) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%590) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %592 = "ttnn.matmul"(%591, %583) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        %593 = "ttnn.typecast"(%592) : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%592) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %594 = "ttnn.permute"(%593) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x512x32x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%593) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %595 = "ttnn.reshape"(%594) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%594) <{force = false}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> ()
        %596 = "ttnn.matmul"(%595, %arg67) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048x2048xbf16, #ttnn_layout1>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %597 = "ttnn.add"(%596, %543) : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%596) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        "ttnn.deallocate"(%543) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %598 = "ttnn.typecast"(%597) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %599 = "ttnn.pow_scalar"(%598) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %600 = "ttnn.mean"(%599) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%599) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %601 = "ttnn.add"(%600, %6) : (tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%600) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %602 = "ttnn.rsqrt"(%601) : (tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%601) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %603 = "ttnn.multiply"(%598, %602) : (tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %604 = "ttnn.typecast"(%603) : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%603) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %605 = "ttnn.rms_norm"(%597, %arg72) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %606 = "ttnn.matmul"(%605, %arg68) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %607 = "ttnn.silu"(%606) : (tensor<1x512x8192xbf16, #ttnn_layout20>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %608 = "ttnn.matmul"(%605, %arg69) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %609 = "ttnn.multiply"(%607, %608) : (tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %610 = "ttnn.matmul"(%609, %arg70) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<2048x8192xbf16, #ttnn_layout4>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %611 = "ttnn.add"(%610, %597) : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%610) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        "ttnn.deallocate"(%597) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %612 = "ttnn.typecast"(%611) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %613 = "ttnn.pow_scalar"(%612) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %614 = "ttnn.mean"(%613) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%613) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %615 = "ttnn.add"(%614, %6) : (tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%614) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %616 = "ttnn.rsqrt"(%615) : (tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%615) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %617 = "ttnn.multiply"(%612, %616) : (tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %618 = "ttnn.typecast"(%617) : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%617) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %619 = "ttnn.rms_norm"(%611, %arg80) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %620 = "ttnn.concat"(%arg73, %arg74, %arg75) <{dim = 0 : si32}> : (tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>) -> tensor<3072x2048xbf16, #ttnn_layout42>
        %621 = "ttnn.matmul"(%619, %620) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<3072x2048xbf16, #ttnn_layout42>) -> tensor<1x512x3072xbf16, #ttnn_layout43>
        "ttnn.deallocate"(%620) <{force = false}> : (tensor<3072x2048xbf16, #ttnn_layout42>) -> ()
        %622 = "ttnn.slice_static"(%621) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 512 : i32, 2048 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %623 = "ttnn.slice_static"(%621) <{begins = [0 : i32, 0 : i32, 2048 : i32], ends = [1 : i32, 512 : i32, 2560 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x512xbf16, #ttnn_layout44>
        %624 = "ttnn.slice_static"(%621) <{begins = [0 : i32, 0 : i32, 2560 : i32], ends = [1 : i32, 512 : i32, 3072 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x512xbf16, #ttnn_layout44>
        "ttnn.deallocate"(%621) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> ()
        %625 = "ttnn.reshape"(%622) <{shape = [1 : i32, 512 : i32, 32 : i32, 64 : i32]}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x32x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%622) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %626 = "ttnn.permute"(%625) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%625) <{force = false}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> ()
        %627 = "ttnn.reshape"(%623) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> tensor<1x512x8x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%623) <{force = false}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> ()
        %628 = "ttnn.permute"(%627) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%627) <{force = false}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> ()
        %629 = "ttnn.reshape"(%624) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> tensor<1x512x8x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%624) <{force = false}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> ()
        %630 = "ttnn.permute"(%629) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%629) <{force = false}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> ()
        %631 = "ttnn.slice_static"(%626) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        %632 = "ttnn.slice_static"(%626) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        %633 = "ttnn.neg"(%632) : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        "ttnn.deallocate"(%632) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        %634 = "ttnn.concat"(%633, %631) <{dim = 3 : si32}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>, tensor<1x32x512x32xbf16, #ttnn_layout48>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%633) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        "ttnn.deallocate"(%631) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        %635 = "ttnn.multiply"(%626, %83) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%626) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %636 = "ttnn.multiply"(%634, %84) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%634) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %637 = "ttnn.add"(%635, %636) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%636) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        "ttnn.deallocate"(%635) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %638 = "ttnn.slice_static"(%628) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        %639 = "ttnn.slice_static"(%628) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        %640 = "ttnn.neg"(%639) : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        "ttnn.deallocate"(%639) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        %641 = "ttnn.concat"(%640, %638) <{dim = 3 : si32}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>, tensor<1x8x512x32xbf16, #ttnn_layout49>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%640) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        "ttnn.deallocate"(%638) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        %642 = "ttnn.multiply"(%628, %83) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%628) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %643 = "ttnn.multiply"(%641, %84) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%641) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %644 = "ttnn.add"(%642, %643) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%643) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        "ttnn.deallocate"(%642) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %645 = "ttnn.reshape"(%644) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout50>
        "ttnn.deallocate"(%644) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %646 = "ttnn.reshape"(%630) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout50>
        "ttnn.deallocate"(%630) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %647 = "ttnn.typecast"(%637) : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%637) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %648 = "ttnn.typecast"(%645) : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%645) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> ()
        %649 = "ttnn.typecast"(%646) : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%646) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> ()
        %650 = "ttnn.repeat"(%649) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> tensor<1x8x4x512x64xf32, #ttnn_layout52>
        "ttnn.deallocate"(%649) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %651 = "ttnn.reshape"(%650) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%650) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> ()
        %652 = "ttnn.multiply"(%647, %3) : (tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%647) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %653 = "ttnn.multiply"(%648, %1) : (tensor<1x8x1x512x64xf32, #ttnn_layout51>, tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%648) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %654 = "ttnn.repeat"(%653) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> tensor<1x8x4x512x64xf32, #ttnn_layout52>
        "ttnn.deallocate"(%653) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %655 = "ttnn.reshape"(%654) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%654) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> ()
        %656 = "ttnn.permute"(%655) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x64x512xf32, #ttnn_layout18>
        %657 = "ttnn.matmul"(%652, %655) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%655) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %658 = "ttnn.add"(%657, %112) : (tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x1x512x512xf32, #ttnn_layout53>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%657) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %659 = "ttnn.softmax"(%658) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%658) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %660 = "ttnn.matmul"(%659, %651) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        %661 = "ttnn.typecast"(%660) : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%660) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %662 = "ttnn.permute"(%661) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x512x32x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%661) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %663 = "ttnn.reshape"(%662) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%662) <{force = false}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> ()
        %664 = "ttnn.matmul"(%663, %arg76) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048x2048xbf16, #ttnn_layout1>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %665 = "ttnn.add"(%664, %611) : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%664) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        "ttnn.deallocate"(%611) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %666 = "ttnn.typecast"(%665) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %667 = "ttnn.pow_scalar"(%666) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %668 = "ttnn.mean"(%667) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%667) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %669 = "ttnn.add"(%668, %6) : (tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%668) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %670 = "ttnn.rsqrt"(%669) : (tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%669) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %671 = "ttnn.multiply"(%666, %670) : (tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %672 = "ttnn.typecast"(%671) : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%671) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %673 = "ttnn.rms_norm"(%665, %arg81) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %674 = "ttnn.matmul"(%673, %arg77) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %675 = "ttnn.silu"(%674) : (tensor<1x512x8192xbf16, #ttnn_layout20>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %676 = "ttnn.matmul"(%673, %arg78) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %677 = "ttnn.multiply"(%675, %676) : (tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %678 = "ttnn.matmul"(%677, %arg79) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<2048x8192xbf16, #ttnn_layout4>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %679 = "ttnn.add"(%678, %665) : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%678) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        "ttnn.deallocate"(%665) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %680 = "ttnn.typecast"(%679) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %681 = "ttnn.pow_scalar"(%680) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %682 = "ttnn.mean"(%681) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%681) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %683 = "ttnn.add"(%682, %6) : (tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%682) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %684 = "ttnn.rsqrt"(%683) : (tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%683) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %685 = "ttnn.multiply"(%680, %684) : (tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %686 = "ttnn.typecast"(%685) : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%685) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %687 = "ttnn.rms_norm"(%679, %arg89) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %688 = "ttnn.concat"(%arg82, %arg83, %arg84) <{dim = 0 : si32}> : (tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>) -> tensor<3072x2048xbf16, #ttnn_layout42>
        %689 = "ttnn.matmul"(%687, %688) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<3072x2048xbf16, #ttnn_layout42>) -> tensor<1x512x3072xbf16, #ttnn_layout43>
        "ttnn.deallocate"(%688) <{force = false}> : (tensor<3072x2048xbf16, #ttnn_layout42>) -> ()
        %690 = "ttnn.slice_static"(%689) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 512 : i32, 2048 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %691 = "ttnn.slice_static"(%689) <{begins = [0 : i32, 0 : i32, 2048 : i32], ends = [1 : i32, 512 : i32, 2560 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x512xbf16, #ttnn_layout44>
        %692 = "ttnn.slice_static"(%689) <{begins = [0 : i32, 0 : i32, 2560 : i32], ends = [1 : i32, 512 : i32, 3072 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x512xbf16, #ttnn_layout44>
        "ttnn.deallocate"(%689) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> ()
        %693 = "ttnn.reshape"(%690) <{shape = [1 : i32, 512 : i32, 32 : i32, 64 : i32]}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x32x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%690) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %694 = "ttnn.permute"(%693) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%693) <{force = false}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> ()
        %695 = "ttnn.reshape"(%691) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> tensor<1x512x8x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%691) <{force = false}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> ()
        %696 = "ttnn.permute"(%695) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%695) <{force = false}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> ()
        %697 = "ttnn.reshape"(%692) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> tensor<1x512x8x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%692) <{force = false}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> ()
        %698 = "ttnn.permute"(%697) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%697) <{force = false}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> ()
        %699 = "ttnn.slice_static"(%694) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        %700 = "ttnn.slice_static"(%694) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        %701 = "ttnn.neg"(%700) : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        "ttnn.deallocate"(%700) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        %702 = "ttnn.concat"(%701, %699) <{dim = 3 : si32}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>, tensor<1x32x512x32xbf16, #ttnn_layout48>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%701) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        "ttnn.deallocate"(%699) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        %703 = "ttnn.multiply"(%694, %83) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%694) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %704 = "ttnn.multiply"(%702, %84) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%702) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %705 = "ttnn.add"(%703, %704) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%704) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        "ttnn.deallocate"(%703) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %706 = "ttnn.slice_static"(%696) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        %707 = "ttnn.slice_static"(%696) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        %708 = "ttnn.neg"(%707) : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        "ttnn.deallocate"(%707) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        %709 = "ttnn.concat"(%708, %706) <{dim = 3 : si32}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>, tensor<1x8x512x32xbf16, #ttnn_layout49>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%708) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        "ttnn.deallocate"(%706) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        %710 = "ttnn.multiply"(%696, %83) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%696) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %711 = "ttnn.multiply"(%709, %84) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%709) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %712 = "ttnn.add"(%710, %711) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%711) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        "ttnn.deallocate"(%710) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %713 = "ttnn.reshape"(%712) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout50>
        "ttnn.deallocate"(%712) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %714 = "ttnn.reshape"(%698) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout50>
        "ttnn.deallocate"(%698) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %715 = "ttnn.typecast"(%705) : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%705) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %716 = "ttnn.typecast"(%713) : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%713) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> ()
        %717 = "ttnn.typecast"(%714) : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%714) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> ()
        %718 = "ttnn.repeat"(%717) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> tensor<1x8x4x512x64xf32, #ttnn_layout52>
        "ttnn.deallocate"(%717) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %719 = "ttnn.reshape"(%718) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%718) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> ()
        %720 = "ttnn.multiply"(%715, %3) : (tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%715) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %721 = "ttnn.multiply"(%716, %1) : (tensor<1x8x1x512x64xf32, #ttnn_layout51>, tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%716) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %722 = "ttnn.repeat"(%721) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> tensor<1x8x4x512x64xf32, #ttnn_layout52>
        "ttnn.deallocate"(%721) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %723 = "ttnn.reshape"(%722) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%722) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> ()
        %724 = "ttnn.permute"(%723) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x64x512xf32, #ttnn_layout18>
        %725 = "ttnn.matmul"(%720, %723) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%723) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %726 = "ttnn.add"(%725, %112) : (tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x1x512x512xf32, #ttnn_layout53>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%725) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %727 = "ttnn.softmax"(%726) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%726) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %728 = "ttnn.matmul"(%727, %719) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        %729 = "ttnn.typecast"(%728) : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%728) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %730 = "ttnn.permute"(%729) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x512x32x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%729) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %731 = "ttnn.reshape"(%730) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%730) <{force = false}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> ()
        %732 = "ttnn.matmul"(%731, %arg85) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048x2048xbf16, #ttnn_layout1>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %733 = "ttnn.add"(%732, %679) : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%732) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        "ttnn.deallocate"(%679) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %734 = "ttnn.typecast"(%733) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %735 = "ttnn.pow_scalar"(%734) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %736 = "ttnn.mean"(%735) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%735) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %737 = "ttnn.add"(%736, %6) : (tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%736) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %738 = "ttnn.rsqrt"(%737) : (tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%737) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %739 = "ttnn.multiply"(%734, %738) : (tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %740 = "ttnn.typecast"(%739) : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%739) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %741 = "ttnn.rms_norm"(%733, %arg90) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %742 = "ttnn.matmul"(%741, %arg86) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %743 = "ttnn.silu"(%742) : (tensor<1x512x8192xbf16, #ttnn_layout20>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %744 = "ttnn.matmul"(%741, %arg87) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %745 = "ttnn.multiply"(%743, %744) : (tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %746 = "ttnn.matmul"(%745, %arg88) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<2048x8192xbf16, #ttnn_layout4>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %747 = "ttnn.add"(%746, %733) : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%746) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        "ttnn.deallocate"(%733) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %748 = "ttnn.typecast"(%747) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %749 = "ttnn.pow_scalar"(%748) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %750 = "ttnn.mean"(%749) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%749) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %751 = "ttnn.add"(%750, %6) : (tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%750) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %752 = "ttnn.rsqrt"(%751) : (tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%751) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %753 = "ttnn.multiply"(%748, %752) : (tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %754 = "ttnn.typecast"(%753) : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%753) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %755 = "ttnn.rms_norm"(%747, %arg98) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %756 = "ttnn.concat"(%arg91, %arg92, %arg93) <{dim = 0 : si32}> : (tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>) -> tensor<3072x2048xbf16, #ttnn_layout42>
        %757 = "ttnn.matmul"(%755, %756) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<3072x2048xbf16, #ttnn_layout42>) -> tensor<1x512x3072xbf16, #ttnn_layout43>
        "ttnn.deallocate"(%756) <{force = false}> : (tensor<3072x2048xbf16, #ttnn_layout42>) -> ()
        %758 = "ttnn.slice_static"(%757) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 512 : i32, 2048 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %759 = "ttnn.slice_static"(%757) <{begins = [0 : i32, 0 : i32, 2048 : i32], ends = [1 : i32, 512 : i32, 2560 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x512xbf16, #ttnn_layout44>
        %760 = "ttnn.slice_static"(%757) <{begins = [0 : i32, 0 : i32, 2560 : i32], ends = [1 : i32, 512 : i32, 3072 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x512xbf16, #ttnn_layout44>
        "ttnn.deallocate"(%757) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> ()
        %761 = "ttnn.reshape"(%758) <{shape = [1 : i32, 512 : i32, 32 : i32, 64 : i32]}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x32x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%758) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %762 = "ttnn.permute"(%761) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%761) <{force = false}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> ()
        %763 = "ttnn.reshape"(%759) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> tensor<1x512x8x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%759) <{force = false}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> ()
        %764 = "ttnn.permute"(%763) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%763) <{force = false}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> ()
        %765 = "ttnn.reshape"(%760) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> tensor<1x512x8x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%760) <{force = false}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> ()
        %766 = "ttnn.permute"(%765) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%765) <{force = false}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> ()
        %767 = "ttnn.slice_static"(%762) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        %768 = "ttnn.slice_static"(%762) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        %769 = "ttnn.neg"(%768) : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        "ttnn.deallocate"(%768) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        %770 = "ttnn.concat"(%769, %767) <{dim = 3 : si32}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>, tensor<1x32x512x32xbf16, #ttnn_layout48>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%769) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        "ttnn.deallocate"(%767) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        %771 = "ttnn.multiply"(%762, %83) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%762) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %772 = "ttnn.multiply"(%770, %84) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%770) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %773 = "ttnn.add"(%771, %772) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%772) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        "ttnn.deallocate"(%771) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %774 = "ttnn.slice_static"(%764) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        %775 = "ttnn.slice_static"(%764) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        %776 = "ttnn.neg"(%775) : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        "ttnn.deallocate"(%775) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        %777 = "ttnn.concat"(%776, %774) <{dim = 3 : si32}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>, tensor<1x8x512x32xbf16, #ttnn_layout49>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%776) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        "ttnn.deallocate"(%774) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        %778 = "ttnn.multiply"(%764, %83) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%764) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %779 = "ttnn.multiply"(%777, %84) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%777) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %780 = "ttnn.add"(%778, %779) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%779) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        "ttnn.deallocate"(%778) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %781 = "ttnn.reshape"(%780) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout50>
        "ttnn.deallocate"(%780) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %782 = "ttnn.reshape"(%766) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout50>
        "ttnn.deallocate"(%766) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %783 = "ttnn.typecast"(%773) : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%773) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %784 = "ttnn.typecast"(%781) : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%781) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> ()
        %785 = "ttnn.typecast"(%782) : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%782) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> ()
        %786 = "ttnn.repeat"(%785) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> tensor<1x8x4x512x64xf32, #ttnn_layout52>
        "ttnn.deallocate"(%785) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %787 = "ttnn.reshape"(%786) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%786) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> ()
        %788 = "ttnn.multiply"(%783, %3) : (tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%783) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %789 = "ttnn.multiply"(%784, %1) : (tensor<1x8x1x512x64xf32, #ttnn_layout51>, tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%784) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %790 = "ttnn.repeat"(%789) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> tensor<1x8x4x512x64xf32, #ttnn_layout52>
        "ttnn.deallocate"(%789) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %791 = "ttnn.reshape"(%790) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%790) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> ()
        %792 = "ttnn.permute"(%791) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x64x512xf32, #ttnn_layout18>
        %793 = "ttnn.matmul"(%788, %791) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%791) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %794 = "ttnn.add"(%793, %112) : (tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x1x512x512xf32, #ttnn_layout53>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%793) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %795 = "ttnn.softmax"(%794) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%794) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %796 = "ttnn.matmul"(%795, %787) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        %797 = "ttnn.typecast"(%796) : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%796) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %798 = "ttnn.permute"(%797) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x512x32x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%797) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %799 = "ttnn.reshape"(%798) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%798) <{force = false}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> ()
        %800 = "ttnn.matmul"(%799, %arg94) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048x2048xbf16, #ttnn_layout1>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %801 = "ttnn.add"(%800, %747) : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%800) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        "ttnn.deallocate"(%747) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %802 = "ttnn.typecast"(%801) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %803 = "ttnn.pow_scalar"(%802) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %804 = "ttnn.mean"(%803) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%803) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %805 = "ttnn.add"(%804, %6) : (tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%804) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %806 = "ttnn.rsqrt"(%805) : (tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%805) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %807 = "ttnn.multiply"(%802, %806) : (tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %808 = "ttnn.typecast"(%807) : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%807) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %809 = "ttnn.rms_norm"(%801, %arg99) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %810 = "ttnn.matmul"(%809, %arg95) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %811 = "ttnn.silu"(%810) : (tensor<1x512x8192xbf16, #ttnn_layout20>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %812 = "ttnn.matmul"(%809, %arg96) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %813 = "ttnn.multiply"(%811, %812) : (tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %814 = "ttnn.matmul"(%813, %arg97) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<2048x8192xbf16, #ttnn_layout4>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %815 = "ttnn.add"(%814, %801) : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%814) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        "ttnn.deallocate"(%801) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %816 = "ttnn.typecast"(%815) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %817 = "ttnn.pow_scalar"(%816) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %818 = "ttnn.mean"(%817) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%817) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %819 = "ttnn.add"(%818, %6) : (tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%818) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %820 = "ttnn.rsqrt"(%819) : (tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%819) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %821 = "ttnn.multiply"(%816, %820) : (tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %822 = "ttnn.typecast"(%821) : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%821) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %823 = "ttnn.rms_norm"(%815, %arg107) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %824 = "ttnn.concat"(%arg100, %arg101, %arg102) <{dim = 0 : si32}> : (tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>) -> tensor<3072x2048xbf16, #ttnn_layout42>
        %825 = "ttnn.matmul"(%823, %824) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<3072x2048xbf16, #ttnn_layout42>) -> tensor<1x512x3072xbf16, #ttnn_layout43>
        "ttnn.deallocate"(%824) <{force = false}> : (tensor<3072x2048xbf16, #ttnn_layout42>) -> ()
        %826 = "ttnn.slice_static"(%825) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 512 : i32, 2048 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %827 = "ttnn.slice_static"(%825) <{begins = [0 : i32, 0 : i32, 2048 : i32], ends = [1 : i32, 512 : i32, 2560 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x512xbf16, #ttnn_layout44>
        %828 = "ttnn.slice_static"(%825) <{begins = [0 : i32, 0 : i32, 2560 : i32], ends = [1 : i32, 512 : i32, 3072 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x512xbf16, #ttnn_layout44>
        "ttnn.deallocate"(%825) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> ()
        %829 = "ttnn.reshape"(%826) <{shape = [1 : i32, 512 : i32, 32 : i32, 64 : i32]}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x32x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%826) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %830 = "ttnn.permute"(%829) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%829) <{force = false}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> ()
        %831 = "ttnn.reshape"(%827) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> tensor<1x512x8x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%827) <{force = false}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> ()
        %832 = "ttnn.permute"(%831) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%831) <{force = false}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> ()
        %833 = "ttnn.reshape"(%828) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> tensor<1x512x8x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%828) <{force = false}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> ()
        %834 = "ttnn.permute"(%833) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%833) <{force = false}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> ()
        %835 = "ttnn.slice_static"(%830) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        %836 = "ttnn.slice_static"(%830) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        %837 = "ttnn.neg"(%836) : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        "ttnn.deallocate"(%836) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        %838 = "ttnn.concat"(%837, %835) <{dim = 3 : si32}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>, tensor<1x32x512x32xbf16, #ttnn_layout48>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%837) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        "ttnn.deallocate"(%835) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        %839 = "ttnn.multiply"(%830, %83) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%830) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %840 = "ttnn.multiply"(%838, %84) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%838) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %841 = "ttnn.add"(%839, %840) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%840) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        "ttnn.deallocate"(%839) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %842 = "ttnn.slice_static"(%832) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        %843 = "ttnn.slice_static"(%832) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        %844 = "ttnn.neg"(%843) : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        "ttnn.deallocate"(%843) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        %845 = "ttnn.concat"(%844, %842) <{dim = 3 : si32}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>, tensor<1x8x512x32xbf16, #ttnn_layout49>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%844) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        "ttnn.deallocate"(%842) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        %846 = "ttnn.multiply"(%832, %83) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%832) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %847 = "ttnn.multiply"(%845, %84) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%845) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %848 = "ttnn.add"(%846, %847) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%847) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        "ttnn.deallocate"(%846) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %849 = "ttnn.reshape"(%848) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout50>
        "ttnn.deallocate"(%848) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %850 = "ttnn.reshape"(%834) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout50>
        "ttnn.deallocate"(%834) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %851 = "ttnn.typecast"(%841) : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%841) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %852 = "ttnn.typecast"(%849) : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%849) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> ()
        %853 = "ttnn.typecast"(%850) : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%850) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> ()
        %854 = "ttnn.repeat"(%853) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> tensor<1x8x4x512x64xf32, #ttnn_layout52>
        "ttnn.deallocate"(%853) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %855 = "ttnn.reshape"(%854) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%854) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> ()
        %856 = "ttnn.multiply"(%851, %3) : (tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%851) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %857 = "ttnn.multiply"(%852, %1) : (tensor<1x8x1x512x64xf32, #ttnn_layout51>, tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%852) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %858 = "ttnn.repeat"(%857) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> tensor<1x8x4x512x64xf32, #ttnn_layout52>
        "ttnn.deallocate"(%857) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %859 = "ttnn.reshape"(%858) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%858) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> ()
        %860 = "ttnn.permute"(%859) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x64x512xf32, #ttnn_layout18>
        %861 = "ttnn.matmul"(%856, %859) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%859) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %862 = "ttnn.add"(%861, %112) : (tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x1x512x512xf32, #ttnn_layout53>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%861) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %863 = "ttnn.softmax"(%862) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%862) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %864 = "ttnn.matmul"(%863, %855) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        %865 = "ttnn.typecast"(%864) : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%864) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %866 = "ttnn.permute"(%865) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x512x32x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%865) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %867 = "ttnn.reshape"(%866) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%866) <{force = false}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> ()
        %868 = "ttnn.matmul"(%867, %arg103) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048x2048xbf16, #ttnn_layout1>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %869 = "ttnn.add"(%868, %815) : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%868) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        "ttnn.deallocate"(%815) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %870 = "ttnn.typecast"(%869) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %871 = "ttnn.pow_scalar"(%870) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %872 = "ttnn.mean"(%871) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%871) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %873 = "ttnn.add"(%872, %6) : (tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%872) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %874 = "ttnn.rsqrt"(%873) : (tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%873) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %875 = "ttnn.multiply"(%870, %874) : (tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %876 = "ttnn.typecast"(%875) : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%875) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %877 = "ttnn.rms_norm"(%869, %arg108) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %878 = "ttnn.matmul"(%877, %arg104) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %879 = "ttnn.silu"(%878) : (tensor<1x512x8192xbf16, #ttnn_layout20>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %880 = "ttnn.matmul"(%877, %arg105) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %881 = "ttnn.multiply"(%879, %880) : (tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %882 = "ttnn.matmul"(%881, %arg106) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<2048x8192xbf16, #ttnn_layout4>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %883 = "ttnn.add"(%882, %869) : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%882) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        "ttnn.deallocate"(%869) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %884 = "ttnn.typecast"(%883) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %885 = "ttnn.pow_scalar"(%884) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %886 = "ttnn.mean"(%885) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%885) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %887 = "ttnn.add"(%886, %6) : (tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%886) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %888 = "ttnn.rsqrt"(%887) : (tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%887) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %889 = "ttnn.multiply"(%884, %888) : (tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %890 = "ttnn.typecast"(%889) : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%889) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %891 = "ttnn.rms_norm"(%883, %arg116) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %892 = "ttnn.concat"(%arg109, %arg110, %arg111) <{dim = 0 : si32}> : (tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>) -> tensor<3072x2048xbf16, #ttnn_layout42>
        %893 = "ttnn.matmul"(%891, %892) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<3072x2048xbf16, #ttnn_layout42>) -> tensor<1x512x3072xbf16, #ttnn_layout43>
        "ttnn.deallocate"(%892) <{force = false}> : (tensor<3072x2048xbf16, #ttnn_layout42>) -> ()
        %894 = "ttnn.slice_static"(%893) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 512 : i32, 2048 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %895 = "ttnn.slice_static"(%893) <{begins = [0 : i32, 0 : i32, 2048 : i32], ends = [1 : i32, 512 : i32, 2560 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x512xbf16, #ttnn_layout44>
        %896 = "ttnn.slice_static"(%893) <{begins = [0 : i32, 0 : i32, 2560 : i32], ends = [1 : i32, 512 : i32, 3072 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x512xbf16, #ttnn_layout44>
        "ttnn.deallocate"(%893) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> ()
        %897 = "ttnn.reshape"(%894) <{shape = [1 : i32, 512 : i32, 32 : i32, 64 : i32]}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x32x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%894) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %898 = "ttnn.permute"(%897) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%897) <{force = false}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> ()
        %899 = "ttnn.reshape"(%895) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> tensor<1x512x8x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%895) <{force = false}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> ()
        %900 = "ttnn.permute"(%899) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%899) <{force = false}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> ()
        %901 = "ttnn.reshape"(%896) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> tensor<1x512x8x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%896) <{force = false}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> ()
        %902 = "ttnn.permute"(%901) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%901) <{force = false}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> ()
        %903 = "ttnn.slice_static"(%898) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        %904 = "ttnn.slice_static"(%898) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        %905 = "ttnn.neg"(%904) : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        "ttnn.deallocate"(%904) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        %906 = "ttnn.concat"(%905, %903) <{dim = 3 : si32}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>, tensor<1x32x512x32xbf16, #ttnn_layout48>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%905) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        "ttnn.deallocate"(%903) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        %907 = "ttnn.multiply"(%898, %83) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%898) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %908 = "ttnn.multiply"(%906, %84) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%906) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %909 = "ttnn.add"(%907, %908) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%908) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        "ttnn.deallocate"(%907) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %910 = "ttnn.slice_static"(%900) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        %911 = "ttnn.slice_static"(%900) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        %912 = "ttnn.neg"(%911) : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        "ttnn.deallocate"(%911) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        %913 = "ttnn.concat"(%912, %910) <{dim = 3 : si32}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>, tensor<1x8x512x32xbf16, #ttnn_layout49>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%912) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        "ttnn.deallocate"(%910) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        %914 = "ttnn.multiply"(%900, %83) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%900) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %915 = "ttnn.multiply"(%913, %84) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%913) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %916 = "ttnn.add"(%914, %915) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%915) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        "ttnn.deallocate"(%914) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %917 = "ttnn.reshape"(%916) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout50>
        "ttnn.deallocate"(%916) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %918 = "ttnn.reshape"(%902) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout50>
        "ttnn.deallocate"(%902) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %919 = "ttnn.typecast"(%909) : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%909) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %920 = "ttnn.typecast"(%917) : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%917) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> ()
        %921 = "ttnn.typecast"(%918) : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%918) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> ()
        %922 = "ttnn.repeat"(%921) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> tensor<1x8x4x512x64xf32, #ttnn_layout52>
        "ttnn.deallocate"(%921) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %923 = "ttnn.reshape"(%922) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%922) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> ()
        %924 = "ttnn.multiply"(%919, %3) : (tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%919) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %925 = "ttnn.multiply"(%920, %1) : (tensor<1x8x1x512x64xf32, #ttnn_layout51>, tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%920) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %926 = "ttnn.repeat"(%925) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> tensor<1x8x4x512x64xf32, #ttnn_layout52>
        "ttnn.deallocate"(%925) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %927 = "ttnn.reshape"(%926) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%926) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> ()
        %928 = "ttnn.permute"(%927) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x64x512xf32, #ttnn_layout18>
        %929 = "ttnn.matmul"(%924, %927) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%927) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %930 = "ttnn.add"(%929, %112) : (tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x1x512x512xf32, #ttnn_layout53>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%929) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %931 = "ttnn.softmax"(%930) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%930) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %932 = "ttnn.matmul"(%931, %923) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        %933 = "ttnn.typecast"(%932) : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%932) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %934 = "ttnn.permute"(%933) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x512x32x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%933) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %935 = "ttnn.reshape"(%934) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%934) <{force = false}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> ()
        %936 = "ttnn.matmul"(%935, %arg112) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048x2048xbf16, #ttnn_layout1>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %937 = "ttnn.add"(%936, %883) : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%936) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        "ttnn.deallocate"(%883) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %938 = "ttnn.typecast"(%937) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %939 = "ttnn.pow_scalar"(%938) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %940 = "ttnn.mean"(%939) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%939) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %941 = "ttnn.add"(%940, %6) : (tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%940) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %942 = "ttnn.rsqrt"(%941) : (tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%941) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %943 = "ttnn.multiply"(%938, %942) : (tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %944 = "ttnn.typecast"(%943) : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%943) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %945 = "ttnn.rms_norm"(%937, %arg117) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %946 = "ttnn.matmul"(%945, %arg113) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %947 = "ttnn.silu"(%946) : (tensor<1x512x8192xbf16, #ttnn_layout20>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %948 = "ttnn.matmul"(%945, %arg114) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %949 = "ttnn.multiply"(%947, %948) : (tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %950 = "ttnn.matmul"(%949, %arg115) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<2048x8192xbf16, #ttnn_layout4>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %951 = "ttnn.add"(%950, %937) : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%950) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        "ttnn.deallocate"(%937) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %952 = "ttnn.typecast"(%951) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %953 = "ttnn.pow_scalar"(%952) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %954 = "ttnn.mean"(%953) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%953) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %955 = "ttnn.add"(%954, %6) : (tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%954) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %956 = "ttnn.rsqrt"(%955) : (tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%955) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %957 = "ttnn.multiply"(%952, %956) : (tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %958 = "ttnn.typecast"(%957) : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%957) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %959 = "ttnn.rms_norm"(%951, %arg125) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %960 = "ttnn.concat"(%arg118, %arg119, %arg120) <{dim = 0 : si32}> : (tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>) -> tensor<3072x2048xbf16, #ttnn_layout42>
        %961 = "ttnn.matmul"(%959, %960) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<3072x2048xbf16, #ttnn_layout42>) -> tensor<1x512x3072xbf16, #ttnn_layout43>
        "ttnn.deallocate"(%960) <{force = false}> : (tensor<3072x2048xbf16, #ttnn_layout42>) -> ()
        %962 = "ttnn.slice_static"(%961) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 512 : i32, 2048 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %963 = "ttnn.slice_static"(%961) <{begins = [0 : i32, 0 : i32, 2048 : i32], ends = [1 : i32, 512 : i32, 2560 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x512xbf16, #ttnn_layout44>
        %964 = "ttnn.slice_static"(%961) <{begins = [0 : i32, 0 : i32, 2560 : i32], ends = [1 : i32, 512 : i32, 3072 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x512xbf16, #ttnn_layout44>
        "ttnn.deallocate"(%961) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> ()
        %965 = "ttnn.reshape"(%962) <{shape = [1 : i32, 512 : i32, 32 : i32, 64 : i32]}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x32x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%962) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %966 = "ttnn.permute"(%965) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%965) <{force = false}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> ()
        %967 = "ttnn.reshape"(%963) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> tensor<1x512x8x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%963) <{force = false}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> ()
        %968 = "ttnn.permute"(%967) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%967) <{force = false}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> ()
        %969 = "ttnn.reshape"(%964) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> tensor<1x512x8x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%964) <{force = false}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> ()
        %970 = "ttnn.permute"(%969) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%969) <{force = false}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> ()
        %971 = "ttnn.slice_static"(%966) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        %972 = "ttnn.slice_static"(%966) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        %973 = "ttnn.neg"(%972) : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        "ttnn.deallocate"(%972) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        %974 = "ttnn.concat"(%973, %971) <{dim = 3 : si32}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>, tensor<1x32x512x32xbf16, #ttnn_layout48>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%973) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        "ttnn.deallocate"(%971) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        %975 = "ttnn.multiply"(%966, %83) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%966) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %976 = "ttnn.multiply"(%974, %84) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%974) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %977 = "ttnn.add"(%975, %976) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%976) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        "ttnn.deallocate"(%975) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %978 = "ttnn.slice_static"(%968) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        %979 = "ttnn.slice_static"(%968) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        %980 = "ttnn.neg"(%979) : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        "ttnn.deallocate"(%979) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        %981 = "ttnn.concat"(%980, %978) <{dim = 3 : si32}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>, tensor<1x8x512x32xbf16, #ttnn_layout49>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%980) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        "ttnn.deallocate"(%978) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        %982 = "ttnn.multiply"(%968, %83) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%968) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %983 = "ttnn.multiply"(%981, %84) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%981) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %984 = "ttnn.add"(%982, %983) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%983) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        "ttnn.deallocate"(%982) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %985 = "ttnn.reshape"(%984) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout50>
        "ttnn.deallocate"(%984) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %986 = "ttnn.reshape"(%970) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout50>
        "ttnn.deallocate"(%970) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %987 = "ttnn.typecast"(%977) : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%977) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %988 = "ttnn.typecast"(%985) : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%985) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> ()
        %989 = "ttnn.typecast"(%986) : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%986) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> ()
        %990 = "ttnn.repeat"(%989) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> tensor<1x8x4x512x64xf32, #ttnn_layout52>
        "ttnn.deallocate"(%989) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %991 = "ttnn.reshape"(%990) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%990) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> ()
        %992 = "ttnn.multiply"(%987, %3) : (tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%987) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %993 = "ttnn.multiply"(%988, %1) : (tensor<1x8x1x512x64xf32, #ttnn_layout51>, tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%988) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %994 = "ttnn.repeat"(%993) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> tensor<1x8x4x512x64xf32, #ttnn_layout52>
        "ttnn.deallocate"(%993) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %995 = "ttnn.reshape"(%994) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%994) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> ()
        %996 = "ttnn.permute"(%995) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x64x512xf32, #ttnn_layout18>
        %997 = "ttnn.matmul"(%992, %995) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%995) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %998 = "ttnn.add"(%997, %112) : (tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x1x512x512xf32, #ttnn_layout53>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%997) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %999 = "ttnn.softmax"(%998) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%998) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %1000 = "ttnn.matmul"(%999, %991) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        %1001 = "ttnn.typecast"(%1000) : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%1000) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %1002 = "ttnn.permute"(%1001) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x512x32x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%1001) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %1003 = "ttnn.reshape"(%1002) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%1002) <{force = false}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> ()
        %1004 = "ttnn.matmul"(%1003, %arg121) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048x2048xbf16, #ttnn_layout1>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %1005 = "ttnn.add"(%1004, %951) : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%1004) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        "ttnn.deallocate"(%951) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %1006 = "ttnn.typecast"(%1005) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %1007 = "ttnn.pow_scalar"(%1006) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %1008 = "ttnn.mean"(%1007) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%1007) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %1009 = "ttnn.add"(%1008, %6) : (tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%1008) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %1010 = "ttnn.rsqrt"(%1009) : (tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%1009) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %1011 = "ttnn.multiply"(%1006, %1010) : (tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %1012 = "ttnn.typecast"(%1011) : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%1011) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %1013 = "ttnn.rms_norm"(%1005, %arg126) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %1014 = "ttnn.matmul"(%1013, %arg122) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %1015 = "ttnn.silu"(%1014) : (tensor<1x512x8192xbf16, #ttnn_layout20>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %1016 = "ttnn.matmul"(%1013, %arg123) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %1017 = "ttnn.multiply"(%1015, %1016) : (tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %1018 = "ttnn.matmul"(%1017, %arg124) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<2048x8192xbf16, #ttnn_layout4>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %1019 = "ttnn.add"(%1018, %1005) : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%1018) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        "ttnn.deallocate"(%1005) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %1020 = "ttnn.typecast"(%1019) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %1021 = "ttnn.pow_scalar"(%1020) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %1022 = "ttnn.mean"(%1021) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%1021) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %1023 = "ttnn.add"(%1022, %6) : (tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%1022) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %1024 = "ttnn.rsqrt"(%1023) : (tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%1023) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %1025 = "ttnn.multiply"(%1020, %1024) : (tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %1026 = "ttnn.typecast"(%1025) : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%1025) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %1027 = "ttnn.rms_norm"(%1019, %arg134) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %1028 = "ttnn.concat"(%arg127, %arg128, %arg129) <{dim = 0 : si32}> : (tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>) -> tensor<3072x2048xbf16, #ttnn_layout42>
        %1029 = "ttnn.matmul"(%1027, %1028) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<3072x2048xbf16, #ttnn_layout42>) -> tensor<1x512x3072xbf16, #ttnn_layout43>
        "ttnn.deallocate"(%1028) <{force = false}> : (tensor<3072x2048xbf16, #ttnn_layout42>) -> ()
        %1030 = "ttnn.slice_static"(%1029) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 512 : i32, 2048 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %1031 = "ttnn.slice_static"(%1029) <{begins = [0 : i32, 0 : i32, 2048 : i32], ends = [1 : i32, 512 : i32, 2560 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x512xbf16, #ttnn_layout44>
        %1032 = "ttnn.slice_static"(%1029) <{begins = [0 : i32, 0 : i32, 2560 : i32], ends = [1 : i32, 512 : i32, 3072 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x512xbf16, #ttnn_layout44>
        "ttnn.deallocate"(%1029) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> ()
        %1033 = "ttnn.reshape"(%1030) <{shape = [1 : i32, 512 : i32, 32 : i32, 64 : i32]}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x32x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%1030) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %1034 = "ttnn.permute"(%1033) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%1033) <{force = false}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> ()
        %1035 = "ttnn.reshape"(%1031) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> tensor<1x512x8x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%1031) <{force = false}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> ()
        %1036 = "ttnn.permute"(%1035) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%1035) <{force = false}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> ()
        %1037 = "ttnn.reshape"(%1032) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> tensor<1x512x8x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%1032) <{force = false}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> ()
        %1038 = "ttnn.permute"(%1037) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%1037) <{force = false}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> ()
        %1039 = "ttnn.slice_static"(%1034) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        %1040 = "ttnn.slice_static"(%1034) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        %1041 = "ttnn.neg"(%1040) : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        "ttnn.deallocate"(%1040) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        %1042 = "ttnn.concat"(%1041, %1039) <{dim = 3 : si32}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>, tensor<1x32x512x32xbf16, #ttnn_layout48>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%1041) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        "ttnn.deallocate"(%1039) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        %1043 = "ttnn.multiply"(%1034, %83) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%1034) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %1044 = "ttnn.multiply"(%1042, %84) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%1042) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %1045 = "ttnn.add"(%1043, %1044) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%1044) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        "ttnn.deallocate"(%1043) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %1046 = "ttnn.slice_static"(%1036) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        %1047 = "ttnn.slice_static"(%1036) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        %1048 = "ttnn.neg"(%1047) : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        "ttnn.deallocate"(%1047) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        %1049 = "ttnn.concat"(%1048, %1046) <{dim = 3 : si32}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>, tensor<1x8x512x32xbf16, #ttnn_layout49>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%1048) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        "ttnn.deallocate"(%1046) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        %1050 = "ttnn.multiply"(%1036, %83) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%1036) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %1051 = "ttnn.multiply"(%1049, %84) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%1049) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %1052 = "ttnn.add"(%1050, %1051) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%1051) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        "ttnn.deallocate"(%1050) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %1053 = "ttnn.reshape"(%1052) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout50>
        "ttnn.deallocate"(%1052) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %1054 = "ttnn.reshape"(%1038) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout50>
        "ttnn.deallocate"(%1038) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %1055 = "ttnn.typecast"(%1045) : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%1045) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %1056 = "ttnn.typecast"(%1053) : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%1053) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> ()
        %1057 = "ttnn.typecast"(%1054) : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%1054) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> ()
        %1058 = "ttnn.repeat"(%1057) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> tensor<1x8x4x512x64xf32, #ttnn_layout52>
        "ttnn.deallocate"(%1057) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %1059 = "ttnn.reshape"(%1058) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%1058) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> ()
        %1060 = "ttnn.multiply"(%1055, %3) : (tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%1055) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %1061 = "ttnn.multiply"(%1056, %1) : (tensor<1x8x1x512x64xf32, #ttnn_layout51>, tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%1056) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %1062 = "ttnn.repeat"(%1061) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> tensor<1x8x4x512x64xf32, #ttnn_layout52>
        "ttnn.deallocate"(%1061) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %1063 = "ttnn.reshape"(%1062) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%1062) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> ()
        %1064 = "ttnn.permute"(%1063) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x64x512xf32, #ttnn_layout18>
        %1065 = "ttnn.matmul"(%1060, %1063) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%1063) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %1066 = "ttnn.add"(%1065, %112) : (tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x1x512x512xf32, #ttnn_layout53>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%1065) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %1067 = "ttnn.softmax"(%1066) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%1066) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %1068 = "ttnn.matmul"(%1067, %1059) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        %1069 = "ttnn.typecast"(%1068) : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%1068) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %1070 = "ttnn.permute"(%1069) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x512x32x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%1069) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %1071 = "ttnn.reshape"(%1070) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%1070) <{force = false}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> ()
        %1072 = "ttnn.matmul"(%1071, %arg130) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048x2048xbf16, #ttnn_layout1>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %1073 = "ttnn.add"(%1072, %1019) : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%1072) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        "ttnn.deallocate"(%1019) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %1074 = "ttnn.typecast"(%1073) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %1075 = "ttnn.pow_scalar"(%1074) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %1076 = "ttnn.mean"(%1075) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%1075) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %1077 = "ttnn.add"(%1076, %6) : (tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%1076) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %1078 = "ttnn.rsqrt"(%1077) : (tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%1077) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %1079 = "ttnn.multiply"(%1074, %1078) : (tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %1080 = "ttnn.typecast"(%1079) : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%1079) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %1081 = "ttnn.rms_norm"(%1073, %arg135) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %1082 = "ttnn.matmul"(%1081, %arg131) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %1083 = "ttnn.silu"(%1082) : (tensor<1x512x8192xbf16, #ttnn_layout20>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %1084 = "ttnn.matmul"(%1081, %arg132) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %1085 = "ttnn.multiply"(%1083, %1084) : (tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %1086 = "ttnn.matmul"(%1085, %arg133) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<2048x8192xbf16, #ttnn_layout4>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %1087 = "ttnn.add"(%1086, %1073) : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%1086) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        "ttnn.deallocate"(%1073) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %1088 = "ttnn.typecast"(%1087) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %1089 = "ttnn.pow_scalar"(%1088) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %1090 = "ttnn.mean"(%1089) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%1089) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %1091 = "ttnn.add"(%1090, %6) : (tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%1090) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %1092 = "ttnn.rsqrt"(%1091) : (tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%1091) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %1093 = "ttnn.multiply"(%1088, %1092) : (tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %1094 = "ttnn.typecast"(%1093) : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%1093) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %1095 = "ttnn.rms_norm"(%1087, %arg143) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %1096 = "ttnn.concat"(%arg136, %arg137, %arg138) <{dim = 0 : si32}> : (tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>) -> tensor<3072x2048xbf16, #ttnn_layout42>
        %1097 = "ttnn.matmul"(%1095, %1096) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<3072x2048xbf16, #ttnn_layout42>) -> tensor<1x512x3072xbf16, #ttnn_layout43>
        "ttnn.deallocate"(%1096) <{force = false}> : (tensor<3072x2048xbf16, #ttnn_layout42>) -> ()
        %1098 = "ttnn.slice_static"(%1097) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 512 : i32, 2048 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %1099 = "ttnn.slice_static"(%1097) <{begins = [0 : i32, 0 : i32, 2048 : i32], ends = [1 : i32, 512 : i32, 2560 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x512xbf16, #ttnn_layout44>
        %1100 = "ttnn.slice_static"(%1097) <{begins = [0 : i32, 0 : i32, 2560 : i32], ends = [1 : i32, 512 : i32, 3072 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> tensor<1x512x512xbf16, #ttnn_layout44>
        "ttnn.deallocate"(%1097) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout43>) -> ()
        %1101 = "ttnn.reshape"(%1098) <{shape = [1 : i32, 512 : i32, 32 : i32, 64 : i32]}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x32x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%1098) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %1102 = "ttnn.permute"(%1101) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%1101) <{force = false}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> ()
        %1103 = "ttnn.reshape"(%1099) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> tensor<1x512x8x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%1099) <{force = false}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> ()
        %1104 = "ttnn.permute"(%1103) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%1103) <{force = false}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> ()
        %1105 = "ttnn.reshape"(%1100) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> tensor<1x512x8x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%1100) <{force = false}> : (tensor<1x512x512xbf16, #ttnn_layout44>) -> ()
        %1106 = "ttnn.permute"(%1105) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%1105) <{force = false}> : (tensor<1x512x8x64xbf16, #ttnn_layout45>) -> ()
        %1107 = "ttnn.slice_static"(%1102) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        %1108 = "ttnn.slice_static"(%1102) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        %1109 = "ttnn.neg"(%1108) : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> tensor<1x32x512x32xbf16, #ttnn_layout48>
        "ttnn.deallocate"(%1108) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        %1110 = "ttnn.concat"(%1109, %1107) <{dim = 3 : si32}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>, tensor<1x32x512x32xbf16, #ttnn_layout48>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%1109) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        "ttnn.deallocate"(%1107) <{force = false}> : (tensor<1x32x512x32xbf16, #ttnn_layout48>) -> ()
        %1111 = "ttnn.multiply"(%1102, %83) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%1102) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %1112 = "ttnn.multiply"(%1110, %84) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%1110) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %1113 = "ttnn.add"(%1111, %1112) : (tensor<1x32x512x64xbf16, #ttnn_layout46>, tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%1112) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        "ttnn.deallocate"(%1111) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %1114 = "ttnn.slice_static"(%1104) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        %1115 = "ttnn.slice_static"(%1104) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        %1116 = "ttnn.neg"(%1115) : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> tensor<1x8x512x32xbf16, #ttnn_layout49>
        "ttnn.deallocate"(%1115) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        %1117 = "ttnn.concat"(%1116, %1114) <{dim = 3 : si32}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>, tensor<1x8x512x32xbf16, #ttnn_layout49>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%1116) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        "ttnn.deallocate"(%1114) <{force = false}> : (tensor<1x8x512x32xbf16, #ttnn_layout49>) -> ()
        %1118 = "ttnn.multiply"(%1104, %83) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%1104) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %1119 = "ttnn.multiply"(%1117, %84) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%1117) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %1120 = "ttnn.add"(%1118, %1119) : (tensor<1x8x512x64xbf16, #ttnn_layout47>, tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x512x64xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%1119) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        "ttnn.deallocate"(%1118) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %1121 = "ttnn.reshape"(%1120) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout50>
        "ttnn.deallocate"(%1120) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %1122 = "ttnn.reshape"(%1106) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout50>
        "ttnn.deallocate"(%1106) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout47>) -> ()
        %1123 = "ttnn.typecast"(%1113) : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%1113) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %1124 = "ttnn.typecast"(%1121) : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%1121) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> ()
        %1125 = "ttnn.typecast"(%1122) : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%1122) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout50>) -> ()
        %1126 = "ttnn.repeat"(%1125) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> tensor<1x8x4x512x64xf32, #ttnn_layout52>
        "ttnn.deallocate"(%1125) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %1127 = "ttnn.reshape"(%1126) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%1126) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> ()
        %1128 = "ttnn.multiply"(%1123, %3) : (tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%1123) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        "ttnn.deallocate"(%3) <{force = false}> : (tensor<1x1x1x1xf32, #ttnn_layout23>) -> ()
        %1129 = "ttnn.multiply"(%1124, %1) : (tensor<1x8x1x512x64xf32, #ttnn_layout51>, tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> tensor<1x8x1x512x64xf32, #ttnn_layout51>
        "ttnn.deallocate"(%1124) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> ()
        %1130 = "ttnn.repeat"(%1129) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> tensor<1x8x4x512x64xf32, #ttnn_layout52>
        "ttnn.deallocate"(%1129) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout51>) -> ()
        %1131 = "ttnn.reshape"(%1130) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%1130) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout52>) -> ()
        %1132 = "ttnn.permute"(%1131) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x64x512xf32, #ttnn_layout18>
        %1133 = "ttnn.matmul"(%1128, %1131) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%1131) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %1134 = "ttnn.add"(%1133, %112) : (tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x1x512x512xf32, #ttnn_layout53>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%1133) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        "ttnn.deallocate"(%112) <{force = false}> : (tensor<1x1x512x512xf32, #ttnn_layout53>) -> ()
        %1135 = "ttnn.softmax"(%1134) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%1134) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %1136 = "ttnn.matmul"(%1135, %1127) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        %1137 = "ttnn.typecast"(%1136) : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xbf16, #ttnn_layout46>
        "ttnn.deallocate"(%1136) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %1138 = "ttnn.permute"(%1137) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> tensor<1x512x32x64xbf16, #ttnn_layout45>
        "ttnn.deallocate"(%1137) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout46>) -> ()
        %1139 = "ttnn.reshape"(%1138) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%1138) <{force = false}> : (tensor<1x512x32x64xbf16, #ttnn_layout45>) -> ()
        %1140 = "ttnn.matmul"(%1139, %arg139) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048x2048xbf16, #ttnn_layout1>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %1141 = "ttnn.add"(%1140, %1087) : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%1140) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        "ttnn.deallocate"(%1087) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %1142 = "ttnn.typecast"(%1141) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %1143 = "ttnn.pow_scalar"(%1142) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %1144 = "ttnn.mean"(%1143) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%1143) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %1145 = "ttnn.add"(%1144, %6) : (tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%1144) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %1146 = "ttnn.rsqrt"(%1145) : (tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%1145) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %1147 = "ttnn.multiply"(%1142, %1146) : (tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %1148 = "ttnn.typecast"(%1147) : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%1147) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %1149 = "ttnn.rms_norm"(%1141, %arg144) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %1150 = "ttnn.matmul"(%1149, %arg140) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %1151 = "ttnn.silu"(%1150) : (tensor<1x512x8192xbf16, #ttnn_layout20>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %1152 = "ttnn.matmul"(%1149, %arg141) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %1153 = "ttnn.multiply"(%1151, %1152) : (tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %1154 = "ttnn.matmul"(%1153, %arg142) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<2048x8192xbf16, #ttnn_layout4>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %1155 = "ttnn.add"(%1154, %1141) : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%1154) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        "ttnn.deallocate"(%1141) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %1156 = "ttnn.typecast"(%1155) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %1157 = "ttnn.pow_scalar"(%1156) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %1158 = "ttnn.mean"(%1157) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%1157) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %1159 = "ttnn.add"(%1158, %6) : (tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%1158) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        "ttnn.deallocate"(%6) <{force = false}> : (tensor<1x1x1xf32, #ttnn_layout10>) -> ()
        %1160 = "ttnn.rsqrt"(%1159) : (tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%1159) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout13>) -> ()
        %1161 = "ttnn.multiply"(%1156, %1160) : (tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %1162 = "ttnn.typecast"(%1161) : (tensor<1x512x2048xf32, #ttnn_layout14>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%1161) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout14>) -> ()
        %1163 = "ttnn.rms_norm"(%1155, %arg145) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%1155) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %1164 = "ttnn.slice_static"(%1163) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 511 : i32, 2048 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x511x2048xbf16, #ttnn_layout15>
        %1165 = "ttnn.matmul"(%1164, %arg0) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x511x2048xbf16, #ttnn_layout15>, tensor<128256x2048xbf16, #ttnn_layout>) -> tensor<1x511x128256xbf16, #ttnn_layout12>
        "ttnn.deallocate"(%1164) <{force = false}> : (tensor<1x511x2048xbf16, #ttnn_layout15>) -> ()
        %1166 = "ttnn.softmax"(%1165) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 2 : si32, numericStable = true}> : (tensor<1x511x128256xbf16, #ttnn_layout12>) -> tensor<1x511x128256xbf16, #ttnn_layout12>
        "ttnn.deallocate"(%1165) <{force = false}> : (tensor<1x511x128256xbf16, #ttnn_layout12>) -> ()
        %1167 = "ttnn.to_layout"(%arg149) : (tensor<1x511x128256xbf16, #ttnn_layout8>) -> tensor<1x511x128256xbf16, #ttnn_layout12>
        "ttnn.deallocate"(%arg149) <{force = false}> : (tensor<1x511x128256xbf16, #ttnn_layout8>) -> ()
        %1168 = "ttnn.multiply"(%1166, %1167) : (tensor<1x511x128256xbf16, #ttnn_layout12>, tensor<1x511x128256xbf16, #ttnn_layout12>) -> tensor<1x511x128256xbf16, #ttnn_layout12>
        %1169 = "ttnn.sum"(%1168) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x511x128256xbf16, #ttnn_layout12>) -> tensor<1x511x1xbf16, #ttnn_layout54>
        "ttnn.deallocate"(%1168) <{force = false}> : (tensor<1x511x128256xbf16, #ttnn_layout12>) -> ()
        %1170 = "ttnn.typecast"(%1169) : (tensor<1x511x1xbf16, #ttnn_layout54>) -> tensor<1x511x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%1169) <{force = false}> : (tensor<1x511x1xbf16, #ttnn_layout54>) -> ()
        %1171 = "ttnn.clamp_scalar"(%1170) <{max = 0x7F800000 : f32, min = 9.99999996E-13 : f32}> : (tensor<1x511x1xf32, #ttnn_layout13>) -> tensor<1x511x1xf32, #ttnn_layout13>
        %1172 = "ttnn.log"(%1171) : (tensor<1x511x1xf32, #ttnn_layout13>) -> tensor<1x511x1xf32, #ttnn_layout13>
        %1173 = "ttnn.to_layout"(%arg150) : (tensor<1x511x1xf32, #ttnn_layout9>) -> tensor<1x511x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%arg150) <{force = false}> : (tensor<1x511x1xf32, #ttnn_layout9>) -> ()
        %1174 = "ttnn.multiply"(%1172, %1173) : (tensor<1x511x1xf32, #ttnn_layout13>, tensor<1x511x1xf32, #ttnn_layout13>) -> tensor<1x511x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%1172) <{force = false}> : (tensor<1x511x1xf32, #ttnn_layout13>) -> ()
        %1175 = "ttnn.sum"(%1174) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [0 : i32, 1 : i32, 2 : i32], keep_dim = true}> : (tensor<1x511x1xf32, #ttnn_layout13>) -> tensor<1x1x1xf32, #ttnn_layout10>
        "ttnn.deallocate"(%1174) <{force = false}> : (tensor<1x511x1xf32, #ttnn_layout13>) -> ()
        %1176 = "ttnn.to_layout"(%arg147) : (tensor<1x512xsi32, #ttnn_layout7>) -> tensor<1x512xsi32, #ttnn_layout11>
        "ttnn.deallocate"(%arg147) <{force = false}> : (tensor<1x512xsi32, #ttnn_layout7>) -> ()
        return %1175, %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg30, %arg31, %arg32, %arg33, %arg34, %arg35, %arg36, %arg37, %arg38, %arg39, %arg40, %arg41, %arg42, %arg43, %arg44, %arg45, %arg46, %arg47, %arg48, %arg49, %arg50, %arg51, %arg52, %arg53, %arg54, %arg55, %arg56, %arg57, %arg58, %arg59, %arg60, %arg61, %arg62, %arg63, %arg64, %arg65, %arg66, %arg67, %arg68, %arg69, %arg70, %arg71, %arg72, %arg73, %arg74, %arg75, %arg76, %arg77, %arg78, %arg79, %arg80, %arg81, %arg82, %arg83, %arg84, %arg85, %arg86, %arg87, %arg88, %arg89, %arg90, %arg91, %arg92, %arg93, %arg94, %arg95, %arg96, %arg97, %arg98, %arg99, %arg100, %arg101, %arg102, %arg103, %arg104, %arg105, %arg106, %arg107, %arg108, %arg109, %arg110, %arg111, %arg112, %arg113, %arg114, %arg115, %arg116, %arg117, %arg118, %arg119, %arg120, %arg121, %arg122, %arg123, %arg124, %arg125, %arg126, %arg127, %arg128, %arg129, %arg130, %arg131, %arg132, %arg133, %arg134, %arg135, %arg136, %arg137, %arg138, %arg139, %arg140, %arg141, %arg142, %arg143, %arg144, %arg145, %1176, %1167, %1173, %64, %68, %68, %70, %71, %83, %84, %106, %107, %111, %115, %115, %119, %122, %126, %126, %128, %129, %130, %131, %132, %133, %136, %140, %140, %142, %143, %83, %84, %175, %176, %180, %183, %183, %187, %190, %194, %194, %196, %197, %198, %199, %200, %201, %204, %208, %208, %210, %211, %83, %84, %243, %244, %248, %251, %251, %255, %258, %262, %262, %264, %265, %266, %267, %268, %269, %272, %276, %276, %278, %279, %83, %84, %311, %312, %316, %319, %319, %323, %326, %330, %330, %332, %333, %334, %335, %336, %337, %340, %344, %344, %346, %347, %83, %84, %379, %380, %384, %387, %387, %391, %394, %398, %398, %400, %401, %402, %403, %404, %405, %408, %412, %412, %414, %415, %83, %84, %447, %448, %452, %455, %455, %459, %462, %466, %466, %468, %469, %470, %471, %472, %473, %476, %480, %480, %482, %483, %83, %84, %515, %516, %520, %523, %523, %527, %530, %534, %534, %536, %537, %538, %539, %540, %541, %544, %548, %548, %550, %551, %83, %84, %583, %584, %588, %591, %591, %595, %598, %602, %602, %604, %605, %606, %607, %608, %609, %612, %616, %616, %618, %619, %83, %84, %651, %652, %656, %659, %659, %663, %666, %670, %670, %672, %673, %674, %675, %676, %677, %680, %684, %684, %686, %687, %83, %84, %719, %720, %724, %727, %727, %731, %734, %738, %738, %740, %741, %742, %743, %744, %745, %748, %752, %752, %754, %755, %83, %84, %787, %788, %792, %795, %795, %799, %802, %806, %806, %808, %809, %810, %811, %812, %813, %816, %820, %820, %822, %823, %83, %84, %855, %856, %860, %863, %863, %867, %870, %874, %874, %876, %877, %878, %879, %880, %881, %884, %888, %888, %890, %891, %83, %84, %923, %924, %928, %931, %931, %935, %938, %942, %942, %944, %945, %946, %947, %948, %949, %952, %956, %956, %958, %959, %83, %84, %991, %992, %996, %999, %999, %1003, %1006, %1010, %1010, %1012, %1013, %1014, %1015, %1016, %1017, %1020, %1024, %1024, %1026, %1027, %83, %84, %1059, %1060, %1064, %1067, %1067, %1071, %1074, %1078, %1078, %1080, %1081, %1082, %1083, %1084, %1085, %1088, %1092, %1092, %1094, %1095, %83, %84, %1127, %1128, %1132, %1135, %1135, %1139, %1142, %1146, %1146, %1148, %1149, %1150, %1151, %1152, %1153, %1156, %1160, %1160, %1162, %1163, %1166, %1170, %1171 : tensor<1x1x1xf32, #ttnn_layout10>, tensor<128256x2048xbf16, #ttnn_layout>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<1x512xsi32, #ttnn_layout11>, tensor<1x511x128256xbf16, #ttnn_layout12>, tensor<1x511x1xf32, #ttnn_layout13>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x511x128256xbf16, #ttnn_layout12>, tensor<1x511x1xf32, #ttnn_layout13>, tensor<1x511x1xf32, #ttnn_layout13>
      }
    }
  }
}

