// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttcore-mark-functions-as-forward --ttnn-decompose-layouts -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// DRAM-sharded layout decompositions for ttnn.to_layout.
//
// ttnn.untilize cannot operate on DRAM-sharded tensors (tt-metal issue
// #43975), so any untilize touching a DRAM-sharded side is routed through
// host. ttnn.tilize accepts DRAM-sharded inputs and outputs via its default
// multicore factory, so tilize tests here are sanity coverage only.
//
// Width-sharded variants use a 1x12 grid matching Wormhole's 12 DRAM banks.

#system_memory = #ttnn.buffer_type<system_memory>
#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>

// Host
#host_rm_bf16    = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x384xbf16, #system_memory>>
#host_rm_f32     = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x384xf32, #system_memory>>
#host_tile_bf16  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x12x!ttcore.tile<32x32, bf16>, #system_memory>>
#host_tile_f32   = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x12x!ttcore.tile<32x32, f32>, #system_memory>>

// DRAM interleaved
#dram_il_tile_f32  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x12x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#dram_il_rm_bf16   = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x384xbf16, #dram>, <interleaved>>
#dram_il_rm_f32    = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x384xf32, #dram>, <interleaved>>

// DRAM width-sharded (32x384, 1x12 grid, shard 32x32)
#dram_ws_tile_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x12>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <width_sharded>, core_ranges = #ttnn.core_range_set<[#ttnn.core_range<(0,0), (11,0)>]>>
#dram_ws_tile_f32  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x12>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <width_sharded>, core_ranges = #ttnn.core_range_set<[#ttnn.core_range<(0,0), (11,0)>]>>
#dram_ws_rm_bf16   = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x12>, memref<32x32xbf16, #dram>, <width_sharded>, core_ranges = #ttnn.core_range_set<[#ttnn.core_range<(0,0), (11,0)>]>>

// DRAM width-sharded with a different grid (32x384, 1x6 grid, shard 32x64),
// used to test the sharded reshard path.
#dram_ws_tile_bf16_grid6 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x6>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <width_sharded>, core_ranges = #ttnn.core_range_set<[#ttnn.core_range<(0,0), (5,0)>]>>

// DRAM height-sharded (256x32, 8x1 grid, shard 32x32)
#dram_hs_tile_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <height_sharded>, core_ranges = #ttnn.core_range_set<[#ttnn.core_range<(0,0), (7,0)>]>>
#dram_hs_rm_bf16   = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x1>, memref<32x32xbf16, #dram>, <height_sharded>, core_ranges = #ttnn.core_range_set<[#ttnn.core_range<(0,0), (7,0)>]>>

// L1 block-sharded (32x384, 1x6 grid, shard 32x64), source for the
// L1-sharded to DRAM-sharded reshard test.
#l1_bs_tile_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x6>, memref<1x2x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, core_ranges = #ttnn.core_range_set<[#ttnn.core_range<(0,0), (5,0)>]>>

// CHECK-DAG: #[[DRAM_WS_RM_BF16:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}<1x12>, memref<32x32xbf16, #dram>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (11,0)>]>>
// CHECK-DAG: #[[DRAM_WS_RM_F32:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}<1x12>, memref<32x32xf32, #dram>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (11,0)>]>>
// CHECK-DAG: #[[DRAM_WS_TILE_BF16:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}<1x12>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (11,0)>]>>
// CHECK-DAG: #[[DRAM_WS_TILE_BF16_GRID6:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}<1x6>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (5,0)>]>>
// CHECK-DAG: #[[DRAM_HS_RM_BF16:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}<8x1>, memref<32x32xbf16, #dram>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,0)>]>>
// CHECK-DAG: #[[DRAM_IL_RM_BF16:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}<1x1>, memref<32x384xbf16, #dram>, <interleaved>>
// CHECK-DAG: #[[DRAM_IL_TILE_BF16:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}<1x1>, memref<1x12x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {} {

    //===------------------------------------------------------------------===//
    // Untilize, no typecast
    //===------------------------------------------------------------------===//

    // Host TILE bf16 to DRAM width-sharded RM bf16. The untilize runs on
    // host and a single to_device places the result directly into the
    // DRAM-sharded memory config.
    func.func @host_tile_bf16_to_dram_ws_rm_bf16(%arg0: tensor<32x384xbf16, #host_tile_bf16>) -> tensor<32x384xbf16, #dram_ws_rm_bf16> {
        // CHECK-LABEL: func.func @host_tile_bf16_to_dram_ws_rm_bf16
        // CHECK: %[[UNTILIZE:.*]] = "ttnn.to_layout"(%arg0)
        // CHECK-SAME: layout = #ttnn.layout<row_major>
        // CHECK: %[[TO_DEV:.*]] = "ttnn.to_device"(%[[UNTILIZE]]
        // CHECK-SAME: -> tensor<32x384xbf16, #[[DRAM_WS_RM_BF16]]>
        // CHECK: return %[[TO_DEV]]
        %0 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#dram, <width_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0, 0), (11, 0)>]>, <32x32>, <row_major>>>}> : (tensor<32x384xbf16, #host_tile_bf16>) -> tensor<32x384xbf16, #dram_ws_rm_bf16>
        return %0 : tensor<32x384xbf16, #dram_ws_rm_bf16>
    }

    // DRAM-sharded TILE input with a non-sharded output forces a host
    // roundtrip even though the output isn't sharded itself.
    func.func @dram_ws_tile_bf16_to_dram_il_rm_bf16(%arg0: tensor<32x384xbf16, #dram_ws_tile_bf16>) -> tensor<32x384xbf16, #dram_il_rm_bf16> {
        // CHECK-LABEL: func.func @dram_ws_tile_bf16_to_dram_il_rm_bf16
        // CHECK: %[[FROM_DEV:.*]] = "ttnn.from_device"(%arg0)
        // CHECK: %[[UNTILIZE:.*]] = "ttnn.to_layout"(%[[FROM_DEV]])
        // CHECK-SAME: layout = #ttnn.layout<row_major>
        // CHECK: %[[TO_DEV:.*]] = "ttnn.to_device"(%[[UNTILIZE]]
        // CHECK-SAME: -> tensor<32x384xbf16, #[[DRAM_IL_RM_BF16]]>
        // CHECK: return %[[TO_DEV]]
        %0 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x384xbf16, #dram_ws_tile_bf16>) -> tensor<32x384xbf16, #dram_il_rm_bf16>
        return %0 : tensor<32x384xbf16, #dram_il_rm_bf16>
    }

    // Both sides DRAM-sharded: host roundtrip with the final to_device
    // landing back in DRAM-sharded RM.
    func.func @dram_ws_tile_bf16_to_dram_ws_rm_bf16(%arg0: tensor<32x384xbf16, #dram_ws_tile_bf16>) -> tensor<32x384xbf16, #dram_ws_rm_bf16> {
        // CHECK-LABEL: func.func @dram_ws_tile_bf16_to_dram_ws_rm_bf16
        // CHECK: %[[FROM_DEV:.*]] = "ttnn.from_device"(%arg0)
        // CHECK: %[[UNTILIZE:.*]] = "ttnn.to_layout"(%[[FROM_DEV]])
        // CHECK-SAME: layout = #ttnn.layout<row_major>
        // CHECK: %[[TO_DEV:.*]] = "ttnn.to_device"(%[[UNTILIZE]]
        // CHECK-SAME: -> tensor<32x384xbf16, #[[DRAM_WS_RM_BF16]]>
        // CHECK: return %[[TO_DEV]]
        %0 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#dram, <width_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0, 0), (11, 0)>]>, <32x32>, <row_major>>>}> : (tensor<32x384xbf16, #dram_ws_tile_bf16>) -> tensor<32x384xbf16, #dram_ws_rm_bf16>
        return %0 : tensor<32x384xbf16, #dram_ws_rm_bf16>
    }

    // DRAM-sharded TILE to host RM. No to_device is emitted after the host
    // untilize; from_device terminates the chain.
    func.func @dram_ws_tile_bf16_to_host_rm_bf16(%arg0: tensor<32x384xbf16, #dram_ws_tile_bf16>) -> tensor<32x384xbf16, #host_rm_bf16> {
        // CHECK-LABEL: func.func @dram_ws_tile_bf16_to_host_rm_bf16
        // CHECK: %[[FROM_DEV:.*]] = "ttnn.from_device"(%arg0)
        // CHECK: %[[UNTILIZE:.*]] = "ttnn.to_layout"(%[[FROM_DEV]])
        // CHECK-SAME: layout = #ttnn.layout<row_major>
        // CHECK: return %[[UNTILIZE]]
        %0 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory>}> : (tensor<32x384xbf16, #dram_ws_tile_bf16>) -> tensor<32x384xbf16, #host_rm_bf16>
        return %0 : tensor<32x384xbf16, #host_rm_bf16>
    }

    // Height-sharded variant of the same host roundtrip; verifies the
    // DRAM-sharded predicate fires regardless of which sharding flavor is
    // used.
    func.func @dram_hs_tile_bf16_to_dram_hs_rm_bf16(%arg0: tensor<256x32xbf16, #dram_hs_tile_bf16>) -> tensor<256x32xbf16, #dram_hs_rm_bf16> {
        // CHECK-LABEL: func.func @dram_hs_tile_bf16_to_dram_hs_rm_bf16
        // CHECK: %[[FROM_DEV:.*]] = "ttnn.from_device"(%arg0)
        // CHECK: %[[UNTILIZE:.*]] = "ttnn.to_layout"(%[[FROM_DEV]])
        // CHECK-SAME: layout = #ttnn.layout<row_major>
        // CHECK: %[[TO_DEV:.*]] = "ttnn.to_device"(%[[UNTILIZE]]
        // CHECK-SAME: -> tensor<256x32xbf16, #[[DRAM_HS_RM_BF16]]>
        // CHECK: return %[[TO_DEV]]
        %0 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#dram, <height_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0, 0), (7, 0)>]>, <32x32>, <row_major>>>}> : (tensor<256x32xbf16, #dram_hs_tile_bf16>) -> tensor<256x32xbf16, #dram_hs_rm_bf16>
        return %0 : tensor<256x32xbf16, #dram_hs_rm_bf16>
    }

    //===------------------------------------------------------------------===//
    // Untilize, with typecast
    //===------------------------------------------------------------------===//

    // Host TILE f32 to DRAM-sharded RM bf16. Both typecast and untilize
    // run on host before a single to_device into the DRAM-sharded output.
    func.func @host_tile_f32_to_dram_ws_rm_bf16(%arg0: tensor<32x384xf32, #host_tile_f32>) -> tensor<32x384xbf16, #dram_ws_rm_bf16> {
        // CHECK-LABEL: func.func @host_tile_f32_to_dram_ws_rm_bf16
        // CHECK: %[[CAST:.*]] = "ttnn.typecast"(%arg0)
        // CHECK-SAME: -> tensor<{{.*}}bf16
        // CHECK: %[[UNTILIZE:.*]] = "ttnn.to_layout"(%[[CAST]])
        // CHECK-SAME: layout = #ttnn.layout<row_major>
        // CHECK: %[[TO_DEV:.*]] = "ttnn.to_device"(%[[UNTILIZE]]
        // CHECK-SAME: -> tensor<32x384xbf16, #[[DRAM_WS_RM_BF16]]>
        // CHECK: return %[[TO_DEV]]
        %0 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#dram, <width_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0, 0), (11, 0)>]>, <32x32>, <row_major>>>}> : (tensor<32x384xf32, #host_tile_f32>) -> tensor<32x384xbf16, #dram_ws_rm_bf16>
        return %0 : tensor<32x384xbf16, #dram_ws_rm_bf16>
    }

    // DRAM-interleaved TILE f32 to DRAM-sharded RM bf16. The typecast
    // runs on device while the tensor is still TILE, then a host roundtrip
    // handles the untilize that DRAM sharding would otherwise FATAL on.
    func.func @dram_il_tile_f32_to_dram_ws_rm_bf16(%arg0: tensor<32x384xf32, #dram_il_tile_f32>) -> tensor<32x384xbf16, #dram_ws_rm_bf16> {
        // CHECK-LABEL: func.func @dram_il_tile_f32_to_dram_ws_rm_bf16
        // CHECK: %[[CAST:.*]] = "ttnn.typecast"(%arg0)
        // CHECK-SAME: -> tensor<{{.*}}bf16
        // CHECK: %[[FROM_DEV:.*]] = "ttnn.from_device"(%[[CAST]])
        // CHECK: %[[UNTILIZE:.*]] = "ttnn.to_layout"(%[[FROM_DEV]])
        // CHECK-SAME: layout = #ttnn.layout<row_major>
        // CHECK: %[[TO_DEV:.*]] = "ttnn.to_device"(%[[UNTILIZE]]
        // CHECK-SAME: -> tensor<32x384xbf16, #[[DRAM_WS_RM_BF16]]>
        // CHECK-NOT: ttnn.to_memory_config
        // CHECK: return %[[TO_DEV]]
        %0 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#dram, <width_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0, 0), (11, 0)>]>, <32x32>, <row_major>>>}> : (tensor<32x384xf32, #dram_il_tile_f32>) -> tensor<32x384xbf16, #dram_ws_rm_bf16>
        return %0 : tensor<32x384xbf16, #dram_ws_rm_bf16>
    }

    // DRAM-sharded TILE f32 to DRAM-interleaved RM bf16. Verifies the
    // typecast can run on a DRAM-sharded TILE input on device, and that
    // from_device produces a clean host TILE (the source-side sharded
    // encoding does not leak through).
    func.func @dram_ws_tile_f32_to_dram_il_rm_bf16(%arg0: tensor<32x384xf32, #dram_ws_tile_f32>) -> tensor<32x384xbf16, #dram_il_rm_bf16> {
        // CHECK-LABEL: func.func @dram_ws_tile_f32_to_dram_il_rm_bf16
        // CHECK: %[[CAST:.*]] = "ttnn.typecast"(%arg0)
        // CHECK-SAME: -> tensor<{{.*}}bf16
        // CHECK: %[[FROM_DEV:.*]] = "ttnn.from_device"(%[[CAST]])
        // CHECK: %[[UNTILIZE:.*]] = "ttnn.to_layout"(%[[FROM_DEV]])
        // CHECK-SAME: layout = #ttnn.layout<row_major>
        // CHECK: "ttnn.to_device"(%[[UNTILIZE]]
        // CHECK-SAME: -> tensor<32x384xbf16, #[[DRAM_IL_RM_BF16]]>
        // CHECK: return
        %0 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x384xf32, #dram_ws_tile_f32>) -> tensor<32x384xbf16, #dram_il_rm_bf16>
        return %0 : tensor<32x384xbf16, #dram_il_rm_bf16>
    }

    //===------------------------------------------------------------------===//
    // Tilize, no typecast (sanity, ttnn.tilize accepts DRAM-sharded)
    //===------------------------------------------------------------------===//

    // Host RM bf16 to DRAM-sharded TILE bf16. ttnn.tilize accepts a
    // DRAM-sharded output directly, so to_device lands the tensor in the
    // sharded config and tilize runs there without staging.
    func.func @host_rm_bf16_to_dram_ws_tile_bf16(%arg0: tensor<32x384xbf16, #host_rm_bf16>) -> tensor<32x384xbf16, #dram_ws_tile_bf16> {
        // CHECK-LABEL: func.func @host_rm_bf16_to_dram_ws_tile_bf16
        // CHECK: %[[TO_DEV:.*]] = "ttnn.to_device"
        // CHECK-SAME: -> tensor<32x384xbf16, #[[DRAM_WS_RM_BF16]]>
        // CHECK: %[[TILIZE:.*]] = "ttnn.to_layout"(%[[TO_DEV]])
        // CHECK-SAME: layout = #ttnn.layout<tile>
        // CHECK-SAME: -> tensor<32x384xbf16, #[[DRAM_WS_TILE_BF16]]>
        // CHECK: return %[[TILIZE]]
        %0 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <width_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0, 0), (11, 0)>]>, <32x32>, <row_major>>>}> : (tensor<32x384xbf16, #host_rm_bf16>) -> tensor<32x384xbf16, #dram_ws_tile_bf16>
        return %0 : tensor<32x384xbf16, #dram_ws_tile_bf16>
    }

    //===------------------------------------------------------------------===//
    // Tilize, with typecast
    //===------------------------------------------------------------------===//

    // Host RM f32 to DRAM-sharded TILE bf16. Tilize and typecast both run
    // on device after the host-to-device transfer.
    func.func @host_rm_f32_to_dram_ws_tile_bf16(%arg0: tensor<32x384xf32, #host_rm_f32>) -> tensor<32x384xbf16, #dram_ws_tile_bf16> {
        // CHECK-LABEL: func.func @host_rm_f32_to_dram_ws_tile_bf16
        // CHECK: %[[TO_DEV:.*]] = "ttnn.to_device"
        // CHECK-SAME: -> tensor<32x384xf32, #[[DRAM_WS_RM_F32]]>
        // CHECK: %[[TILIZE:.*]] = "ttnn.to_layout"(%[[TO_DEV]])
        // CHECK-SAME: layout = #ttnn.layout<tile>
        // CHECK: %[[CAST:.*]] = "ttnn.typecast"(%[[TILIZE]])
        // CHECK-SAME: -> tensor<32x384xbf16, #[[DRAM_WS_TILE_BF16]]>
        // CHECK: return %[[CAST]]
        %0 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <width_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0, 0), (11, 0)>]>, <32x32>, <row_major>>>}> : (tensor<32x384xf32, #host_rm_f32>) -> tensor<32x384xbf16, #dram_ws_tile_bf16>
        return %0 : tensor<32x384xbf16, #dram_ws_tile_bf16>
    }

    // DRAM-interleaved RM f32 to DRAM-sharded TILE bf16. Tilize runs on
    // the interleaved input, typecast on the tilized result, and a final
    // reshard lands the tensor in DRAM-sharded.
    func.func @dram_il_rm_f32_to_dram_ws_tile_bf16(%arg0: tensor<32x384xf32, #dram_il_rm_f32>) -> tensor<32x384xbf16, #dram_ws_tile_bf16> {
        // CHECK-LABEL: func.func @dram_il_rm_f32_to_dram_ws_tile_bf16
        // CHECK: %[[TILIZE:.*]] = "ttnn.to_layout"(%arg0)
        // CHECK-SAME: layout = #ttnn.layout<tile>
        // CHECK: %[[CAST:.*]] = "ttnn.typecast"(%[[TILIZE]])
        // CHECK-SAME: -> tensor<{{.*}}bf16
        // CHECK: %[[RESHARD:.*]] = "ttnn.to_memory_config"(%[[CAST]])
        // CHECK-SAME: -> tensor<32x384xbf16, #[[DRAM_WS_TILE_BF16]]>
        // CHECK: return %[[RESHARD]]
        %0 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <width_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0, 0), (11, 0)>]>, <32x32>, <row_major>>>}> : (tensor<32x384xf32, #dram_il_rm_f32>) -> tensor<32x384xbf16, #dram_ws_tile_bf16>
        return %0 : tensor<32x384xbf16, #dram_ws_tile_bf16>
    }

    //===------------------------------------------------------------------===//
    // TILE to TILE (no layout change): typecast and reshard scenarios
    //===------------------------------------------------------------------===//

    // DRAM-sharded TILE to DRAM-interleaved TILE with a dtype change.
    // ttnn.typecast requires input and output memory layouts to match, so
    // the unshard must happen before the typecast.
    func.func @dram_ws_tile_bf16_to_dram_il_tile_f32_unshards_before_typecast(%arg0: tensor<32x384xbf16, #dram_ws_tile_bf16>) -> tensor<32x384xf32, #dram_il_tile_f32> {
        // CHECK-LABEL: func.func @dram_ws_tile_bf16_to_dram_il_tile_f32_unshards_before_typecast
        // CHECK: %[[TO_MEM_CONFIG:.*]] = "ttnn.to_memory_config"(%arg0)
        // CHECK-SAME: -> tensor<32x384xbf16, #[[DRAM_IL_TILE_BF16]]>
        // CHECK-NEXT: %[[TYPECAST:.*]] = "ttnn.typecast"(%[[TO_MEM_CONFIG]])
        // CHECK-SAME: -> tensor<{{.*}}f32
        // CHECK-NEXT: return %[[TYPECAST]]
        %0 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x384xbf16, #dram_ws_tile_bf16>) -> tensor<32x384xf32, #dram_il_tile_f32>
        return %0 : tensor<32x384xf32, #dram_il_tile_f32>
    }

    // DRAM-sharded to DRAM-sharded reshard with a different grid (1x12 to
    // 1x6). Pure memory config change, no layout or dtype involved.
    func.func @dram_ws_to_dram_ws_different_grid_emits_to_memory_config(%arg0: tensor<32x384xbf16, #dram_ws_tile_bf16>) -> tensor<32x384xbf16, #dram_ws_tile_bf16_grid6> {
        // CHECK-LABEL: func.func @dram_ws_to_dram_ws_different_grid_emits_to_memory_config
        // CHECK: %[[TO_MEM_CONFIG:.*]] = "ttnn.to_memory_config"(%arg0)
        // CHECK-SAME: -> tensor<32x384xbf16, #[[DRAM_WS_TILE_BF16_GRID6]]>
        // CHECK-NOT: ttnn.to_layout
        // CHECK-NOT: ttnn.typecast
        // CHECK: return %[[TO_MEM_CONFIG]]
        %0 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <width_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0, 0), (5, 0)>]>, <32x64>, <row_major>>>}> : (tensor<32x384xbf16, #dram_ws_tile_bf16>) -> tensor<32x384xbf16, #dram_ws_tile_bf16_grid6>
        return %0 : tensor<32x384xbf16, #dram_ws_tile_bf16_grid6>
    }

    // L1 block-sharded to DRAM width-sharded reshard. Sharded-to-sharded
    // across memories with no layout or dtype change.
    func.func @l1_bs_to_dram_ws_emits_to_memory_config(%arg0: tensor<32x384xbf16, #l1_bs_tile_bf16>) -> tensor<32x384xbf16, #dram_ws_tile_bf16> {
        // CHECK-LABEL: func.func @l1_bs_to_dram_ws_emits_to_memory_config
        // CHECK: %[[TO_MEM_CONFIG:.*]] = "ttnn.to_memory_config"(%arg0)
        // CHECK-SAME: -> tensor<32x384xbf16, #[[DRAM_WS_TILE_BF16]]>
        // CHECK-NOT: ttnn.to_layout
        // CHECK-NOT: ttnn.typecast
        // CHECK: return %[[TO_MEM_CONFIG]]
        %0 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <width_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0, 0), (11, 0)>]>, <32x32>, <row_major>>>}> : (tensor<32x384xbf16, #l1_bs_tile_bf16>) -> tensor<32x384xbf16, #dram_ws_tile_bf16>
        return %0 : tensor<32x384xbf16, #dram_ws_tile_bf16>
    }
}
