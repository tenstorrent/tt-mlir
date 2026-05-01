// RUN: ttmlir-opt --ttcore-register-device --d2m-lower-to-layout -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verify that host<->DRAM transfers lower to a single d2m.to_device /
// d2m.to_host pair that targets DRAM directly, without materializing an
// L1 intermediate.

#dram = #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, dram, sharded>

// CHECK-LABEL: func.func @host_to_dram
func.func @host_to_dram(%arg0: tensor<256x256xf32>) -> tensor<1x1x256x256xf32, #dram> {
  %0 = d2m.empty() : tensor<1x1x256x256xf32, #dram>

  // Host->DRAM should land directly in a DRAM-encoded intermediate and
  // emit a single d2m.to_device.  No L1 intermediate should be created.
  // CHECK: %[[DST:.*]] = d2m.empty() : tensor<1x1x256x256xf32, #[[DRAM:layout[0-9]*]]>
  // CHECK: d2m.to_device %arg0, %[[DST]] layout = #[[DRAM]] : tensor<256x256xf32> into tensor<1x1x256x256xf32, #[[DRAM]]> -> tensor<1x1x256x256xf32, #[[DRAM]]>
  // CHECK-NOT: d2m.generic
  // CHECK-NOT: l1, sharded

  %1 = d2m.to_layout %arg0, %0 : tensor<256x256xf32> into tensor<1x1x256x256xf32, #dram>
    -> tensor<1x1x256x256xf32, #dram>

  return %1 : tensor<1x1x256x256xf32, #dram>
}

// CHECK-LABEL: func.func @dram_to_host
func.func @dram_to_host(%arg0: tensor<1x1x256x256xf32, #dram>) -> tensor<256x256xf32> {
  %0 = d2m.empty() : tensor<256x256xf32>

  // DRAM->host should emit a single d2m.to_host reading the DRAM buffer
  // without any intermediate device copy.
  // CHECK: %[[HOST:.*]] = d2m.empty() : tensor<256x256xf32>
  // CHECK: d2m.to_host %arg0, %[[HOST]] layout = #{{.*}} : tensor<1x1x256x256xf32, #{{.*}}> into tensor<256x256xf32> -> tensor<256x256xf32>
  // CHECK-NOT: d2m.generic
  // CHECK-NOT: d2m.to_device

  %1 = d2m.to_layout %arg0, %0 : tensor<1x1x256x256xf32, #dram> into tensor<256x256xf32>
    -> tensor<256x256xf32>

  return %1 : tensor<256x256xf32>
}

// CHECK-LABEL: func.func @host_to_sharded_dram
func.func @host_to_sharded_dram(%arg0: tensor<256x256xf32>) -> tensor<2x2x128x128xf32, #dram> {
  %0 = d2m.empty() : tensor<2x2x128x128xf32, #dram>

  // Non-1x1 DRAM grids should also lower directly to DRAM. The TTMetal
  // flatbuffer lowering carries the page distribution for host interop.
  // CHECK: %[[DST:.*]] = d2m.empty() : tensor<2x2x128x128xf32, #[[DRAM2:layout[0-9]*]]>
  // CHECK: d2m.to_device %arg0, %[[DST]] layout = #[[DRAM2]] : tensor<256x256xf32> into tensor<2x2x128x128xf32, #[[DRAM2]]> -> tensor<2x2x128x128xf32, #[[DRAM2]]>
  // CHECK-NOT: l1, sharded

  %1 = d2m.to_layout %arg0, %0 : tensor<256x256xf32> into tensor<2x2x128x128xf32, #dram>
    -> tensor<2x2x128x128xf32, #dram>

  return %1 : tensor<2x2x128x128xf32, #dram>
}

// CHECK-LABEL: func.func @sharded_dram_to_host
func.func @sharded_dram_to_host(%arg0: tensor<2x2x128x128xf32, #dram>) -> tensor<256x256xf32> {
  %0 = d2m.empty() : tensor<256x256xf32>

  // CHECK: %[[HOST:.*]] = d2m.empty() : tensor<256x256xf32>
  // CHECK: d2m.to_host %arg0, %[[HOST]] layout = #{{.*}} : tensor<2x2x128x128xf32, #{{.*}}> into tensor<256x256xf32> -> tensor<256x256xf32>
  // CHECK-NOT: d2m.generic

  %1 = d2m.to_layout %arg0, %0 : tensor<2x2x128x128xf32, #dram> into tensor<256x256xf32>
    -> tensor<256x256xf32>

  return %1 : tensor<256x256xf32>
}

// Tiled variants: tilize/untilize must happen in L1. Scalar data is written
// to DRAM first (host->DRAM), then bounced DRAM->L1, tilized in L1, and
// finally moved L1->DRAM. The reverse (DRAM tiled->host) untilizes in L1
// before the to_host transfer.

#dram_tiled = #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, dram, sharded>

// CHECK-LABEL: func.func @host_to_dram_tiled
func.func @host_to_dram_tiled(%arg0: tensor<256x256xf32>) -> tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #dram_tiled> {
  %0 = d2m.empty() : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #dram_tiled>

  // Scalar data goes directly to DRAM via to_device (no L1 landing).
  // CHECK: d2m.to_device %arg0
  // Tilize path: DRAM->L1 bounce, tilize in L1, L1->DRAM.
  // CHECK-COUNT-3: d2m.generic

  %1 = d2m.to_layout %arg0, %0 : tensor<256x256xf32> into tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #dram_tiled>
    -> tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #dram_tiled>

  return %1 : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #dram_tiled>
}

// CHECK-LABEL: func.func @dram_to_host_tiled
func.func @dram_to_host_tiled(%arg0: tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #dram_tiled>) -> tensor<256x256xf32> {
  %0 = d2m.empty() : tensor<256x256xf32>

  // DRAM tiled->host: untilize must happen in L1, then to_host.
  // No DRAM->L1 explicit bounce op needed; the untilize generic outputs L1.
  // CHECK: d2m.generic
  // CHECK: d2m.to_host
  // CHECK-NOT: d2m.to_device

  %1 = d2m.to_layout %arg0, %0 : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #dram_tiled> into tensor<256x256xf32>
    -> tensor<256x256xf32>

  return %1 : tensor<256x256xf32>
}
