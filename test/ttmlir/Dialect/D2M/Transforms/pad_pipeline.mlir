// RUN: ttmlir-opt --d2m-fe-pipeline %s | FileCheck %s
// RUN: ttmlir-opt --d2m-fe-pipeline --d2m-be-pipeline %s | FileCheck %s --check-prefix=BACKEND

// Smoke test: pad survives the full D2M frontend pipeline (including
// SplitUnifiedThread, ScheduleDMA, LowerLoadStoreOpsToDMA in the backend).

// CHECK-LABEL: func.func @pad_then_add
// BACKEND-LABEL: func.func @pad_then_add
func.func @pad_then_add(%arg0: tensor<224x224xbf16>, %arg1: tensor<224x256xbf16>) -> tensor<224x256xbf16> {
  // Frontend lowers to memref + d2m.generic with explicit DMA ops.
  // CHECK: memref.alloc
  // CHECK: d2m.fill_buffer
  // CHECK: d2m.generic
  %1 = "ttir.pad"(%arg0) <{padding = array<i32: 0, 0, 0, 32>, value = 1.0 : f32}> : (tensor<224x224xbf16>) -> tensor<224x256xbf16>
  %2 = "ttir.add"(%1, %arg1) : (tensor<224x256xbf16>, tensor<224x256xbf16>) -> tensor<224x256xbf16>
  return %2 : tensor<224x256xbf16>
}
