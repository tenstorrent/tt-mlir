// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="mesh-shape=1,2 enable-trace=true" --mlir-print-local-scope %s | FileCheck %s

// Verify that KV cache arguments are correctly detected and kept on device
// during trace hoisting (not moved to host).

module {
    // In the capture function, the KV cache arg should remain on DRAM (device),
    // while the regular input should be on system_memory (host).
    // Constants are hoisted as the first arguments.
    //
    // The slot allocation function takes only the device-resident arguments -
    // the KV cache and the const-eval'd constant - as pass-through slots.
    // CHECK-LABEL: func.func private @allocate_slots_trace_0_main

    // CHECK-SAME: %arg0: tensor<1x32x64x512xbf16
    // CHECK-SAME: buffer_type<dram>
    // CHECK-SAME: ttcore.kv_cache

    // CHECK-SAME: %arg1:
    // CHECK-SAME: ttcore.argument_type = #ttcore.argument_type<constant>

    // The capture function takes the host-staged inputs first, then every slot.
    // CHECK-LABEL: func.func private @run_and_capture_trace_0_main

    // CHECK-SAME: %arg0: tensor<5x128x512xbf16
    // CHECK-SAME: buffer_type<system_memory>
    // CHECK-SAME: ttcore.argument_type = #ttcore.argument_type<input>

    // CHECK-SAME: %arg2: tensor<1x32x64x512xbf16
    // CHECK-SAME: buffer_type<dram>
    // CHECK-SAME: ttcore.argument_type = #ttcore.argument_type<input>
    // CHECK-SAME: ttcore.kv_cache

    // CHECK-SAME: %arg3:
    // CHECK-SAME: ttcore.argument_type = #ttcore.argument_type<constant>

    // CHECK: "ttnn.begin_trace_capture"
    // CHECK: "ttnn.end_trace_capture"
    // CHECK: "ttnn.execute_trace"

    // CHECK-LABEL: func.func @main(
    func.func @main(
        %arg0: tensor<5x128x512xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<presharded>} loc("p0.2"),
        %cache: tensor<1x32x64x512xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<presharded>} loc("p1.3"),
        %update_input: tensor<1x32x1x512xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}
    ) -> (tensor<5x128x512xbf16> {ttcore.shard_status = #ttcore.shard_status<presharded>},
          tensor<1x32x64x512xbf16> {ttcore.shard_status = #ttcore.shard_status<presharded>}) {
        %idx = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
        "ttir.update_cache"(%cache, %update_input, %idx) <{batch_offset = 0 : i32}> : (tensor<1x32x64x512xbf16>, tensor<1x32x1x512xbf16>, tensor<1xi32>) -> ()
        %1 = "ttir.add"(%arg0, %arg0) : (tensor<5x128x512xbf16>, tensor<5x128x512xbf16>) -> tensor<5x128x512xbf16>
        return %1, %cache : tensor<5x128x512xbf16>, tensor<1x32x64x512xbf16>
    }
}
