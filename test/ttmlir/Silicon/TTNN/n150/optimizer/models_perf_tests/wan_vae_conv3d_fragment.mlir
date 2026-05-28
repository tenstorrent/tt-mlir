// REQUIRES: opmodel, perf
// XFAIL: *
//
// Wan VAE Conv3d perf gate — exercises four representative Conv3d shapes
// (3x3x3 main, 3x1x1 temporal, 1x1x1 pointwise, 1x2x2 patchify) end-to-end
// on n150 silicon. Optimizer attaches a Conv3dConfig per op via Phase 5
// scoring, then ttrt runs the flatbuffer.
//
// Conv3d optimizer support is currently wired through the chain-based
// optimizer (not the greedy one); see plan/nifty-gathering-umbrella.md.
// We pin optimization-level=1 + enable-greedy-optimizer=false so the
// Phase 5 scoring attaches non-null conv3d_config to each ttnn.conv3d
// before silicon execution.
//
// Currently XFAIL: tt-metal's conv3d kernel rejects the optimizer-emitted
// IR with "Layout mismatch, expected TILE, got ROW_MAJOR" at runtime, even
// when block sizes match tt-metal defaults. The compile+translate steps
// succeed; only ttrt run fails. See conv3d_optimizer.mlir for the same
// issue on a minimal single-op fragment. Remove XFAIL once tt-metal
// integration is resolved.
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=1 enable-greedy-optimizer=false" -o wan_vae_conv3d_fragment_ttnn.mlir %models/wan_vae_conv3d_fragment.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn wan_vae_conv3d_fragment_ttnn.mlir
// RUN: ttrt run %t.ttnn
