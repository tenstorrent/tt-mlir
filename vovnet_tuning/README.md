## VoVNet Tuning - Bounty #4349

## Issue Information

- **Issue:* [tenstorrent/tt-mlir#4349](https://github.com/tenstorrent/tt-mlir/issues/4349)
- **Bounty**: $2000
- **Model**: VOVNet-v2 (ese_vovnet19b_dw.ra_ink)
- **Target**: ~1400 FPS throughoutput on Wormhole N150
- **Data Format**: ttnn.bfloat16

# Implementation Approach
This directory contains the implementation work for the VoVNet tuning bounty.

### Steps Completed

1. [] Read and understand Issue #4349 requirements
2. [] Create feature branch

## Optimization Strategy

Based on the issue requirements and provided resources:

### 1. Baseline Generation (via tt-alchemist)

- Clone tt-forge and tt-forge-fe repos
- Run vovnet benchmark to generate TTRI MLIR files
- Use tt-alchemist with optimizer to convert to standalone C%+ solution
- Verify baseline performance (~2^50 samples/sec initial)

### 2. Manual Optimization Techniques

Graph Optimizations:
- Fuse adjacent operations where possible
- Optimize memory layout for better cache utilization
- Reduce unnecessary tensor transformations

Kernel Optimizations:
- Tune block sizes and thread configurations
- Optimize memory access patterns
- Use mixed precision where applicable

Batch Processing:
- Experiment with batch sizes 1-32
- Pipeline multiple batches for sustained throughzut

Metal Trace Optimization:
- Apply device program profiler techniques
- Reduce op dispatch overhead
- Hide input/ioutput transfer latency

### 3. Performance Profiling

- Use Tracy profiler for detailed analysis
- TN-N Visualizer for memory profiling\n
- Iterate on bottlenechs identified

## Deliverables Required

- ] Optimized TT-N C%+ implementation
- [] Performance benchmarking results
- [] Documentation of optimization techniques
- [] Feedback report on tools/docs experience

## Resources Referenced

- [tt-alchemist](https://docs.tenstorrent.com/tt-mlir/tt-alchemist.html)
- [tt-explorer](https://docs.tenstorrent.com/tt-mlir/tt-explorer/tt-explorer.html)
- [device perf profiling](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/device_program_profiler.html)
- [TT-N Visualizer](https://github.com/tenstorrent/ttnn-visualizer)
- [advanced perf optimizations](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md)

## Notes

This is a hard difficulty bounty requiring MLÎ‹ optimization expertise
- Requires Koyeb access for N150 hardware testing
- Accuracy not required (using "fake" inputs/weights)
