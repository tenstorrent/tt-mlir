# Milestone: Multi-Chip Chisel Debugging

## Description

Extend Chisel to support multi-chip (multi-device) models end-to-end. Currently
Chisel assumes single-device execution — tensor retrieval, golden comparison,
and reporting all operate on a single tensor per op output. Multi-chip models
shard tensors across logical devices, so Chisel must handle per-device shards
throughout the pipeline: input stashing, golden execution, comparison, and
reporting.

## Goals

- Chisel produces correct per-op accuracy reports for multi-device model
  inference (e.g., multi-chip Qwen, LLaMA, or Falcon)
- Per-device shard comparison with device ID in report rows
- Golden execution handles sharded inputs (either replicated golden or
  per-shard golden, depending on op semantics)
- No regression on single-chip models

## Prerequisites

- issue_05: Multi-device tensor access via `retrieve_tensor_from_pool`
  (returns `Dict[int, Tensor]` instead of single tensor)
- issue_07: Single program flow (cross-op golden chaining must work before
  extending to multi-device)

## Key Challenges

1. **Shard-aware tensor pool**: `TensorPool` must store per-device shards
   keyed by `(tensor_id, device_id)` or store a map per tensor ID
2. **Golden execution for sharded ops**: Some ops are replicated across devices
   (same golden per shard), others have device-specific semantics (e.g.,
   all-gather, reduce-scatter). Need to determine which ops need special
   handling vs. which can run golden per-shard independently
3. **Input reassembly**: Some golden functions expect the full (un-sharded)
   tensor. May need a reassemble step before golden execution and a
   re-shard step after for comparison
4. **Reporting**: Report rows need device ID column; summary metrics should
   aggregate across devices and flag per-device outliers
5. **Multi-program coordination**: Multi-chip models may use multiple programs
   with cross-program tensor passing — `global_tensor_pool` must handle
   sharded tensors

## Test Models

- Multi-chip variants of models already tested on single chip
- Galaxy (multi-chip mesh) configurations if available in CI

## Acceptance Criteria

- `chisel` produces per-op accuracy report for a multi-device model with
  per-device shard comparisons
- Report includes device ID per comparison row
- Single-chip models continue to work without changes (backwards compatible)
- Golden pool correctly tracks sharded tensors across ops within a program
- At least one multi-chip model tested end-to-end in CI or manual validation
