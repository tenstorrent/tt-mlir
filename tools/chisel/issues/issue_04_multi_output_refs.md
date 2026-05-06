# Multi-Output Op Ref Support

## Description

Change `getOpOutputRef()` to return `std::vector<TensorRef>` instead of `std::optional<TensorRef>`. Ops like `SortOp`, `MaxPool2dWithIndicesOp`, and `BatchNormTrainingOp` produce multiple outputs but currently return `std::nullopt`.

Not required for initial Chisel integration — without this, Chisel skips multi-output ops gracefully.

## Tasks

- [ ] Change `getOpOutputRef()` return type to `std::vector<TensorRef>` in `runtime.cpp`
- [ ] Handle multi-output ops: `SortOp`, `MaxPool2dWithIndicesOp`, `BatchNormTrainingOp`
- [ ] Update all callers of `getOpOutputRef()` for the new return type
