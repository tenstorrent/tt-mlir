# Missing from dump.md

## Initialization

- No mention of `module_provider` callback for handling different binaries
- Missing: the `caching` option for tensor pools (disk caching of `.pt` files)

## PreProgram

- Missing: matching preserved golden tensors with incoming device TensorRefs (cross-program tensor reuse)
- Missing: reset `_op_index` to 0

## PostOp

- Missing: **multi-output ops** — some ops (Sort, MaxPool2dWithIndices, BatchNormTraining) produce multiple outputs; the plan assumes single output per op

## PostProgram

- Missing: **asymmetric reset** — must clear `device_tensor_pool` (stale TensorRefs) but **preserve** `golden_tensor_pool`
- Missing: aggregate metrics logging (min/max/mean PCC across ops)
- Missing: reset `_op_index`

## Cross-cutting concerns not mentioned

- **Multi-chip** handling (per-device tensor lists, per-device comparison)
- **Cross-binary tensor identity** via `Tensor::globalId` (for tt-xla flows where outputs from one binary become inputs to another)
- **Unmapped ops** — what happens when an op has no entry in `GOLDEN_MAPPINGS`? (fail-hard vs warn-and-skip)

## Open questions

- The question about `load_cache` and `funcCall` ops is noted but the docs don't answer it either — this remains an open item.
