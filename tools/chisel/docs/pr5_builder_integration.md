# PR 5: Builder Integration

**Superseded.** Builder integration has been merged into
[PR 1: Single Op Isolation](pr1_single_op_isolation.md).

The `enable_chisel` parameter in `execute_fb()`, mutual exclusivity check,
`chisel.bind()`/`chisel.unbind()` lifecycle, and all builder API forwarding
are now part of PR 1's scope.

See the updated [PR 1 doc](pr1_single_op_isolation.md) for implementation
details and test plan.
