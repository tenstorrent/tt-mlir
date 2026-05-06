# Program-Level Hooks for DebugHooks

## Description

Add `preProgram`/`postProgram` callbacks to `Hooks` alongside the existing `preOperator`/`postOperator` callbacks, and call them before/after the op loop in `ProgramExecutor::execute()`. This enables Chisel to set up/tear down per-program state around the op loop.

Program-level callbacks use a different signature than op callbacks: they receive `(Binary, CallbackContext)` — no `OpContext`, since there is no specific op in scope.

## Acceptance Criteria

- All 4 hooks fire in correct order: `preProgram → (preOp → op → postOp)* → postProgram`
- Python test confirms callback execution order
- Existing op-level callback behavior is unchanged
- `unregister_hooks()` clears all callbacks (op + program)
