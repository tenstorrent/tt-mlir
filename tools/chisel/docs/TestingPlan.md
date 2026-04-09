# Chisel — Testing Plan

## Section 1 — Overview

This document describes the testing strategy for Chisel. The focus is on
**integration tests** that exercise Chisel through the real builder compilation
and execution pipeline, and through `ttrt run`. Unit tests are deferred to a
separate PR (1.5).

All integration tests require TT hardware (silicon). There are no CI-only
integration tests — Chisel's value is in comparing golden CPU outputs against
real device outputs, which requires a device.

### 1.1 Test Categories

| Category | Dependencies | Description |
| :--- | :--- | :--- |
| Builder Integration | builder + ttmlir + device | `compile_and_execute_ttnn(enable_chisel=True)` with real ops |
| Golden Test Suite | golden tests + `--enable-chisel` | Run existing golden tests with chisel enabled |
| ttrt run Integration | ttrt + ttmlir + device | Compile to flatbuffer, `ttrt run` with chisel bound |
| Multi-Op Graph | builder + device | Multi-op graphs exercising op\_iter across several ops |
| Output Verification | builder + device | Verify chisel log format, PCC values, op counts |

### 1.2 Running Tests

**Run existing golden tests with chisel enabled:**

```bash
pytest test/python/golden/test_ttnn_ops.py --enable-chisel
```

The `--enable-chisel` flag is already wired into `test/python/golden/conftest.py`
(lines 277–278, 376–380). It sets `enable_chisel=True` in the kwargs passed to
`compile_and_execute_ttnn()`. This means every existing golden test can serve as
a chisel integration test with zero new test code.

**Run chisel-specific integration tests:**

```bash
pytest tools/chisel/tests/ --tb=short
```

## Section 2 — Golden Test Suite Integration

The fastest path to broad chisel coverage is running the existing golden test
suite with `--enable-chisel`. This exercises chisel against every op and model
that the golden tests already cover.

### 2.1 How It Works

1. `test/python/golden/conftest.py:get_request_kwargs()` reads `--enable-chisel`
   and adds `enable_chisel=True` to the kwargs dict
2. Each golden test calls `compile_and_execute_ttnn(module, **kwargs, device=device)`
3. `builder_runtime.py:execute_fb()` sees `enable_chisel=True`, calls
   `chisel.bind()` which registers preOp/postOp callbacks
4. During execution, chisel callbacks fire for every TTNN op, running golden
   comparison and logging PCC/atol/rtol

### 2.2 Key Test Files

| Test File | Ops Covered |
| :--- | :--- |
| `test/python/golden/test_ttnn_ops.py` | clamp, repeat, reshape, concat, eltwise, etc. |
| `test/python/golden/test_stablehlo_ops.py` | StableHLO frontend ops lowered to TTNN |
| `test/python/golden/test_ttir_ops.py` | TTIR ops lowered to TTNN |
| `test/python/golden/test_generic.py` | Generic op patterns |
| `test/python/golden/test_ttir_models.py` | Full model graphs (resnet, etc.) |

### 2.3 What This Validates

- Chisel callbacks fire without crashing for every TTNN op in the suite
- IRModule parsing succeeds for compiler-generated MLIR (not just hand-written fixtures)
- `execute_golden()` finds golden functions for all ops via `GOLDEN_MAPPINGS`
- Op iterator stays in sync with callback dispatch across multi-op programs
- No interference with builder's own golden comparison

## Section 3 — Builder Integration Tests

Dedicated tests that validate specific chisel behavior through the builder
pipeline. These go beyond "doesn't crash" to verify chisel's output.

### 3.1 Single-Op Tests

Each test compiles and executes a single TTNN op with chisel enabled, then
verifies the chisel log output.

| Test | Op | What It Validates |
| :--- | :--- | :--- |
| `test_chisel_sigmoid` | sigmoid | Unary op. Log contains `ttnn.sigmoid` with PCC > 0.99. |
| `test_chisel_relu` | relu | Unary op. Numerically exact — PCC should be 1.0. |
| `test_chisel_add` | add | Binary op. Two inputs stashed and forwarded correctly. |
| `test_chisel_matmul` | matmul | Binary op with shape change (e.g. 32×64 @ 64×16 → 32×16). |
| `test_chisel_exp` | exp | Unary op. Validates dtype preservation through chisel. |
| `test_chisel_softmax` | softmax | Unary op with attribute (dim). Validates `_build_golden_args` attribute forwarding. |

**Test pattern:**

```python
def test_chisel_sigmoid(device, caplog):
    from builder.ttnn.ttnn_builder import TTNNBuilder
    from builder.base.builder_apis import compile_and_execute_ttnn
    from builder.base.builder_utils import Operand

    def module(builder: TTNNBuilder):
        @builder.func([(32, 32)], [torch.float32])
        def func(in0: Operand, builder: TTNNBuilder):
            return builder.sigmoid(in0)

    with caplog.at_level(logging.INFO, logger="chisel"):
        compile_and_execute_ttnn(
            module, device=device, enable_chisel=True,
        )

    assert "ttnn.sigmoid" in caplog.text
    pcc_match = re.search(r"PCC=(\d+\.\d+)", caplog.text)
    assert pcc_match and float(pcc_match.group(1)) > 0.99
```

### 3.2 Mutual Exclusivity

| Test | Description |
| :--- | :--- |
| `test_chisel_and_verify_raises` | `execute_fb(enable_chisel=True, enable_intermediate_verification=True)` raises `ValueError` |

## Section 4 — Multi-Op Graph Tests

Tests that validate chisel iterating over multiple ops in a single program.

| Test | Graph | What It Validates |
| :--- | :--- | :--- |
| `test_chisel_add_relu` | add → relu | Two-op chain. Chisel logs 2 ops in correct order. Each op's PCC independent. |
| `test_chisel_matmul_softmax` | matmul → softmax | Shape-changing op followed by attribute op. op\_iter advances correctly. |
| `test_chisel_three_op_chain` | relu → add → sigmoid | Three ops. Exact op count in chisel output matches. |

**Test pattern:**

```python
def test_chisel_add_relu(device, caplog):
    def module(builder: TTNNBuilder):
        @builder.func([(32, 32), (32, 32)], [torch.float32, torch.float32])
        def func(in0: Operand, in1: Operand, builder: TTNNBuilder):
            added = builder.add(in0, in1)
            return builder.relu(added)

    with caplog.at_level(logging.INFO, logger="chisel"):
        compile_and_execute_ttnn(
            module, device=device, enable_chisel=True,
        )

    # Verify both ops logged
    assert "ttnn.add" in caplog.text
    assert "ttnn.relu" in caplog.text

    # Verify PCC for each
    pcc_values = re.findall(r"PCC=(\d+\.\d+)", caplog.text)
    assert len(pcc_values) >= 2
    for pcc in pcc_values:
        assert float(pcc) > 0.99
```

## Section 5 — ttrt run Integration Tests

Tests that validate chisel hooked into `ttrt run` execution rather than builder.

### 5.1 Approach

The `ttrt run` path at `tools/ttrt/common/run.py:621–625` registers its own
DebugHooks callbacks. Chisel can replace these via `chisel.bind()` before
calling `ttrt run` programmatically.

The test flow:
1. Compile a model to flatbuffer via builder (`skip_exec=True`)
2. Call `chisel.bind()` to register chisel callbacks
3. Execute the flatbuffer via `ttrt.runtime` (mirrors the `ttrt run` path)
4. Call `chisel.unbind()` for cleanup
5. Verify chisel log output

| Test | Description |
| :--- | :--- |
| `test_chisel_ttrt_single_op` | Compile single-op flatbuffer, execute via ttrt with chisel. Verify callbacks fire. |
| `test_chisel_ttrt_multi_op` | Multi-op flatbuffer via ttrt with chisel. Verify all ops processed. |

### 5.2 Future: `--enable-chisel` flag for `ttrt run`

Currently `ttrt run` has no `--enable-chisel` CLI flag. Adding this would allow:

```bash
ttrt run model.ttnn --enable-chisel
```

This is a natural follow-up after the programmatic integration is validated.

## Section 6 — Output Verification Tests

Verify the format and content of chisel's output.

| Test | Description |
| :--- | :--- |
| `test_chisel_log_format` | Each log line matches `{op_name}: PCC={float}, atol={sci}, rtol={sci}` |
| `test_chisel_all_ops_reported` | For a known N-op graph, exactly N log entries |
| `test_chisel_exact_ops_pcc` | For numerically exact ops (relu, abs), PCC = 1.0 |
