# Strategy: Verify MLIR Walk Matches Flatbuffer Runtime Walk

## Goal

Confirm that iterating over TTNN ops via MLIR python bindings (parsing the MLIR
string from the flatbuffer) produces the same op sequence as iterating via the
flatbuffer's `get_program_ops_as_json()` / runtime callbacks. This is critical
for Chisel: the MLIR walk builds the op registry at init time, and the runtime
callbacks fire in flatbuffer execution order. If these diverge, Chisel will
correlate the wrong op with the wrong callback.

## The Two Walks

### 1. MLIR Walk (Chisel's approach)

Parse the TTNN MLIR string embedded in the flatbuffer, then walk the function
body linearly (regions → blocks → operations):

```python
from ttmlir.ir import Context, Module
from ttmlir.dialects import ttnn as ttnn_dialect

# Get MLIR source from flatbuffer
mlir_text = fbb.mlir.source

with Context() as ctx:
    ttnn_dialect.register_dialect(ctx)
    module = Module.parse(mlir_text, ctx)
    for func_op in module.body.operations:
        for block in func_op.regions[0].blocks:
            for op in block.operations:
                print(op.name, op.location)
```

Each op yields: `op.name` (e.g. `"ttnn.matmul"`), `op.location`, operands,
results with shapes/dtypes.

### 2. Flatbuffer Walk (runtime's approach)

Iterate the serialized operation list via `ttrt.binary`:

```python
import json, ttrt.binary

fbb = ttrt.binary.load_binary_from_path("model.ttnn")
for program_idx in range(fbb.get_num_programs()):
    ops = json.loads(fbb.get_program_ops_as_json(program_idx))
    for op in ops:
        print(op["type_type"], op["loc_info"])
```

Each op yields: `type_type` (e.g. `"MatmulOp"`), `loc_info`, `debug_info`,
input/output tensor refs.

This is the same order the runtime's `ProgramExecutor::execute()` uses when
firing `DebugHooks` callbacks — it literally iterates `program->operations()`.

## What to Compare

| Property       | MLIR walk                            | Flatbuffer walk                  | Join key? |
|----------------|--------------------------------------|----------------------------------|-----------|
| Op name        | `op.name` → `"ttnn.matmul"`         | `type_type` → `"MatmulOp"`      | Yes (after normalization) |
| Location       | `op.location` → `loc(...)` string   | `loc_info` → same format        | **Primary join key** |
| Op count       | `len(ops)`                           | `len(json_ops)`                  | Must match |
| Order          | Linear block iteration               | Flat array index                 | Must match |

### Name Normalization

MLIR names are dialect-qualified snake_case (`ttnn.matmul`), flatbuffer names
are PascalCase with `Op` suffix (`MatmulOp`). Convert with:

```python
import re

def fb_name_to_mlir(fb_name: str) -> str:
    """MatmulOp -> ttnn.matmul"""
    name = fb_name.removesuffix("Op")
    return "ttnn." + re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
```

## Verification Script

```python
"""
Compare MLIR walk vs flatbuffer walk for a compiled .ttnn binary.

Usage:
    source env/activate
    python tools/chisel/docs/strategy_mlir_vs_flatbuffer_walk.py model.ttnn
"""
import json
import re
import sys

import ttrt.binary
from ttmlir.ir import Context, Module
from ttmlir.dialects import ttnn as ttnn_dialect

IGNORED_OPS = {"ttnn.deallocate", "func.return"}


def fb_name_to_mlir(fb_name: str) -> str:
    name = fb_name.removesuffix("Op")
    return "ttnn." + re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def mlir_walk(mlir_text: str):
    """Walk MLIR module, return list of (op_name, loc_string)."""
    ops = []
    with Context() as ctx:
        ttnn_dialect.register_dialect(ctx)
        module = Module.parse(mlir_text, ctx)
        for func_op in module.body.operations:
            for block in func_op.regions[0].blocks:
                for op in block.operations:
                    if op.name in IGNORED_OPS:
                        continue
                    loc_str = str(op.location)
                    ops.append((op.name, loc_str))
    return ops


def flatbuffer_walk(fbb, program_idx=0):
    """Walk flatbuffer ops, return list of (op_name, loc_string)."""
    ops = []
    json_ops = json.loads(fbb.get_program_ops_as_json(program_idx))
    for op in json_ops:
        mlir_name = fb_name_to_mlir(op["type_type"])
        if mlir_name in IGNORED_OPS:
            continue
        ops.append((mlir_name, op.get("loc_info", "")))
    return ops


def compare(mlir_ops, fb_ops):
    """Compare two op lists, print mismatches."""
    if len(mlir_ops) != len(fb_ops):
        print(f"MISMATCH: MLIR has {len(mlir_ops)} ops, "
              f"flatbuffer has {len(fb_ops)} ops")

    max_len = max(len(mlir_ops), len(fb_ops))
    mismatches = 0
    for i in range(max_len):
        mlir = mlir_ops[i] if i < len(mlir_ops) else ("(missing)", "")
        fb = fb_ops[i] if i < len(fb_ops) else ("(missing)", "")
        match = mlir[0] == fb[0]
        status = "OK" if match else "MISMATCH"
        if not match:
            mismatches += 1
        print(f"  [{i:3d}] {status:8s}  mlir={mlir[0]:30s}  fb={fb[0]:30s}"
              f"  loc_match={mlir[1] == fb[1]}")

    if mismatches == 0:
        print(f"ALL {max_len} ops match in name and order.")
    else:
        print(f"{mismatches}/{max_len} ops MISMATCHED.")
    return mismatches == 0


if __name__ == "__main__":
    path = sys.argv[1]
    fbb = ttrt.binary.load_binary_from_path(path)
    mlir_text = fbb.mlir.source

    for prog_idx in range(fbb.get_num_programs()):
        print(f"\n=== Program {prog_idx} ===")
        m_ops = mlir_walk(mlir_text)
        f_ops = flatbuffer_walk(fbb, prog_idx)
        compare(m_ops, f_ops)
```

## Known Differences to Watch For

1. **Ignored ops** — The flatbuffer may not serialize all MLIR ops (e.g.
   `func.return`, `ttnn.deallocate`). Filter these from the MLIR walk to get a
   fair comparison.

2. **GetDeviceOp** — Present in flatbuffer but may not appear in MLIR text.
   Check if it's synthesized during serialization.

3. **Const-eval programs** — The flatbuffer can contain multiple programs
   (const-eval + main). The MLIR text may only show the main function. Compare
   per-program, matching by function name.

4. **Location format** — MLIR `op.location` and flatbuffer `loc_info` should
   use the same format (`loc("file":line:col)`), but verify string equality
   vs semantic equality (whitespace, nested locs).

## Success Criteria

- Op count matches per program (after filtering ignored ops)
- Op names match at every index (after normalization)
- Location strings match at every index
- This holds across multiple test binaries (simple model, multi-program,
  model with fusion)
