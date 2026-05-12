# Multi-Dialect Builder - Option 6 Implementation Guide

## Overview

Option 6 (Delegation Pattern) enables creating modules with operations from multiple dialects (TTIR, TTNN, StableHLO, etc.) using a single unified builder interface.

## Key Concept: State Sharing

The core idea is that **all dialect builders share the same internal state** by sharing the same `__dict__`:

```python
# Create a new builder instance WITHOUT calling __init__
builder = TTIRBuilder.__new__(TTIRBuilder)

# Share the entire state by pointing to the same __dict__
builder.__dict__ = self.__dict__

# Now all operations on 'builder' modify the shared state!
```

This means:
- All dialect builders use the same MLIR Context
- All goldens are stored in the same map
- All function operations are tracked together
- Module insertion points are shared

## Implementation

### Core Structure

```python
class MultiDialectBuilder(Builder):
    def __init__(self, ctx, location, dialects=["ttir", "ttnn"], ...):
        # Initialize the base Builder with all shared state
        super().__init__(ctx, location, ...)

        # Create dialect builders that share this state
        self._dialect_builders = {}
        for dialect in dialects:
            builder_cls = self._dialect_map[dialect]  # TTIRBuilder, TTNNBuilder, etc.
            builder = builder_cls.__new__(builder_cls)
            builder.__dict__ = self.__dict__  # KEY: Share state!
            self._dialect_builders[dialect] = builder

    def __getattr__(self, name):
        # Automatically delegate to the right dialect builder
        for builder in self._dialect_builders.values():
            if hasattr(builder, name):
                return getattr(builder, name)
        raise AttributeError(f"No builder has method {name}")
```

### How Method Delegation Works

1. User calls `builder.sigmoid(x)`
2. Python looks for `sigmoid` on `MultiDialectBuilder` instance
3. Not found, so Python calls `__getattr__('sigmoid')`
4. `__getattr__` searches through `_dialect_builders`
5. Finds `sigmoid` in `TTIRBuilder`
6. Returns the method bound to the shared state
7. Method executes with shared context, goldens, etc.

## Usage Examples

### Example 1: Basic Multi-Dialect Module

```python
from builder.multi_dialect_example import MultiDialectBuilder

ctx = Context()
loc = Location.unknown(ctx)

# Create builder with multiple dialects
builder = MultiDialectBuilder(
    ctx,
    loc,
    dialects=["ttir", "ttnn"],  # Enable these dialects
    mesh_dict=OrderedDict([("x", 1), ("y", 1)])
)

# Build module using ops from both dialects
def module(builder):
    @builder.func([(32, 32), (32, 32)], [torch.float32, torch.float32])
    def mixed_function(in0, in1, builder):
        # Use TTIR operations
        x = builder.sigmoid(in0)     # ttir.sigmoid
        y = builder.relu(in1)        # ttir.relu

        # Use operations (resolved automatically)
        result = builder.add(x, y)   # Whichever dialect has 'add'

        return result

# Compile
with ctx, loc:
    new_module = Module.create()
    builder._root_module_insertion_point = new_module.body
    builder._current_module_insertion_point = new_module.body

    with InsertionPoint(new_module.body):
        module(builder)

print(new_module)
```

### Example 2: Using with build_module API

To integrate with the existing `build_module` API:

```python
from builder.base.builder_apis import build_module

def my_module(builder: MultiDialectBuilder):
    @builder.func([(64, 64)], [torch.float32])
    def forward(x, builder):
        x = builder.abs(x)       # Could be from any enabled dialect
        x = builder.sigmoid(x)   # Automatically finds the right implementation
        return x

# Would need to extend build_module to support "multi" type:
# module, builder = build_module(
#     my_module,
#     builder_type="multi",
#     dialects=["ttir", "ttnn", "stablehlo"]
# )
```

### Example 3: Introspection

```python
builder = MultiDialectBuilder(ctx, loc, dialects=["ttir", "ttnn", "stablehlo"])

# Check which dialects are enabled
print(list(builder._dialect_builders.keys()))
# Output: ['ttir', 'ttnn', 'stablehlo']

# Check if a method exists
if hasattr(builder, 'sigmoid'):
    print("sigmoid is available")

# Find which dialect provides a method (if you add this helper)
dialect = builder.get_method_dialect('sigmoid')  # Returns 'ttir'
```

## Advantages of Option 6

1. **Transparent API**: Users don't need to know which dialect an op comes from
   ```python
   builder.sigmoid(x)  # Just works, no need for builder.ttir.sigmoid(x)
   ```

2. **Shared State**: All operations work on the same context and goldens
   - No need to manually sync state between builders
   - Golden tensors are automatically tracked across dialects

3. **Backward Compatible**: Existing code using single-dialect builders still works
   - `TTIRBuilder` can still be used standalone
   - `MultiDialectBuilder` is opt-in

4. **Flexible**: Can enable any combination of dialects
   ```python
   MultiDialectBuilder(ctx, loc, dialects=["ttir", "ttnn"])  # Two dialects
   MultiDialectBuilder(ctx, loc, dialects=["stablehlo"])     # Single dialect
   MultiDialectBuilder(ctx, loc, dialects=["ttir", "ttnn", "stablehlo"])  # All three
   ```

5. **No Refactoring**: Doesn't require changing existing builder implementations

## Potential Issues and Solutions

### Issue 1: Method Name Conflicts

If multiple dialects have methods with the same name but different signatures:

**Problem:**
```python
# Both TTIR and TTNN have 'add', but with different parameters
ttir_add(a, b)
ttnn_add(a, b, memory_config=None)
```

**Solution 1:** First match wins (current implementation)
- Simple, predictable based on dialect order
- Document which dialect takes precedence

**Solution 2:** Add explicit dialect access
```python
class MultiDialectBuilder:
    @property
    def ttir(self):
        return self._dialect_builders['ttir']

    @property
    def ttnn(self):
        return self._dialect_builders['ttnn']

# Usage
builder.ttir.add(a, b)  # Explicitly use TTIR add
builder.ttnn.add(a, b)  # Explicitly use TTNN add
builder.add(a, b)       # Use default (first match)
```

### Issue 2: Dialect-Specific Initialization

Some dialect builders have unique initialization (e.g., `TTNNBuilder` has `create_tensor_encoding`):

**Solution:** Override `__init__` to handle dialect-specific setup
```python
def __init__(self, ctx, location, dialects, ...):
    super().__init__(ctx, location, ...)

    # Create dialect builders
    for dialect in dialects:
        builder = self._create_dialect_builder(dialect)
        self._dialect_builders[dialect] = builder

        # Apply dialect-specific initialization
        if dialect == "ttnn":
            self.create_tensor_encoding = builder._create_tensor_encoding
```

### Issue 3: Type Hints

Type checkers won't know about delegated methods:

**Solution 1:** Use `typing.Protocol` or `@overload`
```python
from typing import Protocol, overload

class TTIRProtocol(Protocol):
    def sigmoid(self, x: Operand) -> OpResult: ...
    def relu(self, x: Operand) -> OpResult: ...

class TTNNProtocol(Protocol):
    def add(self, a: Operand, b: Operand) -> OpResult: ...

class MultiDialectBuilder(Builder, TTIRProtocol, TTNNProtocol):
    ...
```

**Solution 2:** Generate stub files for IDE support

## Testing

Run the test script:
```bash
cd /home/jgrim/wh-01-src/tt-mlir
python tools/builder/test_multi_dialect.py
```

This will:
1. Create a multi-dialect builder
2. Generate MLIR using ops from multiple dialects
3. Verify state sharing works correctly
4. Test method resolution

## Migration Path

To adopt this in your codebase:

1. **Phase 1**: Implement `MultiDialectBuilder` alongside existing builders
2. **Phase 2**: Use it for new code that needs multi-dialect support
3. **Phase 3**: Optionally migrate existing code
4. **Phase 4**: Keep single-dialect builders for simple use cases

## Files Created

- `/home/jgrim/wh-01-src/tt-mlir/tools/builder/multi_dialect_example.py` - Full implementation with examples
- `/home/jgrim/wh-01-src/tt-mlir/tools/builder/test_multi_dialect.py` - Runnable test script
- `/home/jgrim/wh-01-src/tt-mlir/tools/builder/MULTI_DIALECT_GUIDE.md` - This guide
