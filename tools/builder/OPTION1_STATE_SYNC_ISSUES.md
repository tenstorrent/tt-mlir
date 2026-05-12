# Option 1 (Composite Builder) - State Synchronization Issues

## Overview

Option 1 creates a composite builder where each dialect builder is a **separate instance** with its **own state**:

```python
class MultiDialectBuilder(Builder):
    def __init__(self, ctx, location, dialects=['ttir', 'ttnn'], ...):
        super().__init__(ctx, location, ...)
        # Each builder is a SEPARATE instance with SEPARATE state
        self.ttir = TTIRBuilder(ctx, location, ...) if 'ttir' in dialects else None
        self.ttnn = TTNNBuilder(ctx, location, ...) if 'ttnn' in dialects else None
```

This creates **multiple independent state containers**, leading to synchronization problems.

---

## Critical State That Needs Synchronization

Based on the `Builder.__init__` code, here are the key state variables that would be duplicated:

### 1. **Golden Tensors** (`_goldens`, `_goldens_to_store`)

```python
# In Builder.__init__:
self._goldens: Dict[Operand, GoldenMapTensor] = {}
self._goldens_to_store: List[Operand] = []
```

**The Problem:**
- Each dialect builder maintains its own `_goldens` dictionary
- When `builder.ttir.sigmoid(x)` creates a golden, it's stored in `ttir._goldens`
- When `builder.ttnn.add(x, y)` needs golden for `x`, it looks in `ttnn._goldens`
- **Result: Golden not found!** ❌

**Example:**
```python
builder = MultiDialectBuilder(ctx, loc, dialects=['ttir', 'ttnn'])

@builder.func([(32, 32)], [torch.float32])
def fn(input, builder):
    # TTIR creates the operand and golden
    x = builder.ttir.sigmoid(input)
    # ttir._goldens[x] = <golden tensor>

    # TTNN tries to use x
    y = builder.ttnn.add(x, input)
    # Looks in ttnn._goldens[x] -> KeyError! ❌
    # TTNN doesn't know about x's golden!
```

### 2. **Function Operations** (`_func_ops_generated`)

```python
self._func_ops_generated: Dict[func.FuncOp, List[List[Operand]]] = {}
```

**The Problem:**
- Tracks inputs and outputs for each function
- Used to build the final golden map
- Each builder has its own tracking

**Example:**
```python
@builder.func([(32, 32)], [torch.float32])
def fn(input, builder):
    x = builder.ttir.relu(input)
    return builder.ttnn.multiply(x, x)

# Which builder's _func_ops_generated gets the return value?
# ttir._func_ops_generated = {func_op: [[input], []]}  # Missing output!
# ttnn._func_ops_generated = {}  # Doesn't know about the function at all!
```

### 3. **Location Tracking** (`_operand_to_loc`, `_loc_to_operand`)

```python
self._operand_to_loc: Dict[Operand, str] = {}
self._loc_to_operand: Dict[str, Operand] = {}
```

**The Problem:**
- Maps operands to their source locations for debugging
- Used for golden comparison and error reporting
- Split across multiple builders

**Example:**
```python
x = builder.ttir.sigmoid(input)  # Location stored in ttir._operand_to_loc
y = builder.ttnn.add(x, input)   # Location stored in ttnn._operand_to_loc

# When debugging or comparing goldens, locations are incomplete
# Can't trace the full computational graph
```

### 4. **Module Insertion Points** (`_root_module_insertion_point`, `_current_module_insertion_point`)

```python
self._root_module_insertion_point = None
self._current_module_insertion_point = None
```

**The Problem:**
- Controls where operations are inserted in the MLIR module
- Each builder might try to insert into different locations
- Can lead to incorrect module structure

**Example:**
```python
# Both builders think they control the insertion point
builder.ttir._current_module_insertion_point = location_A
builder.ttnn._current_module_insertion_point = location_B

# Operations might end up in wrong places or duplicate locations
```

### 5. **Global ID Counter** (`_global_id`)

```python
self._global_id = -1
```

**The Problem:**
- Used to generate unique IDs for operations
- Each builder increments its own counter
- Can lead to ID collisions

**Example:**
```python
x = builder.ttir.sigmoid(input)   # ttir._global_id = 0
y = builder.ttnn.add(x, input)    # ttnn._global_id = 0  # Collision! ❌
```

### 6. **Bypass Operations** (`_bypass_ops`)

```python
self._bypass_ops: List[str] = []
```

**The Problem:**
- Tracks which operations to skip during golden comparison
- User might bypass an op in one dialect but not see it in another

**Example:**
```python
x = builder.ttir.sigmoid(input)
builder.ttir.bypass(x)  # Only added to ttir._bypass_ops

# Later, when golden comparison happens:
# ttnn builder doesn't know x should be bypassed
```

### 7. **Deallocations** (`_deallocated_goldens`, `_op_deallocations`)

```python
self._deallocated_goldens: Dict[Operand, str] = {}
self._op_deallocations: Dict[Union[OpView, Operand], List[Union[OpView, Operand]]] = {}
```

**The Problem:**
- Manages memory deallocation of golden tensors
- Each builder manages its own deallocations
- Can lead to premature deallocation or memory leaks

---

## Concrete Failure Scenarios

### Scenario 1: Cross-Dialect Data Flow

```python
builder = MultiDialectBuilder(ctx, loc, dialects=['ttir', 'ttnn'])

@builder.func([(32, 32), (32, 32)], [torch.float32, torch.float32])
def mixed_ops(in0, in1, builder):
    # Step 1: TTIR creates intermediate result
    x = builder.ttir.sigmoid(in0)
    # ttir._goldens[x] = GoldenMapTensor(...)
    # ttir._operand_to_loc[x] = "line:15"

    # Step 2: TTNN tries to use x
    y = builder.ttnn.add(x, in1)
    # Needs golden for x to compute golden for y
    # Looks in ttnn._goldens[x] -> KeyError! ❌

    return y
```

**What breaks:**
- TTNN can't compute golden for `y` because it doesn't have golden for `x`
- Runtime golden comparison will fail
- Debugging information is incomplete

### Scenario 2: Golden Map Generation

```python
# After building the module, get goldens for runtime:
input_output_goldens, intermediate_goldens = builder.golden_map

# Problem: Which builder's golden_map property is called?
# - If MultiDialectBuilder: Only has its own (empty) goldens
# - If ttir builder: Only has TTIR goldens
# - If ttnn builder: Only has TTNN goldens
# - Missing: Combined view of all goldens! ❌
```

### Scenario 3: Function Decorator

```python
@builder.func([(32, 32)], [torch.float32])
def fn(input, builder):
    x = builder.ttir.relu(input)
    return builder.ttnn.multiply(x, x)

# Who owns the function?
# - builder.func is on MultiDialectBuilder
# - It might register with MultiDialectBuilder._func_ops_generated
# - But ttir and ttnn operations are tracked separately
# - Function inputs/outputs are split across builders ❌
```

### Scenario 4: Preshard Arguments

```python
@builder.func([(64, 64)], [torch.float32])
def fn(input, builder):
    # Preshard the input
    builder.preshard_arg(input, shard_dims=[0])
    # This modifies input's attributes and golden

    # Use in TTIR op
    x = builder.ttir.sigmoid(input)

    # Use in TTNN op
    y = builder.ttnn.add(x, input)

    return y

# Problem: Which builder's preshard_arg is called?
# - If MultiDialectBuilder: Modifies MultiDialectBuilder._goldens
# - ttir and ttnn don't see the presharded golden ❌
```

---

## Workarounds and Their Costs

### Workaround 1: Manual State Synchronization

```python
class MultiDialectBuilder(Builder):
    def __init__(self, ctx, location, dialects, ...):
        super().__init__(ctx, location, ...)
        self.ttir = TTIRBuilder(ctx, location, ...)
        self.ttnn = TTNNBuilder(ctx, location, ...)

    def sync_goldens(self):
        """Manually sync golden tensors across builders."""
        # Copy goldens from ttir to ttnn
        self.ttnn._goldens.update(self.ttir._goldens)
        # Copy goldens from ttnn to ttir
        self.ttir._goldens.update(self.ttnn._goldens)
        # Also sync to parent
        self._goldens.update(self.ttir._goldens)
        self._goldens.update(self.ttnn._goldens)
```

**Problems:**
- ❌ Must call after every operation
- ❌ Expensive (copying dictionaries)
- ❌ Error-prone (easy to forget)
- ❌ Doesn't handle all state (insertion points, IDs, etc.)

### Workaround 2: Proxy All Operations

```python
class MultiDialectBuilder(Builder):
    def __init__(self, ctx, location, dialects, ...):
        super().__init__(ctx, location, ...)
        self.ttir = TTIRBuilder(ctx, location, ...)
        self.ttnn = TTNNBuilder(ctx, location, ...)

    def _intercept_operation(self, dialect_builder, op_name, *args, **kwargs):
        """Intercept every operation and sync state."""
        # Call the operation
        result = getattr(dialect_builder, op_name)(*args, **kwargs)

        # Sync golden for result
        if result in dialect_builder._goldens:
            self._goldens[result] = dialect_builder._goldens[result]
            self.ttir._goldens[result] = dialect_builder._goldens[result]
            self.ttnn._goldens[result] = dialect_builder._goldens[result]

        # Sync locations
        if result in dialect_builder._operand_to_loc:
            loc = dialect_builder._operand_to_loc[result]
            self._operand_to_loc[result] = loc
            self.ttir._operand_to_loc[result] = loc
            self.ttnn._operand_to_loc[result] = loc

        # ... sync other state ...

        return result
```

**Problems:**
- ❌ Extremely complex and fragile
- ❌ Performance overhead on every operation
- ❌ Must know about all state variables
- ❌ Breaks encapsulation

### Workaround 3: Shared State Objects

```python
class SharedState:
    """Container for shared state."""
    def __init__(self):
        self.goldens = {}
        self.goldens_to_store = []
        self.operand_to_loc = {}
        # ... all other state ...

class MultiDialectBuilder(Builder):
    def __init__(self, ctx, location, dialects, ...):
        super().__init__(ctx, location, ...)
        shared = SharedState()

        # Inject shared state into builders
        self.ttir = TTIRBuilder(ctx, location, ...)
        self.ttir._goldens = shared.goldens
        self.ttir._goldens_to_store = shared.goldens_to_store
        # ... inject all other state ...

        self.ttnn = TTNNBuilder(ctx, location, ...)
        self.ttnn._goldens = shared.goldens
        self.ttnn._goldens_to_store = shared.goldens_to_store
        # ... inject all other state ...
```

**Problems:**
- ❌ Must know about all internal state variables
- ❌ Breaks encapsulation
- ❌ Fragile (breaks if Builder adds new state)
- ❌ Essentially reimplementing Option 6 poorly

---

## Why Option 6 Solves These Problems

Option 6 (Delegation Pattern) avoids all these issues by **sharing the entire `__dict__`**:

```python
class MultiDialectBuilder(Builder):
    def __init__(self, ctx, location, dialects, ...):
        super().__init__(ctx, location, ...)  # Initialize state once

        # Create dialect builders that share this state
        for dialect in dialects:
            builder_cls = self._dialect_map[dialect]
            builder = builder_cls.__new__(builder_cls)
            builder.__dict__ = self.__dict__  # ✓ Share EVERYTHING
            self._dialect_builders[dialect] = builder
```

**Benefits:**
- ✅ **Zero synchronization overhead**: All builders read/write the same memory
- ✅ **Automatic**: No manual syncing required
- ✅ **Complete**: All state variables are shared automatically
- ✅ **Future-proof**: New state variables are automatically shared
- ✅ **Simple**: One line of code (`builder.__dict__ = self.__dict__`)

---

## Comparison Matrix

| Aspect | Option 1 (Composite) | Option 6 (Delegation) |
|--------|---------------------|----------------------|
| **State Sharing** | Manual sync required ❌ | Automatic ✅ |
| **Performance** | Overhead from syncing ❌ | No overhead ✅ |
| **Complexity** | Complex sync logic ❌ | Simple ✅ |
| **Maintenance** | Breaks when state changes ❌ | Future-proof ✅ |
| **Correctness** | Easy to get wrong ❌ | Correct by design ✅ |
| **Golden Tracking** | Incomplete ❌ | Complete ✅ |
| **API Clarity** | `builder.ttir.op()` ✅ | `builder.op()` ✅ |
| **Debugging** | Split traces ❌ | Complete traces ✅ |

---

## Conclusion

Option 1's state synchronization issues are **fundamental and difficult to solve**:

1. **Too many state variables** to synchronize (10+ attributes)
2. **Operations happen frequently** (every op needs sync)
3. **State is interconnected** (goldens depend on operands depend on locations)
4. **Workarounds are complex** and defeat the purpose of separation

**Recommendation:** Use **Option 6** instead, which solves all these problems elegantly through state sharing.
