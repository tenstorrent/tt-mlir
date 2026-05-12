# Builder Architecture Proposal: Plugin-Based System

## Problem Statement

Current builder architecture has:
- **High complexity**: 5+ builder classes, 40K+ lines of code
- **Code duplication**: Similar patterns repeated across builders
- **Maintenance burden**: Changes require updates to multiple classes
- **Scaling issues**: Adding new dialects requires new builder classes

## Proposed Solution: Plugin-Based Architecture

### Core Concept

```python
# Single Builder class for state management
# Dialects are plugins that register operations
# Operations are bound methods that work with shared state
```

### Architecture Diagram

```
                    ┌──────────────┐
                    │   Builder    │
                    │              │
                    │ • Context    │
                    │ • Goldens    │
                    │ • State      │
                    └──────┬───────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
            ▼              ▼              ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │ TTIRDialect  │ │ TTNNDialect  │ │StableHLODial │
    │              │ │              │ │              │
    │ • sigmoid()  │ │ • add()      │ │ • abs()      │
    │ • relu()     │ │ • multiply() │ │ • reshape()  │
    │ • matmul()   │ │ • conv2d()   │ │ • ...        │
    └──────────────┘ └──────────────┘ └──────────────┘
         (plugin)        (plugin)         (plugin)
```

### Implementation

#### 1. Core Builder (Simplified)

```python
# tools/builder/base/builder.py

from typing import Protocol, Dict, Callable
from ttmlir.ir import *

class DialectPlugin(Protocol):
    """Protocol for dialect plugins."""

    def register_ops(self, builder: 'Builder') -> Dict[str, Callable]:
        """Return dict of operation_name -> operation_function."""
        ...

    def create_tensor_encoding(
        self, shape, element_type
    ) -> Optional[Attribute]:
        """Optional: Create dialect-specific tensor encoding."""
        ...


class DialectProxy:
    """Proxy to make dialect ops accessible as builder.dialect.op()"""

    def __init__(self, builder: 'Builder', plugin: DialectPlugin):
        self._builder = builder
        self._plugin = plugin
        self._ops = plugin.register_ops(builder)

    def __getattr__(self, name: str):
        if name in self._ops:
            return self._ops[name]
        raise AttributeError(
            f"Dialect has no operation '{name}'"
        )


class Builder:
    """Unified builder with plugin-based dialect support."""

    def __init__(
        self,
        ctx: Context,
        location: Location,
        dialects: Optional[List[str]] = None,
        mesh_name: str = "mesh",
        mesh_dict: Dict[str, int] = {"x": 1, "y": 1},
    ):
        # Core state (same as current Builder)
        self._ctx = ctx
        self._loc = location
        self._goldens = {}
        self._func_ops_generated = {}
        # ... all other state ...

        # Plugin management
        self._dialect_plugins = {}
        self._dialect_proxies = {}

        # Auto-register requested dialects
        if dialects:
            for dialect_name in dialects:
                self.register_dialect(dialect_name)

    def register_dialect(self, name: str, plugin: Optional[DialectPlugin] = None):
        """Register a dialect plugin."""
        if plugin is None:
            # Auto-load from dialects module
            plugin = self._load_dialect_plugin(name)

        self._dialect_plugins[name] = plugin
        self._dialect_proxies[name] = DialectProxy(self, plugin)

    @property
    def ttir(self) -> DialectProxy:
        """Access TTIR dialect operations."""
        if "ttir" not in self._dialect_proxies:
            self.register_dialect("ttir")
        return self._dialect_proxies["ttir"]

    @property
    def ttnn(self) -> DialectProxy:
        """Access TTNN dialect operations."""
        if "ttnn" not in self._dialect_proxies:
            self.register_dialect("ttnn")
        return self._dialect_proxies["ttnn"]

    # ... rest of shared builder logic ...
```

#### 2. Dialect Plugin Example

```python
# tools/builder/dialects/ttir.py

from typing import Dict, Callable, Optional
from ttmlir.ir import *
from ttmlir.dialects import ttir
from builder.base.builder import DialectPlugin

class TTIRDialect(DialectPlugin):
    """TTIR dialect operations."""

    def register_ops(self, builder: 'Builder') -> Dict[str, Callable]:
        """Register all TTIR operations."""
        return {
            'sigmoid': lambda *args, **kwargs: self.sigmoid(builder, *args, **kwargs),
            'relu': lambda *args, **kwargs: self.relu(builder, *args, **kwargs),
            'matmul': lambda *args, **kwargs: self.matmul(builder, *args, **kwargs),
            # ... register all ops ...
        }

    def create_tensor_encoding(self, shape, element_type):
        """TTIR uses no special encoding."""
        return None

    def sigmoid(
        self,
        builder: 'Builder',
        input: Operand,
        loc: Optional[str] = None,
        **kwargs
    ) -> OpResult:
        """TTIR sigmoid operation."""
        with builder.context, builder.location:
            # Get or compute golden
            input_golden = builder._get_golden_tensor(input)
            output_golden = torch.sigmoid(input_golden)

            # Create output tensor type
            output_type = self._create_output_type(
                builder, input, output_golden.shape
            )

            # Create TTIR op
            if loc:
                loc = Location.name(loc, context=builder.context)
            else:
                loc = builder.location

            op = ttir.SigmoidOp(output_type, input, loc=loc)

            # Store golden
            builder._set_golden_tensor(op.result, output_golden)

            return op.result

    # ... implement other ops ...
```

#### 3. Usage

```python
from builder import Builder

# Create builder with desired dialects
builder = Builder(ctx, loc, dialects=["ttir", "ttnn"])

# Use operations
@builder.func([(32, 32)], [torch.float32])
def my_model(input, builder):
    # Explicit dialect access (clear!)
    x = builder.ttir.sigmoid(input)
    y = builder.ttir.relu(x)
    z = builder.ttnn.specialized_op(y)
    return z
```

### Migration Strategy

#### Phase 1: Create Plugin Interface (Week 1)
- [ ] Define `DialectPlugin` protocol
- [ ] Create `DialectProxy` class
- [ ] Update `Builder` to support plugins
- [ ] Maintain backward compatibility

#### Phase 2: Convert One Dialect (Week 2)
- [ ] Create `dialects/ttir.py` plugin
- [ ] Move operations from `TTIRBuilder` to plugin
- [ ] Test thoroughly
- [ ] Deprecate `TTIRBuilder` (still works, warns)

#### Phase 3: Convert Remaining Dialects (Week 3-4)
- [ ] Convert `TTNNBuilder` → `dialects/ttnn.py`
- [ ] Convert `StableHLOBuilder` → `dialects/stablehlo.py`
- [ ] Convert `D2MBuilder` → `dialects/d2m.py`

#### Phase 4: Cleanup (Week 5)
- [ ] Remove old builder classes
- [ ] Update all documentation
- [ ] Update all examples and tests
- [ ] Remove `MultiDialectBuilder` (no longer needed!)

### Benefits After Migration

#### 1. Reduced Complexity
```python
# Before: 5+ classes, complex inheritance
TTIRBuilder → Builder
TTNNBuilder → Builder
MultiDialectBuilder → Builder (manages TTIRBuilder + TTNNBuilder)

# After: 1 class, simple plugins
Builder + TTIRDialect + TTNNDialect
```

#### 2. Clear Responsibilities
```python
# Builder: State management only
# - Context, goldens, function tracking

# Dialect Plugins: Operations only
# - Implementation of each operation
# - No state management concerns
```

#### 3. Easy Extensions
```python
# Add new dialect:
# 1. Create dialects/new_dialect.py
# 2. Implement DialectPlugin
# 3. Done! No builder class needed.

class MyDialect(DialectPlugin):
    def register_ops(self, builder):
        return {'my_op': self.my_op}

    def my_op(self, builder, input):
        # Implementation
        ...
```

#### 4. Better Testing
```python
# Test dialects independently
def test_ttir_sigmoid():
    builder = Builder(ctx, loc)
    dialect = TTIRDialect()

    # Test just the op
    result = dialect.sigmoid(builder, input)
    assert_golden(result, expected)
```

#### 5. No MultiDialectBuilder Needed
```python
# Before: Need special MultiDialectBuilder
builder = MultiDialectBuilder(ctx, loc, dialects=["ttir", "ttnn"])

# After: Builder handles all dialects natively
builder = Builder(ctx, loc, dialects=["ttir", "ttnn"])
# Same API, simpler implementation!
```

### Comparison Table

| Aspect | Current | Plugin-Based |
|--------|---------|-------------|
| **Builder Classes** | 5+ | 1 |
| **Lines of Code** | ~41K | ~31K |
| **Add New Dialect** | New class + inheritance | New plugin file |
| **Multi-Dialect** | Special `MultiDialectBuilder` | Built-in |
| **Code Duplication** | High | Low |
| **Testing** | Test entire builders | Test plugins independently |
| **Maintainability** | Complex | Simple |
| **API** | `builder.ttir.op()` | `builder.ttir.op()` (same!) |

### Risks and Mitigations

#### Risk 1: Breaking Changes
**Mitigation:** Keep old builders as wrappers during transition
```python
class TTIRBuilder(Builder):
    def __init__(self, *args, **kwargs):
        warnings.warn("Deprecated: Use Builder with ttir dialect")
        super().__init__(*args, dialects=["ttir"], **kwargs)
        # Forward all attribute access to self.ttir
```

#### Risk 2: Performance
**Mitigation:** Plugin registration is one-time cost, operation calls are direct

#### Risk 3: Migration Effort
**Mitigation:** Gradual migration, one dialect at a time, with extensive testing

### Next Steps

1. **Review this proposal** with the team
2. **Prototype** the plugin interface with one dialect
3. **Measure** impact (performance, complexity, usability)
4. **Decide** whether to proceed with full migration
5. **Execute** migration plan if approved

### Questions for Discussion

1. Is the plugin abstraction the right level?
2. Should we keep old builders indefinitely for backward compat?
3. What's the acceptable timeline for this migration?
4. Are there other pain points we should address?
5. Should operations be functions or still bound methods?

### Conclusion

The plugin-based architecture offers:
- ✅ **Simpler**: 1 builder class instead of 5+
- ✅ **Cleaner**: Clear separation of state vs operations
- ✅ **Extensible**: Easy to add new dialects
- ✅ **Maintainable**: Less duplication, easier to change
- ✅ **Same API**: Users don't need to relearn

**Recommendation**: Proceed with phased migration to plugin architecture.
