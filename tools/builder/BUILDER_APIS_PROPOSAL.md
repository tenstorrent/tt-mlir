# Builder APIs Proposal: Plugin Architecture

## Overview

This document proposes changes to `builder_apis.py` under Option 3 (Plugin Architecture), where we eliminate separate builder classes in favor of a single `Builder` class with dialect plugins.

---

## Current API vs. Plugin-Based API

### Current API Structure

```python
# Multiple builder types
build_module(fn, builder_type="ttir"|"ttnn"|"stablehlo"|"d2m"|"multi")

# Separate compile functions per dialect
compile_ttir_to_flatbuffer(fn, ...)
compile_ttnn_to_flatbuffer(fn, ...)
compile_stablehlo_to_flatbuffer(fn, ...)
compile_d2m_to_flatbuffer(fn, ...)
compile_multi_to_flatbuffer(fn, ...)

# Separate execute functions per dialect
compile_and_execute_ttir(fn, ...)
compile_and_execute_ttnn(fn, ...)
compile_and_execute_shlo(fn, ...)
compile_and_execute_d2m(fn, ...)
compile_and_execute_multi(fn, ...)
```

**Problems:**
- 5+ compile functions (code duplication)
- 5+ execute functions (more duplication)
- Need new functions for each new dialect
- Special "multi" handling everywhere

### Proposed Plugin-Based API

```python
# Single unified builder (no "type" needed for multi-dialect)
build_module(
    fn,
    dialects=["ttir", "ttnn"],  # Just specify which dialects
    ...
)

# Single compile function for all dialects
compile_to_flatbuffer(
    fn,
    dialects=["ttir", "ttnn"],  # Works for any dialect combination
    ...
)

# Single execute function for all dialects
compile_and_execute(
    fn,
    dialects=["ttir", "ttnn"],  # Works for any dialect combination
    ...
)
```

**Benefits:**
- 1 compile function instead of 5+
- 1 execute function instead of 5+
- No special "multi" handling
- Adding new dialect = zero new API functions

---

## Detailed API Changes

### 1. `build_module()` - Simplified

#### Before (Current)
```python
def build_module(
    mod: Callable,
    builder_type: Literal["ttir", "stablehlo", "ttnn", "d2m", "multi"],
    dialects: Optional[List[str]] = None,  # Only used if builder_type="multi"
    mesh_name: str = "mesh",
    mesh_dict: OrderedDict[str, int] = OrderedDict([("x", 1), ("y", 1)]),
    save_artifacts: bool = False,
    artifact_dir: str = ".",
) -> Tuple[Module, Union[TTIRBuilder, StableHLOBuilder, TTNNBuilder, D2MBuilder, MultiDialectBuilder]]:
    """Build with specific builder type."""

    if builder_type == "ttir":
        builder = TTIRBuilder(ctx, loc, mesh_name, mesh_dict)
    elif builder_type == "stablehlo":
        builder = StableHLOBuilder(ctx, loc, mesh_name, mesh_dict)
    elif builder_type == "ttnn":
        builder = TTNNBuilder(ctx, loc, mesh_name, mesh_dict)
    elif builder_type == "d2m":
        builder = D2MBuilder(ctx, loc, mesh_name, mesh_dict)
    elif builder_type == "multi":
        from builder.base.multi_dialect_builder import MultiDialectBuilder
        if dialects is None:
            dialects = ["ttir", "ttnn"]
        builder = MultiDialectBuilder(ctx, loc, dialects, mesh_name, mesh_dict)
    # ...
```

#### After (Plugin-Based)
```python
def build_module(
    mod: Callable,
    dialects: Union[str, List[str]] = "ttir",  # Default to TTIR for backward compat
    mesh_name: str = "mesh",
    mesh_dict: OrderedDict[str, int] = OrderedDict([("x", 1), ("y", 1)]),
    save_artifacts: bool = False,
    artifact_dir: str = ".",
) -> Tuple[Module, Builder]:
    """
    Build an MLIR Module from a Python function.

    Parameters
    ----------
    mod : Callable
        Function that receives a builder and emits ops
    dialects : Union[str, List[str]]
        Dialect(s) to enable. Examples:
        - "ttir"                    # Single dialect
        - ["ttir", "ttnn"]          # Multiple dialects
        - "ttir,ttnn"               # Comma-separated string
    mesh_name : str
        Mesh name for distributed ops
    mesh_dict : OrderedDict[str, int]
        Mesh shape specification
    save_artifacts : bool
        Whether to save artifacts
    artifact_dir : str
        Directory for artifacts

    Returns
    -------
    Tuple[Module, Builder]
        The MLIR module and builder instance

    Examples
    --------
    Single dialect (backward compatible):
        >>> module, builder = build_module(my_fn, dialects="ttir")

    Multiple dialects:
        >>> module, builder = build_module(my_fn, dialects=["ttir", "ttnn"])
        >>> # Use explicit API
        >>> x = builder.ttir.sigmoid(input)
        >>> y = builder.ttnn.add(x, input)
    """
    ctx = Context()

    try:
        fname = inspect.getfile(mod)
        line_no = inspect.getsourcelines(mod)[1]
        loc = Location.file(fname, line_no, 0, ctx)
    except (OSError, TypeError):
        loc = Location.unknown(ctx)

    # Normalize dialects to list
    if isinstance(dialects, str):
        dialects = [d.strip() for d in dialects.split(",")]

    # Create single unified builder
    builder = Builder(
        ctx,
        loc,
        dialects=dialects,
        mesh_name=mesh_name,
        mesh_dict=mesh_dict
    )

    with ctx, loc:
        new_module = _compile(mod, builder)

        print(f"`{mod.__name__}` successfully transformed into a MLIR module.")
        print(new_module)

        if save_artifacts:
            os.makedirs(artifact_dir, exist_ok=True)
            dialects_str = "_".join(dialects)
            filename = os.path.join(artifact_dir, f"{dialects_str}_module.mlir")
            with open(filename, "w") as f:
                f.write(str(module))

    return new_module, builder
```

**Key Changes:**
- ❌ Removed `builder_type` parameter
- ✅ `dialects` is now primary parameter
- ✅ Returns single `Builder` type (not Union of 5 types)
- ✅ Simplified implementation (no if/elif chain)

---

### 2. `compile_to_flatbuffer()` - Unified

#### Before (Current)
```python
# Separate function for each dialect
compile_ttir_to_flatbuffer(fn, ...)       # ~100 lines
compile_ttnn_to_flatbuffer(fn, ...)       # ~100 lines
compile_stablehlo_to_flatbuffer(fn, ...)  # ~150 lines
compile_d2m_to_flatbuffer(fn, ...)        # ~100 lines
compile_multi_to_flatbuffer(fn, ...)      # ~100 lines
# = ~550 lines of mostly duplicated code!
```

#### After (Plugin-Based)
```python
def compile_to_flatbuffer(
    fn: Callable,
    dialects: Union[str, List[str]] = "ttir",
    system_desc_path: str = "ttrt-artifacts/system_desc.ttsys",
    artifact_dir: str = ".",
    target: Literal["ttnn", "ttmetal", "emitc", "emitpy"] = "ttnn",
    mesh_name: str = "mesh",
    mesh_dict: OrderedDict[str, int] = OrderedDict([("x", 1), ("y", 1)]),
    save_artifacts: bool = False,
    argument_types_string: Optional[str] = None,
    custom_pipeline: Optional[Union[Callable, str]] = None,
    pipeline_options: Optional[List[str]] = None,
    print_ir: Union[bool, str] = False,
) -> Tuple[Builder, Any, Dict, Dict]:
    """
    Compile a builder function to flatbuffer format.

    Works with any dialect combination. The pipeline is automatically
    selected based on the dialects used in the module.

    Parameters
    ----------
    fn : Callable
        Builder function to compile
    dialects : Union[str, List[str]]
        Dialect(s) to enable. Examples:
        - "ttir"                    # TTIR only
        - ["ttir", "ttnn"]          # Mixed TTIR/TTNN
        - "stablehlo"               # StableHLO
    target : Literal["ttnn", "ttmetal", "emitc", "emitpy"]
        Target backend
    ... (other params same as before)

    Returns
    -------
    Tuple[Builder, compiled_bin, input_output_goldens, intermediate_goldens]

    Examples
    --------
    Single dialect:
        >>> builder, bin, *goldens = compile_to_flatbuffer(
        ...     my_ttir_fn,
        ...     dialects="ttir"
        ... )

    Multiple dialects:
        >>> builder, bin, *goldens = compile_to_flatbuffer(
        ...     my_mixed_fn,
        ...     dialects=["ttir", "ttnn"]
        ... )
    """

    # Build module
    try:
        module, builder = build_module(
            fn,
            dialects=dialects,
            mesh_name=mesh_name,
            mesh_dict=mesh_dict,
            save_artifacts=save_artifacts,
            artifact_dir=artifact_dir,
        )
    except Exception as e:
        raise TTBuilderCompileException(e)

    # Compile to flatbuffer
    # Pipeline selection is automatic based on module content
    return builder, *compile_module_to_flatbuffer(
        module,
        builder,
        system_desc_path=system_desc_path,
        artifact_dir=artifact_dir,
        target=target,
        mesh_dict=mesh_dict,
        save_artifacts=save_artifacts,
        argument_types_string=argument_types_string,
        custom_pipeline=custom_pipeline,
        pipeline_options=pipeline_options,
        print_ir=print_ir,
    )
```

**Key Changes:**
- ✅ One function replaces 5+ dialect-specific functions
- ✅ `dialects` parameter works same as `build_module()`
- ✅ ~550 lines → ~50 lines (90% reduction!)
- ✅ No special handling for "multi"

---

### 3. `compile_and_execute()` - Unified

#### Before (Current)
```python
# 5 separate functions
compile_and_execute_ttir(fn, ...)     # ~80 lines
compile_and_execute_ttnn(fn, ...)     # ~80 lines
compile_and_execute_shlo(fn, ...)     # ~80 lines
compile_and_execute_d2m(fn, ...)      # ~80 lines
compile_and_execute_multi(fn, ...)    # ~80 lines
# = ~400 lines of duplicated code!
```

#### After (Plugin-Based)
```python
def compile_and_execute(
    fn: Callable,
    dialects: Union[str, List[str]] = "ttir",
    system_desc_path: str = "ttrt-artifacts/system_desc.ttsys",
    test_base: str = "test",
    output_root: str = ".",
    target: Literal["ttnn", "ttmetal", "emitc", "emitpy"] = "ttnn",
    mesh_name: str = "mesh",
    mesh_dict: OrderedDict[str, int] = OrderedDict([("x", 1), ("y", 1)]),
    save_artifacts: bool = False,
    argument_types_string: Optional[str] = None,
    custom_pipeline: Optional[Union[Callable, str]] = None,
    pipeline_options: Optional[List[str]] = None,
    print_ir: Union[bool, str] = False,
    device=None,
    pcc: float = 0.99,
    atol: float = 1e-08,
    rtol: float = 1e-05,
    disable_golden: bool = False,
    skip_exec: bool = False,
    check_pcc: bool = True,
    check_atol: bool = False,
    check_rtol: bool = False,
    enable_intermediate_verification: bool = False,
    dump_memory: bool = False,
) -> str:
    """
    Compile and execute a builder function through complete pipeline.

    Works with any dialect combination.

    Parameters
    ----------
    fn : Callable
        Builder function to compile and execute
    dialects : Union[str, List[str]]
        Dialect(s) to enable
    ... (other params same as before)

    Returns
    -------
    str
        Path to compiled MLIR file

    Examples
    --------
    Single dialect:
        >>> compile_and_execute(my_ttir_fn, dialects="ttir")

    Multiple dialects:
        >>> compile_and_execute(
        ...     my_mixed_fn,
        ...     dialects=["ttir", "ttnn"],
        ...     target="ttnn"
        ... )
    """
    # Normalize dialects
    if isinstance(dialects, str):
        dialects = [d.strip() for d in dialects.split(",")]

    dialects_str = "_".join(dialects)
    artifact_dir = get_artifact_dir(
        output_root, f"Builder[{dialects_str}]", test_base, save_artifacts
    )

    _compile_and_execute(
        compile_fn=compile_to_flatbuffer,
        fn=fn,
        dialects=dialects,
        system_desc_path=system_desc_path,
        artifact_dir=artifact_dir,
        target=target,
        mesh_name=mesh_name,
        mesh_dict=mesh_dict,
        save_artifacts=save_artifacts,
        argument_types_string=argument_types_string,
        custom_pipeline=custom_pipeline,
        pipeline_options=pipeline_options,
        print_ir=print_ir,
        device=device,
        pcc=pcc,
        atol=atol,
        rtol=rtol,
        disable_golden=disable_golden,
        skip_exec=skip_exec,
        check_pcc=check_pcc,
        check_atol=check_atol,
        check_rtol=check_rtol,
        enable_intermediate_verification=enable_intermediate_verification,
        dump_memory=dump_memory,
    )

    return os.path.join(artifact_dir, f"{target}_compiled.mlir")
```

**Key Changes:**
- ✅ One function replaces 5+ dialect-specific functions
- ✅ ~400 lines → ~60 lines (85% reduction!)
- ✅ Same functionality, cleaner implementation

---

## Migration Strategy

### Phase 1: Add New APIs (Parallel to Old)

```python
# builder_apis.py

# NEW: Unified APIs
def build_module(mod, dialects=...):  # New signature
    ...

def compile_to_flatbuffer(fn, dialects=...):  # New function
    ...

def compile_and_execute(fn, dialects=...):  # New function
    ...

# OLD: Deprecated but still working
@deprecated("Use build_module(fn, dialects='ttir') instead")
def build_module_ttir(fn, ...):  # Old function
    return build_module(fn, dialects="ttir", ...)

@deprecated("Use compile_to_flatbuffer(fn, dialects='ttir') instead")
def compile_ttir_to_flatbuffer(fn, ...):
    return compile_to_flatbuffer(fn, dialects="ttir", ...)

# ... similar for ttnn, stablehlo, d2m, multi
```

### Phase 2: Update Tests Incrementally

```python
# Before
module, builder = build_module(my_fn, "ttir")

# After
module, builder = build_module(my_fn, dialects="ttir")

# Or multi-dialect
module, builder = build_module(my_fn, dialects=["ttir", "ttnn"])
```

### Phase 3: Remove Old APIs

After migration complete, remove deprecated functions.

---

## Complete Example: Before vs After

### Before (Current - Requires Multiple Functions)

```python
# File: test_ttir_ops.py
from builder import build_module, compile_ttir_to_flatbuffer, compile_and_execute_ttir

def test_single_dialect():
    def my_fn(builder):
        @builder.func([(32, 32)], [torch.float32])
        def forward(x, builder):
            return builder.sigmoid(x)

    # Need specific function for TTIR
    compile_and_execute_ttir(my_fn, target="ttnn")

def test_multi_dialect():
    def my_fn(builder):
        @builder.func([(32, 32)], [torch.float32])
        def forward(x, builder):
            x = builder.ttir.sigmoid(x)
            return builder.ttnn.add(x, x)

    # Need different function for multi
    compile_and_execute_multi(my_fn, target="ttnn")
```

### After (Plugin-Based - One Function)

```python
# File: test_ttir_ops.py
from builder import build_module, compile_to_flatbuffer, compile_and_execute

def test_single_dialect():
    def my_fn(builder):
        @builder.func([(32, 32)], [torch.float32])
        def forward(x, builder):
            return builder.ttir.sigmoid(x)  # or builder.sigmoid(x)

    # Same function works for single dialect
    compile_and_execute(my_fn, dialects="ttir", target="ttnn")

def test_multi_dialect():
    def my_fn(builder):
        @builder.func([(32, 32)], [torch.float32])
        def forward(x, builder):
            x = builder.ttir.sigmoid(x)
            return builder.ttnn.add(x, x)

    # Same function works for multi dialect!
    compile_and_execute(my_fn, dialects=["ttir", "ttnn"], target="ttnn")

def test_different_combination():
    def my_fn(builder):
        @builder.func([(32, 32)], [torch.float32])
        def forward(x, builder):
            x = builder.stablehlo.abs(x)
            return builder.ttir.sigmoid(x)

    # No new function needed for new combination!
    compile_and_execute(my_fn, dialects=["stablehlo", "ttir"], target="ttnn")
```

---

## Backward Compatibility

### Option A: Wrapper Functions (Recommended)

```python
# builder_apis.py

# New unified API
def build_module(mod, dialects="ttir", **kwargs):
    """New unified API."""
    ...

# Backward compatibility wrappers
def build_module_ttir(mod, **kwargs):
    """Deprecated: Use build_module(mod, dialects='ttir')"""
    warnings.warn(
        "build_module_ttir is deprecated. Use build_module(mod, dialects='ttir')",
        DeprecationWarning,
        stacklevel=2
    )
    return build_module(mod, dialects="ttir", **kwargs)

def build_module_ttnn(mod, **kwargs):
    """Deprecated: Use build_module(mod, dialects='ttnn')"""
    warnings.warn(
        "build_module_ttnn is deprecated. Use build_module(mod, dialects='ttnn')",
        DeprecationWarning,
        stacklevel=2
    )
    return build_module(mod, dialects="ttnn", **kwargs)

# Similar for stablehlo, d2m, etc.
```

### Option B: Preserve Old Signatures

```python
def build_module(
    mod,
    # Support both old and new signatures
    builder_type: Optional[Literal["ttir", "ttnn", "stablehlo", "d2m", "multi"]] = None,
    dialects: Union[str, List[str], None] = None,
    **kwargs
):
    """Support both old and new APIs."""

    # New API: dialects parameter
    if dialects is not None:
        if builder_type is not None:
            raise ValueError("Cannot specify both builder_type and dialects")
        # Use new implementation
        ...

    # Old API: builder_type parameter
    elif builder_type is not None:
        warnings.warn(
            f"builder_type='{builder_type}' is deprecated. "
            f"Use dialects='{builder_type}' instead",
            DeprecationWarning
        )
        if builder_type == "multi":
            raise ValueError("For multi-dialect, use dialects=['ttir', 'ttnn']")
        dialects = builder_type
        # Use new implementation
        ...

    # Default
    else:
        dialects = "ttir"

    # Rest of implementation...
```

---

## Benefits Summary

### Code Reduction

| Current | Plugin-Based | Savings |
|---------|-------------|---------|
| 5 `build_module_*` functions | 1 `build_module` | 80% |
| 5 `compile_*_to_flatbuffer` | 1 `compile_to_flatbuffer` | 90% |
| 5 `compile_and_execute_*` | 1 `compile_and_execute` | 85% |
| **~1000 lines** | **~150 lines** | **~85%** |

### Extensibility

```python
# Current: Add new dialect requires new functions
compile_newdialect_to_flatbuffer()      # New function
compile_and_execute_newdialect()        # New function
# = 200+ lines of code

# Plugin-based: Just register the plugin
class NewDialect(DialectPlugin):
    ...
# = 0 new API functions needed!
```

### API Consistency

```python
# Current: Different function for each case
compile_ttir_to_flatbuffer(fn1)
compile_ttnn_to_flatbuffer(fn2)
compile_multi_to_flatbuffer(fn3)  # Special case

# Plugin-based: Same function for all cases
compile_to_flatbuffer(fn1, dialects="ttir")
compile_to_flatbuffer(fn2, dialects="ttnn")
compile_to_flatbuffer(fn3, dialects=["ttir", "ttnn"])  # Not special!
```

---

## Implementation Checklist

### Phase 1: Core Changes (Week 1)
- [ ] Refactor `build_module()` to use `dialects` parameter
- [ ] Create unified `compile_to_flatbuffer()`
- [ ] Create unified `compile_and_execute()`
- [ ] Add deprecation warnings to old functions

### Phase 2: Helper Updates (Week 2)
- [ ] Update `_compile()` to handle plugin builders
- [ ] Update `compile_module_to_flatbuffer()` for plugin builders
- [ ] Ensure pipeline selection works with any dialect combo

### Phase 3: Testing (Week 3)
- [ ] Test single dialect: "ttir", "ttnn", "stablehlo", "d2m"
- [ ] Test multi-dialect: ["ttir", "ttnn"], ["stablehlo", "ttir"], etc.
- [ ] Test backward compatibility with old APIs
- [ ] Performance testing

### Phase 4: Migration (Week 4-6)
- [ ] Update all tests to use new APIs
- [ ] Update documentation
- [ ] Update examples

### Phase 5: Cleanup (Week 7)
- [ ] Remove deprecated functions
- [ ] Final documentation pass
- [ ] Announce changes

---

## Conclusion

The plugin-based API offers:

1. **Massive Simplification**: ~85% less code in builder_apis.py
2. **Better UX**: One API for all cases (no more "which function do I use?")
3. **Extensibility**: New dialects require zero new API functions
4. **Maintainability**: One implementation to maintain instead of 15+
5. **No Loss**: Backward compatibility maintained during transition

**Recommendation**: Adopt plugin architecture with phased migration.
