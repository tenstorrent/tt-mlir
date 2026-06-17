# Pattern Testing Architecture

## High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pattern Definition File                       │
│         tools/d2m-jit/patterns/my_pattern.py                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  @d2m.kernel                                                     │
│  def my_kernel(...):                                             │
│      # Kernel implementation                                     │
│                                                                  │
│  @d2m.pattern(...)                                               │
│  def my_pattern(...):                                            │
│      # Pattern rewrite logic                                     │
│                                                                  │
│  PATTERN_TEST_METADATA = {                                       │
│      "pattern_name": "...",                                      │
│      "lit_tests": [...],      ← Test definitions                │
│      "e2e_tests": [...],      ← live here!                      │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Discovered by
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Test Framework                              │
│         test/d2m-jit/pattern_tests/                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  discovery.py                                                    │
│  ├─ discover_pattern_modules()                                  │
│  ├─ load_pattern_metadata()                                     │
│  └─ discover_all_pattern_tests()                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
          │                                │
          │                                │
          ▼                                ▼
┌──────────────────────┐       ┌──────────────────────┐
│  LIT Test Runner     │       │  E2E Test Runner     │
├──────────────────────┤       ├──────────────────────┤
│                      │       │                      │
│  test_lit_generated  │       │  test_e2e_generated  │
│         .py          │       │         .py          │
│                      │       │                      │
│  For each lit_test:  │       │  For each e2e_test:  │
│  1. Parse module     │       │  1. Generate inputs  │
│  2. Apply patterns   │       │  2. Run kernel       │
│  3. Verify rewrite   │       │  3. Check PCC        │
│                      │       │                      │
└──────────────────────┘       └──────────────────────┘
          │                                │
          │                                │
          ▼                                ▼
    ✓ Pattern                        ✓ Kernel
      rewrite                          correctness
      correct                          verified
```

## Data Flow

### LIT Test Flow

```
Pattern File
    │
    ├─ lit_tests[i].module_text
    │       │
    │       ▼
    │   MLIR Parser ──────► IR Module
    │                          │
    │                          ▼
    │                   d2m.apply_patterns()
    │                          │
    │                          ▼
    │                   Rewritten IR
    │                          │
    ├─ lit_tests[i].file_checks │
    │       │                   │
    │       └──────► FileCheck ─┘
    │                    │
    │                    ▼
    │               ✓ or ✗ (assertion)
```

### E2E Test Flow

```
Pattern File
    │
    ├─ e2e_tests[i].input_generator()
    │       │
    │       ▼
    │   Input Tensors (torch)
    │       │
    │       ├──────────────────────────┐
    │       │                          │
    │       ▼                          ▼
    │   reference_fn()            kernel_fn()
    │       │                          │
    │       ▼                          ▼
    │   Expected Output           d2m.to_layout()
    │       │                          │
    │       │                          ▼
    │       │                   Device Buffers
    │       │                          │
    │       │                          ▼
    │       │                   Execute on Device
    │       │                          │
    │       │                          ▼
    │       │                   to_host()
    │       │                          │
    │       │                          ▼
    │       │                   Actual Output
    │       │                          │
    │       └──────────► assert_pcc() ─┘
    │                          │
    │                          ▼
    │                     ✓ or ✗ (assertion)
```

## Test Discovery

```
Pattern Discovery Process:

1. discover_pattern_modules()
   └─> Glob: tools/d2m-jit/patterns/*.py
       ├─> eltwise_exp_to_kernel.py
       ├─> eltwise_add_exp_to_kernel.py
       └─> my_new_pattern.py

2. For each pattern file:
   load_pattern_metadata(file)
   └─> Import module dynamically
       └─> Extract PATTERN_TEST_METADATA
           ├─> pattern_name
           ├─> description
           ├─> lit_tests []
           └─> e2e_tests []

3. discover_all_pattern_tests()
   └─> Returns: [metadata1, metadata2, ...]

4. Pytest parametrization
   └─> Generates one test per lit_test/e2e_test entry
       ├─> test_pattern_lit_style[eltwise_exp::exp_pattern_positive]
       ├─> test_pattern_lit_style[eltwise_add_exp::add_exp_pattern_positive]
       ├─> test_pattern_kernel_e2e[eltwise_exp::test_pattern_exp_kernel_on_device]
       └─> test_pattern_kernel_e2e[eltwise_add_exp::test_pattern_add_exp_kernel_on_device]
```

## Module Relationships

```
tools/d2m-jit/patterns/
│
├─ __init__.py
│   └─ Empty (patterns imported explicitly)
│
├─ eltwise_exp_to_kernel.py
│   ├─ exp_fused (kernel)
│   ├─ lower_exp (pattern)
│   └─ PATTERN_TEST_METADATA
│       ├─ lit_tests[0]: exp_pattern_positive
│       └─ e2e_tests[0]: test_pattern_exp_kernel_on_device
│
└─ eltwise_add_exp_to_kernel.py
    ├─ add_exp_fused (kernel)
    ├─ fuse_add_exp (pattern)
    └─ PATTERN_TEST_METADATA
        ├─ lit_tests[0]: add_exp_pattern_positive
        ├─ lit_tests[1]: add_exp_pattern_negative
        └─ e2e_tests[0]: test_pattern_add_exp_kernel_on_device


test/d2m-jit/pattern_tests/
│
├─ __init__.py
│   └─ Package marker
│
├─ discovery.py
│   ├─ discover_pattern_modules()
│   ├─ load_pattern_metadata()
│   └─ discover_all_pattern_tests()
│       └─ Used by test files
│
├─ test_lit_generated.py
│   ├─ Imports: discovery, d2m_jit, ttmlir
│   ├─ _generate_lit_test_params()
│   │   └─> [(metadata, lit_test), ...]
│   └─ test_pattern_lit_style(metadata, lit_test)
│       └─> Parametrized test function
│
├─ test_e2e_generated.py
│   ├─ Imports: discovery, d2m_jit, utils
│   ├─ _generate_test_params()
│   │   └─> [(metadata, e2e_test), ...]
│   └─ test_pattern_kernel_e2e(metadata, e2e_test)
│       └─> Parametrized test function
│
├─ lit_generator.py
│   ├─ generate_lit_test_file(metadata, output_dir)
│   └─ generate_all_lit_tests()
│       └─> Creates: test/d2m-jit/lit_generated/*.py
│
├─ conftest.py
│   ├─ validate_pattern_discovery()
│   └─> Pytest fixtures and configuration
│
└─ Documentation
    ├─ README.md                    (Full documentation)
    ├─ REFACTORING_SUMMARY.md      (Before/after comparison)
    ├─ QUICK_REFERENCE.md          (Cheat sheet)
    ├─ PATTERN_TEMPLATE.py         (Annotated example)
    └─ validate_refactoring.py     (Validation script)
```

## Execution Modes

### Mode 1: Direct Pytest (Recommended for development)

```bash
pytest test/d2m-jit/pattern_tests/test_lit_generated.py
pytest test/d2m-jit/pattern_tests/test_e2e_generated.py
```

- ✓ Fast iteration
- ✓ No file generation needed
- ✓ Detailed pytest output
- ✓ Easy debugging

### Mode 2: Generated LIT Files (For CI integration)

```bash
python -m test.d2m_jit.pattern_tests.lit_generator
# Generates: test/d2m-jit/lit_generated/*.py

lit test/d2m-jit/lit_generated/
```

- ✓ Standard LIT workflow
- ✓ FileCheck integration
- ✓ Can be checked into version control

## Benefits of Architecture

1. **Single Source of Truth**
   - Pattern code and tests in one file
   - Reduces inconsistency

2. **Automatic Discovery**
   - New patterns automatically tested
   - No manual test registration

3. **Flexible Execution**
   - Run as pytest (fast)
   - Generate LIT files (CI)

4. **Easy to Extend**
   - Add new test types by extending metadata schema
   - Add new test runners without changing patterns

5. **DRY Principle**
   - Test logic shared across all patterns
   - No per-pattern test boilerplate
