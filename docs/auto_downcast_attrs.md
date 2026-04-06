# Auto-Downcasting Custom Dialect Attributes in Python Bindings

## Problem

When accessing attributes on parsed MLIR operations via Python (e.g., `op.attributes[i].attr`),
built-in MLIR attributes like `IntegerAttr` automatically return their specialized Python class.
Custom tt-mlir dialect attributes (e.g., `ReduceTypeAttr`, `MetalLayoutAttr`) return a generic
`mlir.ir.Attribute` object instead.

This is because `tt_attribute_class<T>` (defined in `TTMLIRModule.h:33`) creates a standalone
nanobind class that is **not registered** with MLIR's global type caster map. Users must manually
call `T.maybe_downcast(attr)` to get the specialized type.

## Root Cause

MLIR's auto-downcast works via `PyGlobals::typeCasterMap` — a global registry mapping
`MlirTypeID -> callable`. When `Attribute.maybeDownCast()` is called (which `.attr` triggers),
it looks up the attribute's type ID in this map.

Upstream MLIR registers built-in attrs via `PyConcreteAttribute::bind()` (C++ CRTP) or
`mlir_attribute_subclass` (C++ helper for downstream dialects). Both call
`PyGlobals::registerTypeCaster()`. Our `tt_attribute_class` does neither.

## Proposed Solution

Replace `tt_attribute_class` with `mlir_attribute_subclass` from
`mlir/Bindings/Python/NanobindAdaptors.h` (already included in `TTMLIRModule.h`).

### What Changes

**`TTMLIRModule.h`** — Remove or deprecate `tt_attribute_class`. Optionally add a thin helper:

```cpp
// Helper that wraps mlir_attribute_subclass with common patterns.
// Returns the pure_subclass so .def() / .def_staticmethod() can be chained.
template <typename AttrT>
mlir_attribute_subclass tt_attr(nb::module_ &m, const char *name) {
  return mlir_attribute_subclass(
      m, name,
      [](MlirAttribute attr) { return mlir::isa<AttrT>(unwrap(attr)); },
      []() { return wrap(AttrT::getTypeID()); });
}
```

**Each `populate*Module` function** — Migrate from `tt_attribute_class` to the new helper.

Before:
```cpp
tt_attribute_class<tt::ttcore::ReduceTypeAttr>(m, "ReduceTypeAttr")
    .def_static("get",
                [](MlirContext ctx, tt::ttcore::ReduceType reduceType) {
                  return wrap(tt::ttcore::ReduceTypeAttr::get(unwrap(ctx),
                                                              reduceType));
                })
    .def_prop_ro("value", [](tt::ttcore::ReduceTypeAttr self) {
      return self.getValue();
    });
```

After:
```cpp
tt_attr<tt::ttcore::ReduceTypeAttr>(m, "ReduceTypeAttr")
    .def_staticmethod("get",
                [](MlirContext ctx, tt::ttcore::ReduceType reduceType) {
                  return wrap(tt::ttcore::ReduceTypeAttr::get(unwrap(ctx),
                                                              reduceType));
                })
    .def("value", [](MlirAttribute self) {
      return mlir::cast<tt::ttcore::ReduceTypeAttr>(unwrap(self)).getValue();
    });
```

### Key API Difference

`mlir_attribute_subclass` inherits from `pure_subclass`, which creates a **Python-level class**
(subclass of `mlir.ir.Attribute`), not a nanobind C++ class. This means:

| `tt_attribute_class` (current) | `mlir_attribute_subclass` (proposed) |
|-------------------------------|--------------------------------------|
| `nb::class_<T>` — C++ class binding | `pure_subclass` — Python class |
| `.def_prop_ro("x", [](T self) {...})` | `.def("x", [](MlirAttribute self) {...})` |
| Lambdas receive C++ type `T` directly | Lambdas receive `MlirAttribute`, must `unwrap`/`cast` |
| `.def_static(...)` | `.def_staticmethod(...)` |
| No auto-downcast | Auto-downcast via type caster registration |

### Scope

| File | `tt_attribute_class` count | `tt_type_class` count |
|------|---------------------------|-----------------------|
| `TTModule.cpp` | 26 | 1 |
| `TTNNModule.cpp` | 18 | 0 |
| `TTKernelModule.cpp` | 3 | 8 |
| `TTIRModule.cpp` | 1 | 0 |
| `D2MModule.cpp` | 1 | 2 |
| **Total** | **49** | **11** |

The same approach applies to `tt_type_class` using `mlir_type_subclass` from the same header.

### Migration Strategy

1. **Start with one attr** — Convert `ReduceTypeAttr` as a proof of concept, verify auto-downcast works.
2. **Create the `tt_attr` / `tt_type` helpers** in `TTMLIRModule.h` to minimize per-callsite boilerplate.
3. **Migrate remaining 48 attrs + 11 types** mechanically.
4. **Remove `tt_attribute_class` / `tt_type_class`** once all callsites are converted.
5. **Update downstream Python code** — Remove any manual `maybe_downcast()` calls that become unnecessary.

### Risks

- **API break**: `maybe_downcast()` static method goes away (replaced by automatic downcasting).
  Any Python code calling `T.maybe_downcast(attr)` needs updating.
- **Lambda signatures change**: All property/method lambdas switch from receiving `T` to
  `MlirAttribute`/`nb::object`, requiring `unwrap` + `cast` boilerplate inside each lambda.
- **nanobind type caster conflicts**: If nanobind already has a type caster for the C++ type `T`
  (from the old `nb::class_<T>`), registering an MLIR type caster for the same type ID could
  cause ambiguity. Removing the old `nb::class_<T>` binding eliminates this.
