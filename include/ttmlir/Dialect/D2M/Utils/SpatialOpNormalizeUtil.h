// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_SPATIALOPNORMALIZEUTIL_H
#define TTMLIR_DIALECT_D2M_UTILS_SPATIALOPNORMALIZEUTIL_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir::tt::d2m {

class GenericOp;

// Bring each d2m.spatial op to canonical form: hoist non-d2m.generic ops out of
// its regions, rebuild yields, refresh ins/outs from nested generics, and when
// result arity matches outs, update spatial result types from output values.
// Idempotent for already-normal IR. Violates invariants only via TT_assertv.
void normalizeSpatialOpsInModule(ModuleOp module);

// If genericOp lies inside a d2m.spatial, apply the same normalization to that
// enclosing spatial only; otherwise no-op. Use after rewrites to a generic so
// other spatials (not yet updated) are left untouched.
void normalizeSpatialOpContainingGeneric(GenericOp genericOp);

} // namespace mlir::tt::d2m

#endif
