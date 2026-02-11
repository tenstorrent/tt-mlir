// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_GENERIC_AFFINE_UTILS_H
#define TTMLIR_DIALECT_D2M_UTILS_GENERIC_AFFINE_UTILS_H

#include "mlir/IR/Builders.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"

namespace mlir::tt::d2m::utils {

/// Convert fused generic to affine-compatible form for dependence analysis.
/// This replaces D2M-specific ops (get_block_factor, block_index) with MLIR
/// constructs that affine analysis can reason about:
/// - get_block_factor → tagged constants with sentinel primes
/// - block_index → affine.apply identity maps with tagged attributes
void convertToAffineCompatibilityForm(GenericOp fusedOp, OpBuilder &builder);

/// Convert fused generic from affine-compatible form back to D2M form.
/// This restores D2M-specific ops from their temporary representations:
/// - tagged constants → get_block_factor ops
/// - tagged affine.apply identity maps → block_index ops
void convertFromAffineCompatibilityForm(GenericOp compatGeneric,
                                        OpBuilder &builder);

} // namespace mlir::tt::d2m::utils

#endif
