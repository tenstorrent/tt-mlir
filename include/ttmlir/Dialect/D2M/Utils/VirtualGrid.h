// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_VIRTUALGRID_H
#define TTMLIR_DIALECT_D2M_UTILS_VIRTUALGRID_H

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/SmallVector.h"

#include <utility>

namespace ttmlir::d2m::utils::grids {

mlir::AffineMap prependResult(mlir::AffineMap map, mlir::AffineExpr result);

mlir::AffineMap extendWithIdentityDimsAndResults(mlir::AffineMap map,
                                                 unsigned extraDims);

mlir::AffineMap createCollapseMap(mlir::MLIRContext *context,
                                  llvm::ArrayRef<int64_t> virtualGrid);

mlir::AffineMap create1DtoNDMap(mlir::MLIRContext *context,
                                llvm::ArrayRef<int64_t> targetGrid);

/// Generates a pair of forward and inverse affine maps that allow
/// implementing a virtual grid as a physical-view pair of tensors/memrefs.
///
/// The view uses the grid x shard forward map to translate pure virtual
/// coordinates to physical coordinates compatible with the physical grid.
///
/// The physical memref/tensor uses the inverse map to perform core
/// virtualization, translating raw physical core locations at runtime into
/// virtual core locations that are compatible with virtual space. The inverse
/// map is restricted to only the grid dimensions; shard dims CANNOT
/// participate in virtual grid dim exprs (and vice-versa) or reblocking will
/// not work reliably.
std::pair<mlir::AffineMap, mlir::AffineMap>
createCoreVirtMaps(mlir::MLIRContext *context,
                   llvm::ArrayRef<int64_t> virtualGrid,
                   llvm::ArrayRef<int64_t> targetGrid);

} // namespace ttmlir::d2m::utils::grids

#endif // TTMLIR_DIALECT_D2M_UTILS_VIRTUALGRID_H
