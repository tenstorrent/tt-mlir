// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_TRANSFORMS_PASSES_H
#define TTMLIR_DIALECT_D2M_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tt::d2m {

// Wrapper factories (initially delegate to TTIR implementations).
struct TTIRGenericApplyInterchangeOptions;
std::unique_ptr<mlir::Pass>
createD2MGenericApplyInterchange(const TTIRGenericApplyInterchangeOptions &);

struct TTIRGenericTileComputeLoopsOptions;
std::unique_ptr<mlir::Pass>
createD2MGenericTileComputeLoops(const TTIRGenericTileComputeLoopsOptions &);

std::unique_ptr<mlir::Pass> createD2MInsertDstRegisterAccess();
std::unique_ptr<mlir::Pass> createD2MGenericLinearizeMemref();
std::unique_ptr<mlir::Pass> createD2MGenericGenerateDatamovement();
std::unique_ptr<mlir::Pass> createD2MGenericLowerDMAs();
std::unique_ptr<mlir::Pass> createD2MGenericHWThreadSelection();
std::unique_ptr<mlir::Pass> createD2MGenericGenerateLoops();
std::unique_ptr<mlir::Pass> createD2MGenericRegionsToFuncs();

} // namespace mlir::tt::d2m

#endif
