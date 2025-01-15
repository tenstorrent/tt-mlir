// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_REGISTERALL_H
#define TTMLIR_REGISTERALL_H

namespace mlir {

class DialectRegistry;

} // namespace mlir

namespace mlir::tt {

void registerAllDialects(mlir::DialectRegistry &registry);
void registerAllExtensions(mlir::DialectRegistry &registry);
void registerAllPasses();

struct MLIRModuleCacher;

} // namespace mlir::tt

#endif
