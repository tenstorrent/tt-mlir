// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_LINALG_TRANSFORMS_SYNCHRONIZABLEOPINTERFACEIMPL_H
#define TTMLIR_DIALECT_LINALG_TRANSFORMS_SYNCHRONIZABLEOPINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;

namespace linalg {
void registerSynchronizableOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace linalg
} // namespace mlir

#endif // TTMLIR_DIALECT_LINALG_TRANSFORMS_SYNCHRONIZABLEOPINTERFACEIMPL_H
