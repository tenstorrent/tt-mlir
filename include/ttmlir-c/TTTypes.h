// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_C_TTKERNELTYPES_H
#define TTMLIR_C_TTKERNELTYPES_H

#include "ttmlir-c/Dialects.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED MlirType ttmlirTTTileTypeGet(MlirContext ctx,
                                                unsigned height, unsigned width,
                                                uint32_t dataType);

// Tuple type declaration
MLIR_CAPI_EXPORTED MlirType ttmlirTTTupleTypeGet(MlirContext ctx,
                                                 MlirType *elements,
                                                 size_t numElements);

#ifdef __cplusplus
}
#endif

#endif // TTMLIR_C_TTKERNELTYPES_H
