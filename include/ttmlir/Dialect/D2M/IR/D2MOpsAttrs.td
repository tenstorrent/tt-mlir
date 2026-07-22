// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_D2M_DIALECT_D2M_IR_D2MOPSATTRS_TD
#define TTMLIR_D2M_DIALECT_D2M_IR_D2MOPSATTRS_TD

include "mlir/IR/AttrTypeBase.td"
include "ttmlir/Dialect/D2M/IR/D2MOpsEnums.td"
include "ttmlir/Dialect/D2M/IR/D2MBase.td"

// Enum-backed attribute wrappers
def D2M_ReduceDimAttr : EnumAttr<D2M_Dialect, D2M_ReduceDim, "reduce_dim">;

def D2M_TileBcastTypeAttr : EnumAttr<D2M_Dialect, D2M_TileBcastType, "tile_bcast_type">;

def D2M_ResolutionStageAttr : EnumAttr<D2M_Dialect, D2M_ResolutionStage, "resolution_stage">;

// NOTE: ThreadType / ThreadAttr / ThreadArrayAttr live in the TTCore dialect
// (see TTCore_ThreadAttr, mnemonic `#ttcore.thread`). They were hoisted there
// so lower layers (e.g. fabric connection config serialization) can reason
// about the assigned DM core without depending on D2M. For source
// compatibility, d2m::ThreadType/ThreadAttr are aliased in D2MOps.h.

#endif // TTMLIR_D2M_DIALECT_D2M_IR_D2MOPSATTRS_TD
