// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTNN_TRANSFORMS_TTNNPASSES_TD
#define TTMLIR_TTMLIR_DIALECT_TTNN_TRANSFORMS_TTNNPASSES_TD

include "mlir/Pass/PassBase.td"

def TTNNForceDRAMInterleaved: Pass<"ttir-force-dram-interleaved", "::mlir::ModuleOp"> {
    let summary = "Place to_memory_config ops on all op outputs that do not have dram interleaved outputs.";
}

#endif
