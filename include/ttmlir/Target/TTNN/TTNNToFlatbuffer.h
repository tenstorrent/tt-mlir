// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTNN_TTNNTOFLATBUFFER_H
#define TTMLIR_TARGET_TTNN_TTNNTOFLATBUFFER_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "ttmlir/Target/Utils/MLIRToFlatbuffer.h"

namespace mlir::tt::ttnn {

// Convert a TTNNIR operation to a flatbuffer
std::shared_ptr<void> ttnnToFlatbuffer(
    Operation *op,
    /* goldenMap has following structure
    {
      loc: {
        device_id: GoldenTensor
      }
    }
    */
    const std::unordered_map<std::string,
                             std::unordered_map<std::uint32_t, GoldenTensor>>
        &goldenMap = {},
    const std::vector<std::pair<std::string, std::string>> &moduleCache = {},
    const std::string &kernelDumpDir = "");

// Convert a TTNNIR operation to a flatbuffer
// This function signature is required in order to register the conversion in
// mlir translation framework
LogicalResult translateTTNNToFlatbuffer(
    Operation *op, llvm::raw_ostream &os,
    const std::unordered_map<std::string,
                             std::unordered_map<std::uint32_t, GoldenTensor>>
        /* goldenMap has following structure
        {
          loc: {
            device_id: GoldenTensor
          }
        }
        */
        &goldenMap = {},
    const std::vector<std::pair<std::string, std::string>> &moduleCache = {},
    const std::string &kernelDumpDir = "");
} // namespace mlir::tt::ttnn

#endif
