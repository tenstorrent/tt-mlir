// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_PASSES_H
#define TTMLIR_CONVERSION_PASSES_H

#ifdef TTMLIR_ENABLE_STABLEHLO
#include "ttmlir/Conversion/ArithToStableHLO/ArithToStableHLO.h"
#include "ttmlir/Conversion/StableHLOToTTIR/ShardyToTTIR.h"
#include "ttmlir/Conversion/StableHLOToTTIR/StableHLOLegalizeComposite.h"
#include "ttmlir/Conversion/StableHLOToTTIR/StableHLOToTTIR.h"
#endif
#include "ttmlir/Conversion/D2MToTTNN/D2MToTTNN.h"
#include "ttmlir/Conversion/SFPIToEmitC/SFPIToEmitC.h"
#include "ttmlir/Conversion/TTIRToLinalg/TTIRToLinalg.h"
#include "ttmlir/Conversion/TTIRToTTIRDecomposition/TTIRToTTIRDecomposition.h"
#include "ttmlir/Conversion/TTIRToTTIRGeneric/TTIRToTTIRGeneric.h"
#include "ttmlir/Conversion/TTIRToTTKernel/TTIRToTTKernel.h"
#include "ttmlir/Conversion/TTIRToTTMetal/TTIRToTTMetal.h"
#include "ttmlir/Conversion/TTIRToTTNN/TTIRToTTNN.h"
#include "ttmlir/Conversion/TTKernelToEmitC/TTKernelToEmitC.h"
#include "ttmlir/Conversion/TTNNToEmitC/TTNNToEmitC.h"
#include "ttmlir/Conversion/TTNNToEmitPy/TTNNToEmitPy.h"
#include "ttmlir/Conversion/TTNNToTTIR/TTNNToTTIR.h"
#include "ttmlir/Conversion/TosaToTTIR/TosaToTTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"

namespace mlir::tt {

#define GEN_PASS_REGISTRATION
#include "ttmlir/Conversion/Passes.h.inc"

struct MLIRModuleLogger {
  mlir::MLIRContext *context;
  std::vector<std::pair<std::string, std::string>> moduleCache;

  void attachContext(mlir::MLIRContext *ctx,
                     std::vector<std::string> passNamesToCache);
};

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_PASSES_H
