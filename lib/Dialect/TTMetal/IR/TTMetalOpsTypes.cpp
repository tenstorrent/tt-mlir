// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::tt::ttmetal;

#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/TTMetal/IR/TTMetalAttrInterfaces.cpp.inc"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.cpp.inc"

void TTMetalDialect::registerTypes() {
  // NOLINTNEXTLINE
  addTypes<
#define GET_TYPEDEF_LIST
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.cpp.inc"
      >();
}

CoreRangeAttr
CoreRangeAttr::getPhysicalCoreRange(MLIRContext *context,
                                    ArrayRef<int64_t> physicalGridShape) {
  // Default offset is (0, 0) -- in the future, we can make it a parameter when
  // we need to offset differently.
  const SmallVector<int64_t> offset = {0, 0};
  return CoreRangeAttr::get(context, offset, physicalGridShape);
}

// ComputeConfig and NocConfig assembly format: hand-rolled so the new
// `num_threads_per_cluster` field can elide on print when it equals 1
// (the WH/BH default). Without elision, every existing CB-flow test
// would diff to add `, num_threads_per_cluster = 1`.
//
// Parser is permissive — the optional `, num_threads_per_cluster = N`
// suffix may appear after the existing comma-separated fields.

mlir::Attribute ComputeConfigAttr::parse(::mlir::AsmParser &parser,
                                          ::mlir::Type) {
  mlir::SymbolRefAttr kernelSymbol;
  CoreRangeAttr coreRange;
  KernelArgsAttr kernelArgs;
  MathFidelity mathFidelity;
  bool fp32DestAccEn = false;
  bool dstFullSyncEn = false;
  bool mathApproxMode = false;
  llvm::SmallVector<UnpackToDestMode> unpackToDestMode;
  uint32_t numThreadsPerCluster = 1;

  if (parser.parseLess() || parser.parseAttribute(kernelSymbol) ||
      parser.parseComma() || parser.parseAttribute(coreRange) ||
      parser.parseComma() || parser.parseAttribute(kernelArgs) ||
      parser.parseComma()) {
    return {};
  }
  auto mathFidelityResult = mlir::FieldParser<MathFidelity>::parse(parser);
  if (mlir::failed(mathFidelityResult)) {
    return {};
  }
  mathFidelity = *mathFidelityResult;
  if (parser.parseComma() || parser.parseInteger(fp32DestAccEn) ||
      parser.parseComma() || parser.parseInteger(dstFullSyncEn) ||
      parser.parseComma() || parser.parseInteger(mathApproxMode) ||
      parser.parseComma() || parser.parseLSquare()) {
    return {};
  }
  if (parser.parseOptionalRSquare().failed()) {
    do {
      auto value = mlir::FieldParser<UnpackToDestMode>::parse(parser);
      if (mlir::failed(value)) {
        return {};
      }
      unpackToDestMode.push_back(*value);
    } while (parser.parseOptionalComma().succeeded());
    if (parser.parseRSquare()) {
      return {};
    }
  }
  if (parser.parseOptionalComma().succeeded()) {
    if (parser.parseKeyword("num_threads_per_cluster") ||
        parser.parseEqual() || parser.parseInteger(numThreadsPerCluster)) {
      return {};
    }
  }
  if (parser.parseGreater()) {
    return {};
  }

  return ComputeConfigAttr::get(parser.getContext(), kernelSymbol, coreRange,
                                kernelArgs, mathFidelity, fp32DestAccEn,
                                dstFullSyncEn, mathApproxMode, unpackToDestMode,
                                numThreadsPerCluster);
}

void ComputeConfigAttr::print(::mlir::AsmPrinter &printer) const {
  printer << "<";
  printer.printAttribute(getKernelSymbol());
  printer << ", ";
  printer.printAttribute(getCoreRange());
  printer << ", ";
  printer.printAttribute(getKernelArgs());
  printer << ", ";
  printer << stringifyEnum(getMathFidelity());
  printer << ", " << getFp32DestAccEn();
  printer << ", " << getDstFullSyncEn();
  printer << ", " << getMathApproxMode();
  printer << ", [";
  llvm::interleaveComma(getUnpackToDestMode(), printer.getStream(),
                        [&](UnpackToDestMode mode) {
                          printer << stringifyEnum(mode);
                        });
  printer << "]";
  if (getNumThreadsPerCluster() != 1) {
    printer << ", num_threads_per_cluster = " << getNumThreadsPerCluster();
  }
  printer << ">";
}

