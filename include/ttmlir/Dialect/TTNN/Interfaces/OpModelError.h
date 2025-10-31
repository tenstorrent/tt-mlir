// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_INTERFACES_OPMODELERROR_H
#define TTMLIR_DIALECT_TTNN_INTERFACES_OPMODELERROR_H

#include "llvm/Support/Error.h"

namespace mlir::tt::ttnn::detail {

enum class ReasonForLackOfSupport {
  NeedsMemoryIO,
  MissingMetalDefinition,
  NeedsMultiDevice,
  NoNeedForConstraintAPI,
  ArchitecturalMismatch,
};

inline std::string getReasonForLackOfSupportStr(ReasonForLackOfSupport reason) {
  switch (reason) {
  case ReasonForLackOfSupport::NeedsMemoryIO:
    return "needs memory IO";
  case ReasonForLackOfSupport::MissingMetalDefinition:
    return "missing metal definition";
  case ReasonForLackOfSupport::NeedsMultiDevice:
    return "needs multi-device";
  case ReasonForLackOfSupport::NoNeedForConstraintAPI:
    return "no need for constraint API";
  case ReasonForLackOfSupport::ArchitecturalMismatch:
    return "architectural mismatch between dialects";
  }
}

class OpNotSupportedError : public llvm::ErrorInfo<OpNotSupportedError> {
public:
  static char ID;

  OpNotSupportedError(llvm::StringRef opName, ReasonForLackOfSupport reason,
                      llvm::StringRef apiType)
      : opName(opName.str()), reason(reason), apiType(apiType.str()) {}

  void log(llvm::raw_ostream &OS) const override {
    OS << apiType << " is not supported for " << opName << ". Reason: ["
       << getReasonForLackOfSupportStr(reason) << "]";
  }

  std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }

  ReasonForLackOfSupport getReason() const { return reason; }
  llvm::StringRef getOpName() const { return opName; }
  llvm::StringRef getAPIType() const { return apiType; }

private:
  std::string opName;
  ReasonForLackOfSupport reason;
  std::string apiType;
};

} // namespace mlir::tt::ttnn::detail

#endif // TTMLIR_DIALECT_TTNN_INTERFACES_OPMODELERROR_H
