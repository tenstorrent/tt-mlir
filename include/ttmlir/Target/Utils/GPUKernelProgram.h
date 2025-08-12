// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_UTILS_GPUKERNELPROGRAM_H
#define TTMLIR_TARGET_UTILS_GPUKERNELPROGRAM_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/raw_ostream.h"
#include <flatbuffers/buffer.h>
#include <string>
#include <vector>

#include "flatbuffers/flatbuffers.h"
#include "ttmlir/Target/TTNN/Target.h"
#include "ttmlir/Target/Utils/FlatbufferObjectCache.h"
#include "ttmlir/Target/Utils/MLIRToFlatbuffer.h"

namespace mlir::tt::cuda {
struct MemRefDesc {
  std::string name;
  mlir::Type type;
  mlir::TypedAttr value;
};

struct Kernel {
  std::string name;
  std::string ptx;
  std::vector<std::string> inputNames;
  int64_t gridSizeX;
  int64_t gridSizeY;
  int64_t gridSizeZ;
  int64_t blockSizeX;
  int64_t blockSizeY;
  int64_t blockSizeZ;
};

struct Program {
  ::flatbuffers::FlatBufferBuilder *fbb;
  std::vector<::flatbuffers::Offset<Kernel>> kernels;
  std::vector<::flatbuffers::Offset<MemRefDesc>> variables;
  flatbuffers::Offset<std::string> returnVariable;
};

} // namespace mlir::tt::cuda

#endif
