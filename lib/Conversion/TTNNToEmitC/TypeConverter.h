// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_LIB_CONVERSION_TTNNTOEMITC_TYPECONVERTER_H
#define TTMLIR_LIB_CONVERSION_TTNNTOEMITC_TYPECONVERTER_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

class TTNNToEmitCTypeConverter : public TypeConverter {
public:
  TTNNToEmitCTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](mlir::tt::DeviceType type) -> Type {
      return emitc::OpaqueType::get(ctx, "ttnn::Device");
    });
    addConversion([ctx](TensorType type) -> Type {
      return emitc::OpaqueType::get(ctx, "ttnn::Tensor");
    });
  }
};

} // namespace

#endif // TTMLIR_LIB_CONVERSION_TTNNTOEMITC_TYPECONVERTER_H
