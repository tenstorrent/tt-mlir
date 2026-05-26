#pragma once

#include <cstdint>

#include <ATen/ATen.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/LLVM.h>
#include <ttmlir/Dialect/TTIR/IR/TTIROps.h>

#include "cast.hpp"
#include "torch/ttir_module_builder.hpp"

namespace tt::kurbla::torch_backend {

// Build `value * tensor` as a TTIR subgraph: a ttir.full of `value` matching
// `tensor`'s shape/dtype, then a ttir.multiply. ttir.full's fill_value attr
// only accepts F32Attr or I32Attr — for integer tensors we refuse a non-
// integral scalar rather than silently truncate it.
inline mlir::Value scale_tensor(ModuleBuilder &mb, mlir::Value tensor, const at::Scalar &value) {
    auto tensor_type = mlir::cast<mlir::RankedTensorType>(tensor.getType());
    mlir::Attribute fill_attr;
    if (mlir::isa<mlir::FloatType>(tensor_type.getElementType())) {
        fill_attr = mb.attrs().getF32FloatAttr(as<float>(value.toDouble()));
    } else {
        TORCH_CHECK(value.isIntegral(/*includeBool=*/false), "tt-kurbla: non-integral scalar (", value.toDouble(),
                    ") on integer tensor would truncate");
        fill_attr = mb.attrs().getI32IntegerAttr(as<std::int32_t>(value.toLong()));
    }
    auto full = mb.create<mlir::tt::ttir::FullOp>(tensor_type, fill_attr);
    auto mul = mb.create<mlir::tt::ttir::MultiplyOp>(tensor_type, tensor, full.getResult());
    return mul.getResult();
}

} // namespace tt::kurbla::torch_backend
