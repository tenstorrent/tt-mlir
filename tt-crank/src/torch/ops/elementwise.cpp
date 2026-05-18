#include <cstdint>
#include <utility>

#include <ATen/ATen.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/LLVM.h>
#include <torch/library.h>
#include <ttmlir/Dialect/TTIR/IR/TTIROps.h>

#include "torch/backend.hpp"
#include "torch/ttir_module_builder.hpp"

namespace tt::kurbla::torch_backend {

namespace {

TensorTypeSpec spec_for(const at::Tensor &t) {
    return TensorTypeSpec{{t.sizes().begin(), t.sizes().end()}, to_runtime_dtype(t.scalar_type())};
}

// Build `value * tensor` as a TTIR subgraph: a ttir.full of `value` matching
// `tensor`'s shape/dtype, then a ttir.multiply. ttir.full's fill_value attr
// only accepts F32Attr or I32Attr — for integer tensors we refuse a non-
// integral scalar rather than silently truncate it.
mlir::Value scale_tensor(ModuleBuilder &mb, mlir::Value tensor, const at::Scalar &value) {
    auto tensor_type = mlir::cast<mlir::RankedTensorType>(tensor.getType());
    mlir::Attribute fill_attr;
    if (mlir::isa<mlir::FloatType>(tensor_type.getElementType())) {
        fill_attr = mb.attrs().getF32FloatAttr(static_cast<float>(value.toDouble()));
    } else {
        TORCH_CHECK(value.isIntegral(/*includeBool=*/false), "tt-kurbla aten::add: non-integral alpha (",
                    value.toDouble(), ") on integer tensors would truncate");
        fill_attr = mb.attrs().getI32IntegerAttr(static_cast<std::int32_t>(value.toLong()));
    }
    auto full = mb.create<mlir::tt::ttir::FullOp>(tensor_type, fill_attr);
    auto mul = mb.create<mlir::tt::ttir::MultiplyOp>(tensor_type, tensor, full.getResult());
    return mul.getResult();
}

at::Tensor tt_add(const at::Tensor &a, const at::Tensor &b, const at::Scalar &alpha) {
    TORCH_CHECK(a.sizes() == b.sizes(), "tt-kurbla aten::add: broadcasting is not yet supported");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), "tt-kurbla aten::add: mixed dtype is not yet supported");

    auto mb = ModuleBuilder::init({spec_for(a), spec_for(b)});
    mlir::Value lhs = mb.args()[0];
    mlir::Value rhs = mb.args()[1];

    // aten::add semantics: lhs + alpha * rhs. When alpha is exactly 1 we elide
    // the scale so the trivial path stays a single-op module.
    if (alpha.toDouble() != 1.0) {
        rhs = scale_tensor(mb, rhs, alpha);
    }

    auto add = mb.create<mlir::tt::ttir::AddOp>(lhs.getType(), lhs, rhs);
    auto module_op = std::move(mb).finalize({add.getResult()});

    auto outputs = compile_and_run(std::move(module_op), {a, b});
    return outputs[0];
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("add.Tensor", TORCH_FN(tt_add));
}

} // namespace tt::kurbla::torch_backend
