#include <cstdint>
#include <utility>

#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/LLVM.h>
#include <torch/library.h>
#include <ttmlir/Dialect/TTIR/IR/TTIROps.h>

#include "cast.hpp"
#include "torch/backend.hpp"
#include "torch/tensor.hpp"
#include "torch/ttir_module_builder.hpp"

namespace tt::kurbla::torch_backend {

namespace {

// Build `value * tensor` as a TTIR subgraph: a ttir.full of `value` matching
// `tensor`'s shape/dtype, then a ttir.multiply. ttir.full's fill_value attr
// only accepts F32Attr or I32Attr — for integer tensors we refuse a non-
// integral scalar rather than silently truncate it.
mlir::Value scale_tensor(ModuleBuilder &mb, mlir::Value tensor, const at::Scalar &value) {
    auto tensor_type = mlir::cast<mlir::RankedTensorType>(tensor.getType());
    mlir::Attribute fill_attr;
    if (mlir::isa<mlir::FloatType>(tensor_type.getElementType())) {
        fill_attr = mb.attrs().getF32FloatAttr(as<float>(value.toDouble()));
    } else {
        TORCH_CHECK(value.isIntegral(/*includeBool=*/false), "tt-kurbla aten::add: non-integral alpha (",
                    value.toDouble(), ") on integer tensors would truncate");
        fill_attr = mb.attrs().getI32IntegerAttr(as<std::int32_t>(value.toLong()));
    }
    auto full = mb.create<mlir::tt::ttir::FullOp>(tensor_type, fill_attr);
    auto mul = mb.create<mlir::tt::ttir::MultiplyOp>(tensor_type, tensor, full.getResult());
    return mul.getResult();
}

at::Tensor tt_add(const at::Tensor &a_in, const at::Tensor &b_in, const at::Scalar &alpha) {
    const auto [a, b] = align_on_tt(a_in, b_in);

    // `promoted` is the user-facing dtype we'll stamp on the result; physical
    // storage will be the rewriter's hardware-backed alias (e.g. f32 when
    // promoted is f64).
    auto mb = ModuleBuilder::init({spec_for(a), spec_for(b)});
    auto [promoted, lhs, rhs] = promote_inputs(mb, a, b);
    const auto promoted_mlir = mlir_element_type_for(promoted);

    // aten::add: lhs + alpha * rhs. Elide the scale when alpha == 1.
    if (alpha.toDouble() != 1.0) {
        rhs = scale_tensor(mb, rhs, alpha);
    }

    // ttir.add broadcasts internally; just give it the broadcasted output shape.
    auto out_shape = at::infer_size(a.sizes(), b.sizes());
    auto result_type = mlir::RankedTensorType::get(out_shape, promoted_mlir);

    auto add = mb.create<mlir::tt::ttir::AddOp>(result_type, lhs, rhs);
    auto module_op = std::move(mb).finalize({add.getResult()});

    auto outputs = compile_and_run(std::move(module_op), {a, b});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, promoted);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("add.Tensor", TORCH_FN(tt_add));
}

} // namespace tt::kurbla::torch_backend
