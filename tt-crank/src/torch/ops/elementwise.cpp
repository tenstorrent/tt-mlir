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
#include "utils.hpp"

namespace tt::kurbla::torch_backend {

namespace {

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

at::Tensor tt_relu(const at::Tensor &self) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::relu: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto input = mb.args()[0];
    auto result_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto relu = mb.create<mlir::tt::ttir::ReluOp>(result_type, input);
    auto module_op = std::move(mb).finalize({relu.getResult()});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), self.scalar_type());
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("add.Tensor", TORCH_FN(tt_add));
    m.impl("relu", TORCH_FN(tt_relu));
}

} // namespace tt::kurbla::torch_backend
