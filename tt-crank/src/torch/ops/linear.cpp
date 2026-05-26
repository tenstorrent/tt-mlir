#include <vector>

#include <ATen/ATen.h>
#include <mlir/IR/BuiltinTypes.h>
#include <torch/library.h>
#include <ttmlir/Dialect/TTIR/IR/TTIROps.h>

#include "torch/backend.hpp"
#include "torch/tensor.hpp"
#include "torch/ttir_module_builder.hpp"
#include "utils.hpp"

namespace tt::kurbla::torch_backend {

namespace {

at::Tensor tt_mm(const at::Tensor &self, const at::Tensor &mat2) {
    const auto [a_in, b_in] = align_on_tt(self, mat2);
    TORCH_CHECK(a_in.dim() == 2 && b_in.dim() == 2, "tt-kurbla aten::mm: inputs must be 2D");
    TORCH_CHECK(a_in.size(1) == b_in.size(0), "tt-kurbla aten::mm: shape mismatch: ", a_in.sizes(), " vs ",
                b_in.sizes());

    auto mb = ModuleBuilder::init({spec_for(a_in), spec_for(b_in)});
    auto [promoted, a, b] = promote_inputs(mb, a_in, b_in);

    std::vector<int64_t> out_shape{a_in.size(0), b_in.size(1)};
    auto result_type = mlir::RankedTensorType::get(out_shape, mlir_element_type_for(promoted));
    auto mm = mb.create<mlir::tt::ttir::MatmulOp>(result_type, a, b, false, false);
    auto module_op = std::move(mb).finalize({mm.getResult()});

    auto outputs = compile_and_run(std::move(module_op), {a_in, b_in});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, promoted);
}

// aten::addmm(bias, mat1, mat2, beta=1, alpha=1) = beta*bias + alpha*(mat1 @ mat2)
at::Tensor tt_addmm(const at::Tensor &self, const at::Tensor &mat1, const at::Tensor &mat2, const at::Scalar &beta,
                    const at::Scalar &alpha) {
    TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "tt-kurbla aten::addmm: mat1/mat2 must be 2D");
    TORCH_CHECK(mat1.size(1) == mat2.size(0), "tt-kurbla aten::addmm: shape mismatch: ", mat1.sizes(), " @ ",
                mat2.sizes());

    const auto [a_in, b_in, bias_in] = align_on_tt(mat1, mat2, self);

    auto mb = ModuleBuilder::init({spec_for(a_in), spec_for(b_in), spec_for(bias_in)});
    auto [promoted, a, b, bias] = promote_inputs(mb, a_in, b_in, bias_in);

    std::vector<int64_t> out_shape{mat1.size(0), mat2.size(1)};
    auto result_type = mlir::RankedTensorType::get(out_shape, mlir_element_type_for(promoted));

    mlir::Value result;
    if (beta.toDouble() == 1.0 && alpha.toDouble() == 1.0) {
        result = mb.create<mlir::tt::ttir::LinearOp>(result_type, a, b, bias, false, false).getResult();
    } else {
        result = mb.create<mlir::tt::ttir::MatmulOp>(result_type, a, b, false, false).getResult();
        if (alpha.toDouble() != 1.0) {
            result = scale_tensor(mb, result, alpha);
        }
        if (beta.toDouble() != 0.0) {
            mlir::Value scaled_bias = beta.toDouble() == 1.0 ? bias : scale_tensor(mb, bias, beta);
            result = mb.create<mlir::tt::ttir::AddOp>(result_type, result, scaled_bias).getResult();
        }
    }

    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {a_in, b_in, bias_in});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, promoted);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("mm", TORCH_FN(tt_mm));
    m.impl("addmm", TORCH_FN(tt_addmm));
}

} // namespace tt::kurbla::torch_backend
