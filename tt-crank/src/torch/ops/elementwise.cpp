#include "torch/ops/builders.hpp"

#include <cstdint>
#include <utility>

#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <mlir/IR/BuiltinAttributes.h>
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

at::Tensor tt_add(const at::Tensor &a_in, const at::Tensor &b_in, const at::Scalar &alpha) {
    const auto [a, b] = align_on_tt(a_in, b_in);

    // `promoted` is the user-facing dtype we'll stamp on the result; physical
    // storage will be the rewriter's hardware-backed alias (e.g. f32 when
    // promoted is f64).
    auto mb = ModuleBuilder::init({spec_for(a), spec_for(b)});
    auto [promoted, lhs, rhs] = promote_inputs(mb, a, b);

    auto result = build_add(mb, lhs, rhs, alpha.toDouble());
    auto out_shape = at::infer_size(a.sizes(), b.sizes());
    auto module_op = std::move(mb).finalize({result});

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

mlir::Value build_scalar(ModuleBuilder &mb, mlir::Type element_type, double value) {
    // Shape `[1]` broadcasts against any rank via numpy-style prepend-1 rules.
    auto tensor_type = mlir::RankedTensorType::get({1}, element_type);
    mlir::DenseElementsAttr value_attr;
    if (auto float_ty = mlir::dyn_cast<mlir::FloatType>(element_type)) {
        // Build the APFloat at the element type's semantics so bf16/f16/f64 keep
        // their native precision in the IR (no F32-attr downsampling hack).
        llvm::APFloat ap(value);
        bool loses_info = false;
        ap.convert(float_ty.getFloatSemantics(), llvm::APFloat::rmNearestTiesToEven, &loses_info);
        value_attr = mlir::DenseElementsAttr::get(tensor_type, ap);
    } else {
        auto int_ty = mlir::cast<mlir::IntegerType>(element_type);
        llvm::APInt ap(int_ty.getWidth(), as<std::int64_t>(value), /*isSigned=*/true);
        value_attr = mlir::DenseElementsAttr::get(tensor_type, ap);
    }
    auto constant = mb.create<mlir::tt::ttir::ConstantOp>(tensor_type, value_attr);
    return constant.getResult();
}

mlir::Value build_add(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs, double alpha) {
    auto lhs_type = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto rhs_type = mlir::cast<mlir::RankedTensorType>(rhs.getType());
    TORCH_INTERNAL_ASSERT(lhs_type.getElementType() == rhs_type.getElementType(),
                          "tt-kurbla build_add: lhs and rhs must share element type — callers must promote first");

    // aten::add: lhs + alpha * rhs. Elide the scale when alpha == 1.
    if (alpha != 1.0) {
        rhs = scale_tensor(mb, rhs, alpha);
    }

    // ttir.add broadcasts internally; just give it the broadcasted output shape.
    auto out_shape = at::infer_size(at::IntArrayRef(lhs_type.getShape().data(), lhs_type.getShape().size()),
                                    at::IntArrayRef(rhs_type.getShape().data(), rhs_type.getShape().size()));
    auto result_type = mlir::RankedTensorType::get(out_shape, lhs_type.getElementType());
    auto add = mb.create<mlir::tt::ttir::AddOp>(result_type, lhs, rhs);
    return add.getResult();
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("add.Tensor", TORCH_FN(tt_add));
    m.impl("relu", TORCH_FN(tt_relu));
}

} // namespace tt::kurbla::torch_backend
