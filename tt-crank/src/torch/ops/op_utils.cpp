#include "torch/ops/builders.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <ttmlir/Dialect/TTIR/IR/TTIROps.h>

namespace tt::kurbla::torch_backend {

mlir::Value scale_tensor(ModuleBuilder &mb, mlir::Value tensor, double value) {
    auto tensor_type = mlir::cast<mlir::RankedTensorType>(tensor.getType());
    auto scalar = build_scalar(mb, tensor_type.getElementType(), value);
    auto mul = mb.create<mlir::tt::ttir::MultiplyOp>(tensor_type, tensor, scalar);
    return mul.getResult();
}

} // namespace tt::kurbla::torch_backend
