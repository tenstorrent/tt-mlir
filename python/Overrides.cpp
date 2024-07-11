// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Bindings/Python/Overrides.h"

namespace mlir::ttmlir::python {

std::unordered_map<std::string,
                   std::function<MlirAttribute(py::dict, MlirOperation)>>
    overrideHandlers;

void registerHandlers() {
  // Make these handlers accessible for further override purposes

  overrideHandlers["operandSegmentSizes"] =
      [](py::dict attributeParams, MlirOperation op) -> MlirAttribute {
    // We know that this is built with only an array that represents the
    // segmentSizes
    std::vector<int32_t> arr = py::cast<std::vector<int32_t>>(
        attributeParams["array"].cast<py::list>());
    return wrap(mlir::DenseI32ArrayAttr::get(
        unwrap(mlirOperationGetContext(op)), llvm::ArrayRef(arr)));
  };

  overrideHandlers["operand_constraints"] =
      [](py::dict attributeParams, MlirOperation op) -> MlirAttribute {
    std::vector<mlir::Attribute> Array;
    mlir::MLIRContext *ctx = unwrap(mlirOperationGetContext(op));
    std::vector<uint32_t> operandConstraints = py::cast<std::vector<uint32_t>>(
        attributeParams["constraintEnums"].cast<py::list>());
    for (auto operandConstraint : operandConstraints)
      Array.push_back(tt::OperandConstraintAttr::get(
          ctx, static_cast<tt::OperandConstraint>(operandConstraint)));
    return wrap(mlir::ArrayAttr::get(ctx, llvm::ArrayRef(Array)));
  };
}

void parseOverride(py::dict override, MlirOperation op) {
  // This dictionary will now be used to determine the override itself, it's
  // easier to manipulate py::dict than a JSON object
  /*
    {
      "attribute_name": {...<named_attribute_parameters>}
    }
  */

  if (overrideHandlers.size() == 0)
    registerHandlers();

  // Apply Override to the MlirOperation object
  for (auto &attr : override) {
    std::string attrName = py::str(attr.first);
    py::dict attrOverride = py::cast<py::dict>(attr.second);

    if (overrideHandlers.find(attrName) != overrideHandlers.end()) {
      MlirAttribute attr_ = overrideHandlers[attrName](attrOverride, op);
      mlirOperationSetInherentAttributeByName(
          op, mlirStringRefCreateFromCString(attrName.c_str()), attr_);
    }
  }
}

void populateOverridesModule(py::module &m) {

  m.def(
      "get_op_ptr",
      [](MlirOperation op) { return reinterpret_cast<uintptr_t>(unwrap(op)); },
      py::arg("op").noconvert());

  m.def("parse_override", [](py::dict overrides) {
    for (auto overrideToApply : overrides) {
      uintptr_t opPtr = py::cast<uintptr_t>(overrideToApply.first);
      py::dict attrOverrides = py::cast<py::dict>(overrideToApply.second);
      mlir::Operation *op_ = (reinterpret_cast<mlir::Operation *>(opPtr));
      MlirOperation op = wrap(op_);
      llvm::outs() << op_->getName().getStringRef() << '\n';
      parseOverride(attrOverrides, op);
    }
  });

  m.def(
      "override_dict",
      [](py::dict opDict) {
        // Iterate through all the ops

        for (auto item : opDict) {
          // Now transform the object into an operation
          py::object pyOp =
              pybind11::detail::mlirApiObjectToCapsule(item.first);
          MlirOperation op = mlirPythonCapsuleToOperation(pyOp.ptr());
          // Loop through to modify the attributes, a py::handle of item.second
          // contains a dict
          py::dict attrs = py::cast<py::dict>(item.second);
          for (auto attr : attrs) {
            std::string name = py::str(attr.first);

            py::object pyAttr =
                pybind11::detail::mlirApiObjectToCapsule(attr.second);
            MlirAttribute attr_ = mlirPythonCapsuleToAttribute(pyAttr.ptr());
            // This defines the new attribute!

            // Set the new value for the attribute, hope we don't seg-fault
            MlirStringRef name_ = mlirStringRefCreateFromCString(name.c_str());
            mlirOperationSetInherentAttributeByName(op, name_, attr_);
          }
        }
      },
      py::arg("op_dict"));
}

} // namespace mlir::ttmlir::python
