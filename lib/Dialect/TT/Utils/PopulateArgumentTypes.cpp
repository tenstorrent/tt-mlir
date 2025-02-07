// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/Utils/PopulateArgumentTypes.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TT/Utils/PassOverrides.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include <llvm/ADT/SmallSet.h>

namespace mlir::tt {

namespace impl {

std::unique_ptr<::mlir::Pass> createTTPopulateArgumentTypes();
std::unique_ptr<::mlir::Pass>
createTTPopulateArgumentTypes(TTPopulateArgumentTypesOptions options);

template <typename DerivedT>
class TTPopulateArgumentTypesBase : public ::mlir::OperationPass<ModuleOp> {
public:
  using Base = TTPopulateArgumentTypesBase;

  TTPopulateArgumentTypesBase()
      : ::mlir::OperationPass<ModuleOp>(::mlir::TypeID::get<DerivedT>()) {}
  TTPopulateArgumentTypesBase(const TTPopulateArgumentTypesBase &other)
      : ::mlir::OperationPass<ModuleOp>(other) {}
  TTPopulateArgumentTypesBase &
  operator=(const TTPopulateArgumentTypesBase &) = delete;
  TTPopulateArgumentTypesBase(TTPopulateArgumentTypesBase &&) = delete;
  TTPopulateArgumentTypesBase &
  operator=(TTPopulateArgumentTypesBase &&) = delete;
  ~TTPopulateArgumentTypesBase() override = default;

  /// Returns the command-line argument attached to this pass.
  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("tt-populate-argument-types");
  }
  ::llvm::StringRef getArgument() const override {
    return "tt-populate-argument-types";
  }

  ::llvm::StringRef getDescription() const override {
    return "Populate argument types.";
  }

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("TTPopulateArgumentTypes");
  }
  ::llvm::StringRef getName() const override {
    return "TTPopulateArgumentTypes";
  }

  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const ::mlir::Pass *pass) {
    return pass->getTypeID() == ::mlir::TypeID::get<DerivedT>();
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<DerivedT>(*static_cast<const DerivedT *>(this));
  }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::TTDialect>();
  }

  /// Explicitly declare the TypeID for this class. We declare an explicit
  /// private instantiation because Pass classes should only be visible by the
  /// current library.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TTPopulateArgumentTypesBase<DerivedT>)

  TTPopulateArgumentTypesBase(TTPopulateArgumentTypesOptions options)
      : TTPopulateArgumentTypesBase() {
    argumentTypeMap = std::move(options.argumentTypeMap);
  }

protected:
  ::mlir::Pass::Option<llvm::StringMap<TTArgumentTypeVector>,
                       ArgumentTypeMapParser>
      argumentTypeMap{
          *this, OptionNames::argumentTypes,
          llvm::cl::desc(
              "Map of function name to argument types. To use this option in "
              "the "
              "command line, you must provide a whitespace-free\n\t string in "
              "the "
              "format: sequences of phrases in the form "
              "\"<FUNC_NAME_STR>=<ARG_TYPES>\" separated by semicolons, "
              "where <FUNC_NAME_STR> is\n\t the name of a function and "
              "<ARG_TYPES>"
              "is a sequence of argument types separated by commas. Each\n\t "
              "of "
              "which must be one of \"input\", \"parameter\" or "
              "\"constant\".\n\t "
              "Example: "
              "\"main1=input,parameter,parameter;main2=input,constant"
              "\"\n\n"),
          llvm::cl::init(llvm::StringMap<tt::TTArgumentTypeVector>())};

private:
  friend std::unique_ptr<::mlir::Pass> createTTPopulateArgumentTypes() {
    return std::make_unique<DerivedT>();
  }

  friend std::unique_ptr<::mlir::Pass>
  createTTPopulateArgumentTypes(TTPopulateArgumentTypesOptions options) {
    return std::make_unique<DerivedT>(std::move(options));
  }
};
} // namespace impl

std::unique_ptr<::mlir::Pass> createTTPopulateArgumentTypes() {
  return impl::createTTPopulateArgumentTypes();
}

std::unique_ptr<::mlir::Pass>
createTTPopulateArgumentTypes(TTPopulateArgumentTypesOptions options) {
  return impl::createTTPopulateArgumentTypes(std::move(options));
}

class TTPopulateArgumentTypes
    : public impl::TTPopulateArgumentTypesBase<TTPopulateArgumentTypes> {
public:
  using impl::TTPopulateArgumentTypesBase<
      TTPopulateArgumentTypes>::TTPopulateArgumentTypesBase;

  void runOnOperation() final {
    // Currently, we will allow compile without assigning argument types.
    auto map = argumentTypeMap.getValue();
    if (map.empty()) {
      llvm::errs()
          << "WARNING: Empty argument type map provided. Skipping argument "
             "type population. This may affect subsequent compile steps.\n";
      return;
    }

    mlir::ModuleOp module = getOperation();
    llvm::SmallSet<StringRef, 8> funcNames;

    // Iterate through every function as we may be assigning argument types to
    // them.
    for (auto func : module.getOps<mlir::func::FuncOp>()) {
      auto funcName = func.getName();
      funcNames.insert(funcName);

      if (map.find(funcName) == map.end()) {
        continue;
      }

      std::vector<mlir::Attribute> argTypeAttrs;
      for (auto argType : map.at(funcName).argumentTypes) {
        argTypeAttrs.push_back(
            mlir::tt::ArgumentTypeAttr::get(&getContext(), argType));
      }
      if (func.getNumArguments() != argTypeAttrs.size()) {
        llvm::errs() << "Function: \"" << funcName
                     << "\" argument count mismatch.\n";
        signalPassFailure();
      }
      // Need to update/create the DictionaryAttr for each corresponding
      // function argument.
      for (uint32_t i = 0; i < func.getNumArguments(); i++) {
        // The current argument may already have attributes, so we need to add
        // the argument type to that DictonaryAttr rather than overwrite it.
        std::vector<mlir::NamedAttribute> newArgAttrs;
        if (auto currentArgAttrDict = func.getArgAttrDict(i)) {
          for (mlir::NamedAttribute currentArgAttr : currentArgAttrDict) {
            // If this argument already has an argumnet type, this pass wil
            // overwrite it. Log a warning.
            if (currentArgAttr.getName() != "tt.argument_type") {
              newArgAttrs.push_back(currentArgAttr);
            } else {
              llvm::errs() << "WARNING: Overwriting existing argument type "
                              "attribute for function: \""
                           << funcName << "\" argument: " << i << "\n";
            }
          }
        }
        mlir::NamedAttribute attr(
            mlir::StringAttr::get(&getContext(), "tt.argument_type"),
            argTypeAttrs[i]);
        newArgAttrs.push_back(attr);

        func.setArgAttrs(i,
                         mlir::DictionaryAttr::get(&getContext(), newArgAttrs));
      }
    }

    for (auto &kv : map) {
      if (!funcNames.contains(kv.first())) {
        llvm::errs() << "Function: \"" << kv.first()
                     << "\" was provided in the argument types map, however it "
                        "was not found in module!\n";
        signalPassFailure();
      }
    }
  }
};

} // namespace mlir::tt
