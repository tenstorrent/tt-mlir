// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/Utils/PopulateArgumentTypes.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallSet.h"
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/Support/LLVM.h>

namespace mlir::tt {

bool ArgumentTypeMapParser::parse(
    llvm::cl::Option &O, llvm::StringRef argName,
    llvm::StringRef commandLineArg,
    llvm::StringMap<SmallVector<ArgumentType>> &val) {
  llvm::StringRef errorMessage =
      "Invalid format. Expected: function=type1,type2;function=type1,type2";
  llvm::StringRef arg = commandLineArg;

  llvm::SmallVector<llvm::StringRef> entries;
  arg.split(entries, ';'); // Split functions by `;`
  // Entries would hold something like ["func1=input,param",
  // "func2=param,param"].

  for (llvm::StringRef entry : entries) {
    auto [funcName, argsStr] = entry.split('=');
    if (argsStr.empty()) {
      llvm::errs() << errorMessage << "\n";
      return true;
    }

    llvm::SmallVector<llvm::StringRef> argNames;
    argsStr.split(argNames, ','); // Split arguments by `,`

    // This allows the user to leave a trailing comma at the end of the argument
    // list i.e "func1=input,param," We must do this instead of setting
    // KeepEmpty=false in argStr.split() because that would not allow empty
    // arguments in the middle of the list.
    if (argNames.back() == "") {
      argNames.pop_back();
    }

    // if an entry is something like "func1=" or "func1=," then it's invalid.
    if (argNames.empty()) {
      llvm::errs() << "Provided empty argument list for funtion name: \""
                   << funcName << "\"" << "\n";
      return true;
    }
    // If this enrty is  "func1=input,param" then argNames would hold ["input",
    // "param"] now.

    // Parse the argument type names into their respective enums.
    llvm::SmallVector<ArgumentType> argTypes;
    for (llvm::StringRef arg : argNames) {
      auto argTypeEnum = ArgumentTypeStringToEnum(arg);
      if (!argTypeEnum.has_value()) {
        llvm::errs() << "Invalid argument type: " << arg << "\n";
        return true;
      }
      argTypes.push_back(argTypeEnum.value());
    }

    val[funcName] = argTypes;
  }

  return false;
}

void ArgumentTypeMapParser::print(
    llvm::raw_ostream &os,
    const llvm::StringMap<SmallVector<ArgumentType>> &argTypeMap) {

  SmallVector<llvm::StringRef> argTypeNames;
  for (auto &kv : argTypeMap) {
    for (auto argType : kv.second) {
      argTypeNames.push_back(ArgumentTypeEnumToString(argType));
    }
  }

  os << ttmlir::utils::join(argTypeNames, ",") << "\n";
}

namespace impl {

namespace {
std::unique_ptr<::mlir::Pass> createTTPopulateArgumentTypes();
std::unique_ptr<::mlir::Pass>
createTTPopulateArgumentTypes(TTArgumentTypeMap options);

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
    return "tt-populate-argument-types";
  }

  ::llvm::StringRef getArgument() const override {
    return "tt-populate-argument-types";
  }

  ::llvm::StringRef getDescription() const override {
    return "Populate argument types.";
  }

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return "TTPopulateArgumentTypes";
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

  TTPopulateArgumentTypesBase(TTArgumentTypeMap argumentTypeMap)
      : TTPopulateArgumentTypesBase() {
    this->argumentTypeMap = std::move(argumentTypeMap);
  }

protected:
  ::mlir::Pass::Option<TTArgumentTypeMap, ArgumentTypeMapParser>
      argumentTypeMap{
          *this, OptionNames::argumentTypes,
          llvm::cl::desc(
              "Map of function name to argument types. To use this option in "
              "the command line, you must provide a whitespace-free string\n\t"
              " which is a sequence of phrases in the form "
              "\"<FUNC_NAME_STR>=<ARG_TYPES>\" separated by semicolons, where "
              "<FUNC_NAME_STR>\n\t"
              " is the name of a function and <ARG_TYPES> is a sequence of "
              "argument types separated by commas. Each of which must be "
              "one\n\t"
              " of \"input\", \"parameter\" or \"constant\". \n\t"
              " Example: "
              "\"argument-types=forward=input,parameter,parameter,constant\""
              "\n\n"),
          llvm::cl::init(TTArgumentTypeMap())};

private:
  friend std::unique_ptr<::mlir::Pass> createTTPopulateArgumentTypes() {
    return std::make_unique<DerivedT>();
  }

  friend std::unique_ptr<::mlir::Pass>
  createTTPopulateArgumentTypes(TTArgumentTypeMap options) {
    return std::make_unique<DerivedT>(options);
  }
};
} // namespace
} // namespace impl

std::unique_ptr<::mlir::Pass> createTTPopulateArgumentTypes() {
  return impl::createTTPopulateArgumentTypes();
}

std::unique_ptr<::mlir::Pass>
createTTPopulateArgumentTypes(TTArgumentTypeMap options) {
  return impl::createTTPopulateArgumentTypes(std::move(options));
}

class TTPopulateArgumentTypes
    : public impl::TTPopulateArgumentTypesBase<TTPopulateArgumentTypes> {
public:
  using impl::TTPopulateArgumentTypesBase<
      TTPopulateArgumentTypes>::TTPopulateArgumentTypesBase;

  void runOnOperation() final {
    // Currently, we will allow compile without assigning argument types.
    if (failed(checkArgumentTypeMapPopulated())) {
      emitWarning(getOperation().getLoc())
          << "Empty argument type map provided. Skipping argument "
             "type population. This may affect subsequent compile steps.\n";
      return;
    }
    TTArgumentTypeMap map = argumentTypeMap.getValue();
    if (failed(checkOnlyValidFunctionNamesProvided(map, getOperation()))) {
      emitError(getOperation().getLoc())
          << "At least one function name provided in the argument type map "
             "does not exist in the module.\n";
      signalPassFailure();
      return;
    }

    // Iterate through every function as we may be assigning argument types to
    // them.
    for (auto func : getOperation().getOps<mlir::func::FuncOp>()) {
      StringRef funcName = func.getName();

      // If the map does not provide argument types for this function then skip
      if (!map.contains(funcName)) {
        continue;
      }

      SmallVector<ArgumentType> argTypes = map[funcName];
      if (failed(checkNumProvidedArgsMatch(func, argTypes))) {
        emitError(func.getLoc())
            << "Did not provide the correct number of argument types for "
               "function: \""
            << func.getName() << "\". Expected: " << func.getNumArguments()
            << " arguments, got: " << argTypes.size() << " arguments.\n";
        signalPassFailure();
        return;
      }

      for (uint32_t argIdx = 0; argIdx < func.getNumArguments(); argIdx++) {
        applyFuncArgumentType(func, argIdx, argTypes[argIdx]);
      }
    }
  }

private:
  LogicalResult checkArgumentTypeMapPopulated() {
    return success(!argumentTypeMap.getValue().empty());
  }

  LogicalResult checkOnlyValidFunctionNamesProvided(TTArgumentTypeMap map,
                                                    ModuleOp module) {
    llvm::SmallSet<StringRef, 8> funcNames;
    for (func::FuncOp func : module.getOps<mlir::func::FuncOp>()) {
      funcNames.insert(func.getName());
    }
    for (StringRef funcName : map.keys()) {
      if (!funcNames.contains(funcName)) {
        emitError(module.getLoc())
            << "Function: \"" << funcName
            << "\" was provided in the argument types map, however it "
               "was not found in module!\n";
        return failure();
      }
    }
    return success();
  }

  LogicalResult
  checkNumProvidedArgsMatch(mlir::func::FuncOp func,
                            SmallVector<ArgumentType> argTypeAttrs) {
    return success(func.getNumArguments() == argTypeAttrs.size());
  }

  // Adds the argument type attribute to the function argument at the given
  // index.
  void applyFuncArgumentType(func::FuncOp func, uint32_t argIdx,
                             ArgumentType argType) {
    // The current argument may already have attributes, so we need to add
    // the argument type to that DictonaryAttr rather than overwrite it.
    SmallVector<mlir::NamedAttribute> newArgAttrs;
    if (auto currentArgAttrDict = func.getArgAttrDict(argIdx)) {
      if (currentArgAttrDict.contains(ArgumentTypeAttr::name)) {
        emitWarning(func.getLoc())
            << "Overwriting existing argument type attribute for "
               "function: \""
            << func->getName() << "\" argument: " << argIdx << "\n";
        llvm::copy_if(
            currentArgAttrDict.getValue(), std::back_inserter(newArgAttrs),
            [&](mlir::NamedAttribute currentArgAttr) {
              return currentArgAttr.getName() != ArgumentTypeAttr::name;
            });
      } else {
        newArgAttrs =
            SmallVector<mlir::NamedAttribute>(currentArgAttrDict.getValue());
      }
    }
    newArgAttrs.emplace_back(
        mlir::StringAttr::get(&getContext(), ArgumentTypeAttr::name),
        ArgumentTypeAttr::get(&getContext(), argType));

    func.setArgAttrs(argIdx,
                     mlir::DictionaryAttr::get(&getContext(), newArgAttrs));
  }
};

} // namespace mlir::tt
