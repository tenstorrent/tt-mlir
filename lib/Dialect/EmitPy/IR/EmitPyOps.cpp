// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/Support/LogicalResult.h"

#include <string_view>

using namespace mlir;
using namespace mlir::tt::emitpy;

//===----------------------------------------------------------------------===//
// EmitPyDialect
//===----------------------------------------------------------------------===//

/// Parse a format string and return a list of its parts.
/// A part is either a StringRef that has to be printed as-is, or
/// a Placeholder which requires printing the next operand of the VerbatimOp.
/// In the format string, all `{}` are replaced by Placeholders, except if the
/// `{` is escaped by `{{` - then it doesn't start a placeholder.
template <class ArgType>
FailureOr<SmallVector<ReplacementItem>>
parseFormatString(StringRef toParse, ArgType fmtArgs,
                  std::optional<llvm::function_ref<mlir::InFlightDiagnostic()>>
                      emitError = {}) {
  SmallVector<ReplacementItem> items;

  // If there are not operands, the format string is not interpreted.
  if (fmtArgs.empty()) {
    items.push_back(toParse);
    return items;
  }

  while (!toParse.empty()) {
    size_t idx = toParse.find('{');
    if (idx == StringRef::npos) {
      // No '{'
      items.push_back(toParse);
      break;
    }
    if (idx > 0) {
      // Take all chars excluding the '{'.
      items.push_back(toParse.take_front(idx));
      toParse = toParse.drop_front(idx);
      continue;
    }
    if (toParse.size() < 2) {
      return (*emitError)()
             << "expected '}' after unescaped '{' at end of string";
    }
    // toParse contains at least two characters and starts with `{`.
    char nextChar = toParse[1];
    if (nextChar == '{') {
      // Double '{{' -> '{' (escaping).
      items.push_back(toParse.take_front(1));
      toParse = toParse.drop_front(2);
      continue;
    }
    if (nextChar == '}') {
      items.push_back(Placeholder{});
      toParse = toParse.drop_front(2);
      continue;
    }

    if (emitError.has_value()) {
      return (*emitError)() << "expected '}' after unescaped '{'";
    }
    return failure();
  }
  return items;
}

/// Check that the type of the initial value is compatible with the operations
/// result type.
static LogicalResult verifyInitializationAttribute(Operation *op,
                                                   Attribute value) {
  assert(op->getNumResults() == 1 && "operation must have 1 result");

  if (llvm::isa<mlir::tt::emitpy::OpaqueAttr>(value)) {
    return success();
  }

  Type resultType = op->getResult(0).getType();
  Type attrType = cast<TypedAttr>(value).getType();
  if (resultType != attrType) {
    return op->emitOpError()
           << "requires attribute to either be an #emitpy.opaque attribute or "
              "it's type ("
           << attrType << ") to match the op's result type (" << resultType
           << ")";
  }

  return success();
}

template <typename SourceOp>
LogicalResult verifyNearestGlobalSymbol(SourceOp op,
                                        SymbolTableCollection &symbolTable) {
  auto global =
      symbolTable.lookupNearestSymbolFrom<GlobalOp>(op, op.getNameAttr());
  if (!global) {
    return op.emitOpError("'")
           << op.getName() << "' does not reference a valid emitpy.global";
  }

  Type resultType = op.getResult().getType();
  Attribute initialValue = global.getInitialValue();

  // If the global has a typed attribute, verify the types match
  if (auto typedAttr = llvm::dyn_cast<TypedAttr>(initialValue)) {
    Type globalType = typedAttr.getType();
    if (resultType != globalType) {
      return op.emitOpError()
             << "result type (" << resultType
             << ") does not match global's type (" << globalType << ")";
    }
  }
  // For opaque attributes, we allow any type since the type is not specified
  // in the attribute itself

  return success();
}

LogicalResult isValidPythonIdentifier(Operation *op, StringRef name) {
  if (name.empty()) {
    return op->emitOpError() << "variable name must not be empty";
  }

  static constexpr std::array<std::string_view, 35> pythonKeywords = {
      "False",  "None",   "True",    "and",      "as",       "assert", "async",
      "await",  "break",  "class",   "continue", "def",      "del",    "elif",
      "else",   "except", "finally", "for",      "from",     "global", "if",
      "import", "in",     "is",      "lambda",   "nonlocal", "not",    "or",
      "pass",   "raise",  "return",  "try",      "while",    "with",   "yield"};

  for (const auto keyword : pythonKeywords) {
    if (static_cast<std::string_view>(name) == keyword) {
      return op->emitOpError() << "variable name must not be a keyword";
    }
  }

  unsigned char first = static_cast<unsigned char>(name[0]);
  if (!(std::isalpha(first) || first == '_')) {
    return op->emitOpError() << "variable name must start with a letter or '_'";
  }

  for (const auto c : name.drop_front()) {
    if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_')) {
      return op->emitOpError() << "variable name may only contain alphanumeric "
                                  "characters and '_'";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// GetAttrOp / SetAttrOp / ClassOp
//===----------------------------------------------------------------------===//

LogicalResult GetAttrOp::verify() {
  return isValidPythonIdentifier(*this, getAttrName());
}

LogicalResult SetAttrOp::verify() {
  return isValidPythonIdentifier(*this, getAttrName());
}

LogicalResult ClassOp::verify() {
  if (failed(isValidPythonIdentifier(*this, getSymName()))) {
    return failure();
  }

  if (auto baseClasses = getBaseClasses()) {
    for (Attribute base : *baseClasses) {
      if (!isa<SymbolRefAttr, OpaqueAttr>(base)) {
        return emitOpError(
            "base_classes must be symbol refs or #emitpy.opaque");
      }
    }
  }

  int initCount = 0;
  Block &body = getBody().front();
  for (Operation &op : body.getOperations()) {
    if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
      StringRef methodKind = "instance";
      if (auto methodKindAttr =
              funcOp->getAttrOfType<StringAttr>("emitpy.method_kind")) {
        methodKind = methodKindAttr.getValue();
        if (methodKind != "instance" && methodKind != "staticmethod" &&
            methodKind != "classmethod") {
          return op.emitOpError("emitpy.method_kind must be one of "
                                "'instance', 'staticmethod', or 'classmethod'");
        }
      }

      if (funcOp.getName() == "__init__") {
        ++initCount;
        if (funcOp.getFunctionType().getNumResults() != 0) {
          return op.emitOpError("__init__ must not return a value");
        }
        if (methodKind != "instance") {
          return op.emitOpError("__init__ must be an instance method");
        }
      }

      unsigned numInputs = funcOp.getFunctionType().getNumInputs();
      if (methodKind == "instance" || methodKind == "classmethod" ||
          funcOp.getName() == "__init__") {
        if (numInputs < 1) {
          return op.emitOpError("instance and class methods must take a "
                                "receiver argument");
        }
        StringRef expectedReceiver =
            methodKind == "classmethod" ? "cls" : "self";
        if (auto argNameAttr =
                funcOp.getArgAttrOfType<StringAttr>(0, "emitpy.name")) {
          if (argNameAttr.getValue() != expectedReceiver) {
            return op.emitOpError() << "first argument must be named '"
                                    << expectedReceiver << "' via emitpy.name";
          }
        }
        if (methodKind != "classmethod") {
          auto classType = dyn_cast<ClassType>(funcOp.getArgument(0).getType());
          if (!classType) {
            return op.emitOpError("self argument must have !emitpy.class type");
          }
          if (classType.getName() != getSymName()) {
            return op.emitOpError() << "self type must match class name '"
                                    << getSymName() << "'";
          }
        }
      }
      continue;
    }

    // Allow EmitPy ops (e.g., class-level assignments/imports) in class body.
    if (op.getDialect() != nullptr &&
        op.getDialect()->getNamespace() ==
            getOperation()->getDialect()->getNamespace()) {
      continue;
    }

    return op.emitOpError("only emitpy or func operations are allowed in a "
                          "class body");
  }

  if (initCount > 1) {
    return emitOpError("class body must have at most one __init__");
  }

  return success();
}

void ClassOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printSymbolName(getSymName());
  if (auto baseClasses = getBaseClasses()) {
    if (!baseClasses->empty()) {
      p << "(";
      llvm::interleaveComma(*baseClasses, p, [&](Attribute baseAttr) {
        p.printAttribute(baseAttr);
      });
      p << ")";
    }
  }

  bool hasExtraAttrs = false;
  for (NamedAttribute attr : getOperation()->getAttrs()) {
    StringRef name = attr.getName();
    if (name == getSymNameAttrName() || name == "base_classes") {
      continue;
    }
    hasExtraAttrs = true;
    break;
  }
  if (hasExtraAttrs) {
    p << " attributes ";
    p.printOptionalAttrDict(
        getOperation()->getAttrs(),
        /*elidedAttrs=*/{getSymNameAttrName(), "base_classes"});
  }
  p << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

ParseResult ClassOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr)) {
    return failure();
  }
  result.addAttribute(mlir::SymbolTable::getSymbolAttrName(), nameAttr);

  if (succeeded(parser.parseOptionalLParen())) {
    SmallVector<Attribute> baseClasses;
    if (failed(parser.parseOptionalRParen())) {
      do {
        Attribute baseAttr;
        if (parser.parseAttribute(baseAttr)) {
          return failure();
        }
        baseClasses.push_back(baseAttr);
      } while (succeeded(parser.parseOptionalComma()));

      if (parser.parseRParen()) {
        return failure();
      }
    }
    if (!baseClasses.empty()) {
      result.addAttribute("base_classes",
                          ArrayAttr::get(parser.getContext(), baseClasses));
    }
  }

  if (succeeded(parser.parseOptionalKeyword("attributes"))) {
    if (parser.parseOptionalAttrDict(result.attributes)) {
      return failure();
    }
  }

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{})) {
    return failure();
  }
  if (body->empty()) {
    body->emplaceBlock();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// CallOpaqueOp
//===----------------------------------------------------------------------===//

LogicalResult CallOpaqueOp::verify() {
  if (getCallee().empty()) {
    return emitOpError("callee must not be empty");
  }

  if (getArgs() && getKeywordArgs() &&
      getArgs()->size() != getKeywordArgs()->size()) {
    return emitOpError("there must be a specified keyword argument string for "
                       "every argument; empty strings are allowed");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ImportOp
//===----------------------------------------------------------------------===//

void ImportOp::print(OpAsmPrinter &p) {
  StringAttr moduleName = getModuleNameAttr();
  p << " ";
  if (getImportAll()) {
    // Print 'from <moduleName> import *' case.
    p << "from " << moduleName << " import *";
  } else if (getMembersToImport()) {
    // Print 'from <moduleName> import <membersToImport> [as <memberAliases>]'
    // case.
    ArrayAttr membersToImport = *getMembersToImport();
    p << "from " << moduleName << " import " << membersToImport[0];
    ArrayAttr member_aliases = nullptr;
    if (getMemberAliases()) {
      member_aliases = *getMemberAliases();
      if (!dyn_cast<StringAttr>(member_aliases[0]).empty()) {
        p << " as " << member_aliases[0];
      }
    }
    for (size_t i = 1; i < membersToImport.size(); i++) {
      p << ", " << membersToImport[i];
      if (member_aliases && !dyn_cast<StringAttr>(member_aliases[i]).empty()) {
        p << " as " << member_aliases[i];
      }
    }
  } else {
    // Print 'import <moduleName> [as <moduleAlias>]' case.
    p << "import " << moduleName;
    if (getModuleAlias()) {
      p << " as " << getModuleAliasAttr();
    }
  }
}

ParseResult ImportOp::parse(::mlir::OpAsmParser &parser,
                            ::mlir::OperationState &result) {
  Builder &builder = parser.getBuilder();

  StringAttr moduleNameAttr;
  StringAttr moduleAliasAttr;
  UnitAttr importAllAttr;
  ArrayAttr membersToImportAttr;
  ArrayAttr memberAliasesAttr;

  if (succeeded(parser.parseOptionalKeyword("from"))) {
    if (parser.parseAttribute(moduleNameAttr)) {
      return parser.emitError(parser.getNameLoc())
             << "expected string attribute for module name";
    }
    result.addAttribute("module_name", moduleNameAttr);

    if (parser.parseOptionalKeyword("import")) {
      return parser.emitError(parser.getNameLoc())
             << "expected string literal 'import'";
    }

    if (succeeded(parser.parseOptionalStar())) {
      // Parse 'from <moduleName> import *' case.
      importAllAttr = builder.getUnitAttr();
      result.addAttribute("import_all", importAllAttr);

      if (succeeded(parser.parseOptionalKeyword("as"))) {
        return parser.emitError(
            parser.getNameLoc(),
            "unexpected 'as' keyword in 'from ... import *' form");
      }
    } else {
      // Parse 'from <moduleName> import <membersToImport> [as <memberAliases>]'
      // case.
      StringAttr member;
      StringAttr member_alias = builder.getStringAttr("");
      SmallVector<StringRef> members;
      SmallVector<StringRef> member_aliases;

      if (!parser.parseOptionalAttribute(member).has_value()) {
        return parser.emitError(parser.getNameLoc())
               << "expected string attribute for member name";
      }
      if (succeeded(parser.parseOptionalKeyword("as"))) {
        if (!parser.parseOptionalAttribute(member_alias).has_value()) {
          return parser.emitError(parser.getNameLoc())
                 << "expected string attribute for alias";
        }
      }
      members.push_back(member.getValue());
      member_aliases.push_back(member_alias.getValue());

      while (succeeded(parser.parseOptionalComma())) {
        if (!parser.parseOptionalAttribute(member).has_value()) {
          return parser.emitError(parser.getNameLoc())
                 << "expected string attribute for member name";
        }
        member_alias = builder.getStringAttr("");
        if (succeeded(parser.parseOptionalKeyword("as"))) {
          if (!parser.parseOptionalAttribute(member_alias).has_value()) {
            return parser.emitError(parser.getNameLoc())
                   << "expected string attribute for alias";
          }
        }
        members.push_back(member.getValue());
        member_aliases.push_back(member_alias.getValue());
      }

      membersToImportAttr = builder.getStrArrayAttr(members);
      result.addAttribute("members_to_import", membersToImportAttr);
      memberAliasesAttr = builder.getStrArrayAttr(member_aliases);
      result.addAttribute("member_aliases", memberAliasesAttr);
    }
  } else {
    // Parse 'import <moduleName> [as <moduleAlias>]' case.
    if (parser.parseKeyword("import")) {
      return parser.emitError(parser.getNameLoc())
             << "expected string literal 'import'";
    }

    if (parser.parseAttribute(moduleNameAttr)) {
      return parser.emitError(parser.getNameLoc())
             << "expected string attribute for module name";
    }
    result.addAttribute("module_name", moduleNameAttr);

    if (succeeded(parser.parseOptionalKeyword("as"))) {
      if (parser.parseAttribute(moduleAliasAttr)) {
        return parser.emitError(parser.getNameLoc())
               << "expected string attribute for module alias";
      }
      result.addAttribute("module_alias", moduleAliasAttr);
    }
  }

  return success();
}

LogicalResult ImportOp::verify() {
  StringRef moduleName = getModuleName();
  ::std::optional<::llvm::StringRef> moduleAlias = getModuleAlias();
  ::std::optional<::mlir::ArrayAttr> membersToImport = getMembersToImport();
  ::std::optional<::mlir::ArrayAttr> memberAliases = getMemberAliases();
  ::std::optional<bool> importAll = getImportAll();

  // <moduleName> must be non-empty.
  if (moduleName.empty()) {
    return emitOpError("module name attribute must be non-empty");
  }

  bool hasModuleAlias = moduleAlias.has_value();
  bool hasMembersToImport = membersToImport.has_value();
  bool hasMemberAliases = memberAliases.has_value();
  bool hasImportAll = importAll.has_value();

  // Verify 'from <moduleName> import *' case.
  if (hasImportAll) {
    if (hasModuleAlias) {
      return emitOpError("cannot specify module alias with *");
    }
    if (hasMembersToImport) {
      return emitOpError("cannot specify members to import with *");
    }
    if (hasMemberAliases) {
      return emitOpError("cannot specify members' aliases with *");
    }
    return success();
  }

  // Verify 'import <moduleName> as <moduleAlias>' case.
  if (hasModuleAlias) {
    if (hasMembersToImport) {
      return emitOpError("cannot specify members to import with module alias");
    }
    if (hasMemberAliases) {
      return emitOpError("cannot specify members' aliases with module alias");
    }
    return success();
  }

  // Verify 'from <moduleName> import <membersToImport> [as <memberAliases>]'
  // case.
  if (hasMembersToImport) {
    // Check individual members' names are not empty.
    for (Attribute member : *membersToImport) {
      StringAttr memberName = dyn_cast<StringAttr>(member);
      if (memberName.empty()) {
        return emitOpError("imported member name cannot be empty");
      }
    }

    // If <memberAliases> are provided, their count must be equal to
    // <membersToImport> count.
    if (hasMemberAliases) {
      if (membersToImport->size() != memberAliases->size()) {
        return emitOpError("the number of members' aliases must be equal to "
                           "the number of members to import; empty string is "
                           "considered valid member alias");
      }
    }
    return success();
  }

  // Verify 'import <moduleName>' case.
  return success();
}

//===----------------------------------------------------------------------===//
// LiteralOp
//===----------------------------------------------------------------------===//

/// The literal op requires a non-empty value.
LogicalResult LiteralOp::verify() {
  if (getValue().empty()) {
    return emitOpError() << "value must not be empty";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// VerbatimOp
//===----------------------------------------------------------------------===//

LogicalResult VerbatimOp::verify() {
  auto errorCallback = [&]() -> InFlightDiagnostic {
    return this->emitOpError();
  };
  FailureOr<SmallVector<ReplacementItem>> fmt =
      ::parseFormatString(getValue(), getFmtArgs(), errorCallback);
  if (failed(fmt)) {
    return failure();
  }
  size_t numPlaceholders = llvm::count_if(*fmt, [](ReplacementItem &item) {
    return std::holds_alternative<Placeholder>(item);
  });

  if (numPlaceholders != getFmtArgs().size()) {
    return emitOpError()
           << "requires operands for each placeholder in the format string";
  }
  return success();
}

FailureOr<SmallVector<ReplacementItem>> VerbatimOp::parseFormatString() {
  // Error checking is done in verify.
  return ::parseFormatString(getValue(), getFmtArgs());
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

LogicalResult ConstantOp::verify() {
  Attribute value = getValueAttr();
  if (failed(verifyInitializationAttribute(getOperation(), value))) {
    return failure();
  }
  if (auto opaqueValue = llvm::dyn_cast<emitpy::OpaqueAttr>(value)) {
    if (opaqueValue.getValue().empty()) {
      return emitOpError() << "value must not be empty";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

static void printEmitPyGlobalOpInitialValue(OpAsmPrinter &p, GlobalOp op,
                                            Attribute initialValue) {
  p << "= ";
  p.printAttribute(initialValue);
}

static ParseResult parseEmitPyGlobalOpInitialValue(OpAsmParser &parser,
                                                   Attribute &initialValue) {
  if (parser.parseEqual()) {
    return parser.emitError(parser.getNameLoc(),
                            "expected '=' after symbol name");
  }

  if (parser.parseAttribute(initialValue)) {
    return parser.emitError(parser.getNameLoc(),
                            "expected initial value for global variable");
  }

  return success();
}

LogicalResult GlobalOp::verify() {
  Attribute value = getInitialValue();
  if (!value) {
    return emitOpError() << "requires initial value for global variable";
  }

  StringRef name = getSymName();
  return isValidPythonIdentifier(getOperation(), name);
}

//===----------------------------------------------------------------------===//
// AssignGlobalOp
//===----------------------------------------------------------------------===//

void AssignGlobalOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printAttributeWithoutType(getNameAttr());
  p << " = " << getValue() << " : " << getValue().getType();
}

ParseResult AssignGlobalOp::parse(::mlir::OpAsmParser &parser,
                                  ::mlir::OperationState &result) {

  StringAttr symName;
  if (parser.parseSymbolName(symName)) {
    return parser.emitError(parser.getNameLoc(), "expected symbol name");
  }
  FlatSymbolRefAttr nameAttr =
      FlatSymbolRefAttr::get(parser.getContext(), symName);
  result.addAttribute("name", nameAttr);

  if (parser.parseEqual()) {
    return parser.emitError(parser.getNameLoc(),
                            "expected '=' after symbol name");
  }

  OpAsmParser::UnresolvedOperand initialValue;
  Type valueType;
  if (parser.parseOperand(initialValue) || parser.parseColonType(valueType) ||
      parser.resolveOperand(initialValue, valueType, result.operands)) {
    return parser.emitError(parser.getNameLoc(),
                            "expected initial value for global variable");
  }
  return success();
}

LogicalResult AssignGlobalOp::verify() {
  StringRef name = getName();
  return isValidPythonIdentifier(getOperation(), name);
}

LogicalResult
AssignGlobalOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto global =
      symbolTable.lookupNearestSymbolFrom<GlobalOp>(*this, getNameAttr());
  if (!global) {
    return emitOpError("'")
           << getName() << "' does not reference a valid emitpy.global";
  }

  Type valueType = getValue().getType();
  Attribute initialValue = global.getInitialValue();

  // If the global has a typed attribute, verify the types match
  if (auto typedAttr = llvm::dyn_cast<TypedAttr>(initialValue)) {
    Type globalType = typedAttr.getType();
    if (valueType != globalType) {
      return emitOpError() << "value type (" << valueType
                           << ") does not match global's type (" << globalType
                           << ")";
    }
  }
  // For opaque attributes, we allow any type since the type is not specified
  // in the attribute itself

  return success();
}

//===----------------------------------------------------------------------===//
// GlobalStatementOp
//===----------------------------------------------------------------------===//

LogicalResult GlobalStatementOp::verify() {
  StringRef name = getName();
  return isValidPythonIdentifier(getOperation(), name);
}

LogicalResult
GlobalStatementOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyNearestGlobalSymbol<GlobalStatementOp>(*this, symbolTable);
}

//===----------------------------------------------------------------------===//
// CreateDictOp
//===----------------------------------------------------------------------===//

LogicalResult CreateDictOp::verify() {
  StringRef dictName = getDictName();
  if (failed(isValidPythonIdentifier(getOperation(), dictName))) {
    return emitOpError() << "dictionary name must be a valid Python identifier";
  }

  if (getLiteralExpr() && !getItems().empty()) {
    return emitOpError(
        "cannot have both literal_expr and items operands; use either "
        "literal_expr for Python dict literals or items for key-value pairs");
  }

  if (!getLiteralExpr() && getItems().size() % 2 != 0) {
    return emitOpError(
        "items must be alternating key-value pairs (even count required)");
  }

  if (getLiteralExpr() && getLiteralExpr()->empty()) {
    return emitOpError("literal_expr must not be empty");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SetValueForDictKeyOp
//===----------------------------------------------------------------------===//

LogicalResult SetValueForDictKeyOp::verify() {
  Type keyType = getKey().getType();

  // If key is an opaque type, verify it represents a string
  if (auto opaqueType = dyn_cast<OpaqueType>(keyType)) {
    StringRef value = opaqueType.getValue();
    if (value != "str") {
      return emitOpError()
             << "key with opaque type must represent a string type "
             << "(!emitpy.opaque<\"str\">), but got: " << opaqueType;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// GetValueForDictKeyOp
//===----------------------------------------------------------------------===//

LogicalResult GetValueForDictKeyOp::verify() {
  Type keyType = getKey().getType();

  // If key is an opaque type, verify it represents a string
  if (auto opaqueType = dyn_cast<OpaqueType>(keyType)) {
    StringRef value = opaqueType.getValue();
    if (value != "str") {
      return emitOpError()
             << "key with opaque type must represent a string type "
             << "(!emitpy.opaque<\"str\">), but got: " << opaqueType;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ExpressionOp
//===----------------------------------------------------------------------===//

LogicalResult ExpressionOp::verify() {
  Type resultType = getResult().getType();
  Region &region = getRegion();

  Block &body = region.front();

  // Ensure the body has a terminator
  if (!body.mightHaveTerminator()) {
    return emitOpError("must yield a value at termination");
  }

  // Ensure the terminator is a yield op
  auto yield = cast<YieldOp>(body.getTerminator());
  Value yieldResult = yield.getResult();

  // Ensure the yield result is valid
  if (!yieldResult) {
    return emitOpError("must yield a value at termination");
  }

  Operation *rootOp = yieldResult.getDefiningOp();

  // Ensure the yield result is defined within the expression
  if (!rootOp) {
    return emitOpError("yielded value has no defining op");
  }

  // Check if the rootOp is in a block (required for getParentOp())
  if (!rootOp->getBlock()) {
    return emitOpError("yielded value's defining op is not in a block");
  }

  // Ensure the yield result op is defined within the expression
  if (rootOp->getParentOp() != getOperation()) {
    return emitOpError("yielded value not defined within expression");
  }

  // Ensure the yielded type matches the expression return type
  Type yieldType = yieldResult.getType();

  if (resultType != yieldType) {
    return emitOpError("requires yielded type to match return type");
  }

  for (Operation &op : region.front().without_terminator()) {
    auto expressionInterface = dyn_cast<PyExpressionInterface>(op);
    // Ensure each operation implements the expression interface
    if (!expressionInterface) {
      return emitOpError("contains an unsupported operation");
    }
    // Ensure each operation has exactly one result
    if (op.getNumResults() != 1) {
      return emitOpError("requires exactly one result for each operation");
    }
    Value result = op.getResult(0);
    // Ensure each operation's result is used at least once
    if (result.use_empty()) {
      return emitOpError("contains an unused operation");
    }
  }

  // Make sure any operation with side effect is only reachable once from
  // the root op, otherwise emission will be replicating side effects.
  SmallPtrSet<Operation *, 16> visited;
  SmallVector<Operation *> worklist;
  worklist.push_back(rootOp);
  while (!worklist.empty()) {
    Operation *op = worklist.back();
    worklist.pop_back();
    if (visited.contains(op)) {
      if (cast<PyExpressionInterface>(op).hasSideEffects()) {
        return emitOpError(
            "requires exactly one use for operations with side effects");
      }
    }
    visited.insert(op);
    for (Value operand : op->getOperands()) {
      if (Operation *def = operand.getDefiningOp()) {
        worklist.push_back(def);
      }
    }
  }

  return success();
}

void ExpressionOp::print(OpAsmPrinter &p) {
  p << " ";
  if (!getOperands().empty()) {
    p << "(";
    llvm::interleaveComma(getOperands(), p);
    p << ") ";
  }

  if (getDoNotInline()) {
    p << "{do_not_inline} ";
  }

  p << ": ";
  if (!getOperands().empty()) {
    p << "(";
    llvm::interleaveComma(getOperands().getTypes(), p);
    p << ") ";
  }
  p << "-> " << getResult().getType() << " ";

  p.printRegion(getBody(), /*printEntryBlockArgs=*/true,
                /*printBlockTerminators=*/true);
}

ParseResult ExpressionOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse operands
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SmallVector<Type> operandTypes;

  if (succeeded(parser.parseOptionalLParen())) {
    if (parser.parseOperandList(operands) || parser.parseRParen()) {
      return failure();
    }
  }

  // Parse optional do_not_inline attribute
  if (succeeded(parser.parseOptionalLBrace())) {
    if (parser.parseKeyword("do_not_inline") || parser.parseRBrace()) {
      return failure();
    }
    result.addAttribute("do_not_inline", parser.getBuilder().getUnitAttr());
  }

  // Parse types
  if (parser.parseColon()) {
    return failure();
  }

  if (!operands.empty()) {
    if (parser.parseLParen() || parser.parseTypeList(operandTypes) ||
        parser.parseRParen()) {
      return failure();
    }
  }

  Type resultType;
  if (parser.parseArrow() || parser.parseType(resultType)) {
    return failure();
  }

  result.addTypes(resultType);

  // Parse region
  Region *body = result.addRegion();
  SmallVector<OpAsmParser::Argument> regionArgs;
  for (auto type : operandTypes) {
    OpAsmParser::Argument arg;
    arg.type = type;
    regionArgs.push_back(arg);
  }

  if (parser.parseRegion(*body, regionArgs)) {
    return failure();
  }

  // Resolve operands
  if (parser.resolveOperands(operands, operandTypes, parser.getNameLoc(),
                             result.operands)) {
    return failure();
  }

  return success();
}

#define GET_OP_CLASSES
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.cpp.inc"
