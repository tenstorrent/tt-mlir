// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h"

#include "ttmlir/Dialect/EmitPy/IR/EmitPyInterfaces.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/Support/LogicalResult.h>

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
    return op->emitOpError("'")
           << op.getName() << "' does not reference a valid emitpy.global";
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
  p.printAttributeWithoutType(initialValue);
}

static ParseResult parseEmitPyGlobalOpInitialValue(OpAsmParser &parser,
                                                   Attribute &initialValue) {
  if (parser.parseEqual()) {
    return failure();
  }

  if (!parser.parseAttribute(initialValue)) {
    return failure();
  }

  return success();
}

static bool isValidPythonIdentifier(StringRef name) {
  if (name.empty()) {
    return false;
  }

  static constexpr const char *pythonKeywords[] = {
      "False",  "None",   "True",    "and",      "as",       "assert", "async",
      "await",  "break",  "class",   "continue", "def",      "del",    "elif",
      "else",   "except", "finally", "for",      "from",     "global", "if",
      "import", "in",     "is",      "lambda",   "nonlocal", "not",    "or",
      "pass",   "raise",  "return",  "try",      "while",    "with",   "yield"};

  for (const char *keyword : pythonKeywords) {
    if (name == keyword) {
      return false;
    }
  }

  char first = name[0];
  if (!((first >= 'a' && first <= 'z') || (first >= 'A' && first <= 'Z') ||
        first == '_')) {
    return false;
  }

  for (char c : name) {
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
          (c >= '0' && c <= '9') || c == '_')) {
      return false;
    }
  }

  return true;
}

LogicalResult GlobalOp::verify() {
  Attribute value = getInitialValue();
  if (!value) {
    return failure();
  }

  StringRef name = getName();
  if (!isValidPythonIdentifier(name)) {
    return emitOpError()
           << "symbol name '" << name
           << "' is not a valid Python identifier (must start with letter or "
              "underscore, contain only letters, digits, and underscores, and "
              "not be a Python keyword)";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// GetGlobalOp
//===----------------------------------------------------------------------===//

LogicalResult
GetGlobalOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto global =
      symbolTable.lookupNearestSymbolFrom<GlobalOp>(*this, getNameAttr());
  if (!global) {
    return emitOpError("'")
           << getName() << "' does not reference a valid emitpy.global ";
  }

  return success();
}

#define GET_OP_CLASSES
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.cpp.inc"
