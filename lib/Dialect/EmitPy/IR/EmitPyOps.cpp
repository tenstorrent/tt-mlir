// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

#include <cstddef>

#define GET_OP_CLASSES
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.cpp.inc"

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

//===----------------------------------------------------------------------===//
// CallOpaqueOp
//===----------------------------------------------------------------------===//

LogicalResult CallOpaqueOp::verify() {
  if (getCallee().empty()) {
    return emitOpError("callee must not be empty");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ImportOp
//===----------------------------------------------------------------------===//

void ImportOp::print(OpAsmPrinter &p) {
  auto moduleName = getModuleName();
  p << " ";
  if (getImportAll()) {
    p << "from "
      << "\"" << moduleName << "\""
      << " import *";
  } else if (getMembersToImport()) {
    auto membersToImport = *getMembersToImport();
    p << "from "
      << "\"" << moduleName << "\""
      << " import ";
    p << "\"" << membersToImport[0] << "\"";
    ArrayAttr member_aliases = nullptr;
    if (getMemberAliases()) {
      member_aliases = *getMemberAliases();
      p << " as "
        << "\"" << member_aliases[0] << "\"";
    }
    for (size_t i = 1; i < membersToImport.size(); i++) {
      p << ", "
        << "\"" << membersToImport[i] << "\"";
      if (!member_aliases.empty() && i < member_aliases.size()) {
        p << " as "
          << "\"" << member_aliases[i] << "\"";
      }
    }
  } else {
    p << "import "
      << "\"" << moduleName << "\"";
    if (getModuleAlias()) {
      p << " as "
        << "\"" << *getModuleAlias() << "\"";
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
      // This is an 'from <moduleName> import *' form
      importAllAttr = builder.getUnitAttr();
      result.addAttribute("import_all", importAllAttr);

      if (succeeded(parser.parseOptionalKeyword("as"))) {
        return parser.emitError(
            parser.getNameLoc(),
            "unexpected 'as' keyword in 'from ... import *' form");
      }
    } else {
      // This is an 'from <moduleName> import <membersToImport> [as <alias>]'
      // form
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
    // This is an 'import <moduleName> [as <alias>]' form
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
