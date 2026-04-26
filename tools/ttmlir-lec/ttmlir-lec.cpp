// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// ttmlir-lec: logical equivalence checker for TTIR functions.
//
// Usage:
//   ttmlir-lec a.mlir b.mlir -c1=foo -c2=bar [--c1-is-smt] [--z3=path]
//
// Default mode: both inputs are TTIR func.func ops; the tool runs TTIRToSMT,
// constructs an LEC (smt.solver), exports SMT-LIB, and invokes z3 to check.
// On non-equivalence, prints the counterexample input/output values.

#include "ttmlir/Conversion/ConstructTTIRLEC/ConstructTTIRLEC.h"
#include "ttmlir/Conversion/TTIRPruneToOutput/TTIRPruneToOutput.h"
#include "ttmlir/Conversion/TTIRToSMT/TTIRToSMT.h"
#include "ttmlir/RegisterAll.h"

#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SMT/IR/SMTDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/SMTLIB/ExportSMTLIB.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <string>
#include <unistd.h>

namespace cl = llvm::cl;

using namespace mlir;
using namespace mlir::tt;

static cl::OptionCategory mainCategory("ttmlir-lec Options");

static cl::opt<std::string>
    firstFunc("c1", cl::Required,
              cl::desc("Name of the first function to compare"),
              cl::value_desc("function name"), cl::cat(mainCategory));

static cl::opt<std::string>
    secondFunc("c2", cl::Required,
               cl::desc("Name of the second function to compare"),
               cl::value_desc("function name"), cl::cat(mainCategory));

static cl::list<std::string> inputFilenames(cl::Positional, cl::OneOrMore,
                                            cl::desc("<input MLIR files>"),
                                            cl::cat(mainCategory));

static cl::opt<std::string>
    outputFilename("o", cl::desc("Output filename"), cl::value_desc("filename"),
                   cl::init("-"), cl::cat(mainCategory));

enum OutputFormat { OutputResult, OutputMLIR, OutputSMTLIB };

static cl::opt<OutputFormat> outputFormat(
    cl::desc("Specify output format"),
    cl::values(
        clEnumValN(OutputResult, "run",
                   "Invoke z3 and print equivalent/non-equivalent result"),
        clEnumValN(OutputMLIR, "emit-mlir", "Emit MLIR with SMT dialect"),
        clEnumValN(OutputSMTLIB, "emit-smtlib", "Emit SMT-LIB script")),
    cl::init(OutputResult), cl::cat(mainCategory));

static cl::opt<std::string>
    z3Path("z3", cl::desc("Path to z3 binary (default: 'z3' from PATH)"),
           cl::init("z3"), cl::cat(mainCategory));

static cl::opt<bool>
    showModel("show-model",
              cl::desc("Print full Z3 model when non-equivalent"),
              cl::init(true), cl::cat(mainCategory));

static cl::opt<unsigned>
    z3Timeout("z3-timeout",
              cl::desc("Z3 timeout in milliseconds (0 = no timeout)"),
              cl::init(0), cl::cat(mainCategory));

static cl::opt<bool> setLogicQFBV(
    "set-logic-qfbv",
    cl::desc("Emit (set-logic QF_BV) — speeds up Z3 on bitvector-only problems"),
    cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> setLogicQFABV(
    "set-logic-qfabv",
    cl::desc("Emit (set-logic QF_ABV) — bitvectors + arrays"),
    cl::init(false), cl::cat(mainCategory));

static cl::opt<std::string> checkOutput(
    "check-output",
    cl::desc("Compare only the output port with this hw.port_name "
             "(default: compare all)"),
    cl::init(""), cl::cat(mainCategory));

static cl::opt<int> checkOutputIdx(
    "check-output-idx",
    cl::desc("Compare only the output at this index (overrides check-output)"),
    cl::init(-1), cl::cat(mainCategory));

//===----------------------------------------------------------------------===//
// Module merging: combine two input modules into one.
//===----------------------------------------------------------------------===//

/// Merge `src` into `dest`. If any symbol in `src` collides with one in
/// `dest`, rename the source symbol to a unique name. If a protected-name
/// (LEC target) collides, the source's copy is dropped — the destination
/// (typically the SMT reference) takes precedence.
static LogicalResult
mergeModule(ModuleOp dest, OwningOpRef<ModuleOp> src,
            ArrayRef<StringRef> protectedNames) {
  SymbolTable destTable(dest), srcTable(*src);
  SmallVector<Operation *> toErase;
  for (auto &op : src->getOps()) {
    auto symbol = dyn_cast<SymbolOpInterface>(op);
    if (!symbol)
      continue;
    StringAttr origName = symbol.getNameAttr();
    Operation *clash = destTable.lookup(origName);
    if (!clash)
      continue; // no collision

    bool isProtected = llvm::is_contained(protectedNames, origName.getValue());
    if (isProtected) {
      // Both modules have a protected symbol of this name. Drop the source
      // copy so the destination (e.g. the SMT reference) wins.
      toErase.push_back(&op);
      continue;
    }
    // Rename to avoid collision.
    if (failed(srcTable.renameToUnique(&op, {&destTable})))
      return src->emitError() << "failed to rename '" << origName.getValue()
                              << "'";
  }
  for (Operation *op : toErase)
    op->erase();

  Block *destBlock = dest.getBody();
  Block *srcBlock = src->getBody();
  destBlock->getOperations().splice(destBlock->end(),
                                    srcBlock->getOperations());
  return success();
}

//===----------------------------------------------------------------------===//
// Main flow
//===----------------------------------------------------------------------===//

static LogicalResult runLEC(MLIRContext &context) {
  if (inputFilenames.size() < 1 || inputFilenames.size() > 2) {
    llvm::errs() << "expected 1 or 2 input files\n";
    return failure();
  }

  // Parse first input file as the base module.
  auto module = parseSourceFile<ModuleOp>(inputFilenames[0], &context);
  if (!module)
    return failure();

  // If a second input file is provided, parse and merge it.
  if (inputFilenames.size() == 2) {
    auto second = parseSourceFile<ModuleOp>(inputFilenames[1], &context);
    if (!second)
      return failure();
    SmallVector<StringRef> protectedNames = {firstFunc, secondFunc};
    if (failed(mergeModule(module.get(), std::move(second), protectedNames)))
      return failure();
  }

  // Run conversion pipeline: optionally prune to one output, TTIR -> SMT,
  // then construct LEC. Pruning before lowering means we never have to
  // lower ops that don't feed the selected output.
  PassManager pm(&context);
  if (failed(applyPassManagerCLOptions(pm)))
    return failure();
  if (!checkOutput.empty()) {
    TTIRPruneToOutputOptions pruneOpts;
    pruneOpts.keepOutput = checkOutput;
    pm.addPass(createTTIRPruneToOutputPass(pruneOpts));
    pm.addPass(createCanonicalizerPass());
    // Drop arguments that are now unused. This is critical for LEC because
    // both circuits must have matching input signatures, and the HW->SMT
    // reference may carry multi-dim array inputs that don't appear in the
    // TTIR side's cone-of-influence for the selected output.
    pm.addPass(createFuncDropUnusedArgsPass());
  }
  pm.addPass(createConvertTTIRToSMTPass());
  ConstructTTIRLECOptions opts;
  opts.firstFunc = firstFunc;
  opts.secondFunc = secondFunc;
  opts.checkOutput = checkOutput;
  opts.checkOutputIdx = checkOutputIdx;
  pm.addPass(createConstructTTIRLECPass(opts));

  if (failed(pm.run(module.get())))
    return failure();

  // Open output file.
  std::string errorMessage;
  auto outputFile = openOutputFile(outputFilename, &errorMessage);
  if (!outputFile) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  if (outputFormat == OutputMLIR) {
    module->print(outputFile->os());
    outputFile->keep();
    return success();
  }

  // Export SMT-LIB.
  std::string smtlib;
  llvm::raw_string_ostream smtlibStream(smtlib);
  if (failed(smt::exportSMTLIB(module.get(), smtlibStream)))
    return failure();

  if (outputFormat == OutputSMTLIB) {
    outputFile->os() << smtlib;
    outputFile->keep();
    return success();
  }

  // OutputResult: invoke z3 on the SMT-LIB.
  // Write a temp SMT-LIB file with model extraction enabled.
  llvm::SmallString<128> tmpPath;
  int tmpFd;
  if (auto err = llvm::sys::fs::createTemporaryFile("ttmlir_lec", "smt2", tmpFd,
                                                    tmpPath)) {
    llvm::errs() << "failed to create temp file: " << err.message() << "\n";
    return failure();
  }
  {
    llvm::raw_fd_ostream tmpOut(tmpFd, /*shouldClose=*/true);
    if (setLogicQFABV)
      tmpOut << "(set-logic QF_ABV)\n";
    else if (setLogicQFBV)
      tmpOut << "(set-logic QF_BV)\n";
    if (z3Timeout > 0)
      tmpOut << "(set-option :timeout " << z3Timeout << ")\n";
    tmpOut << "(set-option :produce-models true)\n";
    // Splice in (get-model) right after (check-sat).
    size_t checkPos = smtlib.find("(check-sat)");
    if (checkPos == std::string::npos) {
      tmpOut << smtlib;
    } else {
      tmpOut << smtlib.substr(0, checkPos);
      tmpOut << "(check-sat)\n(get-model)\n";
      // Skip past "(check-sat)\n"
      size_t resumePos = checkPos + std::string("(check-sat)").size();
      if (resumePos < smtlib.size() && smtlib[resumePos] == '\n')
        ++resumePos;
      tmpOut << smtlib.substr(resumePos);
    }
  }

  // Invoke z3.
  llvm::SmallString<128> z3OutPath;
  int z3OutFd;
  if (auto err = llvm::sys::fs::createTemporaryFile("ttmlir_lec_out", "txt",
                                                    z3OutFd, z3OutPath)) {
    llvm::errs() << "failed to create temp output: " << err.message() << "\n";
    return failure();
  }
  ::close(z3OutFd);

  // Build z3 args. Pass -T:<seconds> as a CLI-level timeout in addition to the
  // SMT-LIB :timeout option, since some Z3 versions only honor one or the
  // other.
  std::string z3TimeoutArg;
  std::vector<llvm::StringRef> args = {z3Path};
  if (z3Timeout > 0) {
    unsigned secs = (z3Timeout + 999) / 1000;
    z3TimeoutArg = "-T:" + std::to_string(secs);
    args.push_back(z3TimeoutArg);
  }
  args.push_back(tmpPath.str());
  std::optional<llvm::StringRef> redirects[3] = {
      std::nullopt, llvm::StringRef(z3OutPath), std::nullopt};
  std::string execErr;
  int rc = llvm::sys::ExecuteAndWait(z3Path, args, /*Env=*/std::nullopt,
                                     redirects, /*SecondsToWait=*/0,
                                     /*MemoryLimit=*/0, &execErr);
  // Read z3 output.
  auto bufferOrErr = llvm::MemoryBuffer::getFile(z3OutPath);
  if (rc < 0 || !bufferOrErr) {
    llvm::errs() << "failed to invoke z3 (" << z3Path << "): " << execErr
                 << "\n";
    if (bufferOrErr)
      llvm::errs() << (*bufferOrErr)->getBuffer();
    llvm::sys::fs::remove(tmpPath);
    llvm::sys::fs::remove(z3OutPath);
    return failure();
  }

  StringRef z3Output = (*bufferOrErr)->getBuffer();
  llvm::sys::fs::remove(tmpPath);
  llvm::sys::fs::remove(z3OutPath);

  // Parse the first line to determine equivalence.
  StringRef firstLine = z3Output.split('\n').first.trim();
  if (firstLine == "unsat") {
    outputFile->os() << "EQUIVALENT (c1 == c2)\n";
    outputFile->keep();
    return success();
  }
  if (firstLine == "sat") {
    outputFile->os() << "NON-EQUIVALENT (c1 != c2)\n";
    if (showModel) {
      outputFile->os() << "Counterexample:\n";
      // Print everything after the first line.
      size_t nl = z3Output.find('\n');
      if (nl != StringRef::npos)
        outputFile->os() << z3Output.substr(nl + 1);
    }
    outputFile->keep();
    return success();
  }

  if (firstLine == "unknown" || firstLine == "timeout") {
    outputFile->os() << "TIMEOUT or UNKNOWN (z3 could not decide)\n";
    outputFile->keep();
    return failure();
  }

  outputFile->os() << "z3 returned UNKNOWN or error:\n" << z3Output;
  outputFile->keep();
  return failure();
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerAsmPrinterCLOptions();
  cl::ParseCommandLineOptions(
      argc, argv,
      "ttmlir-lec - logical equivalence checker for TTIR functions\n");

  DialectRegistry registry;
  mlir::tt::registerAllDialects(registry);
  mlir::tt::registerAllExtensions(registry);
  registry.insert<mlir::smt::SMTDialect>();
  MLIRContext context(registry);
  context.allowUnregisteredDialects(false);

  return failed(runLEC(context));
}
