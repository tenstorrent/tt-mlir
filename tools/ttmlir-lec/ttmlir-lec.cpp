// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// ttmlir-lec: logical equivalence checker for TTIR functions.
//
// Usage:
//   ttmlir-lec a.mlir b.mlir -c1=foo -c2=bar
//   ttmlir-lec a.mlir b.mlir -c1=foo -c2=bar --shared-libs=/path/to/z3
//   ttmlir-lec a.mlir b.mlir -c1=foo -c2=bar --emit-smtlib -o out.smt2
//
// Default mode: both inputs are TTIR (`func.func` ops on `tensor<NxiM>`).
// The tool merges the two modules, runs the `convert-ttir-to-smt` pass,
// constructs an LEC (`smt.solver`), exports SMT-LIB, and invokes an SMT
// solver to check `(check-sat)`. If the result is `unsat` the two functions
// are equivalent; if `sat` the model is printed as a counterexample.
//
// One or both inputs may already be in the SMT dialect (e.g. produced by
// `circt-opt --convert-hw-to-smt --convert-comb-to-smt`). The `convert-
// ttir-to-smt` pass is a no-op on already-SMT functions, so mixed inputs
// work without any extra flag.
//
// The SMT solver is invoked as a subprocess. By default the tool searches
// `PATH` for a binary named `z3`. Pass `--shared-libs=/path/to/solver` to
// override (the flag accepts a comma-separated list; the first entry is
// used as the solver binary). The solver is expected to accept a single
// SMT-LIB script on the command line and print `sat`/`unsat`/`unknown`
// followed by a model when applicable. This is the standard SMT-LIB CLI
// contract, so any compatible solver (z3, cvc5, ...) works.

#include "ttmlir/Conversion/ConstructTTIRLEC/ConstructTTIRLEC.h"
#include "ttmlir/Conversion/TTIRPruneToOutput/TTIRPruneToOutput.h"
#include "ttmlir/Conversion/TTIRToSMT/TTIRToSMT.h"
#include "ttmlir/RegisterAll.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SMT/IR/SMTDialect.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/SMTLIB/ExportSMTLIB.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DynamicLibrary.h"
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

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
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

static cl::list<std::string> sharedLibs(
    "shared-libs",
    cl::desc("Path(s) to the SMT solver binary. The first entry is used "
             "as the solver; defaults to 'z3' on PATH if omitted. "
             "Comma-separated."),
    cl::value_desc("path"), cl::MiscFlags::CommaSeparated,
    cl::cat(mainCategory));

static cl::opt<bool>
    showModel("show-model",
              cl::desc("Print full solver model when non-equivalent"),
              cl::init(true), cl::cat(mainCategory));

static cl::opt<unsigned>
    solverTimeout("solver-timeout",
                  cl::desc("Solver timeout in milliseconds (0 = no timeout)"),
                  cl::init(0), cl::cat(mainCategory));

static cl::opt<bool> setLogicQFBV(
    "set-logic-qfbv",
    cl::desc("Emit (set-logic QF_BV) — speeds up bitvector-only problems"),
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
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

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
    if (!symbol) {
      continue;
    }
    StringAttr origName = symbol.getNameAttr();
    Operation *clash = destTable.lookup(origName);
    if (!clash) {
      continue; // no collision
    }

    bool isProtected = llvm::is_contained(protectedNames, origName.getValue());
    if (isProtected) {
      // Both modules have a protected symbol of this name. Drop the source
      // copy so the destination (e.g. the SMT reference) wins.
      toErase.push_back(&op);
      continue;
    }
    // Rename to avoid collision.
    if (failed(srcTable.renameToUnique(&op, {&destTable}))) {
      return src->emitError() << "failed to rename '" << origName.getValue()
                              << "'";
    }
  }
  for (Operation *op : toErase) {
    op->erase();
  }

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
  if (!module) {
    return failure();
  }

  // If a second input file is provided, parse and merge it.
  if (inputFilenames.size() == 2) {
    auto second = parseSourceFile<ModuleOp>(inputFilenames[1], &context);
    if (!second) {
      return failure();
    }
    SmallVector<StringRef> protectedNames = {firstFunc, secondFunc};
    if (failed(mergeModule(module.get(), std::move(second), protectedNames))) {
      return failure();
    }
  }

  // Run conversion pipeline: optionally prune to one output, TTIR -> SMT,
  // then construct LEC. Pruning before lowering means we never have to
  // lower ops that don't feed the selected output.
  PassManager pm(&context);
  if (failed(applyPassManagerCLOptions(pm))) {
    return failure();
  }
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

  if (failed(pm.run(module.get()))) {
    return failure();
  }

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
  if (failed(smt::exportSMTLIB(module.get(), smtlibStream))) {
    return failure();
  }

  if (outputFormat == OutputSMTLIB) {
    outputFile->os() << smtlib;
    outputFile->keep();
    return success();
  }

  // OutputResult: invoke the solver on the SMT-LIB.
  //
  // Resolve solver path. First entry of --shared-libs wins; falls back to 'z3'
  // on PATH.
  std::string solverPath = sharedLibs.empty() ? "z3" : sharedLibs.front();

  // Detect shared-library paths (.so / .so.N / .dylib) and use the Z3 C API
  // directly via dlopen instead of spawning a subprocess.
  auto isSoPath = [](StringRef p) -> bool {
    return p.ends_with(".so") || p.contains(".so.") ||
           p.ends_with(".dylib") || p.contains(".dylib.");
  };

  // Locate (check-sat) in the exported SMT-LIB.
  size_t checkPos = smtlib.find("(check-sat)");
  size_t afterCheck = checkPos;
  if (checkPos != std::string::npos) {
    afterCheck = checkPos + std::string("(check-sat)").size();
    if (afterCheck < smtlib.size() && smtlib[afterCheck] == '\n') {
      ++afterCheck;
    }
  }

  // Build the SMT-LIB preamble (logic declaration + options).
  std::string preamble;
  {
    llvm::raw_string_ostream ss(preamble);
    if (setLogicQFABV) {
      ss << "(set-logic QF_ABV)\n";
    } else if (setLogicQFBV) {
      ss << "(set-logic QF_BV)\n";
    }
    if (solverTimeout > 0) {
      ss << "(set-option :timeout " << solverTimeout << ")\n";
    }
    ss << "(set-option :produce-models true)\n";
  }

  std::string solverOutput;

  if (isSoPath(solverPath)) {
    // --- Library mode: dlopen and call Z3_eval_smtlib2_string ---
    //
    // Z3's default error handler calls exit() on errors (e.g. when (get-model)
    // is called after an unsat result).  To avoid that, we drive (check-sat)
    // and (get-model) as separate calls on the same context so we can
    // conditionally skip (get-model) when the result is not 'sat'.
    std::string loadErr;
    if (llvm::sys::DynamicLibrary::LoadLibraryPermanently(solverPath.c_str(),
                                                          &loadErr)) {
      llvm::errs() << "failed to load '" << solverPath << "': " << loadErr
                   << "\n";
      return failure();
    }

    // Opaque typedefs matching Z3's C API.
    using Z3_config_t = void *;
    using Z3_context_t = void *;
    using Z3_mk_config_fn = Z3_config_t (*)();
    using Z3_mk_context_fn = Z3_context_t (*)(Z3_config_t);
    using Z3_del_config_fn = void (*)(Z3_config_t);
    using Z3_del_context_fn = void (*)(Z3_context_t);
    using Z3_eval_smtlib2_fn = const char *(*)(Z3_context_t, const char *);

    auto lookup = [&](const char *sym) -> void * {
      void *addr = llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(sym);
      if (!addr) {
        llvm::errs() << "symbol '" << sym << "' not found in '" << solverPath
                     << "'\n";
      }
      return addr;
    };

    auto *z3_mk_config =
        reinterpret_cast<Z3_mk_config_fn>(lookup("Z3_mk_config"));
    auto *z3_mk_context =
        reinterpret_cast<Z3_mk_context_fn>(lookup("Z3_mk_context"));
    auto *z3_del_config =
        reinterpret_cast<Z3_del_config_fn>(lookup("Z3_del_config"));
    auto *z3_del_context =
        reinterpret_cast<Z3_del_context_fn>(lookup("Z3_del_context"));
    auto *z3_eval =
        reinterpret_cast<Z3_eval_smtlib2_fn>(lookup("Z3_eval_smtlib2_string"));
    if (!z3_mk_config || !z3_mk_context || !z3_del_config || !z3_del_context ||
        !z3_eval) {
      return failure();
    }

    Z3_config_t cfg = z3_mk_config();
    Z3_context_t ctx = z3_mk_context(cfg);
    z3_del_config(cfg);

    // Step 1: run up through (check-sat) — do NOT include (get-model) yet.
    // We stop right after (check-sat) and leave the (reset) out so the solver
    // state is preserved for a potential (get-model) call.
    std::string checkScript = preamble;
    if (checkPos != std::string::npos) {
      checkScript += smtlib.substr(0, checkPos) + "(check-sat)\n";
    } else {
      checkScript += smtlib;
    }

    const char *checkResult = z3_eval(ctx, checkScript.c_str());
    solverOutput = checkResult ? checkResult : "";

    // Step 2: if 'sat', retrieve the model before resetting.
    StringRef checkLine = StringRef(solverOutput).split('\n').first.trim();
    if (checkLine == "sat" && showModel) {
      const char *modelResult = z3_eval(ctx, "(get-model)\n");
      if (modelResult) {
        solverOutput += modelResult;
      }
    }

    z3_del_context(ctx);

  } else {
    // For subprocess mode, splice (get-model) right after (check-sat) so the
    // solver prints both the result and the model in one pass.
    std::string fullScript = preamble;
    if (checkPos != std::string::npos) {
      fullScript += smtlib.substr(0, checkPos);
      fullScript += "(check-sat)\n(get-model)\n";
      fullScript += smtlib.substr(afterCheck);
    } else {
      fullScript += smtlib;
    }
    // --- Subprocess mode: write temp file, exec solver, read stdout ---
    llvm::SmallString<128> tmpPath;
    int tmpFd;
    if (auto err = llvm::sys::fs::createTemporaryFile("ttmlir_lec", "smt2",
                                                      tmpFd, tmpPath)) {
      llvm::errs() << "failed to create temp file: " << err.message() << "\n";
      return failure();
    }
    { llvm::raw_fd_ostream(tmpFd, /*shouldClose=*/true) << fullScript; }

    llvm::SmallString<128> outPath;
    int outFd;
    if (auto err = llvm::sys::fs::createTemporaryFile("ttmlir_lec_out", "txt",
                                                      outFd, outPath)) {
      llvm::errs() << "failed to create temp output: " << err.message() << "\n";
      llvm::sys::fs::remove(tmpPath);
      return failure();
    }
    ::close(outFd);

    // Pass -T:<seconds> as a CLI-level timeout in addition to the SMT-LIB
    // :timeout option — some Z3 versions only honour one or the other.
    std::string cliTimeoutArg;
    std::vector<llvm::StringRef> args = {solverPath};
    if (solverTimeout > 0) {
      unsigned secs = (solverTimeout + 999) / 1000;
      cliTimeoutArg = "-T:" + std::to_string(secs);
      args.push_back(cliTimeoutArg);
    }
    args.push_back(tmpPath.str());

    std::optional<llvm::StringRef> redirects[3] = {
        std::nullopt, llvm::StringRef(outPath), std::nullopt};
    std::string execErr;
    int rc = llvm::sys::ExecuteAndWait(solverPath, args, /*Env=*/std::nullopt,
                                       redirects, /*SecondsToWait=*/0,
                                       /*MemoryLimit=*/0, &execErr);
    auto bufOrErr = llvm::MemoryBuffer::getFile(outPath);
    llvm::sys::fs::remove(tmpPath);
    llvm::sys::fs::remove(outPath);

    if (rc < 0 || !bufOrErr) {
      llvm::errs() << "failed to invoke SMT solver (" << solverPath
                   << "): " << execErr << "\n";
      if (bufOrErr) {
        llvm::errs() << (*bufOrErr)->getBuffer();
      }
      return failure();
    }
    solverOutput = (*bufOrErr)->getBuffer().str();
  }

  // Parse the first line of solver output to determine equivalence.
  StringRef firstLine = StringRef(solverOutput).split('\n').first.trim();
  if (firstLine == "unsat") {
    outputFile->os() << "EQUIVALENT (c1 == c2)\n";
    outputFile->keep();
    return success();
  }
  if (firstLine == "sat") {
    outputFile->os() << "NON-EQUIVALENT (c1 != c2)\n";
    if (showModel) {
      outputFile->os() << "Counterexample:\n";
      size_t nl = solverOutput.find('\n');
      if (nl != std::string::npos) {
        outputFile->os() << solverOutput.substr(nl + 1);
      }
    }
    outputFile->keep();
    return success();
  }
  if (firstLine == "unknown" || firstLine == "timeout") {
    outputFile->os() << "TIMEOUT or UNKNOWN (solver could not decide)\n";
    outputFile->keep();
    return failure();
  }

  outputFile->os() << "solver returned UNKNOWN or error:\n" << solverOutput;
  outputFile->keep();
  return failure();
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Hide unrelated LLVM/MLIR options so --help only surfaces what's
  // actually relevant to ttmlir-lec.
  cl::HideUnrelatedOptions(mainCategory);

  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerAsmPrinterCLOptions();
  cl::ParseCommandLineOptions(
      argc, argv,
      "ttmlir-lec - logical equivalence checker for TTIR functions\n\n"
      "\tThis tool compares two function-style MLIR descriptions and "
      "reports whether they are logically equivalent. Inputs are TTIR "
      "by default; SMT inputs are also accepted (the TTIR-to-SMT lowering "
      "is a no-op on already-SMT functions).\n\n"
      "Examples:\n"
      "  ttmlir-lec a.mlir b.mlir -c1=foo -c2=bar\n"
      "  ttmlir-lec a.mlir b.mlir -c1=foo -c2=bar --emit-smtlib -o out.smt2\n"
      "  ttmlir-lec a.mlir b.mlir -c1=foo -c2=bar "
      "--shared-libs=/path/to/z3\n");

  DialectRegistry registry;
  mlir::tt::registerAllDialects(registry);
  mlir::tt::registerAllExtensions(registry);
  registry.insert<mlir::smt::SMTDialect>();
  MLIRContext context(registry);
  context.allowUnregisteredDialects(false);

  return failed(runLEC(context));
}
