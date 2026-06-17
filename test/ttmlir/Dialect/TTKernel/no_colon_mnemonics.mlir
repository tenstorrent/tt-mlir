// RUN: %python -c 'import pathlib,re; text=pathlib.Path("%S/../../../../include/ttmlir/Dialect/TTKernel/IR/TTKernelOps.td").read_text(); bad=re.findall(r"TTKernel_(?:Op|FPUOp|SFPUOp|InitOp)<\"[^\"]*[<>::][^\"]*\"", text); assert not bad, bad[:5]'

// TTKernel MLIR op mnemonics must not contain C++ spelling syntax. Bare custom
// op printing treats these characters specially and can emit text that does not
// parse back.
// This file is intentionally not valid MLIR; it is a lit-only source check.
