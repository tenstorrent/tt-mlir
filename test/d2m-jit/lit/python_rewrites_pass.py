# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s %ttmlir_tools/ttmlir-opt 2>&1 | FileCheck %s
# REQUIRES: d2m-jit

"""End-to-end test for the `--d2m-python-rewrites` C++ pass.

Spawns `ttmlir-opt --d2m-python-rewrites=module-path=...` on a tiny
ttir.exp module and confirms the rewritten module contains the fused
d2m subgraph the bundled `eltwise_exp_to_kernel` pattern produces.

Validates the C++ pass plumbing end-to-end:
  - Embedded CPython initialization in MLIRD2MTransforms.
  - Loading a user-supplied .py file via importlib at run time
    (no rebuild of ttmlir-opt needed when adding a new pattern).
  - Text round-trip: ttmlir-opt prints the host module, hands it to
    `d2m_jit.apply_patterns_text`, re-parses the result back into the
    host context, and splices it over the original module body.
"""

import os
import subprocess
import sys
import tempfile


# Build a temp .mlir input and run ttmlir-opt on it.
TTIR_INPUT = """\
module {
  func.func @forward(%x: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %r = "ttir.exp"(%x) : (tensor<32x32xf32>) -> tensor<32x32xf32>
    return %r : tensor<32x32xf32>
  }
}
"""


def find_tt_mlir_root():
    # The test file lives at $repo/test/d2m-jit/lit/python_rewrites_pass.py.
    here = os.path.abspath(__file__)
    return os.path.normpath(os.path.join(os.path.dirname(here), "..", "..", ".."))


def main():
    root = find_tt_mlir_root()
    pattern_file = os.path.join(
        root, "tools", "d2m-jit", "patterns", "eltwise_exp_to_kernel.py"
    )
    if not os.path.exists(pattern_file):
        print(f"FATAL: pattern file missing: {pattern_file}", file=sys.stderr)
        sys.exit(1)

    if len(sys.argv) < 2:
        print("FATAL: pass ttmlir-opt path as argv[1]", file=sys.stderr)
        sys.exit(1)
    ttmlir_opt = sys.argv[1]
    if not os.path.exists(ttmlir_opt):
        print(f"FATAL: ttmlir-opt not found at {ttmlir_opt}", file=sys.stderr)
        sys.exit(1)

    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as tmp:
        tmp.write(TTIR_INPUT)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [
                ttmlir_opt,
                f"--d2m-python-rewrites=module-path={pattern_file}",
                tmp_path,
            ],
            check=True,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
    finally:
        os.unlink(tmp_path)

    print(result.stdout)


if __name__ == "__main__":
    main()


# CHECK-LABEL: func.func @forward
# CHECK-NOT:    ttir.exp
# CHECK:        d2m.generic
# CHECK:        d2m.tile_exp
# CHECK:        return %{{.*}} : tensor<32x32xf32>
