# ttmlir-lec

Logical equivalence checker for TTIR functions.

`ttmlir-lec` takes two `func.func` operations (TTIR by default; SMT
already-lowered also accepted), lowers them to SMT, builds a miter
(`smt.solver` that asserts existential difference of outputs),
exports SMT-LIB, and invokes an SMT solver to decide equivalence. On
non-equivalence, the model is printed as a counterexample.

The tool is modelled after `circt-lec`. Flags and switch names follow
the same conventions where possible.

## Usage

```
ttmlir-lec <input1.mlir> [input2.mlir] -c1=<func> -c2=<func> [options]
```

- One or two MLIR input files. When two are passed they are merged into
  a single module (with symbol-collision renaming, except for the
  `c1`/`c2` symbols themselves which are protected — collisions on
  those are resolved by dropping the source-file copy, so the first
  file wins).
- `-c1=<name>` / `-c2=<name>`: the two function symbols to compare.
- Returns 0 on `EQUIVALENT`, non-zero on `NON-EQUIVALENT`, `TIMEOUT`,
  or solver error.

## Common workflows

### TTIR vs TTIR (default)

Both inputs are plain TTIR `func.func`s on `tensor<NxiM>`. No
pre-lowering is required.

```
ttmlir-lec a.mlir b.mlir -c1=foo -c2=foo_impl
```

The tool runs `convert-ttir-to-smt` on the merged module, then
`construct-ttir-lec`, and proceeds. This is the fastest path and the
one to start with.

### SMT vs TTIR (mixed)

`convert-ttir-to-smt` is a no-op on functions that are already in the
SMT dialect, so a TTIR file can be compared against an SMT reference
without any extra flag:

```
ttmlir-lec ref_smt.mlir impl_tt.mlir -c1=foo -c2=foo_impl
```

The reference SMT can come from any source — for example, a hand-
written `smt.solver` body, or a `func.func` in the SMT dialect emitted
by another tool.

### Compare a single output port (large modules)

For modules with many outputs, prune to one port before lowering. The
SMT problem only contains the cone-of-influence of that one output.

```
ttmlir-lec ref.mlir impl.mlir -c1=foo -c2=foo_impl \
  --check-output="<port_name>"
```

`--check-output` triggers `--ttir-prune-to-output`, then
`canonicalize`, then `func-drop-unused-args` so the two circuits keep
matching input signatures after pruning.

`--check-output-idx=<i>` selects by result index instead of port name.

## Output formats

| Flag           | What you get                                                                                  |
| -------------- | --------------------------------------------------------------------------------------------- |
| (default)      | Invoke the solver. Print `EQUIVALENT` or `NON-EQUIVALENT` with a counterexample.              |
| `--emit-mlir`  | The merged + lowered module with the LEC miter. Useful for inspecting what the tool sees.    |
| `--emit-smtlib`| The SMT-LIB script. Useful for piping into a solver by hand or saving a regression input.    |

`-o <path>` redirects output. Default is stdout.

## Solver selection

`ttmlir-lec` invokes the SMT solver as a subprocess and parses
`sat`/`unsat`/`unknown` from the first line. The solver only needs to
implement the standard SMT-LIB CLI contract (`solver foo.smt2`).

```
--shared-libs=/path/to/z3
```

The first entry of `--shared-libs` is used as the solver binary path.
If omitted, the tool searches `PATH` for `z3`. The flag accepts a
comma-separated list, matching `circt-lec` naming, but only the first
entry is consulted today.

`--solver-timeout=<ms>` sets a wall-time budget. For Z3 specifically
the value is also passed via `-T:<seconds>` on the CLI, since some Z3
versions only honor one of the two timeout mechanisms.

## Other useful flags

| Flag                       | Purpose                                                              |
| -------------------------- | -------------------------------------------------------------------- |
| `--solver-timeout=<ms>`    | Wall-time budget. 0 disables.                                        |
| `--show-model`             | Print the solver model on `NON-EQUIVALENT` (default on).             |
| `--set-logic-qfbv`         | Emit `(set-logic QF_BV)`. Speeds up bitvector-only problems.         |
| `--set-logic-qfabv`        | Emit `(set-logic QF_ABV)` — bitvectors plus arrays.                  |
| `--check-output=<port>`    | Prune to a single output by `hw.port_name`.                          |
| `--check-output-idx=<i>`   | Prune to a single output by index (overrides `--check-output`).      |

`--help` shows all flags.

## Examples

A worked example — equivalence of two trivially-equal TTIR functions:

```mlir
// equivalent.mlir
func.func @add_ab(%a: tensor<1xi32>, %b: tensor<1xi32>) -> tensor<1xi32> {
  %0 = "ttir.add"(%a, %b) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  return %0 : tensor<1xi32>
}

func.func @add_ba(%a: tensor<1xi32>, %b: tensor<1xi32>) -> tensor<1xi32> {
  %0 = "ttir.add"(%b, %a) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  return %0 : tensor<1xi32>
}
```

Run:

```
ttmlir-lec equivalent.mlir -c1=add_ab -c2=add_ba
# EQUIVALENT (c1 == c2)
```

Inspect the SMT-LIB:

```
ttmlir-lec equivalent.mlir -c1=add_ab -c2=add_ba --emit-smtlib -o eq.smt2
```

## Adding new TTIR ops

If `ttmlir-lec` errors out with `failed to legalize` on a TTIR op,
the op is missing a TTIRToSMT pattern. Add the conversion in
`lib/Conversion/TTIRToSMT/TTIRToSMT.cpp` and a unit test under
`test/ttmlir/Conversion/TTIRToSMT/`.
