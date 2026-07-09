# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""`ttnn-advise` -- agent-facing CLI for the ttnn-jit L1 shard advisor.

Two ways to get an L1 sharding report:

  ttnn-advise mlir <file.ttir.mlir>            # advise on existing TTIR (no device)
  ttnn-advise capture <module.py:func>         # trace a ttnn fn on device, then advise

Both write a browsable artifact set to the output directory:
  report.json    machine-readable summary (ops, layouts, reshards, spill)
  report.txt     human-readable report
  final_ir.mlir  authoritative final TTNN IR
  decision_trace/*_decision_trace.json

stdout stays clean: a one-line summary + the artifact paths. Read report.json
for the structured result.

The optimizer's OpModel queries run against a mock device built from
SYSTEM_DESC_PATH -- set it first (ttrt query --save-artifacts). `capture` also
opens a real ttnn mesh device to trace the function.
"""
import argparse
import contextlib
import importlib.util
import os
import shutil
import sys
import tempfile
import warnings

# The ttnn extension re-registers nanobind types on import and emits a wall of
# RuntimeWarnings; silence them so the CLI's stdout stays agent-readable.
warnings.filterwarnings("ignore", message=r".*already registered.*")


@contextlib.contextmanager
def _quiet_native_output(log_path: str):
    """Redirect fd 1/2 to a log file for the duration of the block.

    The pipeline / device layers log copiously to the native stdout/stderr file
    descriptors (bypassing sys.stdout), which would otherwise bury the CLI's own
    summary and flood an agent's context. Capture it to a file instead.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out, saved_err = os.dup(1), os.dup(2)
    logf = open(log_path, "w")
    try:
        os.dup2(logf.fileno(), 1)
        os.dup2(logf.fileno(), 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        logf.close()


def _advise_quietly(thunk):
    """Run an advise thunk with native output captured; relocate the log next to
    the report artifacts. Returns the AdvisorReport. On failure, surfaces the
    error and the captured-log path."""
    tmp_log = tempfile.NamedTemporaryFile(
        mode="w", suffix=".pipeline.log", delete=False
    ).name
    try:
        with _quiet_native_output(tmp_log):
            report = thunk()
    except Exception as e:
        print(f"error: advise failed: {e}", file=sys.stderr)
        print(f"       pipeline log: {tmp_log}", file=sys.stderr)
        raise
    dest = os.path.join(report.out_dir, "pipeline.log")
    shutil.move(tmp_log, dest)
    return report


def _summarize(report) -> None:
    t = report.trace
    out_dir = report.out_dir
    print(
        f"[ttnn-advise] ops={t.total_ops} final_choices={len(t.final_choices)} "
        f"spill.ran={t.spill.ran} total_spills={t.spill.total_spills}"
    )
    print(f"[ttnn-advise] artifacts in: {out_dir}")
    print(f"  report.json    {os.path.join(out_dir, 'report.json')}")
    print(f"  report.txt     {os.path.join(out_dir, 'report.txt')}")
    print(f"  final_ir.mlir  {os.path.join(out_dir, 'final_ir.mlir')}")


def _cmd_mlir(args) -> int:
    def _do():
        # Import inside the captured region so the ttnn import banner is caught.
        from ttnn_jit._src.shard_advisor import ShardAdvisor

        return ShardAdvisor.advise_mlir_file(
            args.file,
            optimization_level=args.opt_level,
            out_dir=args.out,
            pipeline=args.pipeline,
            verbose=False,
        )

    report = _advise_quietly(_do)
    _summarize(report)
    return 0


def _load_module(path: str):
    spec = importlib.util.spec_from_file_location("_advise_target", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cmd_capture(args) -> int:
    import ttnn
    from ttnn_jit._src.shard_advisor import ShardAdvisor

    mod_path, _, func_name = args.target.partition(":")
    if not func_name:
        print(
            "error: capture target must be <module.py:func>, e.g. model.py:decode",
            file=sys.stderr,
        )
        return 2
    module = _load_module(mod_path)
    func = getattr(module, func_name, None)
    if func is None:
        print(f"error: {func_name!r} not found in {mod_path}", file=sys.stderr)
        return 2
    make_inputs = getattr(module, "make_inputs", None)
    if make_inputs is None:
        print(
            f"error: {mod_path} must define make_inputs(device) returning the "
            f"ttnn.Tensor args to trace {func_name!r} with",
            file=sys.stderr,
        )
        return 2

    def _do():
        device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
        try:
            inputs = make_inputs(device)
            if not isinstance(inputs, (tuple, list)):
                inputs = (inputs,)
            return ShardAdvisor(
                func,
                optimization_level=args.opt_level,
                out_dir=args.out,
                tracer="interception",
                pipeline=args.pipeline,
                verbose=False,
            ).run(*inputs)
        finally:
            ttnn.close_mesh_device(device)

    report = _advise_quietly(_do)
    _summarize(report)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="ttnn-advise",
        description="L1 shard advisor: report per-op TTNN layouts + reshards.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--opt-level", type=int, default=2, help="optimization level (default 2)"
    )
    common.add_argument(
        "--out", default=None, help="output directory for the artifact set"
    )
    common.add_argument(
        "--pipeline",
        choices=["scoped", "full"],
        default="scoped",
        help="scoped (1:1, optimizer only) or full backend pipeline",
    )

    m = sub.add_parser("mlir", parents=[common], help="advise on an existing .ttir.mlir")
    m.add_argument("file", help="path to a TTIR .mlir file")
    m.set_defaults(fn=_cmd_mlir)

    c = sub.add_parser(
        "capture", parents=[common], help="trace a ttnn fn on device, then advise"
    )
    c.add_argument("target", help="<module.py:func>; module must define make_inputs(device)")
    c.set_defaults(fn=_cmd_capture)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
