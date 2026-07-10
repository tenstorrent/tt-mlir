# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Analysis-only orchestrator: trace -> greedy optimizer -> decision report."""

import dataclasses
import inspect
import json
import os
import re

from ttnn_jit.ttmlir.passes import (
    ttir_to_ttnn_runtime_pipeline,
    run_pipeline,
)
from ttnn_jit._src.ir_generator import generate_ir
from ttnn_jit._src import (
    get_current_system_desc,
    create_runtime_device_from_ttnn,
)
from ttnn_jit._src.decision_trace_parser import (
    load_decision_trace,
    DecisionTraceReport,
)
from ttnn_jit._src.advisor_report import render_text_report
from ttnn_jit._src.ir_layout_summary import (
    render_ir_summary,
    parse_ir_summary,
)
from ttnn_jit._src.interception_tracer import trace_intercepted


class AdvisorReport:
    """Result of a shard-advisor run.

    Attributes:
        trace: parsed greedy-optimizer decision trace (beam candidates, scores,
            spill accounting) - useful as *rationale*, but it only records the
            reshards the greedy pass itself decided.
        ttnn_mlir: the final optimized TTNN IR. This is the ground truth: every
            reshard is an explicit ttnn.to_memory_config / ttnn.to_layout op and
            every tensor carries its chosen layout, including reshards inserted
            by post-greedy passes (input relayouts, output-layout reverts) that
            never reach the decision trace.
        ir_summary: human-readable layout + reshard summary derived from
            ttnn_mlir (authoritative).
        text: ir_summary followed by the decision-trace rationale.
    """

    def __init__(
        self,
        trace: DecisionTraceReport,
        text: str,
        ttnn_mlir: str,
        ir_summary: str = "",
        out_dir: str = "",
    ):
        self.trace = trace
        self.text = text
        self.ttnn_mlir = ttnn_mlir
        self.ir_summary = ir_summary
        self.out_dir = out_dir

    def __str__(self):
        return self.text


class ShardAdvisor:
    """Trace a ttnn function, run the greedy optimizer, report layout decisions.

    Analysis-only: never lowers to a flatbuffer and never executes on device.
    The optimizer itself runs on the mock device from a saved system descriptor.
    """

    def __init__(
        self,
        func,
        optimization_level: int = 2,
        debug: bool = False,
        out_dir: str = None,
        extra_pipeline_options: str = "",
        tracer: str = "ttnn",
        pipeline: str = "scoped",
        verbose: bool = True,
    ):
        self.func = func
        self.optimization_level = optimization_level
        self.debug = debug
        self.tracer = tracer
        # A ttnn-framework model *is* the TTNN dialect, so the direct-TTNN tracer
        # is the default: it emits TTNN and runs the no-lowering ttnn-input
        # pipeline (a TTIR-input pipeline can't consume its output). A ttnn op
        # with no handler is a genuine missing-op signal (loud fail), not a
        # reason to detour through TTIR -- fall back to tracer="interception"
        # only for ops not yet in the direct-TTNN path.
        if tracer == "ttnn" and pipeline == "scoped":
            pipeline = "ttnn"
        self.pipeline = pipeline
        self.verbose = verbose
        # func may be None when advising an existing .mlir file (no tracing).
        name = getattr(func, "__name__", None) or "advise"
        self.out_dir = out_dir or os.path.join("generated", "ttnn-jit", name, "advisor")
        self.extra_pipeline_options = extra_pipeline_options
        os.makedirs(self.out_dir, exist_ok=True)

    @classmethod
    def advise_mlir_file(
        cls,
        path: str,
        name: str = None,
        optimization_level: int = 2,
        out_dir: str = None,
        pipeline: str = "scoped",
        extra_pipeline_options: str = "",
        verbose: bool = True,
    ) -> AdvisorReport:
        """Advise on an existing TTIR .mlir file -- no tracing, no ttnn device.

        The greedy optimizer's OpModel queries run against the mock device the
        pipeline opens from SYSTEM_DESC_PATH, so only that env var is required.
        """
        from ttnn_jit.ttmlir import ir as _ir

        system_desc_path = os.environ.get("SYSTEM_DESC_PATH")
        if not system_desc_path:
            raise RuntimeError(
                "advise_mlir_file requires SYSTEM_DESC_PATH to be set "
                "(the mock device the optimizer queries is built from it). "
                "Generate one with: ttrt query --save-artifacts"
            )
        if name is None:
            base = os.path.basename(path)
            for ext in (".ttir.mlir", ".mlir"):
                if base.endswith(ext):
                    base = base[: -len(ext)]
                    break
            name = base or "advise"

        advisor = cls(
            func=None,
            optimization_level=optimization_level,
            out_dir=out_dir,
            extra_pipeline_options=extra_pipeline_options,
            pipeline=pipeline,
            verbose=verbose,
        )
        with open(path) as f:
            module_text = f.read()
        ctx = _ir.Context()
        module = _ir.Module.parse(module_text, ctx)
        return advisor._advise_ir(module, name, system_desc_path)

    def _ensure_system_desc(self, device) -> str:
        path = os.environ.get("SYSTEM_DESC_PATH")
        if path:
            return path
        runtime_device = create_runtime_device_from_ttnn(device)
        system_desc = get_current_system_desc(mesh_device=runtime_device)
        path = os.path.join(self.out_dir, "system_desc.ttsys")
        system_desc.store(path)
        os.environ["SYSTEM_DESC_PATH"] = path
        return path

    def _build_options(self, system_desc_path: str, trace_dir: str) -> str:
        mem_layout = "true" if self.optimization_level >= 2 else "false"
        opts = (
            f"system-desc-path={system_desc_path} "
            f"optimization-level={self.optimization_level} "
            f"enable-optimizer=true "
            f"enable-greedy-optimizer=true "
            f"memory-layout-analysis-enabled={mem_layout} "
            f"enable-decision-trace=true "
            f"decision-trace-dir={trace_dir}"
        )
        if self.extra_pipeline_options:
            opts += f" {self.extra_pipeline_options}"
        return opts

    def run(self, *args, **kwargs) -> AdvisorReport:
        device = args[0].device() if args else None
        assert (
            device is not None
        ), "ShardAdvisor requires at least one ttnn.Tensor argument on a device"

        system_desc_path = self._ensure_system_desc(device)

        if self.tracer == "ttnn":
            # Direct-TTNN interception: emit TTNN dialect straight from the
            # traced ops (no TTIR), consumed by the ttnn-to-ttnn-l1-advisor.
            from ttnn_jit._src.ttnn_emit_tracer import trace_ttnn

            ir, _output_type = trace_ttnn(self.func, *args)
        elif self.tracer == "interception":
            # Global ttnn-op interception: works across function boundaries.
            ir, _output_type = trace_intercepted(self.func, *args)
        else:
            # Source-rewrite tracer. Mirror JitFunction: pass runtime tensor
            # metadata. insert_output_layout=False lets the optimizer assign the
            # output layout (forcing block-sharded fails the pipeline).
            sig = inspect.signature(self.func)
            param_names = list(sig.parameters.keys())
            kwargs = dict(kwargs)
            kwargs["_tensor_args"] = {param_names[i]: args[i] for i in range(len(args))}
            ir, _output_type = generate_ir(
                self.func, self.debug, None, *args, insert_output_layout=False, **kwargs
            )

        return self._advise_ir(ir, self.func.__name__, system_desc_path)

    def _advise_ir(self, ir, name: str, system_desc_path: str) -> AdvisorReport:
        """Run the pipeline on an already-built TTIR module and persist the
        report set (report.txt, report.json, final_ir.mlir, decision trace).

        Shared by run() (traced graph) and advise_mlir_file() (existing IR).
        """
        trace_dir = os.path.join(self.out_dir, "decision_trace")
        os.makedirs(trace_dir, exist_ok=True)
        options = self._build_options(system_desc_path, trace_dir)
        if self.debug:
            print(f"[shard-advisor] pipeline options: {options}")

        # Mutates `ir` in place: TTIR -> TTNN with greedy optimizer + trace.
        if self.pipeline == "scoped":
            try:
                run_pipeline(ir, "ttir-to-ttnn-l1-advisor", options)
            except RuntimeError as e:
                raise RuntimeError(
                    "scoped ttir-to-ttnn-l1-advisor pipeline failed to lower the "
                    "traced graph (an op may require decomposition). Retry with "
                    f"pipeline='full'. Underlying error: {e}"
                ) from e
        elif self.pipeline == "ttnn":
            # Input module is already TTNN (direct-TTNN producer); no lowering.
            run_pipeline(ir, "ttnn-to-ttnn-l1-advisor", options)
        elif self.pipeline == "full":
            ttir_to_ttnn_runtime_pipeline(ir, options)
        else:
            raise ValueError(
                f"unknown pipeline {self.pipeline!r}; expected 'scoped', "
                "'ttnn', or 'full'"
            )

        # The pipeline names the trace after the MLIR func symbol, which may not
        # match `name` (e.g. an existing .mlir file whose func is @main). Prefer
        # the exact name, else fall back to the single trace this run produced.
        trace_path = os.path.join(trace_dir, f"{name}_decision_trace.json")
        if not os.path.exists(trace_path):
            import glob

            produced = sorted(
                glob.glob(os.path.join(trace_dir, "*_decision_trace.json"))
            )
            if not produced:
                raise RuntimeError(
                    f"Decision trace not produced in {trace_dir}. The optimizer "
                    f"may be disabled or the build may lack OpModel support "
                    f"(-DTTMLIR_ENABLE_OPMODEL=ON). Pipeline options: {options}"
                )
            trace_path = produced[-1]

        trace = load_decision_trace(trace_path)
        ttnn_mlir = str(ir)
        # The final IR is authoritative for layouts + reshards; the decision
        # trace is kept as greedy-phase rationale below it.
        summary = parse_ir_summary(ttnn_mlir)
        ir_summary = render_ir_summary(summary)
        rationale = render_text_report(trace)

        # Ops the validation-fallback pass could not make valid are flagged in
        # the IR (ttnn.validation_unfixable) rather than failing the compile, so
        # analysis continues and the rest of the report stays useful. Surface
        # them prominently -- the agent should treat these as "skip / use other
        # hints" rather than trusting a layout for them.
        unfixable = []
        for line in ttnn_mlir.splitlines():
            if "ttnn.validation_unfixable" not in line:
                continue
            op_m = re.search(r'"(ttnn\.[\w.]+)"', line)
            why_m = re.search(r'ttnn\.validation_unfixable = "((?:[^"\\]|\\.)*)"', line)
            reason = why_m.group(1) if why_m else ""
            # MLIR escapes newlines/tabs in string attrs as \0A / \09; unescape
            # to a single readable line.
            reason = re.sub(r"\\0[aA9]", " ", reason)
            reason = re.sub(r"\s+", " ", reason).strip()
            unfixable.append(
                {"op": op_m.group(1) if op_m else "unknown", "reason": reason}
            )
        banner = ""
        if unfixable:
            lines = [
                f"!! {u['op']}: no valid config (left broken) -- {u['reason']}"
                for u in unfixable
            ]
            banner = (
                f"=== {len(unfixable)} UNFIXABLE OP(S) -- skip these / use other "
                "hints ===\n" + "\n".join(lines) + "\n\n"
            )
        text = banner + ir_summary + "\n" + rationale
        report = AdvisorReport(trace, text, ttnn_mlir, ir_summary, self.out_dir)

        # Persist every artifact next to the decision trace so a run leaves a
        # complete, browsable set: the ground-truth IR, the rendered report, a
        # machine-readable report.json (for agent consumers), and (already
        # written by the pipeline) the raw decision-trace JSON.
        final_ir_path = os.path.join(self.out_dir, "final_ir.mlir")
        report_txt_path = os.path.join(self.out_dir, "report.txt")
        report_json_path = os.path.join(self.out_dir, "report.json")
        with open(final_ir_path, "w") as f:
            f.write(ttnn_mlir)
        with open(report_txt_path, "w") as f:
            f.write(text)
        report_json = {
            "function": name,
            "optimization_level": self.optimization_level,
            "pipeline": self.pipeline,
            "total_ops": trace.total_ops,
            "final_choices": len(trace.final_choices),
            "spill": {
                "ran": trace.spill.ran,
                "total_spills": trace.spill.total_spills,
            },
            "unfixable_ops": unfixable,
            "ops": [
                {
                    "index": o.index,
                    "op": o.op_name,
                    "layout": o.result_layout,
                    # program config the optimizer chose for its sharding strategy
                    # (e.g. matmul 1d multicast); "" when the op carries none.
                    "program_config": o.program_config,
                }
                for o in summary.ops
            ],
            "reshards": [
                {
                    "kind": r.kind,
                    "producer": r.producer,
                    "consumer": r.consumer,
                    "from": r.from_layout,
                    "to": r.to_layout,
                    "output_revert": r.is_output_revert,
                }
                for r in summary.reshards
            ],
            "artifacts": {
                "final_ir": final_ir_path,
                "report_txt": report_txt_path,
                "decision_trace": trace_path,
            },
        }
        with open(report_json_path, "w") as f:
            json.dump(report_json, f, indent=2)

        if self.verbose:
            print(text)
        return report
