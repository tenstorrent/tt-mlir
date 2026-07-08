# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Analysis-only orchestrator: trace -> greedy optimizer -> decision report."""

import inspect
import os

from ttnn_jit.ttmlir.passes import ttir_to_ttnn_runtime_pipeline
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
from ttnn_jit._src.ir_layout_summary import render_ir_summary_from_mlir
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
    ):
        self.trace = trace
        self.text = text
        self.ttnn_mlir = ttnn_mlir
        self.ir_summary = ir_summary

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
        tracer: str = "rewrite",
    ):
        self.func = func
        self.optimization_level = optimization_level
        self.debug = debug
        self.tracer = tracer
        self.out_dir = out_dir or os.path.join(
            "generated", "ttnn-jit", func.__name__, "advisor"
        )
        self.extra_pipeline_options = extra_pipeline_options
        os.makedirs(self.out_dir, exist_ok=True)

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

        if self.tracer == "interception":
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

        trace_dir = os.path.join(self.out_dir, "decision_trace")
        os.makedirs(trace_dir, exist_ok=True)
        options = self._build_options(system_desc_path, trace_dir)
        if self.debug:
            print(f"[shard-advisor] pipeline options: {options}")

        # Mutates `ir` in place: TTIR -> TTNN with greedy optimizer + trace.
        ttir_to_ttnn_runtime_pipeline(ir, options)

        trace_path = os.path.join(
            trace_dir, f"{self.func.__name__}_decision_trace.json"
        )
        if not os.path.exists(trace_path):
            raise RuntimeError(
                f"Decision trace not produced at {trace_path}. The optimizer may "
                f"be disabled or the build may lack OpModel support "
                f"(-DTTMLIR_ENABLE_OPMODEL=ON). Pipeline options: {options}"
            )

        trace = load_decision_trace(trace_path)
        ttnn_mlir = str(ir)
        # The final IR is authoritative for layouts + reshards; the decision
        # trace is kept as greedy-phase rationale below it.
        ir_summary = render_ir_summary_from_mlir(ttnn_mlir)
        rationale = render_text_report(trace)
        text = ir_summary + "\n" + rationale
        report = AdvisorReport(trace, text, ttnn_mlir, ir_summary)

        # Persist every artifact next to the decision trace so a run leaves a
        # complete, browsable set: the ground-truth IR, the rendered report, and
        # (already written by the pipeline) the raw decision-trace JSON.
        with open(os.path.join(self.out_dir, "final_ir.mlir"), "w") as f:
            f.write(ttnn_mlir)
        with open(os.path.join(self.out_dir, "report.txt"), "w") as f:
            f.write(text)

        print(text)
        return report
