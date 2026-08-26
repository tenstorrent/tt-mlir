#!/usr/bin/env python3
"""Compile the Llama 3.2 1B seq-512 forward+loss graph at several optimization
levels and print the loss each compiled binary produces on device.

Repro harness for tenstorrent/tt-mlir#9189 ("Optimizer level 2 hangs on fused
Llama 3.2 1B training step"), where optimization-level=2 returned loss 27.59
against 5.84 at optimization-level=1 on the identical batch.

The pipeline options mirror the ones the tt-kurbla SST-2 experiment used (from
the issue's option table); only `optimization-level` varies between runs.

Inputs come either from a local HF snapshot of the trained weights plus a real
token sequence (`--weights`, the regime the issue reports from), or, by default,
from a synthetic N(0, 0.02) init that makes the model a near-uniform predictor
sitting at ln(128256) = 11.76. Either way every level is handed byte-identical
inputs, so a level that disagrees with the others is miscompiled.
`--cpu-reference` recomputes the same loss in torch for an independent answer.

Usage (from an activated env):
    python run_loss.py                  # levels 1 and 2
    python run_loss.py --levels 0 1 2
    python run_loss.py --levels 2 --keep-compiled
    python run_loss.py --weights ~/.cache/huggingface/hub/models--meta-llama--\
Llama-3.2-1B-Instruct/snapshots/<sha> --cpu-reference
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TTIR = HERE / "ttir_llama_3_2_1b_512_fwd.mlir"

# Graph constants, read off the TTIR signature.
VOCAB = 128256
HIDDEN = 2048
SEQ = 512
LABEL_LEN = SEQ - 1
HEAD_DIM = 64
ROPE_THETA = 500000.0
WEIGHT_STD = 0.02

# Input positions, in func argument order.
ARG_INPUT_IDS = 147
ARG_ATTENTION_MASK = 148
ARG_LABELS_ONE_HOT = 149
ARG_LOSS_WEIGHT = 150


BUILD = REPO / os.environ.get("BUILD_DIR", "build")
METAL = REPO / "third_party" / "tt-metal" / "src" / "tt-metal"


def tool(name):
    env = os.environ.get(name.upper().replace("-", "_"))
    if env:
        return env
    build = BUILD / "bin" / name
    return str(build) if build.exists() else name


def detect_arch(system_desc_path):
    """Name the output directory after the card.

    A Wormhole run and a Blackhole run of the same harness then land in
    different directories instead of overwriting each other.
    """
    try:
        import ttrt.binary

        desc = ttrt.binary.fbb_as_dict(
            ttrt.binary.load_system_desc_from_path(str(system_desc_path))
        )
        arch = desc["system_desc"]["chip_descs"][0]["arch"]
    except Exception:
        return "results"
    return arch.split("_")[0].lower()


def tool_env():
    """Environment for the child processes, repaired to point at the repo.

    `env/activate` derives TT_METAL_HOME, PATH and PYTHONPATH from `$(pwd)`, so
    an env sourced from anywhere but the repo root sends the op-model's metal
    runtime looking for `third_party/tt-metal/...` under the wrong prefix. Pin
    them here so the script works whatever directory it was activated from.
    """
    env = os.environ.copy()
    env["TT_MLIR_HOME"] = str(REPO)
    env["TT_METAL_RUNTIME_ROOT"] = str(METAL)
    env["TT_METAL_HOME"] = str(METAL)
    env["TT_METAL_BUILD_HOME"] = str(METAL / "build")
    prepend = [
        BUILD / "python_packages",
        BUILD / "runtime" / "python",
        REPO / ".local" / "toolchain" / "python_packages" / "mlir_core",
        METAL,
        METAL / "ttnn",
        METAL / "tt_eager",
        METAL / "build" / "tools" / "profiler" / "bin",
    ]
    existing = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    seen, ordered = set(), []
    for entry in [str(p) for p in prepend] + existing:
        if entry not in seen:
            seen.add(entry)
            ordered.append(entry)
    env["PYTHONPATH"] = os.pathsep.join(ordered)
    env["PATH"] = os.pathsep.join([str(BUILD / "bin"), env.get("PATH", "")])
    return env


def pipeline_options(level, system_desc, perf_metrics_file):
    """Non-default options from the issue's table, plus the varying level.

    compute-cfg-* are spelled out on purpose: the issue calls them no-ops
    because they match the level-0 defaults, but resolveOptimizationLevelOptions
    turns an *unset* fidelity into `Undefined` whenever level > 0, so leaving
    them off would not reproduce the experiment at levels 1 and 2.
    """
    opts = [
        f"system-desc-path={system_desc}",
        f"optimization-level={level}",
        "enable-const-eval=false",
        "enable-permute-matmul-fusion=true",
        "compute-cfg-math-fidelity=hifi4",
        "compute-cfg-fp32-dest-acc-en=true",
        "mesh-shape=1,1",
        # One topology per mesh dimension; a bare `linear` fails the
        # meshTopology-size-matches-meshShape-rank check.
        "mesh-topology=linear,linear",
    ]
    if perf_metrics_file:
        opts += [
            "ttnn-perf-metrics-enabled=true",
            f"ttnn-perf-metrics-output-file={perf_metrics_file}",
        ]
    return " ".join(opts)


def run_cmd(cmd, log_path):
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    start = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.run(
            cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO, env=tool_env()
        )
    return proc.returncode, time.time() - start


def compile_level(level, args, outdir):
    ttnn_mlir = outdir / f"ttnn_opt{level}.mlir"
    flatbuffer = outdir / f"opt{level}.ttnn"
    if args.keep_compiled and flatbuffer.exists():
        print(f"  reusing {flatbuffer.name}", flush=True)
        return flatbuffer, 0.0

    options = pipeline_options(
        level,
        args.system_desc,
        None if args.no_perf_metrics else outdir / f"perf_metrics_opt{level}",
    )
    rc, opt_secs = run_cmd(
        [
            tool("ttmlir-opt"),
            f"--ttir-to-ttnn-backend-pipeline={options}",
            str(TTIR),
            "-o",
            str(ttnn_mlir),
        ],
        outdir / f"compile_opt{level}.log",
    )
    if rc != 0:
        return None, opt_secs

    rc, translate_secs = run_cmd(
        [
            tool("ttmlir-translate"),
            "--ttnn-to-flatbuffer",
            str(ttnn_mlir),
            "-o",
            str(flatbuffer),
        ],
        outdir / f"translate_opt{level}.log",
    )
    if rc != 0:
        return None, opt_secs + translate_secs
    return flatbuffer, opt_secs + translate_secs


class FixedInputs:
    """Deterministic stand-ins for the real batch, keyed by argument index.

    Each tensor is drawn from a generator seeded with (seed, argument index),
    so every optimization level is handed byte-identical inputs regardless of
    how the compiler reorders anything.
    """

    def __init__(self, seed):
        self.seed = seed
        self.index = -1

    def _generator(self, index):
        import torch

        return torch.Generator().manual_seed(self.seed * 1000003 + index)

    def _target_ids(self):
        import torch

        # Shares a generator with nothing else so the labels stay fixed even if
        # the argument order changes.
        gen = torch.Generator().manual_seed(self.seed + 977)
        return torch.randint(0, VOCAB, (1, LABEL_LEN), generator=gen)

    def __call__(self, shape, dtype):
        import torch

        self.index += 1
        index = self.index
        shape = list(shape)

        # The four runtime inputs, keyed by argument index: the token ids and
        # the mask share a shape, so shape alone cannot tell them apart.
        if index == ARG_INPUT_IDS and shape == [1, SEQ]:
            return torch.randint(
                0, VOCAB, shape, dtype=dtype, generator=self._generator(index)
            )
        if index == ARG_ATTENTION_MASK and shape == [1, SEQ]:
            return torch.ones(shape, dtype=dtype)
        if index == ARG_LABELS_ONE_HOT and shape == [1, LABEL_LEN, VOCAB]:
            out = torch.zeros(shape, dtype=dtype)
            out.scatter_(2, self._target_ids().unsqueeze(-1), 1.0)
            return out
        # Per-token loss weight; negative so that -mean(log p) comes out positive.
        if index == ARG_LOSS_WEIGHT and shape == [1, LABEL_LEN, 1]:
            return torch.full(shape, -1.0 / LABEL_LEN, dtype=dtype)

        # RMSNorm gains: the real ones are ~1.
        if shape == [HIDDEN]:
            return torch.ones(shape, dtype=dtype)

        # Rotary inv_freq buffer.
        if shape == [HEAD_DIM // 2]:
            exponent = torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM
            return (1.0 / (ROPE_THETA**exponent)).to(dtype)

        # Embedding table and every projection weight.
        if len(shape) == 2 and dtype.is_floating_point:
            out = torch.empty(shape, dtype=dtype)
            return out.normal_(0.0, WEIGHT_STD, generator=self._generator(index))

        raise RuntimeError(
            f"no fixed input defined for argument {index} shape={shape} dtype={dtype}"
        )


def execute(flatbuffer, seed, result_path, weights=None):
    """Run one flatbuffer on device and write {loss: ...} to result_path.

    This drives ttrt.runtime directly instead of going through `ttrt run`
    because the graph returns the 146 weights alongside the loss: `ttrt run`
    reads every output back to host, which is 2.5 GB of pass-through traffic
    we have no use for. Only output 0 — the loss — is read here.
    """
    import ttrt.runtime
    from ttrt.common.util import (
        Binary,
        FileManager,
        Logger,
        convert_input_layouts,
        convert_runtime_to_torch_tensor,
        create_tensor,
    )

    stem = Path(flatbuffer).stem
    logger = Logger(str(Path(result_path).parent / f"ttrt_{stem}.log"))
    binary = Binary(logger, FileManager(logger), str(flatbuffer))
    ttrt.runtime.set_compatible_device_runtime(binary.fbb)

    program = binary.get_program(0)
    if weights:
        from real_inputs import RealInputs

        program.populate_inputs(RealInputs(weights))
    else:
        program.populate_inputs(FixedInputs(seed))

    mesh_options = ttrt.runtime.MeshDeviceOptions()
    mesh_options.mesh_shape = program.mesh_shape
    device = ttrt.runtime.open_mesh_device(mesh_options)
    try:
        inputs = [
            create_tensor(shards, program.mesh_shape)
            for shards in program.input_tensors
        ]
        inputs = convert_input_layouts(device, inputs, binary.fbb, 0)

        start = time.time()
        outputs = ttrt.runtime.submit(device, binary.fbb, 0, inputs)
        ttrt.runtime.wait(outputs)
        submit_secs = time.time() - start

        loss_host = ttrt.runtime.to_host(outputs[0], untilize=True)
        loss = convert_runtime_to_torch_tensor(loss_host[0]).flatten()[0].item()
        for output in outputs:
            ttrt.runtime.deallocate_tensor(output, force=True)
    finally:
        ttrt.runtime.close_mesh_device(device)

    print(f"loss={loss} submit={submit_secs:.1f}s", flush=True)
    Path(result_path).write_text(
        json.dumps(
            {
                "loss": loss,
                "seed": None if weights else seed,
                "weights": weights,
                "submit_seconds": submit_secs,
            }
        )
    )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[1, 2],
        help="optimization levels to compile and run (default: 1 2)",
    )
    parser.add_argument("--seed", type=int, default=23, help="input seed")
    parser.add_argument(
        "--system-desc",
        default=str(REPO / "ttrt-artifacts" / "system_desc.ttsys"),
    )
    parser.add_argument(
        "--keep-compiled",
        action="store_true",
        help="reuse an existing opt<N>.ttnn instead of recompiling",
    )
    parser.add_argument(
        "--no-perf-metrics",
        action="store_true",
        help="drop the ttnn-perf-metrics-* options the experiment set",
    )
    parser.add_argument(
        "--weights",
        help="HF snapshot directory (model.safetensors + config.json + "
        "tokenizer.json). Uses the trained weights and a real token sequence "
        "instead of the synthetic ones.",
    )
    parser.add_argument(
        "--cpu-reference",
        action="store_true",
        help="also compute the loss in torch on CPU (requires --weights)",
    )
    parser.add_argument(
        "--outdir",
        help="directory for this run's artifacts, relative to the harness "
        "(default: the card's architecture, e.g. wormhole / blackhole)",
    )
    parser.add_argument("--compile-only", action="store_true")
    # Internal: one device run, isolated in its own process.
    parser.add_argument("--exec", help=argparse.SUPPRESS)
    parser.add_argument("--exec-result", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.exec:
        return execute(args.exec, args.seed, args.exec_result, args.weights)

    if not TTIR.exists():
        sys.exit(f"missing input graph: {TTIR}")
    if not Path(args.system_desc).exists():
        sys.exit(
            f"missing system desc: {args.system_desc}\n"
            "run `ttrt query --save-artifacts` first"
        )

    if args.cpu_reference and not args.weights:
        sys.exit("--cpu-reference needs --weights")

    outdir = HERE / (args.outdir or detect_arch(args.system_desc))
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"writing results to {outdir.relative_to(HERE)}/", flush=True)

    reference = None
    if args.cpu_reference:
        print("[cpu] computing reference loss in torch", flush=True)
        from real_inputs import cpu_reference_loss

        start = time.time()
        reference = cpu_reference_loss(args.weights, log=lambda _: None)
        print(f"[cpu] reference loss = {reference} ({time.time() - start:.0f}s)",
              flush=True)

    results = {}
    for level in args.levels:
        print(f"[opt{level}] compiling", flush=True)
        flatbuffer, secs = compile_level(level, args, outdir)
        if flatbuffer is None:
            print(f"[opt{level}] compile FAILED after {secs:.1f}s "
                  f"(see compile_opt{level}.log)", flush=True)
            results[level] = {"loss": None, "note": "compile failed"}
            continue
        print(f"[opt{level}] compiled in {secs:.1f}s -> {flatbuffer.name}", flush=True)
        if args.compile_only:
            continue

        print(f"[opt{level}] running on device", flush=True)
        result_path = outdir / f"loss_opt{level}.json"
        rc, run_secs = run_cmd(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--exec",
                str(flatbuffer),
                "--exec-result",
                str(result_path),
                "--seed",
                str(args.seed),
            ]
            + (["--weights", args.weights] if args.weights else []),
            outdir / f"run_opt{level}.log",
        )
        if rc != 0 or not result_path.exists():
            print(f"[opt{level}] run FAILED after {run_secs:.1f}s "
                  f"(see run_opt{level}.log)", flush=True)
            results[level] = {"loss": None, "note": "run failed"}
            continue
        payload = json.loads(result_path.read_text())
        results[level] = {"loss": payload["loss"], "note": f"{run_secs:.1f}s"}
        print(f"[opt{level}] loss = {payload['loss']}", flush=True)

    print()
    if args.weights:
        print(f"loss, trained weights from {args.weights}")
    else:
        print(f"loss, seed={args.seed}, random weights (uniform predictor "
              f"would give ln({VOCAB}) = 11.762)")
    print(f"{'level':>6}  {'loss':>14}  note")
    if reference is not None:
        print(f"{'cpu':>6}  {reference:>14.6f}  torch reference")
    for level in args.levels:
        entry = results.get(level, {"loss": None, "note": "not run"})
        loss = "—" if entry["loss"] is None else f"{entry['loss']:.6f}"
        print(f"{level:>6}  {loss:>14}  {entry['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
