#!/usr/bin/env python
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Time exact Llama 3 8B prefill graphs on compiler backends.

The benchmark can run either the full composed layer or only its decoder block.
The decoder sequence is configurable for same-shape comparisons with other
implementations. Each timed iteration starts with torch tensors on the host and
copies every program output back to host memory.

The current branch fuses each chain of 32 per-user TTIR cache fills into one
batched fill.  That form is valid for D2M but unsupported by TTNN.  For the
compiler-TTNN measurement, only the 64 input slices feeding K/V cache writes
are made dynamic so the compiler retains the original per-user fills.

When exact binaries are supplied for every requested backend, inputs are
derived from their descriptors and no model pattern discovery or compilation
is required. This is the preferred mode for reproducible measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "test/d2m-jit"))
sys.path.insert(0, str(REPO_ROOT / "tools/d2m-jit"))

from runner import (  # noqa: E402
    _RT_STR_TO_TORCH,
    _TORCH_TO_RT,
    _gen_tensor,
    _rt,
    compile_spec_to_fbb,
    compile_ttir_to_ttnn_fbb,
    discover,
    execute_ttm_in_process,
    parse_func_io,
)

SPEC_NAMES = {
    "full": "llama_prefill_layer_composed_e2e",
    "decoder": "llama_prefill_decoder_composed_e2e",
}
DEFAULT_SEQUENCE = 18
BATCH = 32
PHYSICAL_DEVICE_IDS = (0,)

_STATIC_SLICE_RE = re.compile(
    r'^(?P<indent>\s*)(?P<result>%\d+) = "ttir\.slice_static"'
    r"\((?P<input>%\d+)\) <\{begins = \[(?P<begins>.*?)\], "
    r"ends = \[(?P<ends>.*?)\], step = \[(?P<step>.*?)\]\}> : "
    r"\((?P<input_type>tensor<.*>)\) -> (?P<output_type>tensor<.*>)"
    r"(?: (?P<loc>loc\(.*\)))?$"
)


def _make_inputs(spec) -> list[torch.Tensor]:
    generator = torch.Generator()
    generator.manual_seed(spec.inputs.seed)
    return [
        _gen_tensor(shape, dtype, spec.inputs.dist, generator)
        for shape, dtype in parse_func_io(spec.ttir)
    ]


def _logical_input_signature(fbb) -> list[dict]:
    input_descs, _ = _parse_descriptors(fbb)
    return [
        {
            "shape": desc["desc"]["shape"],
            "dtype": desc["desc"]["layout"]["memory_desc"]["data_type"],
        }
        for desc in input_descs
    ]


def _make_binary_inputs(fbb, seed: int) -> list[torch.Tensor]:
    _rt()
    generator = torch.Generator()
    generator.manual_seed(seed)
    inputs = []
    for desc in _logical_input_signature(fbb):
        dtype = _RT_STR_TO_TORCH[desc["dtype"]]
        shape = desc["shape"]
        if dtype.is_floating_point:
            inputs.append(torch.randn(shape, generator=generator, dtype=dtype))
        else:
            inputs.append(torch.zeros(shape, dtype=dtype))
    return inputs


def _assert_input_parity(binaries: list[tuple[str, object]]) -> list[dict]:
    reference_label, reference_fbb = binaries[0]
    reference = _logical_input_signature(reference_fbb)
    for label, fbb in binaries[1:]:
        candidate = _logical_input_signature(fbb)
        if candidate != reference:
            raise RuntimeError(
                f"logical input mismatch between {reference_label} and {label}: "
                f"{reference!r} != {candidate!r}"
            )
    return reference


def _integer_values(text: str) -> str:
    return ", ".join(
        value.split(":", maxsplit=1)[0].strip() for value in text.split(",")
    )


def _rewrite_sequence_length(ttir_text: str, sequence: int) -> str:
    if sequence == DEFAULT_SEQUENCE:
        return ttir_text

    replacements = (
        (DEFAULT_SEQUENCE, sequence),
        (BATCH * DEFAULT_SEQUENCE, BATCH * sequence),
    )
    rewritten_lines = []
    for line in ttir_text.splitlines():
        slice_match = _STATIC_SLICE_RE.match(line)
        if slice_match:
            input_dims = [
                int(value)
                for value in re.findall(r"(\d+)(?=x)", slice_match.group("input_type"))
            ]
            ends = slice_match.group("ends").split(", ")
            for index, input_dim in enumerate(input_dims):
                if (
                    input_dim == DEFAULT_SEQUENCE
                    and int(ends[index].split(":", maxsplit=1)[0]) == DEFAULT_SEQUENCE
                ):
                    ends[index] = ends[index].replace(
                        str(DEFAULT_SEQUENCE), str(sequence), 1
                    )
            line = line.replace(
                f'ends = [{slice_match.group("ends")}]',
                f'ends = [{", ".join(ends)}]',
            )

        for old, new in replacements:
            if "shape = [" in line or "broadcast_dimensions = array<i64:" in line:
                line = re.sub(
                    rf"(?<![%\d]){old}(?=\s*(?::|,|>|\]))",
                    str(new),
                    line,
                )
            line = line.replace(f"end = {old} : si64", f"end = {new} : si64")

            def replace_tensor_dimension(match: re.Match) -> str:
                body = re.sub(
                    rf"(?P<prefix>^|x){old}(?=x)",
                    lambda dimension: f'{dimension.group("prefix")}{new}',
                    match.group("body"),
                )
                return f"tensor<{body}>"

            line = re.sub(r"tensor<(?P<body>[^>]+)>", replace_tensor_dimension, line)

        rewritten_lines.append(line)

    return "\n".join(rewritten_lines) + "\n"


def _block_batched_fill_cache_fusion(ttir_text: str) -> str:
    lines = ttir_text.splitlines()
    rewritten = []
    replaced = 0
    for index, line in enumerate(lines):
        match = _STATIC_SLICE_RE.match(line)
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        if (
            not match
            or f', {match.group("result")})' not in next_line
            or '"ttir.fill_cache"' not in next_line
        ):
            rewritten.append(line)
            continue

        indent = match.group("indent")
        begins = _integer_values(match.group("begins"))
        ends = _integer_values(match.group("ends"))
        begin_value = f"%cache_begin_{replaced}"
        end_value = f"%cache_end_{replaced}"
        rewritten.extend(
            (
                f'{indent}{begin_value} = "ttir.constant"() <{{value = dense<[{begins}]> : tensor<4xi32>}}> : () -> tensor<4xi32>',
                f'{indent}{end_value} = "ttir.constant"() <{{value = dense<[{ends}]> : tensor<4xi32>}}> : () -> tensor<4xi32>',
                f'{indent}{match.group("result")} = "ttir.slice_dynamic"'
                f'({match.group("input")}, {begin_value}, {end_value}) '
                f'<{{step = [{match.group("step")}]}}> : '
                f'({match.group("input_type")}, tensor<4xi32>, tensor<4xi32>) -> '
                f'{match.group("output_type")}'
                + (f' {match.group("loc")}' if match.group("loc") else ""),
            )
        )
        replaced += 1

    if replaced != 64:
        raise RuntimeError(
            f"expected to rewrite 64 cache input slices, rewrote {replaced}"
        )
    return "\n".join(rewritten) + "\n"


def _parse_descriptors(fbb, program_index: int = 0):
    def parse(raw: str):
        return json.loads(
            re.sub(r"\binf\b", "Infinity", re.sub(r"\bnan\b", "NaN", raw))
        )

    return (
        parse(fbb.get_program_inputs_as_json(program_index)),
        parse(fbb.get_program_outputs_as_json(program_index)),
    )


def _load_binary(path: Path):
    from _ttmlir_runtime import binary

    return binary.load_binary_from_path(str(path))


def _binary_record(path: Path | None) -> dict:
    if path is None:
        return {"source": "compiled_in_process"}
    return {
        "source": "loaded_from_path",
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def _execute_ttm_timed(fbb, inputs, device, program_index: int = 0) -> dict:
    runtime = _rt()
    total_begin = time.perf_counter_ns()

    begin = time.perf_counter_ns()
    input_descs, output_descs = _parse_descriptors(fbb, program_index)
    descriptor_ms = (time.perf_counter_ns() - begin) / 1_000_000

    begin = time.perf_counter_ns()
    rt_inputs = []
    host_inputs = []
    for tensor, input_desc in zip(inputs, input_descs, strict=True):
        desc = input_desc["desc"]
        expected_dtype = _RT_STR_TO_TORCH[desc["layout"]["memory_desc"]["data_type"]]
        if tensor.dtype != expected_dtype:
            tensor = tensor.to(expected_dtype)
        tensor = tensor.contiguous()
        host_inputs.append(tensor)
        rt_input = runtime.create_borrowed_host_tensor(
            tensor.data_ptr(),
            list(tensor.shape),
            list(tensor.stride()),
            tensor.element_size(),
            _TORCH_TO_RT[tensor.dtype],
        )
        layout = runtime.get_layout(fbb, program_index, len(rt_inputs))
        rt_inputs.append(runtime.to_layout(rt_input, device, layout, True))
    input_enqueue_ms = (time.perf_counter_ns() - begin) / 1_000_000

    begin = time.perf_counter_ns()
    runtime.set_compatible_device_runtime(fbb)
    rt_outputs = runtime.submit(device, fbb, program_index, rt_inputs)
    runtime.wait(rt_outputs)
    submit_wait_ms = (time.perf_counter_ns() - begin) / 1_000_000

    begin = time.perf_counter_ns()
    for rt_output, output_desc in zip(rt_outputs, output_descs, strict=True):
        desc = output_desc["desc"]
        shape = desc["shape"]
        dtype = _RT_STR_TO_TORCH[desc["layout"]["memory_desc"]["data_type"]]
        host_tensor = torch.empty(shape, dtype=dtype)
        rt_host = runtime.create_borrowed_host_tensor(
            host_tensor.data_ptr(),
            list(host_tensor.shape),
            list(host_tensor.stride()),
            host_tensor.element_size(),
            _TORCH_TO_RT[dtype],
        )
        host_view = runtime.to_host(rt_output, untilize=True)[0]
        runtime.memcpy(rt_host, host_view)
        runtime.deallocate_tensor(rt_output, force=True)
    output_ms = (time.perf_counter_ns() - begin) / 1_000_000

    return {
        "descriptor_ms": descriptor_ms,
        "input_enqueue_ms": input_enqueue_ms,
        "submit_wait_ms": submit_wait_ms,
        "output_ms": output_ms,
        "total_ms": (time.perf_counter_ns() - total_begin) / 1_000_000,
    }


def _benchmark(
    label: str,
    fbb,
    inputs,
    warmup: int,
    loops: int,
    boundary: str,
    sequence: int,
    phase_breakdown: bool,
    binary_record: dict,
) -> dict:
    runtime = _rt()
    runtime.set_compatible_device_runtime(fbb)
    mesh_shape = list(fbb.get_program_mesh_shape(0))
    mesh_volume = 1
    for dimension in mesh_shape:
        mesh_volume *= dimension
    if mesh_volume != len(PHYSICAL_DEVICE_IDS):
        raise RuntimeError(
            f"program mesh {mesh_shape} requires {mesh_volume} devices, but "
            f"the benchmark is configured for {len(PHYSICAL_DEVICE_IDS)}"
        )
    device_options = runtime.MeshDeviceOptions()
    device_options.mesh_shape = mesh_shape
    device_options.device_ids = list(PHYSICAL_DEVICE_IDS)
    device = runtime.open_mesh_device(device_options)
    program_cache_enabled = device.is_program_cache_enabled()
    try:
        for _ in range(warmup):
            execute_ttm_in_process(fbb, inputs, device)

        phase_samples = []
        samples_ms = []
        for _ in range(loops):
            if phase_breakdown:
                phases = _execute_ttm_timed(fbb, inputs, device)
                phase_samples.append(phases)
                samples_ms.append(phases["total_ms"])
            else:
                begin = time.perf_counter_ns()
                execute_ttm_in_process(fbb, inputs, device)
                samples_ms.append((time.perf_counter_ns() - begin) / 1_000_000)
    finally:
        runtime.close_mesh_device(device)

    result = {
        "backend": label,
        "boundary": boundary,
        "precision": "BF16 activations and weights",
        "physical_device_ids": PHYSICAL_DEVICE_IDS,
        "program_cache_enabled": program_cache_enabled,
        "binary": binary_record,
        "batch": BATCH,
        "sequence": sequence,
        "tokens": BATCH * sequence,
        "warmup": warmup,
        "loops": loops,
        "measurement": "host-to-host, including input layout/transfers and all output readback",
        "samples_ms": samples_ms,
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }
    if phase_samples:
        result["phase_medians_ms"] = {
            phase: statistics.median(sample[phase] for sample in phase_samples)
            for phase in phase_samples[0]
        }
        result["phase_samples_ms"] = phase_samples
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("d2m", "compiler-ttnn", "both"),
        default="both",
    )
    parser.add_argument("--workload", choices=("full", "decoder"), default="full")
    parser.add_argument("--sequence", type=int, default=DEFAULT_SEQUENCE)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--loops", type=int, default=5)
    parser.add_argument("--phase-breakdown", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--d2m-binary", type=Path)
    parser.add_argument("--compiler-ttnn-binary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.warmup < 0 or args.loops <= 0:
        parser.error("--warmup must be non-negative and --loops must be positive")
    if args.sequence <= 0:
        parser.error("--sequence must be positive")
    if args.workload == "full" and args.sequence != DEFAULT_SEQUENCE:
        parser.error("only the decoder workload supports changing --sequence")
    for path in (args.d2m_binary, args.compiler_ttnn_binary):
        if path is not None and not path.is_file():
            parser.error(f"binary does not exist: {path}")

    boundary = {
        "full": "embedding through full-token LM head",
        "decoder": "hidden states through attention, cache updates, and residual MLP",
    }[args.workload]

    d2m_fbb = None
    if args.backend in ("d2m", "both") and args.d2m_binary is not None:
        d2m_fbb = _load_binary(args.d2m_binary)
    ttnn_fbb = None
    if (
        args.backend in ("compiler-ttnn", "both")
        and args.compiler_ttnn_binary is not None
    ):
        ttnn_fbb = _load_binary(args.compiler_ttnn_binary)

    requested_binaries = [
        ("D2M-JIT", d2m_fbb),
        ("compiler TTNN", ttnn_fbb),
    ]
    requested_binaries = [
        (label, fbb) for label, fbb in requested_binaries if fbb is not None
    ]
    needs_compilation = (
        args.backend in ("d2m", "both")
        and d2m_fbb is None
        or args.backend in ("compiler-ttnn", "both")
        and ttnn_fbb is None
    )
    spec = None
    if needs_compilation:
        specs, _ = discover()
        spec_name = SPEC_NAMES[args.workload]
        try:
            spec = next(spec for spec in specs if spec.name == spec_name)
        except StopIteration as error:
            raise RuntimeError(
                f"model spec {spec_name!r} is unavailable; provide exact binaries "
                "for every requested backend"
            ) from error
        if args.sequence != DEFAULT_SEQUENCE:
            spec = replace(
                spec,
                name=f"{spec.name}_s{args.sequence}",
                ttir=_rewrite_sequence_length(spec.ttir, args.sequence),
            )
        inputs = _make_inputs(spec)
    else:
        input_signature = _assert_input_parity(requested_binaries)
        inputs = _make_binary_inputs(requested_binaries[0][1], args.seed)

    if d2m_fbb is None and args.backend in ("d2m", "both"):
        d2m_fbb = compile_spec_to_fbb(spec)
    if ttnn_fbb is None and args.backend in ("compiler-ttnn", "both"):
        ttnn_fbb = compile_ttir_to_ttnn_fbb(_block_batched_fill_cache_fusion(spec.ttir))

    active_binaries = [
        ("D2M-JIT", d2m_fbb),
        ("compiler TTNN", ttnn_fbb),
    ]
    active_binaries = [
        (label, fbb) for label, fbb in active_binaries if fbb is not None
    ]
    input_signature = _assert_input_parity(active_binaries)

    results = []
    if args.backend in ("d2m", "both"):
        results.append(
            _benchmark(
                "D2M-JIT",
                d2m_fbb,
                inputs,
                args.warmup,
                args.loops,
                boundary,
                args.sequence,
                args.phase_breakdown,
                _binary_record(args.d2m_binary),
            )
        )
    if args.backend in ("compiler-ttnn", "both"):
        results.append(
            _benchmark(
                "compiler TTNN (per-user cache-fill workaround)",
                ttnn_fbb,
                inputs,
                args.warmup,
                args.loops,
                boundary,
                args.sequence,
                args.phase_breakdown,
                _binary_record(args.compiler_ttnn_binary),
            )
        )

    if len(results) == 2:
        ratio = results[0]["median_ms"] / results[1]["median_ms"]
        print(f"D2M-JIT / compiler TTNN median: {ratio:.3f}x", flush=True)
    if args.output:
        from _ttmlir_runtime import __file__ as runtime_module

        report = {
            "runtime_module": runtime_module,
            "input_seed": args.seed,
            "logical_input_signature": input_signature,
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
