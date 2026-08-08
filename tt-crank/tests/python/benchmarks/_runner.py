"""
Benchmarking helpers for the tt-kurbla torch backend.

Every measurement uses host-side wall-clock via `time.perf_counter_ns` with
explicit fences via `_sync` (recursive `.cpu()` on tensor leaves).

Results carry a `measurements` array of `{name, value, unit}` entries
rather than a fixed set of fields, so new metrics (TTFT, ITL percentiles,
device kernel duration from tracy, etc.) can be added without breaking the
JSON schema that downstream dashboards key on.
"""

from __future__ import annotations

import os
import re
import statistics
import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

import torch
import torch.nn as nn
from tt_kurbla.torch._compile import CompileOption

try:
    import tracy as _tracy
    if not hasattr(_tracy, "signpost"):
        raise ImportError
except ImportError:
    _tracy = None


def _signpost(name: str) -> None:
    """Emit a tracy signpost if tracy is available; no-op otherwise.

    Lets us label phase boundaries (prefill, decode step i, drain, etc.) so
    tt-perf-report can slice device-side perf between markers, without
    forcing tracy to be importable in non-instrumented runs.
    """
    if _tracy is not None:
        _tracy.signpost(name)


@contextmanager
def _maybe_profile(trace_path: str | None) -> Iterator[None]:
    """If `trace_path` is set, capture a torch.profiler trace into it."""
    if not trace_path:
        yield
        return
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=False,
    ) as prof:
        yield
    prof.export_chrome_trace(trace_path)


_LABEL_SANITIZE = re.compile(r"[^A-Za-z0-9._-]+")


def _resolve_trace_path(
    profile_enabled: bool, profile_dir: str, label: str, suffix: str
) -> str | None:
    if not profile_enabled:
        return None
    os.makedirs(profile_dir, exist_ok=True)
    safe = _LABEL_SANITIZE.sub("_", label).strip("_")
    return os.path.join(profile_dir, f"{safe}_{suffix}.json")


def prepare_model(model: nn.Module, mode: str, options: dict [CompileOption, str | int | bool] | None = None) -> nn.Module:
    """Return `model` wrapped according to the requested execution mode.

    `"eager"` returns the model unchanged. `"compile"` wraps it with
    `torch.compile(backend="tt")` after freezing gradients.
    """
    if mode == "eager":
        return model
    if mode == "compile":
        model.requires_grad_(False)
        # dynamic=False is needed for the --accuracy path: causes SymInts to appear
        # in our graph, which we don't support at the moment.
        return torch.compile(model, backend="tt", dynamic=False, options=options)
    raise ValueError(f"unknown mode {mode!r}; expected 'eager' or 'compile'")


def _sync(obj: Any) -> None:
    """Force a host-side fence by materializing every tensor leaf on CPU.

    Raises `TypeError` for unrecognized container types - otherwise a new
    output shape (e.g. a custom Cache class with tensor attributes that
    isn't iterable) could silently skip the fence and quietly produce bogus
    timings.
    """
    if isinstance(obj, torch.Tensor):
        obj.cpu()
        return
    # Scalar / primitive leaves: nothing to fence.
    if obj is None or isinstance(obj, (int, float, bool, str, bytes)):
        return
    if isinstance(obj, Mapping):
        for v in obj.values():
            _sync(v)
        return
    if isinstance(obj, (list, tuple)) or (
        isinstance(obj, Iterable) and not isinstance(obj, (str, bytes))
    ):
        for v in obj:
            _sync(v)
        return
    # HuggingFace ModelOutput exposes its tensor fields as attributes; fall
    # through to the generic dict-like iteration via .to_tuple() if available.
    to_tuple = getattr(obj, "to_tuple", None)
    if callable(to_tuple):
        _sync(to_tuple())
        return
    raise TypeError(
        f"_sync: don't know how to fence {type(obj).__name__}; "
        "extend the helper if a new container shape was added to model outputs"
    )


@dataclass
class Measurement:
    name: str
    value: float
    unit: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkResult:
    label: str
    mode: str
    device: str
    warmup: int
    iters: int
    measurements: list[Measurement] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "mode": self.mode,
            "device": self.device,
            "warmup": self.warmup,
            "iters": self.iters,
            "measurements": [m.as_dict() for m in self.measurements],
        }

    def format_card(self) -> str:
        """Render this result as a multi-line card for terminal output."""
        header = (
            f"{self.label}  "
            f"[mode={self.mode}, device={self.device}, "
            f"warmup={self.warmup}, iters={self.iters}]"
        )
        rows = [header]
        for m in self.measurements:
            value = f"{m.value:>12.0f}" if m.unit == "count" else f"{m.value:>12.3f}"
            rows.append(f"  {m.name:<24} {value} {m.unit}")
        return "\n".join(rows)


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    k = int(round(pct / 100.0 * (len(sorted_values) - 1)))
    k = max(0, min(len(sorted_values) - 1, k))
    return sorted_values[k]


def _infer_batch_size(inputs: Sequence[Any]) -> int:
    """Pick the batch dim from the first tensor in `inputs`.

    Throughput is reported per-sample, so a benchmark with batch 64 and 1500
    iters/s shows up as 96000 samples/s.
    """
    for x in inputs:
        if isinstance(x, torch.Tensor) and x.ndim >= 1:
            return int(x.shape[0])
    return 1


def _extract_primary_tensor(out: Any) -> torch.Tensor:
    """Pick the canonical comparison tensor out of a model forward result.

    nn.Module → the tensor itself. HF causal-LM output → `.logits`. Tuple/
    list → first element. Anything else is a benchmark-author bug.
    """
    if isinstance(out, torch.Tensor):
        return out
    logits = getattr(out, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    if isinstance(out, (list, tuple)) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    raise TypeError(
        f"don't know how to extract a comparison tensor from {type(out).__name__}; "
        "the benchmark expected either a tensor, an object with a .logits "
        "attribute, or a non-empty sequence whose first element is a tensor"
    )


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation coefficient between two tensors of matching shape.

    Operands go to CPU then f64 (for more robust pcc calculation).
    """
    a = a.detach().cpu().to(torch.float64).flatten()
    b = b.detach().cpu().to(torch.float64).flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = float((a.norm() * b.norm()).item())
    if denom < 1e-12:
        return 1.0 if torch.allclose(a, b, atol=1e-6) else 0.0
    return float((a * b).sum().item() / denom)


def _pcc_against_reference(
    device_out: Any, ref_model: nn.Module, ref_inputs: Sequence[Any]
) -> float:
    """Run `ref_model` once and PCC its primary tensor against `device_out`."""
    with torch.no_grad():
        ref_out = ref_model(*ref_inputs)
    return compute_pcc(_extract_primary_tensor(device_out), _extract_primary_tensor(ref_out))


_PCC_TARGET = 0.94  # default accuracy gate for the --accuracy reference check


def run_benchmark(
    model: nn.Module,
    inputs: Sequence[Any],
    *,
    warmup: int,
    iters: int,
    label: str,
    mode: str,
    device: str,
    reference_model: nn.Module | None = None,
    reference_inputs: Sequence[Any] | None = None,
    pcc_target: float = _PCC_TARGET,
    profile_enabled: bool = False,
    profile_dir: str = "./profile_data",
) -> BenchmarkResult:
    """Time `iters` forward passes dispatched back-to-back with a single
    final sync.

    Issues every `model(*inputs)` call without waiting in between, so an
    async runtime can pipeline them. Then materializes every output on CPU
    in a single fenced pass - that final `_sync` is where pipelined device
    work actually drains.

    The warmup phase is timed separately and reported as `warmup_total_ms`;
    in compile mode without `--accuracy` that's where graph compilation
    lands (with `--accuracy` the untimed pre-warmup reference pass compiles
    first).

    If `reference_model` is given, runs an extra (untimed) forward pass
    before and after warmup, emits `pcc_before_warmup` / `pcc_after_warmup`
    measurements against the reference's output, and asserts both clear
    `pcc_target` so a regression fails the run. The reference is expected to
    live on CPU with the same weights and inputs.

    When `profile_enabled` is true, the timed region is captured into a
    Chrome trace at `<profile_dir>/<label>_perf.json`.
    """
    accuracy_measurements: list[Measurement] = []
    have_ref = reference_model is not None and reference_inputs is not None
    outputs: list[Any] = []
    trace_path = _resolve_trace_path(profile_enabled, profile_dir, label, "perf")

    cold_out = warm_out = None
    with torch.no_grad():
        if have_ref:
            cold_out = model(*inputs)
            _sync(cold_out)

        _signpost("warmup_start")
        warmup_t0 = time.perf_counter_ns()
        for _ in range(warmup):
            out = model(*inputs)
            _sync(out)
        warmup_ns = time.perf_counter_ns() - warmup_t0
        _signpost("warmup_end")

        if have_ref:
            warm_out = model(*inputs)
            _sync(warm_out)

        _signpost("dispatch_start")
        with _maybe_profile(trace_path):
            t0 = time.perf_counter_ns()
            for _ in range(iters):
                outputs.append(model(*inputs))
            _signpost("drain_start")
            for out in outputs:
                _sync(out)
            total_ns = time.perf_counter_ns() - t0
        _signpost("end")

    # PCC + gate outside the timed region so the reference run and assert don't
    # perturb the measurement.
    if have_ref:
        pcc_before = _pcc_against_reference(cold_out, reference_model, reference_inputs)
        pcc_after = _pcc_against_reference(warm_out, reference_model, reference_inputs)
        accuracy_measurements.append(Measurement("pcc_before_warmup", pcc_before, "pcc"))
        accuracy_measurements.append(Measurement("pcc_after_warmup", pcc_after, "pcc"))
        assert pcc_before >= pcc_target, f"{label}: pcc_before_warmup {pcc_before:.4f} < {pcc_target}"
        assert pcc_after >= pcc_target, f"{label}: pcc_after_warmup {pcc_after:.4f} < {pcc_target}"

    total_ms = total_ns / 1e6
    iter_mean_ms = total_ms / iters if iters > 0 else 0.0
    batch_size = _infer_batch_size(inputs)
    samples_per_sec = (iters * batch_size * 1e9 / total_ns) if total_ns > 0 else 0.0

    return BenchmarkResult(
        label=label,
        mode=mode,
        device=device,
        warmup=warmup,
        iters=iters,
        measurements=[
            *accuracy_measurements,
            Measurement("warmup_total_ms", warmup_ns / 1e6, "ms"),
            Measurement("total_ms", total_ms, "ms"),
            Measurement("iter_mean_ms", iter_mean_ms, "ms"),
            Measurement("samples_per_sec", samples_per_sec, "samples/s"),
        ],
    )


def run_llm_benchmark(
    model: nn.Module,
    prompt_input_ids: torch.Tensor,
    past_key_values: Any,
    cache_position: torch.Tensor,
    *,
    warmup_steps: int,
    total_steps: int,
    label: str,
    mode: str,
    device: str,
    profile_enabled: bool = False,
    profile_dir: str = "./profile_data",
) -> BenchmarkResult:
    """Autoregressive generate loop over one StaticCache; step 0 = prefill.

    `model` is an `LLMSamplingWrapper` returning `(next_token,
    next_cache_position)`, so between steps only the next token crosses to
    host - that transfer is the per-step fence. Step 0 prefills the full
    prompt into the pre-allocated cache (-> `ttft_ms`); the remaining
    `total_steps - 1` decode steps yield the inter-token-latency
    distribution and `tokens_per_sec` (per user, so batch-size independent).

    Warmup runs the same loop for `warmup_steps`, then the cache is
    `reset()` outside the timed region so timing starts from a clean cache.
    Warmup wall-clock is reported as `warmup_total_ms`; in compile mode
    that's where prefill + decode graph compilation lands.
    """

    def _generate(steps: int) -> list[int]:
        token, position = prompt_input_ids, cache_position
        times_ns: list[int] = []
        for step in range(steps):
            name = "prefill" if step == 0 else f"decode_{step - 1}"
            _signpost(f"{name}_start")
            t0 = time.perf_counter_ns()
            token, position = model(token, past_key_values, position)
            _sync(token)
            times_ns.append(time.perf_counter_ns() - t0)
            _signpost(f"{name}_end")
        return times_ns

    trace_path = _resolve_trace_path(profile_enabled, profile_dir, label, "perf")

    with torch.no_grad():
        _signpost("warmup_start")
        warmup_t0 = time.perf_counter_ns()
        _generate(warmup_steps)
        warmup_ns = time.perf_counter_ns() - warmup_t0
        _signpost("warmup_end")
        past_key_values.reset()

        with _maybe_profile(trace_path):
            step_ns = _generate(total_steps)
        _signpost("end")

    decode_ms = sorted(t / 1e6 for t in step_ns[1:])
    decode_total_ns = sum(step_ns[1:])
    tokens_per_sec = (
        len(decode_ms) * 1e9 / decode_total_ns if decode_total_ns > 0 else 0.0
    )

    return BenchmarkResult(
        label=label,
        mode=mode,
        device=device,
        warmup=warmup_steps,
        iters=total_steps,
        measurements=[
            Measurement("warmup_total_ms", warmup_ns / 1e6, "ms"),
            Measurement("ttft_ms", step_ns[0] / 1e6, "ms"),
            Measurement("itl_mean_ms", statistics.fmean(decode_ms) if decode_ms else 0.0, "ms"),
            Measurement("itl_p50_ms", _percentile(decode_ms, 50.0), "ms"),
            Measurement("itl_p95_ms", _percentile(decode_ms, 95.0), "ms"),
            Measurement("tokens_per_sec", tokens_per_sec, "tokens/s"),
            Measurement("decode_total_ms", decode_total_ns / 1e6, "ms"),
        ],
    )
