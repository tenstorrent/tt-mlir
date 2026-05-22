"""
Benchmarking helpers for the tt-kurbla torch backend.

Every measurement uses host-side wall-clock via ``time.perf_counter_ns`` with
explicit fences via ``_sync`` (recursive ``.cpu()`` on tensor leaves).

Results carry a ``measurements`` array of ``{name, value, unit, target}``
entries rather than a fixed set of fields. That shape mirrors what tt-xla
emits and makes it cheap to tack on new metrics (TTFT, ITL percentiles,
device kernel duration from tracy, etc.) without breaking the JSON schema
that downstream dashboards key on.
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
    """If ``trace_path`` is set, capture a torch.profiler trace into it."""
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


def prepare_model(model: nn.Module, mode: str) -> nn.Module:
    """Return ``model`` wrapped according to the requested execution mode.

    ``"eager"`` returns the model unchanged. ``"compile"`` wraps it with
    ``torch.compile(backend="tt_kurbla")``; until the tt_kurbla compile
    backend is registered with torch._dynamo, the first forward pass raises.
    Benchmark tests deliberately do not catch this - failures make the gap
    visible until compile support lands.
    """
    if mode == "eager":
        return model
    if mode == "compile":
        return torch.compile(model, backend="tt_kurbla")
    raise ValueError(f"unknown mode {mode!r}; expected 'eager' or 'compile'")


def _sync(obj: Any) -> None:
    """Force a host-side fence by materializing every tensor leaf on CPU.

    Raises ``TypeError`` for unrecognized container types - otherwise a new
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
    target: float = -1.0   # -1.0 = no regression threshold set

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
            rows.append(f"  {m.name:<24} {m.value:>12.3f} {m.unit}")
        return "\n".join(rows)


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    k = int(round(pct / 100.0 * (len(sorted_values) - 1)))
    k = max(0, min(len(sorted_values) - 1, k))
    return sorted_values[k]


def _infer_batch_size(inputs: Sequence[Any]) -> int:
    """Pick the batch dim from the first tensor in ``inputs``.

    Matches tt-xla's convention: throughput is reported per-sample, so a
    benchmark with batch 64 and 1500 iters/s shows up as 96000 samples/s.
    """
    for x in inputs:
        if isinstance(x, torch.Tensor) and x.ndim >= 1:
            return int(x.shape[0])
    return 1


def _extract_primary_tensor(out: Any) -> torch.Tensor:
    """Pick the canonical comparison tensor out of a model forward result.

    nn.Module → the tensor itself. HF causal-LM output → ``.logits``. Tuple/
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

    Robust to bf16 precision noise: brings operands to CPU first, *then*
    casts to f32 so the dtype conversion never has to go through a device
    backend that may not support it. Returns 1.0 for the degenerate case
    where one operand has zero variance and the two operands are
    elementwise close.
    """
    a = a.detach().cpu().to(torch.float32).flatten()
    b = b.detach().cpu().to(torch.float32).flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = float((a.norm() * b.norm()).item())
    if denom < 1e-12:
        return 1.0 if torch.allclose(a, b, atol=1e-6) else 0.0
    return float((a * b).sum().item() / denom)


def _pcc_against_reference(
    device_out: Any, ref_model: nn.Module, ref_inputs: Sequence[Any]
) -> float:
    """Run ``ref_model`` once and PCC its primary tensor against ``device_out``."""
    with torch.no_grad():
        ref_out = ref_model(*ref_inputs)
    return compute_pcc(_extract_primary_tensor(device_out), _extract_primary_tensor(ref_out))


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
    profile_enabled: bool = False,
    profile_dir: str = "./profile_data",
) -> BenchmarkResult:
    """Time ``iters`` forward passes dispatched back-to-back with a single
    final sync.

    Issues every ``model(*inputs)`` call without waiting in between, so an
    async runtime can pipeline them. Then materializes every output on CPU
    in a single fenced pass - that final ``_sync`` is where pipelined device
    work actually drains.

    If ``reference_model`` is given, runs an extra (untimed) forward pass
    before and after warmup, and emits ``pcc_before_warmup`` /
    ``pcc_after_warmup`` measurements against the reference's output. The
    reference is expected to live on CPU with the same weights and inputs.

    When ``profile_enabled`` is true, the timed region is captured into a
    Chrome trace at ``<profile_dir>/<label>_perf.json``.
    """
    accuracy_measurements: list[Measurement] = []
    have_ref = reference_model is not None and reference_inputs is not None
    outputs: list[Any] = []
    trace_path = _resolve_trace_path(profile_enabled, profile_dir, label, "perf")

    with torch.no_grad():
        if have_ref:
            cold_out = model(*inputs)
            _sync(cold_out)
            pcc_before = _pcc_against_reference(cold_out, reference_model, reference_inputs)
            accuracy_measurements.append(Measurement("pcc_before_warmup", pcc_before, "pcc"))

        _signpost("warmup_start")
        for _ in range(warmup):
            out = model(*inputs)
            _sync(out)
        _signpost("warmup_end")

        if have_ref:
            warm_out = model(*inputs)
            _sync(warm_out)
            pcc_after = _pcc_against_reference(warm_out, reference_model, reference_inputs)
            accuracy_measurements.append(Measurement("pcc_after_warmup", pcc_after, "pcc"))

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
            Measurement("total_ms", total_ms, "ms"),
            Measurement("iter_mean_ms", iter_mean_ms, "ms"),
            Measurement("samples_per_sec", samples_per_sec, "samples/s"),
        ],
    )


def run_decode_benchmark(
    model: nn.Module,
    prompt_input_ids: torch.Tensor,
    *,
    warmup: int,
    iters: int,
    label: str,
    mode: str,
    device: str,
    reference_model: nn.Module | None = None,
    reference_prompt_input_ids: torch.Tensor | None = None,
    profile_enabled: bool = False,
    profile_dir: str = "./profile_data",
) -> BenchmarkResult:
    """Autoregressive decode benchmark with per-step timing.

    Decode is inherently serial - each step depends on the previous step's
    sampled token plus the growing KV cache - so there's no host-side
    pipelining to give up by fencing per step. We exploit that to capture
    per-token timing, which lets us derive TTFT (time-to-first-token, i.e.
    prefill latency) plus ITL (inter-token-latency) distribution rather
    than just a single aggregate.

    Shape:
      1. ``warmup`` discarded prefill+decode round to warm JIT/kernel caches.
      2. One timed prefill → ``ttft_ms``.
      3. ``iters`` timed decode steps, each fenced via ``_sync(next_token)``.

    Reported measurements: ttft_ms, itl_mean_ms, itl_p50_ms, itl_p95_ms,
    decode_throughput_tps, decode_total_ms.
    """

    def _decode_step(token: torch.Tensor, past: Any) -> tuple[Any, torch.Tensor]:
        out = model(token, past_key_values=past, use_cache=True)
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        return out, next_tok

    accuracy_measurements: list[Measurement] = []
    have_ref = reference_model is not None and reference_prompt_input_ids is not None
    trace_path = _resolve_trace_path(profile_enabled, profile_dir, label, "perf")

    with torch.no_grad():
        if have_ref:
            cold_prefill = model(prompt_input_ids, use_cache=True)
            _sync(cold_prefill.logits)
            pcc_before = _pcc_against_reference(
                cold_prefill, reference_model, (reference_prompt_input_ids,)
            )
            accuracy_measurements.append(Measurement("pcc_before_warmup", pcc_before, "pcc"))

        _signpost("warmup_start")
        if warmup > 0:
            warm_prefill = model(prompt_input_ids, use_cache=True)
            past = warm_prefill.past_key_values
            tok = warm_prefill.logits[:, -1:, :].argmax(dim=-1)
            for _ in range(warmup):
                warm_out, tok = _decode_step(tok, past)
                past = warm_out.past_key_values
            _sync(tok)
        _signpost("warmup_end")

        if have_ref:
            warm_check = model(prompt_input_ids, use_cache=True)
            _sync(warm_check.logits)
            pcc_after = _pcc_against_reference(
                warm_check, reference_model, (reference_prompt_input_ids,)
            )
            accuracy_measurements.append(Measurement("pcc_after_warmup", pcc_after, "pcc"))

        with _maybe_profile(trace_path):
            # Prefill → TTFT.
            _signpost("prefill_start")
            t_prefill0 = time.perf_counter_ns()
            prefill = model(prompt_input_ids, use_cache=True)
            start_token = prefill.logits[:, -1:, :].argmax(dim=-1)
            _sync(start_token)
            ttft_ns = time.perf_counter_ns() - t_prefill0
            _signpost("prefill_end")

            # Per-step decode timing.
            past = prefill.past_key_values
            next_token = start_token
            step_times_ns: list[int] = []
            for step_idx in range(iters):
                _signpost(f"decode_{step_idx}_start")
                t0 = time.perf_counter_ns()
                out, next_token = _decode_step(next_token, past)
                _sync(next_token)
                step_times_ns.append(time.perf_counter_ns() - t0)
                past = out.past_key_values
                _signpost(f"decode_{step_idx}_end")
        _signpost("end")

    step_times_ms = [t / 1e6 for t in step_times_ns]
    sorted_ms = sorted(step_times_ms)
    itl_mean_ms = statistics.fmean(step_times_ms) if step_times_ms else 0.0
    itl_p50_ms = _percentile(sorted_ms, 50.0)
    itl_p95_ms = _percentile(sorted_ms, 95.0)
    decode_total_ns = sum(step_times_ns)
    tokens_per_sec = (
        iters * 1e9 / decode_total_ns if decode_total_ns > 0 else 0.0
    )
    ttft_ms = ttft_ns / 1e6
    decode_total_ms = decode_total_ns / 1e6

    return BenchmarkResult(
        label=label,
        mode=mode,
        device=device,
        warmup=warmup,
        iters=iters,
        measurements=[
            *accuracy_measurements,
            Measurement("ttft_ms", ttft_ms, "ms"),
            Measurement("itl_mean_ms", itl_mean_ms, "ms"),
            Measurement("itl_p50_ms", itl_p50_ms, "ms"),
            Measurement("itl_p95_ms", itl_p95_ms, "ms"),
            Measurement("tokens_per_sec", tokens_per_sec, "tokens/s"),
            Measurement("decode_total_ms", decode_total_ms, "ms"),
        ],
    )
