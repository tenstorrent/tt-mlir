"""Benchmark: a small (~1B) Llama causal-LM - separate prefill and decode rows.

LLM inference has two phases with very different perf characteristics:
  * prefill: one forward over the full prompt. Compute-bound. Pipelinable -
    each prompt is independent - so we use the same dispatch loop as the
    CNN benchmarks (``run_benchmark``).
  * decode: one forward per generated token, attending to a growing KV cache.
    Memory-bandwidth-bound and inherently serial - each step depends on the
    previous token. Timed end-to-end via ``run_decode_benchmark``.

Loads a real published Llama (default ``meta-llama/Llama-3.2-1B``) via
HuggingFace ``transformers``. Override the id with ``--llama-model=<id>`` if
the default is gated. Skips cleanly if the model can't load (auth / network /
disk).
"""

import pytest
import torch

from ._runner import (
    prepare_model,
    run_benchmark,
    run_decode_benchmark,
)

transformers = pytest.importorskip("transformers")
from transformers import AutoModelForCausalLM  # noqa: E402


_DTYPE = torch.bfloat16
_PROMPT_LEN = 128


def _load_model(model_id: str) -> torch.nn.Module:
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=_DTYPE
        ).eval()
    except Exception as e:
        pytest.skip(f"could not load {model_id!r}: {e}")


def _build_inputs(
    model_id: str, device: torch.device | str
) -> tuple[torch.nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    model = _load_model(model_id).to(device)
    vocab = int(model.config.vocab_size)
    input_ids = torch.randint(0, vocab, (1, _PROMPT_LEN), dtype=torch.long).to(device)
    return model, input_ids


@pytest.mark.benchmark
def test_llama_3_2_1b(
    mode: str,
    warmup: int,
    iters: int,
    cpu_baseline: bool,
    accuracy: bool,
    llama_model_id: str,
    record_bench,
    tt_device: torch.device,
) -> None:
    model, input_ids = _build_inputs(llama_model_id, tt_device)
    model = prepare_model(model, mode)

    ref_model, ref_input_ids = (None, None)
    if accuracy:
        ref_model, ref_input_ids = _build_inputs(llama_model_id, "cpu")

    record_bench(
        run_benchmark(
            model, (input_ids,), warmup=warmup, iters=iters,
            label=f"llama/{llama_model_id}/prefill", mode=mode, device="tt",
            reference_model=ref_model,
            reference_inputs=(ref_input_ids,) if ref_input_ids is not None else None,
        )
    )
    record_bench(
        run_decode_benchmark(
            model, input_ids, warmup=warmup, iters=iters,
            label=f"llama/{llama_model_id}/decode", mode=mode, device="tt",
            reference_model=ref_model,
            reference_prompt_input_ids=ref_input_ids,
        )
    )

    if cpu_baseline:
        cpu_model, cpu_input_ids = _build_inputs(llama_model_id, "cpu")
        record_bench(
            run_benchmark(
                cpu_model, (cpu_input_ids,), warmup=warmup, iters=iters,
                label=f"llama/{llama_model_id}/prefill", mode="eager", device="cpu",
            )
        )
        record_bench(
            run_decode_benchmark(
                cpu_model, cpu_input_ids, warmup=warmup, iters=iters,
                label=f"llama/{llama_model_id}/decode", mode="eager", device="cpu",
            )
        )
