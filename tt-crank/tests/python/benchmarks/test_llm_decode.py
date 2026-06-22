"""Benchmark: decoder-LM autoregressive generate loop.

One test per model (select with e.g. ``-k gemma``), all sharing
``_run_decode_benchmark``:

  Models:   meta-llama/Llama-3.2-1B-Instruct (test_llama_3_2_1b),
            Qwen/Qwen2.5-0.5B-Instruct (test_qwen2_5_0_5b),
            google/gemma-3-1b-it (test_gemma_3_1b) - all bfloat16
  Inputs:   a fixed prompt, unpadded (~16 tokens), batch 32 (``--llm-batch-size``)
  Cache:    StaticCache, 128 slots; decode fills the cache (~112 steps)
  Loop:     LLMSamplingWrapper keeps argmax + position increment on device;
            per-step fence is the next-token transfer; step 0 -> TTFT

The generic ``--warmup``/``--iters`` flags are ignored: warmup is a fixed
token count and the step count is set by cache capacity.
``--llm-max-output-tokens`` trims runs (warmup shrinks with it). With
``--accuracy``, prefill / first-decode logits are PCC'd against a CPU run.
"""

import pytest
import torch

from ._runner import Measurement, compute_pcc, prepare_model, run_llm_benchmark

transformers = pytest.importorskip("transformers")
from ._llama import (  # noqa: E402
    BENCHMARK_PROMPT,
    LLMSamplingWrapper,
    init_static_cache,
    load_model,
    load_tokenizer,
)

_MAX_CACHE_LEN = 128  # max_cache_len = input sequence length
_MIN_WARMUP_STEPS = 16  # warmup token count (capped at total_steps)
_PCC_TARGET = 0.94  # required PCC for the --accuracy check


def _two_step_logits(wrapper, prompt_ids, cache, cache_position, force_second_token=None):
    """Prefill + one decode step, returning (prefill_logits, decode_logits, prefill_token) on CPU."""
    with torch.no_grad():
        tok, pos, prefill_logits = wrapper(prompt_ids, cache, cache_position)
        second = tok if force_second_token is None else force_second_token
        _, _, decode_logits = wrapper(second, cache, pos)
    return prefill_logits.detach().cpu(), decode_logits.detach().cpu(), tok.detach().cpu()


def _run_decode_benchmark(
    llm_model_id: str,
    *,
    mode: str,
    cpu_baseline: bool,
    accuracy: bool,
    llm_batch_size: int,
    llm_max_output_tokens: int | None,
    profile_enabled: bool,
    profile_dir: str,
    record_bench,
    tt_device: torch.device,
) -> None:
    model = load_model(llm_model_id)
    tokenizer = load_tokenizer(llm_model_id)
    prompt_ids = tokenizer(
        [BENCHMARK_PROMPT] * llm_batch_size,
        return_tensors="pt",
        max_length=_MAX_CACHE_LEN,
        truncation=True,
    ).input_ids
    prompt_len = prompt_ids.shape[1]
    total_steps = llm_max_output_tokens or (_MAX_CACHE_LEN - prompt_len)
    warmup_steps = min(_MIN_WARMUP_STEPS, total_steps)

    def _fresh_cache(device: torch.device | str):
        return init_static_cache(
            model.config,
            batch_size=llm_batch_size,
            max_cache_len=_MAX_CACHE_LEN,
            device=device,
        )

    # CPU reference must run before model.to(tt_device) - .to() moves in-place.
    cpu_prefill_logits = cpu_decode_logits = cpu_prefill_token = None
    if accuracy:
        cpu_prefill_logits, cpu_decode_logits, cpu_prefill_token = _two_step_logits(
            LLMSamplingWrapper(model, return_logits=True),
            prompt_ids,
            _fresh_cache("cpu"),
            torch.arange(prompt_len),
        )

    model = model.to(tt_device)
    input_ids = prompt_ids.to(tt_device)
    cache_position = torch.arange(prompt_len).to(tt_device)

    perf_model = prepare_model(LLMSamplingWrapper(model), mode)
    result = run_llm_benchmark(
        perf_model,
        input_ids,
        _fresh_cache(tt_device),
        cache_position,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        label=f"llm_decode/{llm_model_id}/bs{llm_batch_size}",
        mode=mode,
        device="tt",
        profile_enabled=profile_enabled,
        profile_dir=profile_dir,
    )

    if accuracy:
        # PCC runs after perf with a fresh cache and a logits wrapper; the CPU
        # prefill token is teacher-forced into the decode step so a poor
        # prefill PCC can't compound into the decode PCC.
        dev_prefill_logits, dev_decode_logits, _ = _two_step_logits(
            prepare_model(LLMSamplingWrapper(model, return_logits=True), mode),
            input_ids,
            _fresh_cache(tt_device),
            cache_position,
            force_second_token=cpu_prefill_token.to(tt_device),
        )
        pcc_prefill = compute_pcc(dev_prefill_logits, cpu_prefill_logits)
        pcc_first_decode = compute_pcc(dev_decode_logits, cpu_decode_logits)
        result.measurements.append(Measurement("pcc_prefill", pcc_prefill, "pcc", target=_PCC_TARGET))
        result.measurements.append(
            Measurement("pcc_first_decode", pcc_first_decode, "pcc", target=_PCC_TARGET)
        )
        # Fail the run on a correctness regression rather than only recording the
        # number: an accuracy drop would otherwise be a green run with a bad metric
        # buried in the card.
        assert pcc_prefill >= _PCC_TARGET, (
            f"{llm_model_id}: prefill PCC {pcc_prefill:.4f} < {_PCC_TARGET}"
        )
        assert pcc_first_decode >= _PCC_TARGET, (
            f"{llm_model_id}: first-decode PCC {pcc_first_decode:.4f} < {_PCC_TARGET}"
        )

    record_bench(result)

    if cpu_baseline:
        # The first model instance now lives on tt; load a fresh one for CPU.
        cpu_model = load_model(llm_model_id)
        record_bench(
            run_llm_benchmark(
                LLMSamplingWrapper(cpu_model),
                prompt_ids,
                _fresh_cache("cpu"),
                torch.arange(prompt_len),
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                label=f"llm_decode/{llm_model_id}/bs{llm_batch_size}",
                mode="eager",
                device="cpu",
            )
        )


# One test per model so the suite is greppable / selectable by name
# (`pytest -k gemma`) and the benchmark cards name the model directly. All
# share _run_decode_benchmark; only the model id differs.
_LLM_FIXTURES = (
    "mode",
    "cpu_baseline",
    "accuracy",
    "llm_batch_size",
    "llm_max_output_tokens",
    "profile_enabled",
    "profile_dir",
    "record_bench",
    "tt_device",
)


@pytest.mark.benchmark
def test_llama_3_2_1b(request: pytest.FixtureRequest) -> None:
    _run_decode_benchmark(
        "meta-llama/Llama-3.2-1B-Instruct",
        **{name: request.getfixturevalue(name) for name in _LLM_FIXTURES},
    )


@pytest.mark.benchmark
def test_qwen2_5_0_5b(request: pytest.FixtureRequest) -> None:
    _run_decode_benchmark(
        "Qwen/Qwen2.5-0.5B-Instruct",
        **{name: request.getfixturevalue(name) for name in _LLM_FIXTURES},
    )


@pytest.mark.benchmark
def test_gemma_3_1b(request: pytest.FixtureRequest) -> None:
    _run_decode_benchmark(
        "google/gemma-3-1b-it",
        **{name: request.getfixturevalue(name) for name in _LLM_FIXTURES},
    )
