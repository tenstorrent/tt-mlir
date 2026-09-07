# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark: decoder-LM autoregressive generate loop (single-chip / DP / TP).

One test per model (select with e.g. ``-k gemma``); each is parametrized over a
parallelism strategy, so the multi-chip scenarios are just variants of the same
decode benchmark (select with e.g. ``-k 'llama and tp'``):

  Models:   meta-llama/Llama-3.2-1B-Instruct (test_llama_3_2_1b),
            Qwen/Qwen2.5-0.5B-Instruct (test_qwen2_5_0_5b),
            google/gemma-3-1b-it (test_gemma_3_1b) - all bfloat16
  Variants: none  single-chip baseline (no mesh, no collectives)
            dp    data-parallel: model replicated, batch sharded over all chips;
                  per-user token rate ~unchanged, aggregate throughput scales
            tp    Megatron tensor-parallel: each decoder layer's attention/MLP
                  linears column/row sharded over all chips, lowering latency
  Inputs:   a fixed prompt, unpadded (~16 tokens), batch 32 (``--llm-batch-size``)
  Cache:    StaticCache, 128 slots; decode fills the cache (~112 steps)
  Loop:     LLMSamplingWrapper keeps argmax + position increment on device;
            per-step fence is the next-token transfer; step 0 -> TTFT

``--llm-batch-size`` (B) is the *per-chip* batch in every variant, so each chip
does the same batch-dim work: none = B (1 chip), dp = B*chips (B per chip,
sharded), tp = B (replicated). dp/tp need ``--mode compile`` and >= 2 chips (the
sharded StaticCache write only has a device path when traced) and skip otherwise.

The generic ``--warmup``/``--iters`` flags are ignored: warmup is a fixed token
count and the step count is set by cache capacity. ``--llm-max-output-tokens``
trims runs and ``--llm-num-layers`` trims the model (smoke knobs). With
``--accuracy``, prefill / first-decode logits are PCC'd against a CPU run (the
sharded variants gather their logits back to host first).
"""

import contextlib

import pytest
import torch
from torch.distributed.tensor import DTensor, distribute_module
from torch.distributed.tensor.experimental import implicit_replication
from tt_kurbla.torch._compile import BfpDtype, CompileOption

from ._runner import Measurement, compute_pcc, prepare_model, run_llm_benchmark

transformers = pytest.importorskip("transformers")
from ._llama import (  # noqa: E402
    BENCHMARK_PROMPT,
    LLMSamplingWrapper,
    init_static_cache,
    load_model,
    load_tokenizer,
)
from ._sharding import (  # noqa: E402
    distribute_inputs,
    shard_static_cache,
    tensor_parallel,
)

_MAX_CACHE_LEN = 128  # max_cache_len = input sequence length
_MIN_WARMUP_STEPS = 16  # warmup token count (capped at total_steps)
_PCC_TARGET = 0.94  # required PCC for the --accuracy check


def _gather(t: torch.Tensor) -> torch.Tensor:
    """Pull a (possibly sharded) device tensor to a full fp32 CPU tensor.

    Sharded variants return logits as a DTensor (replicated under TP, batch
    sharded under DP); ``full_tensor`` reconstructs the global tensor so it can
    be PCC'd against the unsharded CPU reference. Plain tensors just move.
    """
    if isinstance(t, DTensor):
        t = t.full_tensor()
    return t.detach().to(torch.float32).cpu()


def _two_step_logits(
    wrapper, prompt_ids, cache, cache_position, force_second_token=None
):
    """Prefill + one decode step -> (prefill_logits, decode_logits, prefill_token) on CPU.

    Gathers sharded outputs. The second token can be teacher-forced so a weak
    prefill PCC can't compound into the decode PCC.
    """
    with torch.no_grad():
        tok, pos, prefill_logits = wrapper(prompt_ids, cache, cache_position)
        second = tok if force_second_token is None else force_second_token
        _, _, decode_logits = wrapper(second, cache, pos)
    return _gather(prefill_logits), _gather(decode_logits), _gather(tok)


def _run_decode_benchmark(
    llm_model_id: str,
    *,
    mode: str,
    parallel: str,
    cpu_baseline: bool,
    accuracy: bool,
    llm_batch_size: int,
    llm_max_output_tokens: int | None,
    llm_num_layers: int | None,
    profile_enabled: bool,
    profile_dir: str,
    record_bench,
    tt_device: torch.device,
    options: dict[CompileOption, str | int | bool] | None = None,
) -> None:
    n = torch.tt.num_chips()
    if parallel != "none" and n < 2:
        pytest.skip("dp/tp require >= 2 chips")

    # llm_batch_size is the per-chip batch; dp fans the total batch across chips.
    total_batch = llm_batch_size * n if parallel == "dp" else llm_batch_size

    model = load_model(llm_model_id, num_layers=llm_num_layers)
    tokenizer = load_tokenizer(llm_model_id)

    if parallel == "tp":
        num_kv_heads = getattr(
            model.config, "num_key_value_heads", model.config.num_attention_heads
        )
        if num_kv_heads % n:
            pytest.skip(
                f"tp needs num_key_value_heads ({num_kv_heads}) divisible by {n}"
            )

    prompt_ids = tokenizer(
        [BENCHMARK_PROMPT] * total_batch,
        return_tensors="pt",
        max_length=_MAX_CACHE_LEN,
        truncation=True,
    ).input_ids
    prompt_len = prompt_ids.shape[1]
    total_steps = llm_max_output_tokens or (_MAX_CACHE_LEN - prompt_len)
    warmup_steps = min(_MIN_WARMUP_STEPS, total_steps)

    mesh = (
        None
        if parallel == "none"
        else torch.tt.init_device_mesh((n,), mesh_dim_names=(parallel,))
    )

    def fresh_cache(device, *, shard: bool = False):
        cache = init_static_cache(
            model.config,
            batch_size=total_batch,
            max_cache_len=_MAX_CACHE_LEN,
            device=device,
        )
        if shard:
            shard_static_cache(cache, mesh, parallel=parallel)
        return cache

    # implicit_replication lets HF's plain 1-D cache_position be treated as a
    # replicated DTensor so the sharded index_copy_ dispatches; a no-op for the
    # single-chip baseline. Fresh per use (these are single-shot context managers).
    def repl():
        return (
            implicit_replication() if parallel != "none" else contextlib.nullcontext()
        )

    # CPU reference for --accuracy must run before the model moves/shards onto tt.
    cpu_prefill = cpu_decode = cpu_tok = None
    if accuracy:
        cpu_prefill, cpu_decode, cpu_tok = _two_step_logits(
            LLMSamplingWrapper(model, return_logits=True),
            prompt_ids,
            fresh_cache("cpu"),
            torch.arange(prompt_len),
        )

    if parallel == "none":
        model = model.to(tt_device)
        input_ids = prompt_ids.to(tt_device)
        cache_position = torch.arange(prompt_len).to(tt_device)
    else:
        # DP just replicates every param across the mesh (the batch is sharded at
        # the input, in distribute_inputs); TP column/row-shards the linears.
        if parallel == "dp":
            model = distribute_module(model.to(tt_device), mesh)
        else:
            model = tensor_parallel(model, mesh)
        input_ids, cache_position = distribute_inputs(
            prompt_ids, torch.arange(prompt_len), mesh, parallel=parallel
        )

    label = f"llm_decode/{llm_model_id}/{parallel}/x{n if parallel != 'none' else 1}/bs{total_batch}"
    perf_model = prepare_model(LLMSamplingWrapper(model), mode, options=options)
    with repl():
        result = run_llm_benchmark(
            perf_model,
            input_ids,
            fresh_cache(tt_device, shard=parallel != "none"),
            cache_position,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            label=label,
            mode=mode,
            device="tt",
            profile_enabled=profile_enabled,
            profile_dir=profile_dir,
        )
    # Geometry, so the report can turn per-user tokens/s into aggregate throughput.
    result.measurements.append(
        Measurement("total_batch", float(total_batch), "samples")
    )
    result.measurements.append(
        Measurement("chips", float(n if parallel != "none" else 1), "chips")
    )

    if accuracy:
        # PCC the device prefill / first-decode logits against the CPU run, teacher-
        # forcing the CPU prefill token so a poor prefill can't compound the decode.
        if parallel == "none":
            forced = cpu_tok.to(torch.long).to(tt_device)
        else:
            forced, _ = distribute_inputs(
                cpu_tok.to(torch.long), torch.arange(1), mesh, parallel=parallel
            )
        with repl():
            dev_prefill, dev_decode, _ = _two_step_logits(
                prepare_model(
                    LLMSamplingWrapper(model, return_logits=True), mode, options=options
                ),
                input_ids,
                fresh_cache(tt_device, shard=parallel != "none"),
                cache_position,
                force_second_token=forced,
            )
        # _two_step_logits gathers sharded outputs back to host, so PCC against the
        # CPU reference covers the dp/tp variants too. Assert >= target so a sharded
        # variant that mis-shards (e.g. a Replicate->Shard redistribute that drops to
        # chunk 0) fails the run instead of reporting a fast but wrong pass.
        pcc_prefill = compute_pcc(dev_prefill, cpu_prefill)
        pcc_first_decode = compute_pcc(dev_decode, cpu_decode)
        result.measurements.append(Measurement("pcc_prefill", pcc_prefill, "pcc"))
        result.measurements.append(
            Measurement("pcc_first_decode", pcc_first_decode, "pcc")
        )
        assert (
            pcc_prefill >= _PCC_TARGET
        ), f"{label}: prefill PCC {pcc_prefill:.4f} < {_PCC_TARGET}"
        assert (
            pcc_first_decode >= _PCC_TARGET
        ), f"{label}: first-decode PCC {pcc_first_decode:.4f} < {_PCC_TARGET}"

    record_bench(result)

    if cpu_baseline:
        # The first model instance now lives on tt; load a fresh one for CPU.
        cpu_model = load_model(llm_model_id, num_layers=llm_num_layers)
        record_bench(
            run_llm_benchmark(
                LLMSamplingWrapper(cpu_model),
                prompt_ids,
                fresh_cache("cpu"),
                torch.arange(prompt_len),
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                label=label,
                mode="eager",
                device="cpu",
            )
        )


# One test per model so the suite is greppable / selectable by name
# (`pytest -k gemma`). Each model declares the parallelism variants it runs via
# its own parametrize list (`-k 'llama and tp'` selects one); all share
# _run_decode_benchmark.
_LLM_FIXTURES = (
    "mode",
    "cpu_baseline",
    "accuracy",
    "llm_batch_size",
    "llm_max_output_tokens",
    "llm_num_layers",
    "profile_enabled",
    "profile_dir",
    "record_bench",
    "tt_device",
)


def _run(request: pytest.FixtureRequest, llm_model_id: str, parallel: str) -> None:
    # dp/tp dispatch DTensor collectives through the tt process group; pull it
    # lazily so the single-chip baseline stays PG-free.
    if parallel != "none":
        request.getfixturevalue("tt_pg")
    # Eager TP is wrong: head-sharded attention mis-shards under DTensor, so the
    # accuracy gate fails (the compile path lowers attention whole and is correct).
    # Non-strict because an accuracy-free run makes no correctness assertion and
    # would otherwise xpass.
    if parallel == "tp" and request.getfixturevalue("mode") == "eager":
        request.node.add_marker(
            pytest.mark.xfail(
                reason="eager TP mis-shards head-sharded SDPA under DTensor",
                strict=False,
            )
        )
    opt_level = request.getfixturevalue("opt_level")
    options = {
        CompileOption.OPT_LEVEL: opt_level if opt_level is not None else 2,
        CompileOption.ENABLE_TRACE: True,
        CompileOption.EXPERIMENTAL_WEIGHT_DTYPE: BfpDtype.BfpBf8,
        CompileOption.EXPERIMENTAL_KV_CACHE_DTYPE: BfpDtype.BfpBf8,
        CompileOption.EXPERIMENTAL_ENABLE_PERMUTE_MATMUL_FUSION: False,
    }
    _run_decode_benchmark(
        llm_model_id,
        parallel=parallel,
        options=options,
        **{name: request.getfixturevalue(name) for name in _LLM_FIXTURES},
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("parallel", ["none", "dp", "tp"])
def test_llama_3_2_1b(parallel: str, request: pytest.FixtureRequest) -> None:
    _run(request, "meta-llama/Llama-3.2-1B-Instruct", parallel)


@pytest.mark.benchmark
@pytest.mark.parametrize("parallel", ["none", "dp", "tp"])
def test_qwen2_5_0_5b(parallel: str, request: pytest.FixtureRequest) -> None:
    _run(request, "Qwen/Qwen2.5-0.5B-Instruct", parallel)


@pytest.mark.benchmark
@pytest.mark.parametrize("parallel", ["none", "dp", "tp"])
def test_gemma_3_1b(parallel: str, request: pytest.FixtureRequest) -> None:
    _run(request, "google/gemma-3-1b-it", parallel)
