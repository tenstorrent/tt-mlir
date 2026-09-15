# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tt_crank.torch import _compile, _native
from tt_crank.torch.testing import strict_no_fallback

_TOLERANCE = {
    torch.bfloat16: dict(rtol=5e-2, atol=6e-2),
    torch.float32: dict(rtol=1e-2, atol=5e-3),
}
_FUSED_KWARGS = dict(
    lr=1e-2,
    beta1=0.9,
    beta2=0.999,
    weight_decay=0.0,
    eps=1e-8,
    amsgrad=False,
    maximize=False,
)
_aten = torch.ops.aten
_LOWERING = _compile._LOWERINGS[_aten._fused_adamw.default]
# At lr=1e-2 the update is below bf16 atol, so a wrong beta^t would hide; lr=1 makes it visible.
_AMPLIFIED_LR = 1.0


def _close(actual, expected, dtype=torch.bfloat16):
    torch.testing.assert_close(actual.cpu(), expected, **_TOLERANCE[dtype])


def _state(shape=(32, 32), dtype=torch.bfloat16, device="tt", fill=None):
    tensor = (
        torch.zeros(shape, dtype=dtype)
        if fill == 0
        else torch.randn(shape, dtype=dtype)
    )
    return tensor.to(device)


def _fused_adamw(
    params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, steps, **overrides
):
    torch._fused_adamw_(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        steps,
        **{**_FUSED_KWARGS, **overrides},
    )


def _cpu_fused_reference(params, grads, steps, **overrides):
    out = [p.clone() for p in params]
    exp_avgs = [torch.zeros_like(p) for p in params]
    exp_avg_sqs = [torch.zeros_like(p) for p in params]
    torch._fused_adamw_(
        out,
        [g.clone() for g in grads],
        exp_avgs,
        exp_avg_sqs,
        [],
        [torch.tensor(float(s)) for s in steps],
        **{**_FUSED_KWARGS, **overrides},
    )
    return out, exp_avgs, exp_avg_sqs


def _tt_fused_step(params, grads, steps, **overrides):
    tt_params = [p.to("tt") for p in params]
    tt_exp_avgs = [torch.zeros_like(p).to("tt") for p in params]
    tt_exp_avg_sqs = [torch.zeros_like(p).to("tt") for p in params]
    _fused_adamw(
        tt_params,
        [g.to("tt") for g in grads],
        tt_exp_avgs,
        tt_exp_avg_sqs,
        [],
        [torch.tensor(float(s)) for s in steps],
        **overrides,
    )
    return tt_params, tt_exp_avgs, tt_exp_avg_sqs


def _run_steps(shapes, dtype=torch.bfloat16, steps=3, lr=1e-2, **options):
    init = [torch.randn(shape, dtype=dtype) for shape in shapes]
    cpu_params = [p.clone() for p in init]
    tt_params = [p.to("tt") for p in init]
    cpu_opt = torch.optim.AdamW(cpu_params, lr=float(lr), fused=False, **options)
    tt_opt = torch.optim.AdamW(tt_params, lr=lr, fused=True, **options)
    for _ in range(steps):
        for cpu_param, tt_param in zip(cpu_params, tt_params):
            grad = torch.randn_like(cpu_param)
            cpu_param.grad = grad
            tt_param.grad = grad.to("tt")
        cpu_opt.step()
        tt_opt.step()

    for cpu_param, tt_param in zip(cpu_params, tt_params):
        _close(tt_param, cpu_param, dtype)
        for key, cpu_state in cpu_opt.state[cpu_param].items():
            tt_state = tt_opt.state[tt_param][key]
            if key == "step":
                assert tt_state.cpu().item() == cpu_state.item()
            else:
                _close(tt_state, cpu_state, dtype)


@pytest.mark.parametrize(
    "shapes",
    [[(32, 32)], [(64, 128)], [(32, 64, 32)], [(32, 32), (64, 128), (32, 64, 32)]],
)
def test_adamw_matches_cpu(shapes: list) -> None:
    _run_steps(shapes)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_adamw_dtypes(dtype: torch.dtype) -> None:
    _run_steps([(32, 32)], dtype=dtype)


@pytest.mark.parametrize(
    "options",
    [
        dict(amsgrad=True),
        dict(maximize=True),
        dict(weight_decay=0.1),
        dict(betas=(0.8, 0.95), eps=1e-6),
        dict(amsgrad=True, maximize=True, weight_decay=0.1),
        dict(lr=torch.tensor(1e-2)),
    ],
)
def test_adamw_options(options: dict) -> None:
    _run_steps([(32, 32), (64, 128)], **options)


@pytest.mark.parametrize(
    "options",
    [
        {},
        dict(amsgrad=True),
        dict(lr=torch.tensor(1e-2)),
        dict(lr=torch.tensor(1e-2, device="tt")),
    ],
)
def test_adamw_compiled_step(options: dict) -> None:
    init = [torch.randn(shape, dtype=torch.bfloat16) for shape in [(32, 32), (64, 128)]]
    cpu_params = [p.clone() for p in init]
    tt_params = [p.to("tt") for p in init]
    cpu_opt = torch.optim.AdamW(cpu_params, fused=False, **{**options, "lr": 1e-2})
    tt_opt = torch.optim.AdamW(tt_params, fused=True, **{"lr": 1e-2, **options})
    graphs = []
    _compile._post_aot_fx_hook = lambda gm: graphs.append(
        [n.target for n in gm.graph.nodes if n.op == "call_function"]
    )
    try:
        step = torch.compile(tt_opt.step, backend="tt")
        for _ in range(3):
            for cpu_param, tt_param in zip(cpu_params, tt_params):
                cpu_param.grad = torch.randn_like(cpu_param)
                tt_param.grad = cpu_param.grad.to("tt")
            cpu_opt.step()
            step()
    finally:
        _compile._post_aot_fx_hook = None
    assert len(graphs) == 1 and any(
        t in (_aten._fused_adamw.default, _aten._fused_adamw.tensor_lr)
        for t in graphs[0]
    )
    for cpu_param, tt_param in zip(cpu_params, tt_params):
        _close(tt_param, cpu_param)
        _close(tt_opt.state[tt_param]["exp_avg"], cpu_opt.state[cpu_param]["exp_avg"])
        assert tt_opt.state[tt_param]["step"].cpu().item() == 3


def test_adamw_lr_schedule() -> None:
    init = torch.randn((32, 32), dtype=torch.bfloat16)
    cpu_param, tt_param = init.clone(), init.to("tt")
    cpu_opt = torch.optim.AdamW([cpu_param], lr=1e-2, fused=False)
    tt_opt = torch.optim.AdamW([tt_param], lr=1e-2, fused=True)
    for lr in (1e-2, 5e-3, 1e-3):
        cpu_opt.param_groups[0]["lr"] = tt_opt.param_groups[0]["lr"] = lr
        cpu_param.grad = torch.randn_like(cpu_param)
        tt_param.grad = cpu_param.grad.to("tt")
        cpu_opt.step()
        tt_opt.step()
    _close(tt_param, cpu_param)


def test_adamw_stays_on_device_and_keeps_state_objects() -> None:
    param = _state()
    param.grad = _state()
    optimizer = torch.optim.AdamW([param], lr=1e-2, fused=True)
    with strict_no_fallback():
        optimizer.step()
    exp_avg = optimizer.state[param]["exp_avg"]
    before = exp_avg.cpu().clone()
    param.grad = _state()
    optimizer.step()
    assert optimizer.state[param]["exp_avg"] is exp_avg
    assert not torch.equal(exp_avg.cpu(), before)


def test_adamw_uploads_cpu_grad() -> None:
    param = _state()
    before = param.cpu().clone()
    _fused_adamw(
        [param],
        [_state(device="cpu")],
        [_state(fill=0)],
        [_state(fill=0)],
        [],
        [torch.tensor(1.0)],
    )
    assert not torch.equal(param.cpu(), before)


def test_adamw_differing_steps_across_params() -> None:
    shapes, steps = [(32, 32), (64, 128)], [1.0, 7.0]
    params = [torch.randn(shape, dtype=torch.bfloat16) for shape in shapes]
    grads = [torch.randn(shape, dtype=torch.bfloat16) for shape in shapes]
    tt = _tt_fused_step(params, grads, steps, lr=_AMPLIFIED_LR)
    cpu = _cpu_fused_reference(params, grads, steps, lr=_AMPLIFIED_LR)
    for tt_group, cpu_group in zip(tt, cpu):
        for actual, expected in zip(tt_group, cpu_group):
            _close(actual, expected)


@pytest.mark.parametrize("step", [100.0, 5000.0])
def test_adamw_large_step_count(step: float) -> None:
    params, grads = [torch.randn((32, 32), dtype=torch.bfloat16)], [
        torch.randn((32, 32), dtype=torch.bfloat16)
    ]
    tt = _tt_fused_step(params, grads, [step], lr=_AMPLIFIED_LR)
    cpu = _cpu_fused_reference(params, grads, [step], lr=_AMPLIFIED_LR)
    for tt_group, cpu_group in zip(tt, cpu):
        _close(tt_group[0], cpu_group[0])


def test_adamw_mixed_dtypes_across_params() -> None:
    dtypes = [torch.bfloat16, torch.float32]
    params = [torch.randn((32, 32), dtype=dtype) for dtype in dtypes]
    grads = [torch.randn((32, 32), dtype=dtype) for dtype in dtypes]
    tt_params, tt_exp_avgs, _ = _tt_fused_step(params, grads, [1.0, 1.0])
    cpu_params, cpu_exp_avgs, _ = _cpu_fused_reference(params, grads, [1.0, 1.0])
    for i, dtype in enumerate(dtypes):
        assert tt_params[i].dtype == dtype
        # grad is cast to bf16 in the kernel, so f32 params only reach bf16 accuracy
        _close(tt_params[i], cpu_params[i])
        _close(tt_exp_avgs[i], cpu_exp_avgs[i])


@pytest.mark.parametrize("position", ["param", "exp_avg", "exp_avg_sq"])
def test_adamw_rejects_host_in_place_operand(position: str) -> None:
    operands = {
        "param": _state(),
        "exp_avg": _state(fill=0),
        "exp_avg_sq": _state(fill=0),
    }
    operands[position] = operands[position].cpu()
    with pytest.raises(RuntimeError, match=f"{position}\\[0\\] is updated in place"):
        _fused_adamw(
            [operands["param"]],
            [_state()],
            [operands["exp_avg"]],
            [operands["exp_avg_sq"]],
            [],
            [torch.tensor(1.0)],
        )


@pytest.mark.parametrize(
    "case, match",
    [
        (
            dict(exp_avgs=[_state(dtype=torch.float32, fill=0)]),
            "exp_avg has type tensor<32x32xf32>",
        ),
        (dict(grads=[_state(shape=(64, 64))]), "grad has type tensor<64x64xbf16>"),
        (dict(params=[_state(), _state()]), "must have equal length"),
        (dict(amsgrad=True), "must have equal length"),
        (dict(grad_scale=torch.tensor(1.0)), "AMP gradient scaling"),
        (dict(found_inf=torch.tensor(1.0)), "AMP gradient scaling"),
    ],
)
def test_adamw_rejects(case: dict, match: str) -> None:
    args = dict(
        params=[_state()],
        grads=[_state()],
        exp_avgs=[_state(fill=0)],
        exp_avg_sqs=[_state(fill=0)],
        max_exp_avg_sqs=[],
        steps=[torch.tensor(1.0)],
    )
    args.update(case)
    with pytest.raises(RuntimeError, match=match):
        _fused_adamw(**args)


def _builder(tensors):
    specs = [_compile._spec_from_tensor(t) for t in tensors]
    mb = _native.ModuleBuilder(specs, [_native.ArgumentType.Input] * len(specs))
    return mb, [mb.arg(i) for i in range(len(specs))]


def _lower(mb, args, count, steps, *, amsgrad=False, lr=1e-2, **kwargs):
    group = 5 if amsgrad else 4
    columns = [[args[i * group + k] for i in range(count)] for k in range(group)]
    return _LOWERING(
        mb,
        columns[0],
        columns[1],
        columns[2],
        columns[3],
        columns[4] if amsgrad else [],
        steps,
        lr=lr,
        **{
            **{k: v for k, v in _FUSED_KWARGS.items() if k != "lr"},
            "amsgrad": amsgrad,
            **kwargs,
        },
    )


def _lowering_tensors(shapes, dtype=torch.bfloat16, amsgrad=False):
    tensors = []
    for shape in shapes:
        tensors += [torch.randn(shape, dtype=dtype), torch.randn(shape, dtype=dtype)]
        tensors += [torch.zeros(shape, dtype=dtype)] * (3 if amsgrad else 2)
    return tensors


def _run(mb, outputs, tensors, dtype=torch.bfloat16):
    program = mb.compile(outputs, _native.CompileOptions()).program
    dtypes = [_compile._to_runtime_dtype(dtype)] * len(outputs)
    return _native.run_program(program, [t.to("tt") for t in tensors], dtypes)


@pytest.mark.parametrize(
    "options",
    [
        {},
        dict(amsgrad=True),
        dict(maximize=True),
        dict(weight_decay=0.1),
        dict(dtype=torch.float32),
    ],
)
def test_compile_lowering_matches_cpu(options: dict) -> None:
    dtype = options.pop("dtype", torch.bfloat16)
    amsgrad = options.get("amsgrad", False)
    tensors = _lowering_tensors([(32, 32)], dtype, amsgrad)
    mb, args = _builder(tensors)
    result = _lower(mb, args, 1, [torch.tensor(1.0)], **options)
    outputs = [result[0][0], result[2][0], result[3][0]] + (
        [result[4][0]] if amsgrad else []
    )
    produced = _run(mb, outputs, tensors, dtype)

    reference = tensors[0].clone()
    reference.grad = tensors[1].clone()
    optimizer = torch.optim.AdamW([reference], lr=1e-2, fused=False, **options)
    optimizer.step()
    _close(produced[0], reference, dtype)
    for out, key in zip(produced[1:], ["exp_avg", "exp_avg_sq", "max_exp_avg_sq"]):
        _close(out, optimizer.state[reference][key], dtype)


@pytest.mark.parametrize("steps", [[1.0, 1.0], [1.0, 4.0]])
def test_compile_lowering_multiple_params(steps: list) -> None:
    shapes = [(32, 32), (64, 128)]
    tensors = _lowering_tensors(shapes)
    mb, args = _builder(tensors)
    result = _lower(mb, args, 2, [torch.tensor(s) for s in steps], lr=_AMPLIFIED_LR)
    produced = _run(mb, result[0] + result[2], tensors)
    params, grads = tensors[0::4], tensors[1::4]
    cpu_params, cpu_exp_avgs, _ = _cpu_fused_reference(
        params, grads, steps, lr=_AMPLIFIED_LR
    )
    for i in range(2):
        assert list(produced[i].shape) == list(shapes[i])
        _close(produced[i], cpu_params[i])
        _close(produced[2 + i], cpu_exp_avgs[i])


def test_compile_lowering_one_program_many_steps() -> None:
    tensors = _lowering_tensors([(32, 32)]) + [
        torch.tensor([1.0]),
        torch.tensor([1e-2]),
    ]
    mb, args = _builder(tensors)
    result = _lower(mb, args, 1, [args[4]], lr=args[5])
    compiled = mb.compile(
        [result[0][0], result[2][0], result[3][0]], _native.CompileOptions(), True
    )
    assert (
        compiled.ttir.count('"ttir.pow"') == 2
        and "weight_decay = 0.000000e+00 : f32" in compiled.ttir
    )
    program = compiled.program

    reference = tensors[0].clone()
    reference.grad = tensors[1].clone()
    optimizer = torch.optim.AdamW([reference], lr=1e-2, fused=False)
    tt = [t.to("tt") for t in tensors]
    for step in range(1, 4):
        tt[4] = torch.tensor([float(step)]).to("tt")
        tt[0], tt[2], tt[3] = _native.run_program(
            program, tt, [_compile._to_runtime_dtype(torch.bfloat16)] * 3
        )
        optimizer.step()
    _close(tt[0], reference)
    _close(tt[2], optimizer.state[reference]["exp_avg"])
    _close(tt[3], optimizer.state[reference]["exp_avg_sq"])


def test_compile_lowering_casts_grad_to_bf16() -> None:
    mb, args = _builder(_lowering_tensors([(32, 32)], torch.float32))
    result = _lower(mb, args, 1, [torch.tensor(1.0)])
    ttir = mb.compile(
        [result[0][0], result[2][0], result[3][0]], _native.CompileOptions(), True
    ).ttir
    assert 'ttir.typecast"(%arg1)' in ttir and "-> tensor<32x32xbf16>" in ttir
    assert (
        "(tensor<32x32xf32>, tensor<32x32xbf16>, tensor<32x32xf32>, tensor<32x32xf32>, "
        "tensor<1xf32>, tensor<1xf32>, tensor<1xf32>)"
    ) in ttir


@pytest.mark.parametrize("kwarg", ["grad_scale", "found_inf"])
def test_compile_lowering_rejects_amp_scaling(kwarg: str) -> None:
    mb, args = _builder(_lowering_tensors([(32, 32)]))
    with pytest.raises(NotImplementedError, match="AMP gradient scaling"):
        _lower(mb, args, 1, [torch.tensor(1.0)], **{kwarg: torch.tensor(1.0)})
