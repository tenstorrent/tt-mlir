# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

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
