import torch

import tt_kurbla.torch  # noqa: F401  — import registers the backend


def tied_model() -> torch.nn.Sequential:
    embedding = torch.nn.Embedding(8, 4)
    head = torch.nn.Linear(4, 8, bias=False)
    head.weight = embedding.weight
    return torch.nn.Sequential(embedding, head)


# Regression test for issue #123: weight tying must survive `Module.to("tt")`.
def test_tying_survives_move_to_tt():
    assert torch.__future__.get_swap_module_params_on_conversion()

    model = tied_model()
    assert model[1].weight is model[0].weight
    numel_cpu = sum(p.numel() for p in model.parameters())

    model.to("tt")

    assert model[0].weight.device.type == "tt"
    assert model[1].weight is model[0].weight
    # parameters() dedups by identity, so severed ties double the count.
    assert sum(p.numel() for p in model.parameters()) == numel_cpu


def test_optimizer_references_survive_move():
    # Swap mode updates parameters in place, so an optimizer built before the
    # move must end up holding the moved (tt) parameters, not orphaned CPU ones.
    model = tied_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    model.to("tt")

    assert all(p.device.type == "tt" for group in optimizer.param_groups for p in group["params"])
    assert optimizer.param_groups[0]["params"][0] is model[0].weight
