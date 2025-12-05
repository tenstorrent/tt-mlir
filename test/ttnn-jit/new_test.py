import ttnn_jit
import ttnn
import torch
import pytest
from utils import (
    _get_ttnn_op,
    all_close_check,
    memory_configs_equal,
    create_dram_tensor,
    create_sharded_tile_tensor,
    run_op_test,
)


def add(input_tensor_a, input_tensor_b):
    return ttnn.add(input_tensor_a, input_tensor_b)


@pytest.mark.parametrize(
    "shape, grid1, grid2",
    [
        # square
        ((64, 64), (0, 0), (1, 1)),
        ((96, 192), (0, 0), (2, 2)),
        ((128, 128), (1, 1), (0, 0)),
        ((192, 192), (2, 2), (0, 0)),
        # non square
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("op", [add])
@pytest.mark.xfail(
    reason="L1 sharded tensors with varying grids are not yet supported.",
)
def test_l1_varying_grids(device, shape, grid1, grid2, dtype, op):

    input_sharded_tensor1 = create_sharded_tile_tensor(
        device, shape, max_grid=grid1, dtype=dtype
    )
    input_sharded_tensor2 = create_sharded_tile_tensor(
        device, shape, max_grid=grid2, dtype=dtype
    )

    golden_op = _get_ttnn_op(op)

    op_jit = ttnn_jit.jit(debug=True)(op)
    output_tensor = op_jit(input_sharded_tensor1, input_sharded_tensor2)
    print("after jit")
    golden_tensor = (golden_op or op)(
        input_sharded_tensor1, input_sharded_tensor2)

    assert all_close_check(output_tensor, golden_tensor)
