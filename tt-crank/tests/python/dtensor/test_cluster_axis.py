"""cluster_axis derivation: each PG infers its runtime mesh axis from its
own rank composition.
"""

import pytest
import torch

pytestmark = pytest.mark.multichip


def test_2d_mesh_axes_derived_from_rank_coords(tt_pg) -> None:
    """On a square 2x2 mesh both dim groups have size 2, so only the rank
    coordinates can tell them apart: dim 0 walks a column ({0,2} → axis 0),
    dim 1 walks a row ({0,1} → axis 1).
    """
    if torch.tt.num_chips() < 4:
        pytest.skip("2x2 mesh needs at least 4 chips")
    mesh = torch.tt.init_device_mesh((2, 2), mesh_dim_names=("dp", "tp"))
    assert mesh.get_group(0).cluster_axis == 0
    assert mesh.get_group(1).cluster_axis == 1
