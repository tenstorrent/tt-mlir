"""cluster_axis derivation: each PG infers its runtime mesh axis from its
own rank composition.
"""

import pytest
import torch

pytestmark = pytest.mark.multichip


def test_2d_mesh_axes_derived_from_rank_coords(tt_pg, mesh_2d_shape) -> None:
    """Each mesh dim's group infers its runtime axis purely from rank
    coordinates: dim 0 walks a column (→ axis 0), dim 1 walks a row (→ axis 1).
    The square 2x2 case (4 chips) is the strongest check — both dims have the
    same size, so only the coordinates can tell them apart.
    """
    mesh = torch.tt.init_device_mesh(mesh_2d_shape, mesh_dim_names=("dp", "tp"))
    assert mesh.get_group(0).cluster_axis == 0
    assert mesh.get_group(1).cluster_axis == 1
