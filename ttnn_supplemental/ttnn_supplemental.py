"""
TTNN Supplemental CCL Operations Auto-Injector

Import this module to automatically inject all ttnn_supplemental CCL operations
into the ttnn namespace.

Usage:
    import ttnn
    import ttnn_supplemental_ccl  # This automatically injects everything

    # Now use directly:
    ttnn.all_gather(...)
    ttnn.mesh_shard(...)
"""

try:
    import ttnn
    import ttnn_supplemental

    # Add enums
    ttnn.MeshShardDirection = ttnn_supplemental.MeshShardDirection
    ttnn.MeshShardType = ttnn_supplemental.MeshShardType

    # Add CCL operations
    ttnn.mesh_shard = ttnn_supplemental.mesh_shard
    ttnn.all_gather = ttnn_supplemental.all_gather
    ttnn.reduce_scatter = ttnn_supplemental.reduce_scatter
    ttnn.collective_permute = ttnn_supplemental.collective_permute
    ttnn.point_to_point = ttnn_supplemental.point_to_point

except ImportError as e:
    import sys
    print(f"Warning: Could not inject ttnn_supplemental into ttnn namespace: {e}", file=sys.stderr)
