# Auto-inject ttnn_supplemental items into ttnn namespace
def inject_into_ttnn():
    """Inject ttnn_supplemental exports into the ttnn namespace"""
    try:
        import ttnn
        from . import ttnn_supplemental

        # Add enums
        ttnn.MeshShardDirection = ttnn_supplemental.MeshShardDirection
        ttnn.MeshShardType = ttnn_supplemental.MeshShardType

        # Add functions
        ttnn.mesh_shard = ttnn_supplemental.mesh_shard

    except ImportError:
        pass  # ttnn not available, skip injection

# Optionally call this on import
# inject_into_ttnn()
