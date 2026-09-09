// RUN: ttmlir-opt --split-input-file --ttcore-register-device --convert-d2m-to-ttkernel --convert-d2m-to-ttmetal -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// After spatial merge, only kernels with setup_fabric_connections get
// fabric_config_index.
#l1 = #ttcore.memory_space<l1>
#fabric = #ttcore.fabric_connection_config<noc_index = noc0, topology = ring, cluster_axis = 1, routing_mode = unidir_ring_torus, num_links = 1>
#vgm_11_inv = affine_map<(d0, d1) -> (0, d0 - 1, d1 - 1)>
#vgm_11_fwd = affine_map<(d0, d1, d2, d3) -> (d0 + 1, d1 + 1, d2, d3)>
module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  // CHECK-LABEL: func.func @spatial_fabric_scoped_to_setup_kernel
  // CHECK-COUNT-1: "ttmetal.enqueue_program"
  // CHECK-DAG: fabricConnectionConfigs = [#ttcore.fabric_connection_config<noc_index = noc0, topology = ring, cluster_axis = 1, routing_mode = unidir_ring_torus, num_links = 1>]
  // CHECK-DAG: #ttmetal.noc_config<@dm_fabric, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< >, dm_core = 1, noc0, fabric_config_index = 0
  // CHECK-DAG: #ttmetal.noc_config<@dm_local, #ttmetal.core_range<1x1, 1x1>, #ttmetal.kernel_args< >, dm_core = 1, noc0>
  // CHECK-NOT: #ttmetal.noc_config<@dm_local{{.*}}fabric_config_index
  // CHECK-NOT: d2m.spatial
  func.func @spatial_fabric_scoped_to_setup_kernel(
      %arg0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>,
      %arg1: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
      -> (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>,
          memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>) {
    %out0 = memref.alloc() {alignment = 64 : i64, address = 0x1000} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %out1 = memref.alloc() {alignment = 64 : i64, address = 0x2000, d2m.virtualGridInverseMapping = #vgm_11_inv, d2m.virtualGridForwardMapping = #vgm_11_fwd} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>

    d2m.spatial {grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(1, 1), (1, 1)>]}
        ins(%arg0, %arg1 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
        outs(%out0, %out1 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>) {
      ^region_0:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_fabric, dm_core = 1>], fabricConnectionConfig = #fabric}
            ins(%arg0 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
            outs(%out0 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
      }, {
      ^region_1:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1, virt_to_physical_map = (d0, d1) -> (0, d0 + 1, d1 + 1), physical_to_virt_map = (d0, d1) -> (0, d0 - 1, d1 - 1)>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_local, dm_core = 1>]}
            ins(%arg1 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
            outs(%out1 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
      }
    return %out0, %out1 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
  }

  func.func private @dm_fabric() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec< >, ttkernel.thread = #ttkernel.thread<noc>} {
    %fcm = "ttkernel.experimental.create_fabric_connection_manager"() : () -> !ttkernel.fabric_connection_manager
    "ttkernel.experimental.setup_fabric_connections"(%fcm) : (!ttkernel.fabric_connection_manager) -> ()
    "ttkernel.experimental.close_fabric_connections"(%fcm) : (!ttkernel.fabric_connection_manager) -> ()
    return
  }
  func.func private @dm_local() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec< >, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
}

// -----

// Spatial merge concatenates per-region fabric tables and remaps indices.
#l1 = #ttcore.memory_space<l1>
#fabric0 = #ttcore.fabric_connection_config<noc_index = noc0, topology = ring, cluster_axis = 1, routing_mode = unidir_ring_torus, num_links = 1>
#fabric1 = #ttcore.fabric_connection_config<noc_index = noc0, topology = linear, cluster_axis = 0, routing_mode = bidir_line_mesh, num_links = 1>
#vgm_11_inv = affine_map<(d0, d1) -> (0, d0 - 1, d1 - 1)>
#vgm_11_fwd = affine_map<(d0, d1, d2, d3) -> (d0 + 1, d1 + 1, d2, d3)>
module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  // CHECK-LABEL: func.func @spatial_fabric_multi_config_index_remap
  // CHECK-COUNT-1: "ttmetal.enqueue_program"
  // CHECK-DAG: fabricConnectionConfigs = [#ttcore.fabric_connection_config<noc_index = noc0, topology = ring, cluster_axis = 1, routing_mode = unidir_ring_torus, num_links = 1>, #ttcore.fabric_connection_config<noc_index = noc0, topology = linear, cluster_axis = 0, routing_mode = bidir_line_mesh, num_links = 1>]
  // CHECK-DAG: #ttmetal.noc_config<@dm_fabric0, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< >, dm_core = 1, noc0, fabric_config_index = 0
  // CHECK-DAG: #ttmetal.noc_config<@dm_fabric1, #ttmetal.core_range<1x1, 1x1>, #ttmetal.kernel_args< >, dm_core = 1, noc0, fabric_config_index = 1
  // CHECK-NOT: d2m.spatial
  func.func @spatial_fabric_multi_config_index_remap(
      %arg0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>,
      %arg1: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
      -> (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>,
          memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>) {
    %out0 = memref.alloc() {alignment = 64 : i64, address = 0x1000} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %out1 = memref.alloc() {alignment = 64 : i64, address = 0x2000, d2m.virtualGridInverseMapping = #vgm_11_inv, d2m.virtualGridForwardMapping = #vgm_11_fwd} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>

    d2m.spatial {grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(1, 1), (1, 1)>]}
        ins(%arg0, %arg1 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
        outs(%out0, %out1 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>) {
      ^region_0:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_fabric0, dm_core = 1>], fabricConnectionConfig = #fabric0}
            ins(%arg0 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
            outs(%out0 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
      }, {
      ^region_1:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1, virt_to_physical_map = (d0, d1) -> (0, d0 + 1, d1 + 1), physical_to_virt_map = (d0, d1) -> (0, d0 - 1, d1 - 1)>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_fabric1, dm_core = 1>], fabricConnectionConfig = #fabric1}
            ins(%arg1 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
            outs(%out1 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
      }
    return %out0, %out1 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
  }

  func.func private @dm_fabric0() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec< >, ttkernel.thread = #ttkernel.thread<noc>} {
    %fcm = "ttkernel.experimental.create_fabric_connection_manager"() : () -> !ttkernel.fabric_connection_manager
    "ttkernel.experimental.setup_fabric_connections"(%fcm) : (!ttkernel.fabric_connection_manager) -> ()
    "ttkernel.experimental.close_fabric_connections"(%fcm) : (!ttkernel.fabric_connection_manager) -> ()
    return
  }
  func.func private @dm_fabric1() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec< >, ttkernel.thread = #ttkernel.thread<noc>} {
    %fcm = "ttkernel.experimental.create_fabric_connection_manager"() : () -> !ttkernel.fabric_connection_manager
    "ttkernel.experimental.setup_fabric_connections"(%fcm) : (!ttkernel.fabric_connection_manager) -> ()
    "ttkernel.experimental.close_fabric_connections"(%fcm) : (!ttkernel.fabric_connection_manager) -> ()
    return
  }
}
