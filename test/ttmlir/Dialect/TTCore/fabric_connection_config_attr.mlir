// RUN: ttmlir-opt %s | ttmlir-opt | FileCheck %s

// The legacy form (no router_cores) must round-trip unchanged: the optional
// router_cores group is absent, so existing CCL kernels are unaffected.
// CHECK-DAG: #ttcore.fabric_connection_config<noc_index = noc0, topology = linear, cluster_axis = 1, routing_mode = bidir_line_mesh, num_links = 1>
#legacy = #ttcore.fabric_connection_config<noc_index = noc0, topology = linear, cluster_axis = 1, routing_mode = bidir_line_mesh, num_links = 1>

// router_cores is a flat list of (y, x) pairs, one per (link, direction) slot.
// CHECK-DAG: #ttcore.fabric_connection_config<noc_index = noc0, topology = ring, cluster_axis = 1, routing_mode = unidir_ring_torus, num_links = 1, router_cores = [0, 0, 1, 0]>
#routed = #ttcore.fabric_connection_config<noc_index = noc0, topology = ring, cluster_axis = 1, routing_mode = unidir_ring_torus, num_links = 1, router_cores = [0, 0, 1, 0]>

module attributes {ttcore.legacy = #legacy, ttcore.routed = #routed} {}
