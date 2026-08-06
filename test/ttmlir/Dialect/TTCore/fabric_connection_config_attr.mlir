// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt %s | ttmlir-opt | FileCheck %s

// CHECK-DAG: #ttcore.fabric_connection_config<noc_index = noc0, topology = linear, cluster_axis = 1, routing_mode = bidir_line_mesh, num_links = 1>
#all_cores = #ttcore.fabric_connection_config<noc_index = noc0, topology = linear, cluster_axis = 1, routing_mode = bidir_line_mesh, num_links = 1>

// CHECK-DAG: #ttcore.fabric_connection_config<noc_index = noc1, topology = ring, cluster_axis = 0, routing_mode = unidir_ring_torus, num_links = 1, router_cores = [0, 0, 1, 0]>
#routed = #ttcore.fabric_connection_config<noc_index = noc1, topology = ring, cluster_axis = 0, routing_mode = unidir_ring_torus, num_links = 1, router_cores = [0, 0, 1, 0]>

module attributes {ttcore.all_cores = #all_cores, ttcore.routed = #routed} {}
