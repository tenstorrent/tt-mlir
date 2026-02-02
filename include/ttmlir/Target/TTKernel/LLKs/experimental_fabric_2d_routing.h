// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef EXPERIMENTAL_FABRIC_2D_ROUTING_H
#define EXPERIMENTAL_FABRIC_2D_ROUTING_H

#ifdef FABRIC_2D

struct UnicastParams {
  // uint16_t dst_dev_id;
  // uint16_t dst_mesh_id,
  uint8_t outgoing_direction;
  uint8_t ns_hops = 0;
  uint8_t ew_hops = 0;
  uint8_t ns_dir;
  uint8_t ew_dir;
};

struct McastParams {
  struct McastParamsPerDirection {
    bool active = false;
    // uint16_t dst_dev_id;
    // uint16_t dst_mesh_id;
    uint16_t e_num_hops = 0;
    uint16_t w_num_hops = 0;
    uint16_t n_num_hops = 0;
    uint16_t s_num_hops = 0;
  };

  std::array<McastParamsPerDirection, MAX_SEND_DIR> params_per_direction;
};

// Forward declarations
FORCE_INLINE UnicastParams get_unicast_params_unidir_ring(
    TopologyInfo &topology, uint16_t my_device_id, uint16_t dst_device_id);
FORCE_INLINE UnicastParams get_unicast_params_unidir_torus(
    TopologyInfo &topology, uint16_t my_device_id, uint16_t dst_device_id);
FORCE_INLINE McastParams get_mcast_params_unidir_ring(
    TopologyInfo &topology, uint16_t my_device_id, uint16_t dst_start_device_id,
    uint16_t dst_end_device_id);
FORCE_INLINE McastParams get_mcast_params_unidir_torus(
    TopologyInfo &topology, uint16_t my_device_id, uint16_t dst_start_device_id,
    uint16_t dst_end_device_id);
FORCE_INLINE McastParams get_mcast_params_line(TopologyInfo &topology,
                                               uint16_t my_device_id,
                                               uint16_t dst_start_device_id,
                                               uint16_t dst_end_device_id);

FORCE_INLINE UnicastParams get_unicast_params(TopologyInfo &topology,
                                              uint16_t my_device_id,
                                              uint16_t dst_device_id) {
  if (topology.topology_type == TopologyInfo::TopologyType::Ring) {
    // Assert: Only UnidirRingTorus routing mode supported for ring topology.
    WAYPOINT("DA34");
    ASSERT(topology.routing_mode == TopologyInfo::RoutingMode::UnidirRingTorus);
    return get_unicast_params_unidir_ring(topology, my_device_id,
                                          dst_device_id);
  } else if (topology.topology_type == TopologyInfo::TopologyType::Torus) {
    // Assert: Only UnidirRingTorus routing mode supported for torus topology.
    WAYPOINT("DA35");
    ASSERT(topology.routing_mode == TopologyInfo::RoutingMode::UnidirRingTorus);
    return get_unicast_params_unidir_torus(topology, my_device_id,
                                           dst_device_id);
  } else {
    // Assert: Unsupported topology type.
    WAYPOINT("DA36");
    ASSERT(false);
    return UnicastParams(); // unreachable, satisfies compiler
  }
}

FORCE_INLINE UnicastParams get_unicast_params_unidir_ring(
    TopologyInfo &topology, uint16_t my_device_id, uint16_t dst_device_id) {
  int32_t my_idx =
      topology.get_logical_mesh_position(my_device_id)[topology.axis];
  int32_t dest_idx =
      topology.get_logical_mesh_position(dst_device_id)[topology.axis];
  int32_t size = topology.mesh_shape[topology.axis];
  WAYPOINT("DA37");
  ASSERT(my_idx != dest_idx);

  bool is_forward =
      (topology.routing_direction == TopologyInfo::RoutingDirection::Forward);
  auto outgoing_direction =
      is_forward ? static_cast<uint32_t>(
                       topology.routing_directions[topology.axis].first)
                 : static_cast<uint32_t>(
                       topology.routing_directions[topology.axis].second);
  auto num_hops = is_forward
                      ? static_cast<uint8_t>((dest_idx - my_idx + size) % size)
                      : static_cast<uint8_t>((my_idx - dest_idx + size) % size);

  UnicastParams result;
  // result.dst_dev_id = dst_device_id;
  // result.dst_mesh_id = 0; // fix???
  if (outgoing_direction == eth_chan_directions::EAST ||
      outgoing_direction == eth_chan_directions::WEST) {
    result.outgoing_direction = outgoing_direction;
    result.ew_hops = num_hops;
    result.ew_dir = outgoing_direction;
  } else if (outgoing_direction == eth_chan_directions::NORTH ||
             outgoing_direction == eth_chan_directions::SOUTH) {
    result.outgoing_direction = outgoing_direction;
    result.ns_hops = num_hops;
    result.ns_dir = outgoing_direction;
  } else {
    WAYPOINT("DA38");
    ASSERT(false);
  }

  return result;
}

FORCE_INLINE UnicastParams get_unicast_params_unidir_torus(
    TopologyInfo &topology, uint16_t my_device_id, uint16_t dst_device_id) {
  WAYPOINT("DA39");
  ASSERT(NUM_DIMS == 2);
  int32_t my_y = topology.get_logical_mesh_position(my_device_id)[0];
  int32_t dest_y = topology.get_logical_mesh_position(dst_device_id)[0];
  int32_t size_y = topology.mesh_shape[0];
  int32_t my_x = topology.get_logical_mesh_position(my_device_id)[1];
  int32_t dest_x = topology.get_logical_mesh_position(dst_device_id)[1];
  int32_t size_x = topology.mesh_shape[1];
  WAYPOINT("DA40");
  ASSERT(!(my_y == dest_y && my_x == dest_x));

  UnicastParams result;
  bool is_forward =
      (topology.routing_direction == TopologyInfo::RoutingDirection::Forward);
  result.ns_dir =
      is_forward ? static_cast<uint32_t>(topology.routing_directions[0].first)
                 : static_cast<uint32_t>(topology.routing_directions[0].second);
  result.ns_hops =
      is_forward ? static_cast<uint8_t>((dest_y - my_y + size_y) % size_y)
                 : static_cast<uint8_t>((my_y - dest_y + size_y) % size_y);
  result.ew_dir =
      is_forward ? static_cast<uint32_t>(topology.routing_directions[1].first)
                 : static_cast<uint32_t>(topology.routing_directions[1].second);
  result.ew_hops =
      is_forward ? static_cast<uint8_t>((dest_x - my_x + size_x) % size_x)
                 : static_cast<uint8_t>((my_x - dest_x + size_x) % size_x);
  result.outgoing_direction =
      result.ns_hops > 0 ? result.ns_dir : result.ew_dir;

  WAYPOINT("DA41");
  ASSERT(result.ns_hops != 0 || result.ew_hops != 0);

  return result;
}

FORCE_INLINE McastParams get_mcast_params(TopologyInfo &topology,
                                          uint16_t my_device_id,
                                          uint16_t dst_start_device_id,
                                          uint16_t dst_end_device_id) {
  if (topology.topology_type == TopologyInfo::TopologyType::Ring) {
    // Assert: Only UnidirRingTorus routing mode supported for ring topology.
    WAYPOINT("DA42");
    ASSERT(topology.routing_mode == TopologyInfo::RoutingMode::UnidirRingTorus);
    return get_mcast_params_unidir_ring(topology, my_device_id,
                                        dst_start_device_id, dst_end_device_id);
  } else if (topology.topology_type == TopologyInfo::TopologyType::Torus) {
    // Assert: Only UnidirRingTorus routing mode supported for torus topology.
    WAYPOINT("DA43");
    ASSERT(topology.routing_mode == TopologyInfo::RoutingMode::UnidirRingTorus);
    return get_mcast_params_unidir_torus(
        topology, my_device_id, dst_start_device_id, dst_end_device_id);
  } else if (topology.topology_type == TopologyInfo::TopologyType::Line) {
    // Assert: Only BidirLineMesh routing mode supported for line topology.
    WAYPOINT("DA44");
    ASSERT(topology.routing_mode == TopologyInfo::RoutingMode::BidirLineMesh);
    return get_mcast_params_line(topology, my_device_id, dst_start_device_id,
                                 dst_end_device_id);
  } else {
    // Assert: Unsupported topology type.
    WAYPOINT("DA45");
    ASSERT(false);
    return McastParams(); // unreachable, satisfies compiler
  }
}

// Note: in line don't use wrap with sender inside; in ring, if sender inside,
// only broadcast to full ring to assure this
FORCE_INLINE McastParams get_mcast_params_line(TopologyInfo &topology,
                                               uint16_t my_device_id,
                                               uint16_t dst_start_device_id,
                                               uint16_t dst_end_device_id) {
  // TODO: check that my_device_id and dst_device_id are in same line/ring
  int32_t my_idx =
      topology.get_logical_mesh_position(my_device_id)[topology.axis];
  int32_t start_idx =
      topology.get_logical_mesh_position(dst_start_device_id)[topology.axis];
  int32_t end_idx =
      topology.get_logical_mesh_position(dst_end_device_id)[topology.axis];
  int32_t size = topology.mesh_shape[topology.axis];

  // Assert: at least one destination (not including sender)
  WAYPOINT("DA46");
  ASSERT(!(my_idx == start_idx && start_idx == end_idx));

  auto [start_fwd, range_fwd, start_bwd, range_bwd] =
      get_line_regions(my_idx, start_idx, end_idx, size);

  McastParams result;
  uint32_t fwd_dir =
      static_cast<uint32_t>(topology.routing_directions[topology.axis].first);
  uint32_t bwd_dir =
      static_cast<uint32_t>(topology.routing_directions[topology.axis].second);

  if (range_fwd != 0) {
    // sender must be inside or adjacent to mcast region
    WAYPOINT("DA47");
    ASSERT(start_fwd == 1);
    result.params_per_direction[fwd_dir].active = true;
    if (fwd_dir == eth_chan_directions::EAST) {
      result.params_per_direction[fwd_dir].e_num_hops = range_fwd;
    } else if (fwd_dir == eth_chan_directions::WEST) {
      result.params_per_direction[fwd_dir].w_num_hops = range_fwd;
    } else if (fwd_dir == eth_chan_directions::NORTH) {
      result.params_per_direction[fwd_dir].n_num_hops = range_fwd;
    } else if (fwd_dir == eth_chan_directions::SOUTH) {
      result.params_per_direction[fwd_dir].s_num_hops = range_fwd;
    } else {
      WAYPOINT("DA48");
      ASSERT(false);
    }
  }
  if (range_bwd != 0) {
    // sender must be inside or adjacent to mcast region
    WAYPOINT("DA49");
    ASSERT(start_bwd == 1);
    result.params_per_direction[bwd_dir].active = true;
    if (bwd_dir == eth_chan_directions::EAST) {
      result.params_per_direction[bwd_dir].e_num_hops = range_bwd;
    } else if (bwd_dir == eth_chan_directions::WEST) {
      result.params_per_direction[bwd_dir].w_num_hops = range_bwd;
    } else if (bwd_dir == eth_chan_directions::NORTH) {
      result.params_per_direction[bwd_dir].n_num_hops = range_bwd;
    } else if (bwd_dir == eth_chan_directions::SOUTH) {
      result.params_per_direction[bwd_dir].s_num_hops = range_bwd;
    } else {
      WAYPOINT("DA50");
      ASSERT(false);
    }
  }

  return result;
}

// 2d fabric mcast limitation: sender must be inside or adjacent to mcast region
FORCE_INLINE McastParams get_mcast_params_unidir_ring(
    TopologyInfo &topology, uint16_t my_device_id, uint16_t dst_start_device_id,
    uint16_t dst_end_device_id) {
  // TODO: check that my_device_id and dst_device_id are in same ring
  int32_t my_idx =
      topology.get_logical_mesh_position(my_device_id)[topology.axis];
  int32_t start_idx =
      topology.get_logical_mesh_position(dst_start_device_id)[topology.axis];
  int32_t end_idx =
      topology.get_logical_mesh_position(dst_end_device_id)[topology.axis];
  int32_t size = topology.mesh_shape[topology.axis];

  // Assert: at least one destination (not including sender)
  WAYPOINT("DA51");
  ASSERT(!(my_idx == start_idx && start_idx == end_idx));

  McastParams result;
  bool is_forward =
      (topology.routing_direction == TopologyInfo::RoutingDirection::Forward);
  uint32_t dir = is_forward
                     ? static_cast<uint32_t>(
                           topology.routing_directions[topology.axis].first)
                     : static_cast<uint32_t>(
                           topology.routing_directions[topology.axis].second);

  auto [start_1, range_1, start_2_gap, range_2] =
      get_ring_regions(my_idx, start_idx, end_idx, size, is_forward);
  // sender must be inside or adjacent to mcast region
  WAYPOINT("DA52");
  ASSERT(start_1 == 1);
  // gap not supported
  WAYPOINT("DA53");
  ASSERT(start_2_gap == 1);

  result.params_per_direction[dir].active = true;
  // result.params_per_direction[dir].dst_dev_id = dst_start_device_id;
  // result.params_per_direction[dir].dst_mesh_id = 0; // fix???
  if (dir == eth_chan_directions::EAST) {
    result.params_per_direction[dir].e_num_hops = range_1 + range_2;
  } else if (dir == eth_chan_directions::WEST) {
    result.params_per_direction[dir].w_num_hops = range_1 + range_2;
  } else if (dir == eth_chan_directions::NORTH) {
    result.params_per_direction[dir].n_num_hops = range_1 + range_2;
  } else if (dir == eth_chan_directions::SOUTH) {
    result.params_per_direction[dir].s_num_hops = range_1 + range_2;
  } else {
    WAYPOINT("DA54");
    ASSERT(false);
  }

  return result;
}

FORCE_INLINE McastParams get_mcast_params_unidir_torus(
    TopologyInfo &topology, uint16_t my_device_id, uint16_t dst_start_device_id,
    uint16_t dst_end_device_id) {
  WAYPOINT("DA55");
  ASSERT(NUM_DIMS == 2);
  int32_t my_y = topology.get_logical_mesh_position(my_device_id)[0];
  int32_t start_y = topology.get_logical_mesh_position(dst_start_device_id)[0];
  int32_t end_y = topology.get_logical_mesh_position(dst_end_device_id)[0];
  int32_t size_y = topology.mesh_shape[0];
  int32_t my_x = topology.get_logical_mesh_position(my_device_id)[1];
  int32_t start_x = topology.get_logical_mesh_position(dst_start_device_id)[1];
  int32_t end_x = topology.get_logical_mesh_position(dst_end_device_id)[1];
  int32_t size_x = topology.mesh_shape[1];

  // Assert: at least one destination (not including sender)
  WAYPOINT("DA56");
  ASSERT(!(my_device_id == dst_start_device_id &&
           dst_start_device_id == dst_end_device_id));

  McastParams result;
  bool is_forward =
      (topology.routing_direction == TopologyInfo::RoutingDirection::Forward);
  uint32_t ns_dir =
      is_forward ? static_cast<uint32_t>(topology.routing_directions[0].first)
                 : static_cast<uint32_t>(topology.routing_directions[0].second);

  uint32_t ew_dir =
      is_forward ? static_cast<uint32_t>(topology.routing_directions[1].first)
                 : static_cast<uint32_t>(topology.routing_directions[1].second);

  bool ns_region_exists = !(my_y == start_y && start_y == end_y);
  bool ew_region_exists = !(my_x == start_x && start_x == end_x);
  bool ns_region_contains_my_y = ((my_y - start_y + size_y) % size_y) <
                                 ((end_y - start_y + size_y) % size_y);
  int32_t ns_start_1 = 0, ns_range_1 = 0, ns_start_2_gap = 0, ns_range_2 = 0;
  int32_t ew_start_1 = 0, ew_range_1 = 0, ew_start_2_gap = 0, ew_range_2 = 0;
  if (ns_region_exists) {
    std::tie(ns_start_1, ns_range_1, ns_start_2_gap, ns_range_2) =
        get_ring_regions(my_y, start_y, end_y, size_y, is_forward);
    // sender must be inside or adjacent to mcast region
    WAYPOINT("DA57");
    ASSERT(ns_start_1 == 1);
    // gap not supported
    WAYPOINT("DA58");
    ASSERT(ns_start_2_gap == 1);
  }
  if (ew_region_exists) {
    std::tie(ew_start_1, ew_range_1, ew_start_2_gap, ew_range_2) =
        get_ring_regions(my_x, start_x, end_x, size_x, is_forward);
    // sender must be inside or adjacent to mcast region
    WAYPOINT("DA59");
    ASSERT(ew_start_1 == 1);
    WAYPOINT("DA60");
    ASSERT(ew_start_2_gap == 1);
  }

  // TODO: i don't like that we're explicitly checking north and south here
  // maybe can have a function function to set correct hop field or abstract to
  // array indexed by dir same issue with setting hops exists in other functions
  if (ns_region_exists) {
    result.params_per_direction[ns_dir].active = true;
    if (ns_dir == eth_chan_directions::NORTH) {
      result.params_per_direction[ns_dir].n_num_hops = ns_range_1 + ns_range_2;
    } else if (ns_dir == eth_chan_directions::SOUTH) {
      result.params_per_direction[ns_dir].s_num_hops = ns_range_1 + ns_range_2;
    } else {
      WAYPOINT("DA61");
      ASSERT(false);
    }

    if (ew_region_exists) {
      if (ew_dir == eth_chan_directions::EAST) {
        result.params_per_direction[ns_dir].e_num_hops =
            ew_range_1 + ew_range_2;
      } else if (ew_dir == eth_chan_directions::WEST) {
        result.params_per_direction[ns_dir].w_num_hops =
            ew_range_1 + ew_range_2;
      } else {
        WAYPOINT("DA62");
        ASSERT(false);
      }
    }
  }

  if (ew_region_exists && (!ns_region_exists || ns_region_contains_my_y)) {
    result.params_per_direction[ew_dir].active = true;
    if (ew_dir == eth_chan_directions::EAST) {
      result.params_per_direction[ew_dir].e_num_hops = ew_range_1 + ew_range_2;
    } else if (ew_dir == eth_chan_directions::WEST) {
      result.params_per_direction[ew_dir].w_num_hops = ew_range_1 + ew_range_2;
    } else {
      WAYPOINT("DA63");
      ASSERT(false);
    }
  }

  WAYPOINT("DA64");
  ASSERT(ns_region_exists || ew_region_exists);

  return result;
}

// Custom API for unidir routing on 2d fabric (mcast uses normal function)
// This doesn't support intermesh routing in 2d fabric
// since that requires explicit support from fabric to enable use a different
// routing path for es/ns mode since routes on intermediate (and dst) meshes are
// generated on routers and not by sender core for consistency to add inter-mesh
// support we would have switch use shortest path routing mode
void fabric_set_unicast_route_custom(
    volatile tt_l1_ptr HybridMeshPacketHeader *packet_header,
    uint16_t dst_dev_id, uint16_t dst_mesh_id, uint8_t ns_hops, uint8_t ew_hops,
    uint8_t ns_dir, uint8_t ew_dir) {
  packet_header->dst_start_node_id =
      ((uint32_t)dst_mesh_id << 16) | (uint32_t)dst_dev_id;
  packet_header->mcast_params_64 = 0;
  packet_header->is_mcast_active = 0;

  // Use canonical 2D encoder to generate route buffer
  // Note: Buffer size is determined by packet header template
  routing_encoding::encode_2d_unicast(
      ns_hops, ew_hops, ns_dir, ew_dir,
      const_cast<uint8_t *>(packet_header->route_buffer),
      FabricHeaderConfig::MESH_ROUTE_BUFFER_SIZE,
      false // not a router
  );

  packet_header->routing_fields.value = 0;
  if (ns_hops > 0 && ew_hops > 0) {
    // 2D routing: turn from NS to EW at turn_point
    if (ew_dir) {
      packet_header->routing_fields.branch_east_offset =
          ns_hops; // turn to EAST after NS
    } else {
      packet_header->routing_fields.branch_west_offset =
          ns_hops; // turn to WEST after NS
    }
  } else if (ns_hops > 0) {
    packet_header->routing_fields.branch_east_offset = ns_hops;
    packet_header->routing_fields.branch_west_offset = ns_hops;
  } else if (ew_hops > 0) {
    // East/West only routing: branch offset is set at position 1 (start_hop +
    // 1)
    if (ew_dir) {
      packet_header->routing_fields.branch_east_offset =
          1; // East only: branch at hop 1
    } else {
      packet_header->routing_fields.branch_west_offset =
          1; // West only: branch at hop 1
    }
  } else {
    WAYPOINT("DA65");
    ASSERT(false);
  }
}

#endif // FABRIC_2D

#endif // EXPERIMENTAL_FABRIC_2D_ROUTING_H
