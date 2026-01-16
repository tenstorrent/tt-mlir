// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0


#ifndef TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_FABRIC_API_H
#define TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_FABRIC_API_H

#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

namespace experimental {

///////////////////////////////////// Helper Functions /////////////////////////////////////

// hardcoded defaults for now
static constexpr uint32_t max_fabric_payload_size = 4352;
static constexpr size_t num_connection_arg_idx = 0;
static constexpr size_t fabric_connection_arg_idx = 1;

FORCE_INLINE uint32_t calculate_initial_direction(uint16_t dst_chip_id, uint16_t my_chip_id) {
    auto* routing_info = reinterpret_cast<tt_l1_ptr intra_mesh_routing_path_t<2, true>*>(ROUTING_PATH_BASE_2D);

    uint32_t initial_dir = static_cast<uint32_t>(eth_chan_directions::EAST);

    const auto& compressed_route = routing_info->paths[dst_chip_id];
    uint8_t ns_hops = compressed_route.get_ns_hops();
    uint8_t ew_hops = compressed_route.get_ew_hops();

    if (ns_hops > 0) {
        // is there another way to know whether it's north or south hops?
        if (dst_chip_id < my_chip_id) {
            initial_dir = static_cast<uint32_t>(eth_chan_directions::NORTH);
        } else {
            initial_dir = static_cast<uint32_t>(eth_chan_directions::SOUTH);
        }
    } else if (ew_hops > 0) {
        // is there another way to know whether it's east or west hops?
        if (dst_chip_id < my_chip_id) {
            initial_dir = static_cast<uint32_t>(eth_chan_directions::WEST);
        } else {
            initial_dir = static_cast<uint32_t>(eth_chan_directions::EAST);
        }
    }

    return initial_dir;
}

FORCE_INLINE std::pair<tt::tt_fabric::RoutingPlaneConnectionManager&, bool> get_or_open_fabric_connections() {
    static tt::tt_fabric::RoutingPlaneConnectionManager connections;
    static bool initialized = false;
    
    if (!initialized) {
        DPRINT << "Build connections with args\n";
        uint32_t num_send_dir = get_arg_val<uint32_t>(num_connection_arg_idx);
        size_t arg_idx = fabric_connection_arg_idx;
        DPRINT << "Num connections is " << num_send_dir << "\n";
        connections = tt::tt_fabric::RoutingPlaneConnectionManager::template build_from_args<
        tt::tt_fabric::RoutingPlaneConnectionManager::BUILD_AND_OPEN_CONNECTION>(arg_idx, num_send_dir);
        initialized = true;
    }

    return {connections, initialized};
}

FORCE_INLINE int get_connection_index_by_tag(RoutingPlaneConnectionManager& fabric_connections, uint32_t tag) {
    for (uint32_t i = 0; i < fabric_connections.active_count(); ++i) {
        DPRINT << "connection tag is " << (uint32_t)fabric_connections.get(i).tag << "\n";
        if (fabric_connections.get(i).tag == tag) {
            return i;
        }
    }

    return -1;
}

FORCE_INLINE volatile tt_l1_ptr PACKET_HEADER_TYPE* get_or_allocate_header(uint32_t dir) {
    static uint8_t route_id = -1;
    if (route_id == -1) {
        uint32_t num_send_dir = get_arg_val<uint32_t>(num_connection_arg_idx);
        route_id = PacketHeaderPool::allocate_header_n(num_send_dir);
    }
    return PacketHeaderPool::header_table[route_id].first + dir;
}

FORCE_INLINE void fabric_fast_write(
    WorkerToFabricEdmSender& connection,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint64_t dest_addr,
    uint32_t len_bytes,
    bool multicast = false,
    uint32_t num_dests = 1) {
    if (multicast) {
        // TODO: Set up multicast header with proper routing
        ASSERT(false);
        while (1) {
        }
    } else {
        packet_header->to_noc_unicast_write(NocUnicastCommandHeader{dest_addr}, len_bytes);
    }

    DPRINT << "waiting for empty write slot\n";
    DPRINT << "connection noc x and y " << (uint32_t)connection.edm_noc_x << " " << (uint32_t)connection.edm_noc_y << "\n";
    connection.wait_for_empty_write_slot();
    DPRINT << "sending payload without header\n";
    connection.send_payload_without_header_non_blocking_from_address(src_addr, len_bytes);
    DPRINT << "sending payload flush\n";
    connection.send_payload_flush_non_blocking_from_address(
        reinterpret_cast<uint32_t>(packet_header), sizeof(PACKET_HEADER_TYPE));

    noc_async_writes_flushed();
}

///////////////////////////////////// APIs /////////////////////////////////////

FORCE_INLINE void close_fabric_connections() {
    DPRINT << "Close start\n";
    auto [connections, initialized] = get_or_open_fabric_connections();
    if (initialized) {
        connections.close();
    }
    DPRINT << "Close done\n";
}

FORCE_INLINE uint16_t get_my_device_id() {
    auto* routing_table = reinterpret_cast<tt_l1_ptr routing_l1_info_t*>(MEM_TENSIX_ROUTING_TABLE_BASE);
    return routing_table->my_device_id;
}

FORCE_INLINE void fabric_fast_write_any_len(
    uint16_t dst_mesh_id,
    uint16_t dst_dev_id,
    uint64_t dest_addr,
    uint32_t src_addr,
    uint32_t len_bytes) {
    tt_l1_ptr routing_l1_info_t* routing_table =
        reinterpret_cast<tt_l1_ptr routing_l1_info_t*>(MEM_TENSIX_ROUTING_TABLE_BASE);
    uint16_t my_chip_id = routing_table->my_device_id;
    uint16_t my_mesh_id = routing_table->my_mesh_id;
    ASSERT(my_mesh_id == dst_mesh_id);  // we dont support inter-mesh routing yet

    uint32_t initial_dir = calculate_initial_direction(dst_dev_id, my_chip_id);
    DPRINT << "initial_dir is " << initial_dir << "\n";
    auto [fabric_connections, is_init] = get_or_open_fabric_connections();
    auto connection_index = get_connection_index_by_tag(fabric_connections, initial_dir);
    DPRINT << "connection_index is " << connection_index << "\n";
    ASSERT(connection_index != -1);
    auto& connection = fabric_connections.get(connection_index).sender;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header = get_or_allocate_header(connection_index);

    // First call the original API by casting to base type
    bool result = fabric_set_unicast_route(
        static_cast<volatile tt_l1_ptr HybridMeshPacketHeader*>(packet_header), dst_dev_id, dst_mesh_id);
    while (len_bytes > max_fabric_payload_size) {
        fabric_fast_write(
            connection, packet_header, src_addr, dest_addr, max_fabric_payload_size);

        src_addr += max_fabric_payload_size;
        dest_addr += max_fabric_payload_size;
        len_bytes -= max_fabric_payload_size;
    }
    fabric_fast_write(connection, packet_header, src_addr, dest_addr, len_bytes);
}

} // namespace experimental

#endif