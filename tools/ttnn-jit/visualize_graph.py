# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import json


def create_graph_from_capture(captured_graph):
    """
    Create a networkx directed graph from the captured graph trace.
    
    Args:
        captured_graph: List of dictionaries representing the graph trace
        
    Returns:
        networkx.DiGraph: The directed graph
    """
    G = nx.DiGraph()
    
    # Add nodes
    for node in captured_graph:
        counter = node['counter']
        node_type = node['node_type']
        params = node.get('params', {})
        
        # Create a label for the node
        label = f"{counter}: {node_type}"
        if 'name' in params:
            label += f"\n{params['name']}"
        if 'shape' in params:
            label += f"\n{params['shape']}"
        if 'size' in params and params['size'] != '0':
            label += f"\nsize: {params['size']}"
        if 'tensor_id' in params:
            label += f"\n(tid: {params['tensor_id']})"
            
        G.add_node(counter, 
                   node_type=node_type,
                   label=label,
                   params=params)
    
    # Add edges
    for node in captured_graph:
        counter = node['counter']
        connections = node.get('connections', [])
        for target in connections:
            G.add_edge(counter, target)
    
    return G


def get_node_color(node_type):
    """Get color for different node types."""
    colors = {
        'capture_start': '#90EE90',  # light green
        'capture_end': '#FFB6C1',    # light pink
        'function_start': '#87CEEB', # sky blue
        'function_end': '#B0C4DE',   # light steel blue
        'tensor': '#FFD700',         # gold
        'buffer': '#FFA500',         # orange
        'buffer_allocate': '#FF6347', # tomato
        'buffer_deallocate': '#FF4500', # orange red
        'circular_buffer_allocate': '#DDA0DD', # plum
        'circular_buffer_deallocate_all': '#DA70D6', # orchid
    }
    return colors.get(node_type, '#D3D3D3')  # light gray default


def create_hierarchical_layout(G):
    """
    Create a tree-like hierarchical layout for the graph.
    Nodes are positioned based on their depth from the root (capture_start).
    """
    # Find the root node (capture_start)
    root = None
    for node in G.nodes():
        if G.nodes[node]['node_type'] == 'capture_start':
            root = node
            break
    
    if root is None:
        # No root found, use node 0
        root = 0
    
    # Calculate levels using BFS
    levels = {root: 0}
    queue = [root]
    while queue:
        current = queue.pop(0)
        current_level = levels[current]
        
        for neighbor in G.successors(current):
            if neighbor not in levels:
                levels[neighbor] = current_level + 1
                queue.append(neighbor)
    
    # Handle any disconnected nodes
    for node in G.nodes():
        if node not in levels:
            levels[node] = 0
    
    # Group nodes by level
    level_nodes = {}
    for node, level in levels.items():
        if level not in level_nodes:
            level_nodes[level] = []
        level_nodes[level].append(node)
    
    # Assign positions
    pos = {}
    max_width = max(len(nodes) for nodes in level_nodes.values())
    
    for level, nodes in level_nodes.items():
        # Center nodes at this level
        y = -level  # Negative so tree goes downward
        num_nodes = len(nodes)
        
        # Sort nodes by counter to maintain order
        nodes = sorted(nodes)
        
        if num_nodes == 1:
            pos[nodes[0]] = (0, y)
        else:
            # Distribute nodes evenly across the width
            width = max(num_nodes * 2, max_width)
            spacing = width / (num_nodes + 1)
            for i, node in enumerate(nodes):
                x = (i + 1) * spacing - width / 2
                pos[node] = (x, y)
    
    return pos


def visualize_graph(G, output_path='graph_trace.png', figsize=(20, 16)):
    """
    Visualize the graph using matplotlib.
    
    Args:
        G: networkx.DiGraph
        output_path: Path to save the visualization
        figsize: Size of the figure (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Use custom hierarchical layout for tree-like visualization
    print("Creating hierarchical tree-like layout...")
    pos = create_hierarchical_layout(G)
    
    # Get node colors based on type
    node_colors = [get_node_color(G.nodes[node]['node_type']) for node in G.nodes()]
    
    # Get labels
    labels = {node: G.nodes[node]['label'] for node in G.nodes()}
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors,
                          node_size=3000,
                          alpha=0.9,
                          node_shape='s')
    
    nx.draw_networkx_labels(G, pos, labels, 
                           font_size=7,
                           font_weight='bold')
    
    nx.draw_networkx_edges(G, pos,
                          edge_color='gray',
                          arrows=True,
                          arrowsize=15,
                          arrowstyle='->',
                          connectionstyle='arc3,rad=0.1',
                          width=1.5)
    
    plt.title("TTNN Graph Trace Visualization", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Graph visualization saved to: {output_path}")
    
    # Show the figure
    plt.show()
    
    return pos


def print_graph_stats(G):
    """Print statistics about the graph."""
    print("\n=== Graph Statistics ===")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    
    # Count node types
    node_types = {}
    for node in G.nodes():
        node_type = G.nodes[node]['node_type']
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    print("\nNode types:")
    for node_type, count in sorted(node_types.items()):
        print(f"  {node_type}: {count}")
    
    # Find longest path (critical path)
    if nx.is_directed_acyclic_graph(G):
        longest_path = nx.dag_longest_path(G)
        print(f"\nLongest path length: {len(longest_path)}")
        print(f"Longest path: {longest_path}")
    else:
        print("\nGraph contains cycles")


def visualize_captured_graph(captured_graph, output_path='graph_trace.png'):
    """
    Main function to visualize a captured graph.
    
    Args:
        captured_graph: List of dictionaries or JSON string
        output_path: Path to save the visualization
    """
    # Handle JSON string input
    if isinstance(captured_graph, str):
        captured_graph = json.loads(captured_graph)
    
    # Create graph
    print("Creating graph from captured trace...")
    G = create_graph_from_capture(captured_graph)
    
    # Print statistics
    print_graph_stats(G)
    
    # Visualize
    print("\nVisualizing graph...")
    visualize_graph(G, output_path)
    
    return G


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        # Load from file
        with open(sys.argv[1], 'r') as f:
            captured_graph = json.load(f)
        output_path = sys.argv[2] if len(sys.argv) > 2 else 'graph_trace.png'
    else:
        # Use sample data from the documentation
        print("Usage: python visualize_graph.py <graph_json_file> [output_path]")
        sys.exit(1)
    
    visualize_captured_graph(captured_graph, output_path)

