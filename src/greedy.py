import numpy as np
import networkx as nx
from typing import List, Tuple

def guided_search(n_nodes: int, cost_matrix: np.array, reliability_matrix: np.array, max_cost: int) -> List[List[Tuple[int, int]]]:
    """
    Finds the top graphs in terms of reliability, subject to a maximum cost constraint, 
    with the number of nodes equal to n_nodes + 1.
    """

    all_edges = []
    for i in range(n_nodes + 1):
        for j in range(i + 1, n_nodes + 1):
            if i < cost_matrix.shape[0] and j < cost_matrix.shape[1]:
                cost = cost_matrix[i, j]
                reliability = reliability_matrix[i, j]
                if cost <= max_cost:
                    value = reliability / cost if cost > 0 else 0
                    all_edges.append(((i, j), value, cost, reliability))
    all_edges.sort(key=lambda x: x[1], reverse=True)
    print(all_edges)
    potential_networks = []
    for edge_info in all_edges:
        edge, value, cost, reliability = edge_info
        current_network = [edge]
        current_cost = cost

        # Try adding other edges without exceeding max_cost
        for next_edge_info in all_edges:
            next_edge, _, next_cost, _ = next_edge_info
            
            if (next_edge not in current_network) and (current_cost + next_cost <= max_cost):
                # Temporarily add the edge to check if it maintains validity
                temp_network = current_network + [next_edge]
                if is_valid_network(n_nodes + 1, temp_network):
                    # If valid, permanently add the edge
                    current_network = temp_network
                    current_cost += next_cost
                
                # Stop trying to add edges once we reach a valid configuration
                if len(set([node for edge in current_network for node in edge])) == n_nodes + 1:
                    break

        # Check if the current network is valid and complete
        if is_valid_network(n_nodes + 1, current_network) and len(set([node for edge in current_network for node in edge])) == n_nodes + 1:
            potential_networks.append(current_network)
            break

    return potential_networks

def is_valid_network(n_nodes: int, network: List[Tuple[int, int]]) -> bool:
    """
    Checks if the network is valid, i.e., it's fully connected and contains exactly n_nodes.
    """
    G = nx.Graph()
    G.add_edges_from(network)
    return nx.is_connected(G) and G.number_of_nodes() == n_nodes
