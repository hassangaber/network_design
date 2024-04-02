import numpy as np
import networkx as nx


def _find(parent, i) -> callable:
    if parent[i] == i:
        return i
    return _find(parent, parent[i])

def _union(parent, rank, x, y) -> None:
    xroot = _find(parent, x)
    yroot = _find(parent, y)
    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1

def guided_search(num_cities:int, cost_matrix:np.array, reliability_matrix:np.array, max_cost: int) -> nx.Graph:
    """
    Finds an optimal network configuration using a greedy heuristic approach and returns it as a NetworkX Graph.
    """
    # Initialize Union-Find data structures
    parent = list(range(num_cities))
    rank = [0] * num_cities

    # Create a graph to represent the network
    optimal_solution = nx.Graph()

    # Initialize variables
    total_cost = 0

    # Create a list of all possible connections with their cost and reliability
    edges = [(i, j, cost_matrix[i][j], reliability_matrix[i][j]) 
                for i in range(num_cities) for j in range(i + 1, num_cities)]

    # Sort edges based on cost-to-reliability ratio (greedy heuristic objective)
    edges.sort(key=lambda x: x[2]/x[3])

    # Iterate over sorted edges and add them if they don't form a cycle and are within the cost limit
    for i, j, cost, _ in edges:
        if total_cost + cost > max_cost:
            break
        if _find(parent, i) != _find(parent, j):
            _union(parent, rank, i, j)
            optimal_solution.add_edge(i, j, weight=cost)
            total_cost += cost

    # Check if the graph is fully connected
    if not nx.is_connected(optimal_solution): raise ValueError("Unable to construct a connected network within the given cost limit.")

    return optimal_solution