import numpy as np
import networkx as nx
from typing import List


def _find(parent: List[int], i: int) -> int:
    """
    Recursively finds the root of the set that element `i` belongs to.
    
    Args:
    - parent: A list where `parent[i]` is the parent of element `i`.
    - i: The element to find the root set for.
    
    Returns:
    - The root element of the set that contains `i`.
    """
    if parent[i] == i:
        return i
    return _find(parent, parent[i])


def _union(parent: List[int], rank: List[int], x: int, y: int) -> None:
    """
    Merges the sets containing elements `x` and `y`.
    
    Args:
    - parent: A list where `parent[i]` represents the parent of element `i`.
    - rank: A list tracking the depth of trees representing subsets.
    - x, y: The elements to unionize.
    """
    xroot = _find(parent, x)
    yroot = _find(parent, y)
    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1


def guided_search(num_cities: int, 
                  cost_matrix: np.array, 
                  reliability_matrix: np.array, 
                  max_cost: int, 
                  spanning_tree_solution: bool = False) -> nx.Graph:
    """
    Finds an optimal network configuration using a greedy heuristic approach. The method first constructs a minimum
    spanning tree (MST) based on the cost-to-reliability ratio. If spanning_tree_solution is False, it then adds
    additional edges to increase reliability, potentially forming cycles.
    
    Args:
    - num_cities: The number of cities (nodes) in the network.
    - cost_matrix: A 2D array representing the cost of connecting cities.
    - reliability_matrix: A 2D array representing the reliability of connections between cities.
    - max_cost: The maximum allowable cost for the network.
    - spanning_tree_solution: A boolean indicating whether to stop at a spanning tree solution or to add more edges.
    
    Returns:
    - A NetworkX Graph object representing the constructed network.
    """
    parent = list(range(num_cities))
    rank = [0] * num_cities
    optimal_solution = nx.Graph()
    total_cost = 0

    edges = [(i, j, cost_matrix[i][j], reliability_matrix[i][j]) for i in range(num_cities) for j in range(i + 1, num_cities)]
    edges.sort(key=lambda x: x[2]/x[3])  # cost-to-reliability ratio is the greedy objective

    # Iterate over all potential edges sorted by the cost-to-reliability ratio.
    # This sorting is key to the greedy heuristic: at each step, we consider the next most cost-effective edge.
    for i, j, cost, _ in edges:
        # Check if adding the current edge would exceed the maximum allowed cost.
        # If so, we stop the process as we cannot add more edges without breaching the cost limit.
        # This termination condition ensures we adhere to the budget constraint.
        if total_cost + cost > max_cost:
            break

        # Check if adding the current edge would create a cycle in the case of spanning_tree_solution being True,
        # or add the edge regardless in the case of spanning_tree_solution being False.
        # The use of the Union-Find structure creates a valid tree structure or forms a more connected network.
        if (_find(parent, i) != _find(parent, j)) or (not spanning_tree_solution):
            # Merge the subsets, indicating that we're either connecting two previously disconnected components
            # (in the case of a spanning tree) or simply adding an edge to increase connectivity.
            _union(parent, rank, i, j)
            
            # Add the current edge to the optimal solution graph, including its cost and reliability as attributes.
            # This step is where the edge becomes part of the final network.
            optimal_solution.add_edge(i, j, weight=cost, reliability=reliability_matrix[i][j])
            
            # Update the total cost of the network to include the cost of the newly added edge.
            total_cost += cost
            
            # For spanning tree solutions, check if the graph is now fully connected with the minimum number of edges
            # (num_cities - 1 edges). If so, we have formed a minimum spanning tree and can terminate early.
            # This check is skipped if spanning_tree_solution is False, allowing for more cycles.
            if spanning_tree_solution and nx.is_connected(optimal_solution) and (optimal_solution.number_of_edges() == num_cities - 1):
                break


    # Ensure the final network is connected; if not, it's not a valid solution
    if not nx.is_connected(optimal_solution):
        raise ValueError("Unable to construct a connected network within the given cost limit.")

    return optimal_solution

