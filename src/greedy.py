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


def guided_search(num_cities: int, cost_matrix: np.array, reliability_matrix: np.array, max_cost: int) -> nx.Graph:
    """
    Utilizes a greedy heuristic approach to find an optimal network configuration within a specified maximum cost.
    
    The greedy heuristic sorts potential connections (edges) based on their cost-to-reliability ratio, preferring edges that
    offer a better balance of low cost and high reliability. It then iteratively adds these edges to the network,
    ensuring no cycles are formed (to maintain a spanning tree structure until necessary), and the total cost does not
    exceed the specified limit.
    
    Args:
    - num_cities: The number of cities (nodes) in the network.
    - cost_matrix: A 2D array where `cost_matrix[i][j]` represents the cost of connecting city `i` to city `j`.
    - reliability_matrix: A 2D array where `reliability_matrix[i][j]` represents the reliability of the connection
      between city `i` and city `j`.
    - max_cost: The maximum total cost allowed for the network.
    
    Returns:
    - A NetworkX Graph object representing the optimal network configuration found.
    """
    parent = list(range(num_cities))
    rank = [0] * num_cities
    optimal_solution = nx.Graph()
    total_cost = 0
    edges = [(i, j, cost_matrix[i][j], reliability_matrix[i][j]) 
             for i in range(num_cities) for j in range(i + 1, num_cities)]

    # Sort edges based on cost-to-reliability ratio
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
    if not nx.is_connected(optimal_solution):
        raise ValueError("Unable to construct a connected network within the given cost limit.")

    return optimal_solution
