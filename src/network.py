from itertools import combinations
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx # May need to add to requirements
from typing import List

class NetworkDesigner:
    def __init__(
        self, 
        num_cities: int, 
        cost_matrix: np.array, 
        reliability_matrix: np.array, 
        max_cost: float):
        
        self.num_cities = num_cities
        self.cost_matrix = cost_matrix
        self.reliability_matrix = reliability_matrix
        self.max_cost = max_cost

    def _generate_all_possible_networks(self) -> list:
        """
        Generates all possible networks as combinations of links between cities.
        Each link is represented by a tuple (i, j) where i and j are city indices.
        This method returns a list of networks, where each network is represented as a list of links.
        """
        links = [(i, j) for i in range(self.num_cities) for j in range(i+1, self.num_cities)]
        all_networks = []
        for r in range(self.num_cities-1, len(links)+1):
            for combo in combinations(links, r):
                # Check if the combination forms a connected graph
                if self._is_connected(combo):
                    all_networks.append(combo)
        return all_networks

    def _is_connected(self, network: list) -> bool:
        """
        Checks if the given network of links forms a connected graph.
        """
        # Initialize an adjacency list
        adjacency_list = {i: [] for i in range(self.num_cities)}
        for i, j in network:
            adjacency_list[i].append(j)
            adjacency_list[j].append(i)
        
        visited = set()
        self._dfs(0, visited, adjacency_list)
        return len(visited) == self.num_cities
    
    def _dfs(self, node: int, visited: set, adjacency_list: dict):
        """
        Depth-First Search to check connectivity.
        """
        if node not in visited:
            visited.add(node)
            for neighbor in adjacency_list[node]:
                self._dfs(neighbor, visited, adjacency_list)

    def _calculate_network_cost(self, network: list) -> float:
        """
        Calculates the total cost of a given network design.
        The network cost is the sum of the costs of all links in the network.
        """
        return sum(self.cost_matrix[i][j-i-1] for i, j in network)
    
    ## Network Reliability calculation utilising the Monte Carlo Approach
    def is_network_connected(self, graph):
        """Check if the network is fully connected."""
        return nx.is_connected(graph)

    def simulate_network_failure(self, graph, link_failure_prob):
        """Simulate random failure of network links."""
        failed_graph = graph.copy()
        for edge in failed_graph.edges:
            if np.random.random() < link_failure_prob:
                failed_graph.remove_edge(*edge)
        return failed_graph

    def monte_carlo_reliability_analysis(self, graph, link_failure_prob, num_simulations=1000):
        """Perform Monte Carlo reliability analysis on a network."""
        connected_count = 0
        for _ in range(num_simulations):
            failed_graph = self.simulate_network_failure(graph, link_failure_prob)
            if self.is_network_connected(failed_graph):
                connected_count += 1
        return connected_count / num_simulations
    
def main():
    print("Beginning Testing")
    # Parameters for the reliability calculation
    num_cities = 4
    reliability = 0.9  # Assuming all edges have the same reliability
    link_failure_prob = 1 - reliability
    num_simulations = 100000

    # Initialize the NetworkDesigner
    designer = NetworkDesigner(num_cities, reliability, link_failure_prob, num_simulations)

    print("Designer Created")

    # Define graph configurations
    graphs = {
        "Fully Connected": nx.complete_graph(num_cities),
        "Linear": nx.path_graph(num_cities),
        "Star": nx.star_graph(num_cities - 1)  # Subtract 1 because star_graph generates an extra node at the center
    }

    plt.figure(figsize=(12, 12))

    for i, (name, graph) in enumerate(graphs.items(), start=1):
        plt.subplot(2, 2, i)
        nx.draw(graph, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700)
        plt.title(name)

        # Calculate and print the reliability of the current graph
        reliability = designer.monte_carlo_reliability_analysis(graph, link_failure_prob, num_simulations)
        print(f"{name} Network Reliability: {reliability:.4f}")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

    # def _calculate_network_reliability(self, network: list) -> float:
    #     """
    #     Calculates the reliability of a given network design.
    #     The network reliability is calculated as the product of the reliabilities of all links in the network,
    #     considering each link's reliability contributes multiplicatively to the overall network reliability.
    #     """
    #     return reduce(lambda x, y: x*y, (self.reliability_matrix[i][j-i-1] for i, j in network))

    # def simple_design(self) -> tuple:
    #     """
    #     Finds the best network design using exhaustive enumeration, under the cost constraint.
    #     """
    #     best_network = None
    #     highest_reliability = 0
    #     all_networks = self._generate_all_possible_networks()
        
    #     for network in all_networks:
    #         cost = self._calculate_network_cost(network)
    #         if cost <= self.max_cost:
    #             reliability = self._calculate_network_reliability(network)
    #             if reliability > highest_reliability:
    #                 highest_reliability = reliability
    #                 best_network = network
        
    #     return best_network, highest_reliability

    # def advanced_design(self) -> tuple:
    #     pass

