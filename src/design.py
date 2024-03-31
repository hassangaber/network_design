from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from typing import List, Tuple
import random

class NetworkDesigner:
    def __init__(self, num_cities: int, cost_matrix: np.array, reliability_matrix: np.array, max_cost: float):
        self.num_cities = num_cities
        self.cost_matrix = cost_matrix
        self.reliability_matrix = reliability_matrix
        self.max_cost = max_cost

    def _generate_all_possible_networks(self) -> List[List[Tuple[int, int]]]:
        connections = [(i, j) for i in range(self.num_cities + 1) for j in range(i + 1, self.num_cities + 1)]
        all_networks = []
        for r in range(self.num_cities, len(connections) + 1):
            all_networks.extend([c for c in combinations(connections, r) if self._is_connected(c)])
        print(all_networks[0])
        return all_networks

    def _is_connected(self, network: List[Tuple[int, int]]) -> bool:
        G = nx.Graph()
        G.add_nodes_from(range(self.num_cities + 1))
        G.add_edges_from(network)
        return bool(nx.is_connected(G) and G.number_of_nodes() == self.num_cities + 1)


    # def _calculate_network_cost(self, network: List[Tuple[int, int]]) -> float:
    #     return sum(self.cost_matrix[i][j] for i, j in network)

    def _calculate_network_cost(self, network: List[Tuple[int, int]]) -> float:
        total_cost = 0
        for i, j in network:
            try:
                if i < j:
                    matrix_row = i
                else:
                    matrix_row = j - 1 

                total_cost += self.cost_matrix[matrix_row][j - i - 1]
            except IndexError:
                print(f"Warning: Trying to access cost_matrix[{matrix_row}][{j - i - 1}], but it's out of bounds.")
        return total_cost


    def _network_to_graph(self, network: List[Tuple[int, int]]):
        """
        Converts a network (list of links) into a NetworkX graph object.
        """
        G = nx.Graph()
        G.add_edges_from(network)
        return G

    def _simulate_network_scenario_with_graph(self, network: List[Tuple[int, int]]):
        """
        Simulates a single scenario of the network, randomly failing links based on their reliability,
        using a graph to check for connectivity.
        """
        G = self._network_to_graph(network)
        for i, j in network:
            if random.random() >= self.reliability_matrix[i][j-1]:
                G.remove_edge(i, j)
                if not nx.is_connected(G): 
                    return False
        return True

    def simulate_network_reliability_with_graph(self, network: List[Tuple[int, int]], simulations=10000) -> float:
        """
        Simulates the reliability of the network over a given number of simulations,
        using a graph-based approach for increased accuracy.
        """
        successful_simulations = 0
        for _ in range(simulations):
            if self._simulate_network_scenario_with_graph(network):
                successful_simulations += 1
        return successful_simulations / simulations
    
    def fit_transform_part_1(self, reliability_simulations:int=1000) -> None:
        """
        Runs the brute-force enumeration of network generation, 
        only filtering based on cost and reliability constraints
        """

        # Step 1: Generate all possible networks
        all_networks = self._generate_all_possible_networks()

        # Step 2 & 3: Calculate costs and filter networks based on max_cost
        filtered_networks = [(network, self._calculate_network_cost(network)) for network in all_networks if self._calculate_network_cost(network) <= self.max_cost]

        # Step 4: Calculate the reliability of each filtered network
        network_reliabilities = [(network, 
                                  cost, 
                                  self.simulate_network_reliability_with_graph(network, simulations=reliability_simulations)) for network, cost in filtered_networks]

        # Step 5: Rank networks based on reliability
        ranked_networks = sorted(network_reliabilities, key=lambda x: x[2], reverse=True)

        # Step 6: Graph the top 3 networks
        self._graph_top_networks(ranked_networks[:3])

    def _graph_top_networks(self, networks):
        for idx, (network, cost, reliability) in enumerate(networks, start=1):
            G = nx.Graph()
            G.add_edges_from(network)
            plt.figure(figsize=(10, 8))
            plt.title(f"Network {idx} - Cost: {cost:.2f}, Reliability: {reliability:.4f}")
            nx.draw(G)
        plt.show()

