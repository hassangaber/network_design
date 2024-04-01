from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from typing import List, Tuple
import random
import logging
from greedy import guided_search

logging.basicConfig(level=logging.INFO)

class NetworkDesigner:
    def __init__(
        self, 
        num_cities: int, 
        cost_matrix: np.array, 
        reliability_matrix: np.array):

        self.num_cities = num_cities
        self.cost_matrix = cost_matrix
        self.reliability_matrix = reliability_matrix

    def _generate_all_possible_networks(self) -> List[List[Tuple[int, int]]]:
        """
        Given the number of cities in a network, generate all the possible networks
        with that number of nodes.
        """
        connections = [(i, j) for i in range(self.num_cities + 1) for j in range(i + 1, self.num_cities + 1)]
        all_networks = []
        for r in range(self.num_cities, len(connections) + 1):
            all_networks.extend([c for c in combinations(connections, r) if self._is_connected(c)])
        logging.info(all_networks[0])
        return all_networks

    def _is_connected(self, network: List[Tuple[int, int]]) -> bool:
        """
        Check if a given graph satisfies the connectivity criteria which implies
        that in the graph any node can be visited wherever you are in the network.
        """
        G = nx.Graph()
        G.add_nodes_from(range(self.num_cities + 1))
        G.add_edges_from(network)
        return bool(nx.is_connected(G) and G.number_of_nodes() == self.num_cities + 1)

    def _calculate_network_cost(self, network: List[Tuple[int, int]]) -> float:
        """
        Given the cost matrix and the network, calculate the network cost as a sum
        of the cost of all the links.
        """
        total_cost = 0
        # iterating through each edge in the network, calculating
        # cost, and then summing to obtain the total cost of the network
        for i, j in network:

            try:
                if i < j:
                    matrix_row = i
                else:
                    matrix_row = j - 1 
                total_cost += self.cost_matrix[matrix_row][j - i - 1]

            except IndexError:
                logging.warning(f"Warning: Trying to access cost_matrix[{matrix_row}][{j - i - 1}], but it's out of bounds.")

        return total_cost

    def _network_to_graph(self, network: List[Tuple[int, int]]) -> nx.Graph:
        """
        Converts a network (list of links) into a NetworkX graph object.
        """
        G = nx.Graph()
        G.add_edges_from(network)
        return G

    def _simulate_network_scenario_with_graph(self, network: List[Tuple[int, int]]) -> bool:
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

    def simulate_network_reliability_with_graph(self, network: List[Tuple[int, int]], simulations:int) -> float:
        """
        Simulates the reliability of the network over a given number of simulations,
        using a graph-based approach for increased accuracy.
        """
        successful_simulations = 0
        for _ in range(simulations):
            if self._simulate_network_scenario_with_graph(network):
                successful_simulations += 1
        return successful_simulations / simulations
    
    def fit_transform_part_1(self, max_cost:int, reliability_simulations:int=1000) -> None:
        """
        Runs the brute-force enumeration of network generation, 
        only filtering based on cost and reliability constraints
        """

        # Step 1: Generate all possible networks
        logging.info(f'Generating all possible networks with {self.num_cities} nodes...')
        all_networks = self._generate_all_possible_networks()

        # Step 2 & 3: Calculate costs and filter networks based on max_cost
        logging.info(f'Filtering generated networks based on max. cost <= {max_cost}')
        filtered_networks = [(network, self._calculate_network_cost(network)) for network in all_networks if self._calculate_network_cost(network) <= max_cost]

        # Step 4: Calculate the reliability of each filtered network
        logging.info(f'Computing network reliability with {reliability_simulations} iterations of Monte Carlo...')
        network_reliabilities = [(network, 
                                  cost, 
                                  self.simulate_network_reliability_with_graph(network, simulations=reliability_simulations)) for network, cost in filtered_networks]

        # Step 5: Rank networks based on reliability
        ranked_networks = sorted(network_reliabilities, key=lambda x: x[2], reverse=True)

        # Step 6: Graph the top 3 networks
        logging.info("Displaying top 3 network solutions...")
        self._graph_top_networks(ranked_networks[:3])

    def _graph_top_networks(self, networks: list) -> None:
        for idx, (network, cost, reliability) in enumerate(networks, start=1):
            G = nx.Graph()
            G.add_edges_from(network)
            plt.figure(figsize=(10, 8))
            plt.title(f"Network {idx} - Cost: {cost:.2f}, Reliability: {reliability:.4f}")
            nx.draw(G)
        plt.show()

    def fit_transform_part_2(self, max_cost:int) -> None:
        solutions = guided_search(self.num_cities, self.cost_matrix, self.reliability_matrix, max_cost)
        print(solutions)
        self._graph_top_networks(solutions)
