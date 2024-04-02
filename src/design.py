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
            nx.draw(G, with_labels=True)
        plt.show()
    
    def _find(self, parent, i):
        if parent[i] == i:
            return i
        return self._find(parent, parent[i])

    def _union(self, parent, rank, x, y):
        xroot = self._find(parent, x)
        yroot = self._find(parent, y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    def guided_search(self, max_cost: int) -> nx.Graph:
        """
        Finds an optimal network configuration using a guided search approach and returns it as a NetworkX Graph.
        """
        # Initialize Union-Find data structures
        parent = list(range(self.num_cities))
        rank = [0] * self.num_cities

        # Create a graph to represent the network
        network_graph = nx.Graph()

        # Initialize variables
        total_cost = 0

        # Create a list of all possible connections with their cost and reliability
        edges = [(i, j, self.cost_matrix[i][j], self.reliability_matrix[i][j]) 
                 for i in range(self.num_cities) for j in range(i + 1, self.num_cities)]

        # Sort edges based on cost-to-reliability ratio
        edges.sort(key=lambda x: x[2]/x[3])

        # Iterate over sorted edges and add them if they don't form a cycle and are within the cost limit
        for i, j, cost, _ in edges:
            if total_cost + cost > max_cost:
                break
            if self._find(parent, i) != self._find(parent, j):
                self._union(parent, rank, i, j)
                network_graph.add_edge(i, j, weight=cost)
                total_cost += cost

        # Check if the graph is fully connected
        if not nx.is_connected(network_graph):
            raise ValueError("Unable to construct a connected network within the given cost limit.")

        return network_graph

    def simulate_network_reliability(self, network_graph: nx.Graph, simulations: int = 1000) -> float:
        """
        Simulates the reliability of the network over a given number of simulations.
        """
        successful_simulations = 0
        for _ in range(simulations):
            # Create a deep copy of the graph for simulation
            G = network_graph.copy()
            for (u, v, reliability) in G.edges.data('reliability'):
                if random.random() > reliability:
                    G.remove_edge(u, v)
            if nx.is_connected(G):
                successful_simulations += 1
        return successful_simulations / simulations

    def fit_transform(self, max_cost: int, reliability_simulations: int = 1000):
        """
        Finds an optimal network configuration, simulates its reliability, and visualizes the result.
        """
        # Step 1: Use guided_search to find an optimal network configuration
        network_graph = self.guided_search(max_cost)

        # Step 2: Calculate the network's reliability
        #reliability = self.simulate_network_reliability(network_graph, reliability_simulations)

        # Step 3: Visualize the network
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(network_graph)
        nx.draw(network_graph, pos, with_labels=True, node_color='skyblue', node_size=700, font_size=15)

        # Annotate edges with reliability
        # edge_labels = dict([((u, v,), f'{reliability:.2f}')
        #                     for u, v, reliability in network_graph.edges.data('reliability')])
        # nx.draw_networkx_edge_labels(network_graph, pos, edge_labels=edge_labels, font_color='red')

        # total_cost = sum(nx.get_edge_attributes(network_graph, 'weight').values())
        #plt.title(f"Optimal Network - Cost: {total_cost:.2f}, Reliability: {reliability:.4f}")
        plt.show()


