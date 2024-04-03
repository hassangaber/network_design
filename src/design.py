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
            if random.random() >= self.reliability_matrix[i][j]:
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
        print("Simple Solution Best Cost:", ranked_networks[0][1], "Best Reliability:", ranked_networks[0][2])

    def _graph_top_networks(self, networks: list) -> None:
        for idx, (network, cost, reliability) in enumerate(networks, start=1):
            G = nx.Graph()
            G.add_edges_from(network)
            plt.figure(figsize=(10, 8))
            plt.title(f"Network {idx} - Cost: {cost:.2f}, Reliability: {reliability:.4f}")
            nx.draw(G, with_labels=True)
        plt.show()


    def fit_transform_part_2(self, max_cost: int, reliability_simulations: int = 10000):
        # Initialization
        remaining_vertices = set(range(self.num_cities))
        spanning_tree = set()
        total_cost = 0

        # Start from an arbitrary vertex
        current_vertex = remaining_vertices.pop()
        connected_vertices = {current_vertex}

        # Prim's algorithm adapted for maximum reliability
        while remaining_vertices:
            edge, cost, reliability = self._find_best_edge(connected_vertices, remaining_vertices, max_cost - total_cost)
            if edge is None:
                break  # No edge found within the cost constraint
            spanning_tree.add(edge)
            total_cost += cost
            current_vertex = edge[1]  # Assuming edge is (u, v) where v is the newly added vertex
            connected_vertices.add(current_vertex)
            remaining_vertices.remove(current_vertex)

        # Augment the network with additional edges for redundancy, without exceeding max_cost
        self._augment_network(spanning_tree, max_cost - total_cost)

        # Convert the spanning tree into a format suitable for simulations
        network_edges = list(spanning_tree)
        reliability = self.simulate_network_reliability_with_graph(network_edges, reliability_simulations)

        final_reliability = self.simulate_network_reliability_with_graph(network_edges, reliability_simulations)
        final_cost = sum(self.cost_matrix[edge[0]][edge[1]] for edge in network_edges)

        print(f"Complex solution Prims for {self.num_cities} cities and max cost {max_cost} has reliability {final_reliability:.4f} and cost {final_cost:.2f}")

        # Visualization
        self._visualize_network(network_edges, total_cost, reliability)

    def fit_transform_part_3(self, max_cost: int, reliability_simulations: int = 10000):
        """
        Finds an optimal network configuration, simulates its reliability, and visualizes the result.
        """
        print(self.reliability_matrix)
        print(self.cost_matrix)

        # Step 1: Use guided_search to find an optimal network configuration
        network_graph = guided_search(cost_matrix=self.cost_matrix, 
                                      reliability_matrix=self.reliability_matrix, 
                                      num_cities=self.num_cities,
                                      max_cost=max_cost)

        # Convert network_graph to a format that simulate_network_reliability_with_graph can process
        network_edges = tuple((u, v) for u, v in network_graph.edges())
        print(network_edges)
        # Step 2: Calculate the network's reliability
        #reliability_simulations = 2 ** network_graph.number_of_edges()
        reliability = self.simulate_network_reliability_with_graph(network_edges, reliability_simulations)

        # Step 3: Visualize the network
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(network_graph)
        nx.draw(network_graph, pos, with_labels=True, node_color='skyblue', node_size=700, font_size=15)

        # Annotate edges with reliability and cost from the matrices directly
        edge_labels = {(u, v): f"r={self.reliability_matrix[u][v]:.2f}, c={self.cost_matrix[u][v]}"
                    for u, v in network_graph.edges()}
        nx.draw_networkx_edge_labels(network_graph, pos, edge_labels=edge_labels, font_color='black')

        # Calculate and display the total cost
        total_cost = sum(self.cost_matrix[u][v] for u, v in network_graph.edges())
        print(f"Optimal Network - Cost: {total_cost:.2f}, Reliability: {reliability:.4f}")
        plt.savefig(f'results/graph_{self.num_cities}_max_cost={max_cost}_c={total_cost}_r={reliability}.jpg')
        plt.show()
        plt.plot()

    def _find_best_edge(self, connected_vertices, remaining_vertices, available_budget):
        best_edge = None
        best_cost = float('inf')
        best_reliability = 0
        for u in connected_vertices:
            for v in remaining_vertices:
                cost = self.cost_matrix[u][v]
                reliability = self.reliability_matrix[u][v]
                if cost <= available_budget and reliability > best_reliability:
                    best_edge = (u, v)
                    best_cost = cost
                    best_reliability = reliability
        return best_edge, best_cost, best_reliability

    def _augment_network(self, spanning_tree, available_budget):
        all_edges = [(i, j) for i in range(self.num_cities) for j in range(i + 1, self.num_cities)]
        potential_edges = [edge for edge in all_edges if edge not in spanning_tree]

        for edge in potential_edges:
            cost = self.cost_matrix[edge[0]][edge[1]]
            if cost <= available_budget:
                # Simulate adding this edge and check if it improves reliability
                spanning_tree.add(edge)
                available_budget -= cost
                # Note: In a real implementation, you'd want to check if adding the edge actually improves reliability
                # before permanently adding it to the spanning_tree

    def _visualize_network(self, network_edges, total_cost, reliability):
        G = nx.Graph()
        G.add_edges_from(network_edges)
        plt.figure(figsize=(10, 8))
        nx.draw(G, with_labels=True, node_color='skyblue', node_size=700, edge_color='black')
        plt.title(f"Network - Cost: {total_cost:.2f}, Reliability: {reliability:.4f}")
        plt.show()

        

