from itertools import combinations
from functools import reduce
import numpy as np
from typing import List

class NetworkDesigner:
    def __init__(self, num_cities: int, cost_matrix: np.array, reliability_matrix: np.array, max_cost: float):
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
            all_networks.extend(combinations(links, r))
        return all_networks

    def _calculate_network_cost(self, network: list) -> float:
        """
        Calculates the total cost of a given network design.
        The network cost is the sum of the costs of all links in the network.
        """
        return sum(self.cost_matrix[i][j-i-1] for i, j in network)

    def _calculate_network_reliability(self, network: list) -> float:
        """
        Calculates the reliability of a given network design.
        The network reliability is calculated as the product of the reliabilities of all links in the network,
        considering each link's reliability contributes multiplicatively to the overall network reliability.
        """
        return reduce(lambda x, y: x*y, (self.reliability_matrix[i][j-i-1] for i, j in network))

    def simple_design(self) -> tuple:
       pass

    def advanced_design(self) -> tuple:
        pass

