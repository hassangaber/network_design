import numpy as np
from design import NetworkDesigner
from read_input import read_network_file

def main():
    
    num_cities, reliability_matrix, cost_matrix = read_network_file("data/5_city.txt")
    max_cost= 70

    designer = NetworkDesigner(num_cities-1, cost_matrix, reliability_matrix, max_cost)

    designer.fit_transform_part_1()

if __name__ == "__main__":
    main()
