import numpy as np
from design import NetworkDesigner
from read_input import read_network_file

FILES=['data/4_city.txt','data/4_city.txt','data/5_city.txt', 'data/5_city.txt', 'data/6_city.txt','data/6_city.txt']
MAX_COSTS=[50,60,60,70,65,85] 

def main():
    
    for file, max_cost in zip(FILES, MAX_COSTS):
        num_cities, reliability_matrix, cost_matrix = read_network_file(file)

        designer = NetworkDesigner(num_cities-1, cost_matrix, reliability_matrix)

        designer.fit_transform_part_1(max_cost=max_cost)

if __name__ == "__main__":
    main()
