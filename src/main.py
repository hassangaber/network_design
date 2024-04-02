import time
from design import NetworkDesigner
from read_input import read_network_file


FILES=['data/4_city.txt','data/4_city.txt','data/5_city.txt', 'data/5_city.txt', 'data/6_city.txt','data/6_city.txt']
MAX_COSTS=[50,60,60,70,65,85] 

def main():
    
    for file, max_cost in zip(FILES, MAX_COSTS):
        num_cities, reliability_matrix, cost_matrix = read_network_file(file)
        designer = NetworkDesigner(num_cities-1, cost_matrix, reliability_matrix)
        start = time.monotonic()
        designer.fit_transform_part_1(max_cost=max_cost, reliability_simulations=10000)
        end = time.monotonic()
        print(f'Simple solution runtime for file={file} max_cost={max_cost} : {end - start: .2f} s')

    file=FILES[4]
    num_cities, reliability_matrix, cost_matrix = read_network_file(file)

    designer = NetworkDesigner(num_cities, cost_matrix, reliability_matrix)

    designer.fit_transform_part_2(max_cost=150)

if __name__ == "__main__":
    main()
