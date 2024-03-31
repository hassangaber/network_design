#! /usr/bin/env python3
import numpy as np
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)

def construct_symmetric_matrix(upper_triangular_components: List[List[float]]) -> np.array:
    """
    Constructs a symmetric matrix from the given upper triangular components, corrected for accessing elements.

    Args:
    upper_triangular_components (list of lists): The upper triangular part of the matrix,
                                                  excluding diagonal elements, given as a nested list.

    Returns:
    list of lists: The symmetric matrix.
    """
    num_rows = len(upper_triangular_components) + 1
    matrix = [[0.0 for _ in range(num_rows)] for _ in range(num_rows)]  

    current_index = 0 
    for i in range(num_rows):
        for j in range(i + 1, num_rows):
            value = upper_triangular_components[i][current_index]
            matrix[i][j] = value
            matrix[j][i] = value  # Mirror the value for symmetry
            current_index += 1
        current_index = 0  

    return np.array(matrix)

def read_network_file(file_path: str) -> Tuple[int, np.array, np.array]:
    """
    Reads the file and constructs symmetric reliability and cost matrices.

    Args:
    file_path (str): The path to the file containing the matrix data.

    Returns:
    tuple: A tuple containing two matrices (lists of lists); the first is the reliability matrix, and
           the second is the cost matrix.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]

    num_nodes = int(lines[0]) 
    R_lines = lines[1:num_nodes]
    C_lines = lines[num_nodes:2*num_nodes]

    R_values = [[float(val) for val in line.split()] for line in R_lines]
    C_values = [[int(val) for val in line.split()] for line in C_lines]

    R = construct_symmetric_matrix(R_values)
    C = construct_symmetric_matrix(C_values)

    logging.info(f'Loaded in {file_path}')

    return (num_nodes, R, C)


if __name__ == "__main__":
    N, R, C = read_network_file("../data/4_city.txt")
    print(N)
    print(C)
