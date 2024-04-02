# ECSE 422 Network Design Assignment

### Description

The main objective of this assignment is to compare a brute-force and guided search approach to solving a network optimization problem. The problem's constraints include a maximum cost constraint, a node constraint, and the objective of maximizing network overall reliability. The two approaches should yield the same solutions with the guided search arriving at the optimal graph faster than the brute-force method.

The brute-force algorithm generates the optimal graph using the following approach: 1. Generate all the possible graphs given the number of nodes and full connectivity of the graphs 2. Filter the graphs based on the edge cost and maximum cost constraint 3. Compute all graph reliabilities and choose the highest reliability graph.

The guided search algorithm finds the optimal graph using the following: 

### Code Structure

### How to Run

- Navigate to the root of the project
- Install packages with `pip install -r requirements.txt`
- Run `python3 src/main.py`
- If a `networkx` random state error arises, run `pip install decorator==5.0.9` and retry.

