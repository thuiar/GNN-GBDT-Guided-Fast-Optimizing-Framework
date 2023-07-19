from functools import cmp_to_key
import numpy as np
import argparse
import pickle
import random
import time
import os

class pair: 
    def __init__(self): 
        self.x = 0
        self.y = 0
        self.val = 0

def cmp(a, b):
    if a.val > b.val: 
        return -1
    else:
        return 1
    
def FENNEL(n, m, k, site):
    '''
    Function Description:
    Use the FENNEL algorithm to partition the bipartite representation of the problem instance.

    Parameters:
    - n: Number of decision variables in the problem instance.
    - m: Number of constraints in the problem instance.
    - k: k[i] represents the number of decision variables in the i-th constraint.
    - site: site[i][j] represents which decision variable the j-th decision variable of the i-th constraint corresponds to.

    Return: 
    The result of the graph partitioning.
    '''
    raise NotImplementedError('FENNEL method should be implemented')

def generate_pair(
    number : int
):
    '''
    Function Description:
    Partition the problem based on the given problem instances and generate training data.

    Parameters:
    - number: Number of problem instances.

    Return: 
    The training data is generated and packaged as data.pickle. The function does not have a return value.
    '''
    for turn in range(number):
        print("=====", turn)
        # Check if data.pickle exists and read it if it exists.
        if(os.path.exists('./example/data' + str(turn) + '.pickle') == False):
            print("No problem file!")
            return 
        with open('./example/data' + str(turn) + '.pickle', "rb") as f:
            problem = pickle.load(f)
        
        # Check if solution.pickle exists and read it if it exists.
        if(os.path.exists('./example/sample' + str(turn) + '.pickle') == False):
            print("No solutuion file!")
            return 
        with open('./example/sample' + str(turn) + '.pickle', "rb") as f:
            solution = pickle.load(f)

        # obj_type represents the problem type (maximization/minimization).
        # n represents the number of decision variables.
        # m represents the number of constraints.
        # k[i] represents the number of decision variables in the i-th constraint.
        # site[i][j] represents which decision variable the j-th decision variable of the i-th constraint corresponds to.
        # value[i][j] represents the coefficient of the j-th decision variable in the i-th constraint.
        # constraint[i] represents the right-hand side value of the i-th constraint.
        # constraint_type[i] represents the type of the i-th constraint, where 1 represents <=, 2 represents >=, and 3 represents =.
        # coefficient[i] represents the coefficient of the i-th decision variable in the objective function.
        obj_type = problem[0]
        n = problem[1]
        m = problem[2]
        k = problem[3]
        site = problem[4]
        value = problem[5]
        constraint = problem[6]
        constraint_type = problem[7]
        coefficient = problem[8]

        variable_features = solution[0]
        constraint_features = solution[1]
        edge_indices = solution[2]
        edge_features = solution[3]
        optX = solution[4]
        
        # Get the graph partitioning result.
        new_color = FENNEL(n, m, k, site)
        
        with open('./example/pair' + str(turn) + '.pickle', 'wb') as f:
            pickle.dump([variable_features, constraint_features, edge_indices, edge_features, new_color, optX], f)

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--number", type = int, default = 10, help = 'The number of instances.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    #print(vars(args))
    generate_pair(**vars(args))