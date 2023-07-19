import numpy as np
import argparse
import pickle
import random
import time
import os


def get_best_solution(n, m, k, site, value, constraint, constraint_type, coefficient, time_limit, obj_type):
    '''
    Function Description:
    Solve the problem using an optimization solver based on the provided problem instance.

    Parameters:
    - n: Number of decision variables in the problem instance.
    - m: Number of constraints in the problem instance.
    - k: k[i] represents the number of decision variables in the i-th constraint.
    - site: site[i][j] represents which decision variable the j-th decision variable of the i-th constraint corresponds to.
    - value: value[i][j] represents the coefficient of the j-th decision variable in the i-th constraint.
    - constraint: constraint[i] represents the right-hand side value of the i-th constraint.
    - constraint_type: constraint_type[i] represents the type of the i-th constraint, where 1 represents <=, 2 represents >=, and 3 represents =..
    - coefficient: coefficient[i] represents the coefficient of the i-th decision variable in the objective function.
    - time_limit: Maximum solving time.
    - obj_type: Whether the problem is a maximization problem or a minimization problem.

    Return: 
    The optimal solution of the problem, represented as a list of values for each decision variable in the optimal solution.
    '''

    raise NotImplementedError('get_best_solution method should be implemented')


def optimize(
    time : int,
    number : int,
):
    '''
    Function Description:
    Based on the specified parameter design, invoke the designated algorithm and solver to optimize the optimization problem in data.pickle in the current directory.

    Parameters:
    - number: Integer value indicating the number of instances to generate.
    - suboptimal: Integer value indicating the number of suboptimal solutions to generate.

    Return: 
    The optimal solution is generated and packaged as data.pickle. The function does not have a return value.
    '''


    for num in range(number):
        # Check if data.pickle exists and read it if it exists.
        if(os.path.exists('./example/data' + str(num) + '.pickle') == False):
            print("No input file!")
            return 
        with open('./example/data' + str(num) + '.pickle', "rb") as f:
            data = pickle.load(f)
    
        # n represents the number of decision variables.
        # m represents the number of constraints.
        # k[i] represents the number of decision variables in the i-th constraint.
        # site[i][j] represents which decision variable the j-th decision variable of the i-th constraint corresponds to.
        # value[i][j] represents the coefficient of the j-th decision variable in the i-th constraint.
        # constraint[i] represents the right-hand side value of the i-th constraint.
        # constraint_type[i] represents the type of the i-th constraint, where 1 represents <=, 2 represents >=, and 3 represents =.
        # coefficient[i] represents the coefficient of the i-th decision variable in the objective function.
        n = data[1]
        m = data[2]
        k = data[3]
        site = data[4]
        value = data[5]
        constraint = data[6]
        constraint_type = data[7]
        coefficient = data[8]
        # IS and CAT are maximization problems.
        # MVC and SC are minimization problems.
        obj_type = data[0]
        optimal_solution = get_best_solution(n, m, k, site, value, constraint, constraint_type, coefficient, time, obj_type)
        

        variable_features = []
        constraint_features = []
        edge_indices = [[], []] 
        edge_features = []

        for i in range(n):
            now_variable_features = []
            now_variable_features.append(coefficient[i])
            now_variable_features.append(0)
            now_variable_features.append(1)
            now_variable_features.append(1)
            now_variable_features.append(random.random())
            variable_features.append(now_variable_features)
        
        for i in range(m):
            now_constraint_features = []
            now_constraint_features.append(constraint[i])
            now_constraint_features.append(constraint_type[i])
            now_constraint_features.append(random.random())
            constraint_features.append(now_constraint_features)
        
        for i in range(m):
            for j in range(k[i]):
                edge_indices[0].append(i)
                edge_indices[1].append(site[i][j])
                edge_features.append([value[i][j]])

        with open('./example/sample' + str(num) + '.pickle', 'wb') as f:
                pickle.dump([variable_features, constraint_features, edge_indices, edge_features, optimal_solution], f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', type = int, default = 10, help = 'Running wall-clock time.')
    parser.add_argument("--number", type = int, default = 10, help = 'The number of instances.')
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()
    #print(vars(args))
    optimize(**vars(args))