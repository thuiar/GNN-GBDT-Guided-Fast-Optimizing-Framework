import os
import time
import torch
import random
import pickle
import argparse
import torch_geometric

import numpy as np
import torch.nn.functional as F
from pytorch_metric_learning import losses

from gurobipy import *
from typing import Union
from pathlib import Path
from functools import cmp_to_key
from model.graphcnn import GNNPolicy
from model.gbdt_regressor import GradientBoostingRegressor

def make(constraint_features,
         edge_index,
         edge_attr,
         variable_features,
         model_path : str,
         device : torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    '''
    Function Description:
    Use the trained GNN to obtain the neural encoding results of the decision variables based on the given problem instances.

    Parameters:
    - constraint_features: Initial feature encoding of constraint points in the bipartite representation of the problem.
    - edge_index: Edges in the bipartite representation of the problem.
    - edge_attr: Edge features in the bipartite representation of the problem.
    - variable_features: Initial feature encoding of decision variable points in the bipartite representation of the problem.
    - model_path: Path to the trained GNN model.
    - device: Select the computing device.

    Return: 
    The neural encoding results of the decision variables.
    '''
    policy = GNNPolicy().to(device)
    policy.load_state_dict(torch.load(model_path, policy.state_dict()))
    logits = policy(
        torch.FloatTensor(constraint_features).to(device),
        torch.LongTensor(edge_index).to(device),
        torch.FloatTensor(edge_attr).to(device),
        torch.FloatTensor(variable_features).to(device),
    )
    #print(logits)
    return logits.tolist()

class pair: 
    def __init__(self): 
        self.site = 0
        self.loss = 0

def cmp(a, b):
    if a.loss < b.loss: 
        return -1
    else:
        return 1

def cmp2(a, b):
    if a.loss > b.loss: 
        return -1
    else:
        return 1

def get_best_solution(n, m, k, site, value, constraint, constraint_type, coefficient, time_limit, obj_type, now_sol, now_col):
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
    - constraint_type: constraint_type[i] represents the type of the i-th constraint, where 1 represents <=, 2 represents >=, and 3 represents =.
    - coefficient: coefficient[i] represents the coefficient of the i-th decision variable in the objective function.
    - time_limit: Maximum solving time.
    - obj_type: Whether the problem is a maximization problem or a minimization problem.

    Return: 
    The optimal solution of the problem.
    '''
    raise NotImplementedError('get_best_solution method should be implemented')

def initial_solution_search(n, m, k, site, value, constraint, constraint_type, coefficient, set_time, obj_type, predict, loss):
    '''
    Function Description:
    Perform an initial solution search using fixed-radius neighborhood search based on the given problem instance and the predicted results from GBDT.

    Parameters:
    - n: Number of decision variables in the problem instance.
    - m: Number of constraints in the problem instance.
    - k: k[i] represents the number of decision variables in the i-th constraint.
    - site: site[i][j] represents which decision variable the j-th decision variable of the i-th constraint corresponds to.
    - value: value[i][j] represents the coefficient of the j-th decision variable in the i-th constraint.
    - constraint: constraint[i] represents the right-hand side value of the i-th constraint.
    - constraint_type: constraint_type[i] represents the type of the i-th constraint, where 1 represents <=, 2 represents >=, and 3 represents =.
    - coefficient: coefficient[i] represents the coefficient of the i-th decision variable in the objective function.
    - time_limit: Maximum solving time.
    - obj_type: Whether the problem is a maximization problem or a minimization problem.
    - predict: Predicted results from GBDT.
    - loss: Prediction loss from GBDT.

    Return: 
    The initial feasible solution of the problem and its corresponding objective function value.
    '''
    raise NotImplementedError('initial_solution_search method should be implemented')

def cross_generate_blocks(n, loss, rate, predict, nowX, GBDT, data):
    '''
    Function Description:
    Obtain the neighborhood partitioning result based on the given problem instance, the predicted results from GBDT, and the current solution.

    Parameters:
    - n: Number of decision variables in the problem instance.
    - loss: Prediction loss from GBDT.
    - rate: Neighborhood radius.
    - predict: Predicted results from GBDT.
    - nowX: Current solution of the problem instance.
    - GBDT: Trained Gradient Boosting Decision Tree.
    - data: Neural encoding results of the decision variables.

    Return: A set of partitioning results of the neighborhood.
    '''
    raise NotImplementedError('cross_generate_blocks method should be implemented')

def cross(n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, rate, solA, blockA, solB, blockB, set_time):
    '''
    Function Description:
    Obtain the crossover solution of two neighborhoods based on the given problem instance, the neighborhood information and search results of neighborhood A, the neighborhood information and search results of neighborhood B.

    Parameters:
    - n: Number of decision variables in the problem instance.
    - m: Number of constraints in the problem instance.
    - k: k[i] represents the number of decision variables in the i-th constraint.
    - site: site[i][j] represents which decision variable the j-th decision variable of the i-th constraint corresponds to.
    - value: value[i][j] represents the coefficient of the j-th decision variable in the i-th constraint.
    - constraint: constraint[i] represents the right-hand side value of the i-th constraint.
    - constraint_type: constraint_type[i] represents the type of the i-th constraint, where 1 represents <=, 2 represents >=, and 3 represents =.
    - coefficient: coefficient[i] represents the coefficient of the i-th decision variable in the objective function.
    - rate: Neighborhood radius.
    - solA: Search result of neighborhood A.
    - blockA: Neighborhood information of neighborhood A.
    - solB: Search result of neighborhood B.
    - blockB: Neighborhood information of neighborhood B.
    - set_time: Set running time.

    Return: 
    The crossover solution of the two neighborhoods and their corresponding objective function values.
    '''
    raise NotImplementedError('cross method should be implemented')
        
        
def optimize(fix : float,
             set_time : int,
             rate : float,
             model_path : str,
             device : torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    begin_time = time.time()

    if(os.path.exists('./example-IS-h/data0.pickle') == False):
        print("No problem file!")

    with open('./example-IS-h/data0.pickle', "rb") as f:
        problem = pickle.load(f)

    obj_type = problem[0]
    n = problem[1]
    m = problem[2]
    k = problem[3]
    site = problem[4]
    value = problem[5]
    constraint = problem[6]
    constraint_type = problem[7]
    coefficient = problem[8]


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
    data = make(constraint_features, edge_indices, edge_features, variable_features, model_path, device)


    if(os.path.exists('./GBDT-IS-h.pickle') == False):
        print("No problem file!")

    with open('./GBDT-IS-h.pickle', "rb") as f:
        GBDT = pickle.load(f)[0]
    predict = GBDT.predict(np.array(data))
    loss =  GBDT.calc(np.array(data))

    
    # Initial solution search.
    ansTime = []
    ansVal = []
    nowX, nowVal = initial_solution_search(n, m, k, site, value, constraint, constraint_type, coefficient, set_time, obj_type, predict, loss) 
    ansTime.append(time.time() - begin_time)
    ansVal.append(nowVal)

    while(time.time() - begin_time < set_time):
        turnX = []
        turnVal = []
        block_list, _, _ = cross_generate_blocks(n, loss, rate, predict, nowX, GBDT, data)
        # GBDT-guided neighborhood partitioning and neighborhood search.
        for i in range(4):
            max_time = set_time - (time.time() - begin_time)
            if(max_time <= 0):
                break
            newX, newVal = get_best_solution(n, m, k, site, value, constraint, constraint_type, coefficient, max_time, obj_type, nowX, block_list[i])
            turnX.append(newX)
            turnVal.append(newVal)
        
        # First-level crossover between neighborhoods.
        if(len(turnX) == 4):
            max_time = set_time - (time.time() - begin_time)
            if(max_time <= 0):
                break
            newX, newVal = cross(n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, rate, turnX[0], block_list[0], turnX[1], block_list[1], max_time)
            if(turnVal != -1):
                turnX.append(newX)
                turnVal.append(newVal)

            newX, newVal = cross(n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, rate, turnX[2], block_list[2], turnX[3], block_list[3],  max_time)
            if(turnVal != -1):
                turnX.append(newX)
                turnVal.append(newVal)
        
        # Second-level crossover between neighborhoods.
        if(len(turnX) == 6):
            max_time = set_time - (time.time() - begin_time)
            if(max_time <= 0):
                break

            block_list.append(np.zeros(n, int))
            for i in range(n):
                if(block_list[0][i] == 1 or block_list[1][i] == 1):
                    block_list[4][i] = 1
            block_list.append(np.zeros(n, int))
            for i in range(n):
                if(block_list[2][i] == 1 or block_list[3][i] == 1):
                    block_list[5][i] = 1
            
            newX, newVal = cross(n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, rate, turnX[4], block_list[4], turnX[5], block_list[5], max_time)
            if(turnVal != -1):
                turnX.append(newX)
                turnVal.append(newVal)
        
        # Update the current solution as the current optimal solution.
        for i in range(len(turnVal)):
            if(obj_type == 'maximize'):
                if(turnVal[i] > nowVal):
                    nowVal = turnVal[i]
                    for j in range(n):
                        nowX[j] = turnX[i][j]
            else:
                if(turnVal[i] < nowVal):
                    nowVal = turnVal[i]
                    for j in range(n):
                        nowX[j] = turnX[i][j]
        
        ansTime.append(time.time() - begin_time)
        ansVal.append(nowVal)
    print(ansTime)
    print(ansVal)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", type = float, default = 0.1, help = 'time.')
    parser.add_argument("--set_time", type = int, default = 100, help = 'set_time.')
    parser.add_argument("--rate", type = float, default = 0.2, help = 'sub rate.')
    parser.add_argument("--model_path", type=str, default="trained_model-IS-h.pkl")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training.")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    #print(vars(args)["model_path"])
    optimize(**vars(args))