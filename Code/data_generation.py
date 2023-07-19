import numpy as np
import argparse
import pickle
import random
import time
import os

def generate_IS(N, M):
    '''
    Function Description:
    Generate instances of the maximum independent set problem in a general graph.
    
    Parameters:
    - N: Number of vertices in the graph.
    - M: Number of edges in the graph.

    Return: 
    Relevant parameters of the generated maximum independent set problem.
    '''
    
    # n represents the number of decision variables, where each vertex in the graph corresponds to a decision variable.
    # m represents the number of constraints, where each edge in the graph corresponds to a constraint.
    # k[i] represents the number of decision variables in the i-th constraint.
    n = N
    m = M
    k = []

    # site[i][j] represents which decision variable the j-th decision variable corresponds to in the i-th constraint.
    # value[i][j] represents the coefficient of the j-th decision variable in the i-th constraint.
    # constraint[i] represents the right-hand side value of the i-th constraint.
    # constraint_type[i] represents the type of the i-th constraint, where 1 represents <=, 2 represents >=, and 3 represents =.
    # coefficient[i] represents the coefficient of the i-th decision variable in the objective function.
    site = []
    value = []
    for i in range(m):
        site.append([])
        value.append([])
        k.append(0)
    constraint = np.zeros(m)
    constraint_type = np.zeros(m)
    coefficient = {}

    # Add constraint: randomly generate an edge and impose a constraint that the vertices connected by the edge cannot be selected simultaneously.
    for i in range(M):
        x = random.randint(0, N - 1)
        y = random.randint(0, N - 1)
        while(x == y) :
            x = random.randint(0, N - 1)
            y = random.randint(0, N - 1)
        site[i].append(x)
        value[i].append(1)
        site[i].append(y) 
        value[i].append(1)
        constraint[i] = 1
        constraint_type[i] = 1
        k[i] = 2
    
    # Set the coefficients of the objective function, where the coefficient value of each decision variable corresponding to a vertex is a random value.
    for i in range(N):
        coefficient[i] = random.random()
    
    return(n, m, k, site, value, constraint, constraint_type, coefficient)

def generate_MVC(N, M):
    '''
    Function Description:
    Generate instances of the minimum vertex cover problem in a general graph.

    Parameters:
    - N: Number of vertices in the graph.
    - M: Number of edges in the graph.

    Return: 
    Relevant parameters of the generated minimum vertex cover problem.
    '''
    
    # n represents the number of decision variables, where each vertex in the graph corresponds to a decision variable.
    # m represents the number of constraints, where each edge in the graph corresponds to a constraint.
    # k[i] represents the number of decision variables in the i-th constraint.
    n = N
    m = M
    k = []

    # site[i][j] represents which decision variable the j-th decision variable corresponds to in the i-th constraint.
    # value[i][j] represents the coefficient of the j-th decision variable in the i-th constraint.
    # constraint[i] represents the right-hand side value of the i-th constraint.
    # constraint_type[i] represents the type of the i-th constraint, where 1 represents <=, 2 represents >=, and 3 represents =.
    # coefficient[i] represents the coefficient of the i-th decision variable in the objective function.
    site = []
    value = []
    for i in range(m):
        site.append([])
        value.append([])
        k.append(0)
    constraint = np.zeros(m)
    constraint_type = np.zeros(m)
    coefficient = {}

    # Add constraint: randomly generate an edge and impose a constraint that at least one of the vertices connected by the edge must be selected.
    for i in range(M):
        x = random.randint(0, N - 1)
        y = random.randint(0, N - 1)
        while(x == y) :
            x = random.randint(0, N - 1)
            y = random.randint(0, N - 1)
        k[i] = 2
        site[i].append(x)
        value[i].append(1)
        site[i].append(y)
        value[i].append(1)
        constraint[i] = 1
        constraint_type[i] = 2
    
    # Set the coefficients of the objective function, where the coefficient value of each decision variable corresponding to a vertex is a random value.
    for i in range(N):
        coefficient[i] = random.random()
    
    return(n, m, k, site, value, constraint, constraint_type, coefficient)

def generate_SC(N, M):
    '''
    Function Description:
    Generate instances of the set cover problem, where each item is guaranteed to appear in exactly 4 sets.

    Parameters:
    - N: Number of sets.
    - M: Number of items.

    Return: 
    Relevant parameters of the generated set cover problem.
    '''

    # n represents the number of decision variables, where each set corresponds to a decision variable.
    # m represents the number of constraints, where each item corresponds to a constraint.
    # k[i] represents the number of decision variables in the i-th constraint.
    n = N
    m = M
    k = []

    # site[i][j] represents which decision variable the j-th decision variable corresponds to in the i-th constraint.
    # value[i][j] represents the coefficient of the j-th decision variable in the i-th constraint.
    # constraint[i] represents the right-hand side value of the i-th constraint.
    # constraint_type[i] represents the type of the i-th constraint, where 1 represents <=, 2 represents >=, and 3 represents =.
    # coefficient[i] represents the coefficient of the i-th decision variable in the objective function.
    site = []
    value = []
    for i in range(m):
        site.append([])
        value.append([])
        k.append(0)
    constraint = np.zeros(m)
    constraint_type = np.zeros(m)
    coefficient = {}

    # Add constraint: At least one of the four sets in which each item appears must be selected.
    for i in range(M):
        vis = {}
        for j in range(4):
            now = random.randint(0, N - 1)
            while(now in vis.keys()):
                now = random.randint(0, N - 1)
            vis[now] = 1

            site[i].append(now)
            value[i].append(1)
        k[i] = 4   
    for i in range(M):
        constraint[i] = 1
        constraint_type[i] = 2
    
    # Set the coefficients of the objective function, where the coefficient value of each decision variable corresponding to a set is a random value.
    for i in range(N):
        coefficient[i] = random.random()
    
    
    return(n, m, k, site, value, constraint, constraint_type, coefficient)

def generate_CAT(N, M):
    '''
    Function Description:
    Generate instances of the set cover problem, where each item is guaranteed to appear in exactly 5 sets.

    Parameters:
    - N: Number of sets.
    - M: Number of items.

    Return: 
    Relevant parameters of the generated set cover problem.
    '''
    # n represents the number of decision variables, where each set corresponds to a decision variable.
    # m represents the number of constraints, where each item corresponds to a constraint.
    # k[i] represents the number of decision variables in the i-th constraint.
    n = N
    m = M
    k = []

    # site[i][j] represents which decision variable the j-th decision variable corresponds to in the i-th constraint.
    # value[i][j] represents the coefficient of the j-th decision variable in the i-th constraint.
    # constraint[i] represents the right-hand side value of the i-th constraint.
    # constraint_type[i] represents the type of the i-th constraint, where 1 represents <=, 2 represents >=, and 3 represents =.
    # coefficient[i] represents the coefficient of the i-th decision variable in the objective function.
    
    site = []
    value = []
    for i in range(m):
        site.append([])
        value.append([])
        k.append(0)
    constraint = np.zeros(m)
    constraint_type = np.zeros(m)
    coefficient = {}

    # Add constraints.
    for i in range(M):
        vis = {}
        for j in range(5):
            now = random.randint(0, N - 1)
            while(now in vis.keys()):
                now = random.randint(0, N - 1)
            vis[now] = 1

            site[i].append(now)
            value[i].append(1)
        k[i] = 5   
    for i in range(M):
        constraint[i] = 1
        constraint_type[i] = 1
    
    # Set the coefficients of the objective function, where the coefficient value of each decision variable corresponding to a set is a random value.
    for i in range(N):
        coefficient[i] = random.random() * 1000
    
    
    return(n, m, k, site, value, constraint, constraint_type, coefficient)

def generate_samples(
    problem_type : str,
    difficulty_mode : str,
    seed : int, 
    number : int
):
    '''
    Function Description:
    Generate problem instances based on the provided parameters and package the output as data.pickle.

    Parameters:
    - problem_type: Available options are ['IS', 'MVC', 'MAXCUT', 'SC'], representing the maximum independent set problem, minimum vertex cover problem, maximum cut problem, minimum set cover problem, and Meituan flash sale problem, respectively.
    - difficulty_mode: Available options are ['easy', 'medium', 'hard'], representing easy (small-scale), medium (medium-scale), and hard (large-scale) difficulties.
    - seed: Integer value indicating the starting random seed used for problem generation.
    - number: Integer value indicating the number of instances to generate.

    Return: 
    The problem instances are generated and packaged as data.pickle. The function does not have a return value.
    '''
    # Set the random seed.
    random.seed(seed) 

    # Check and create using the os module.
    dir_name = 'example'
    if not os.path.exists(dir_name):  
        os.mkdir(dir_name)

    for i in range(number):
        # Randomly generate instances of the maximum independent set problem and package the output.
        if(problem_type == 'IS'):
            if(difficulty_mode == 'easy'):
                N = 10000
                M = 30000
            elif(difficulty_mode == 'medium'):
                N = 100000
                M = 300000
            else:
                N = 1000000
                M = 3000000  
            n, m, k, site, value, constraint, constraint_type, coefficient = generate_IS(N, M)
            with open('./example/data' + str(i) + '.pickle', 'wb') as f:
                pickle.dump(['maximize', n, m, k, site, value, constraint, constraint_type, coefficient], f)
        
        # Randomly generate instances of the minimum vertex cover problem and package the output.
        if(problem_type == 'MVC'):
            if(difficulty_mode == 'easy'):
                N = 10000
                M = 30000
            elif(difficulty_mode == 'medium'):
                N = 100000
                M = 300000
            else:
                N = 1000000
                M = 3000000 
            n, m, k, site, value, constraint, constraint_type, coefficient = generate_MVC(N, M)
            with open('./example/data' + str(i) + '.pickle', 'wb') as f:
                pickle.dump(['minimize', n, m, k, site, value, constraint, constraint_type, coefficient], f)
        
        # Randomly generate instances of the minimum set cover problem and package the output.
        if(problem_type == 'SC'):
            if(difficulty_mode == 'easy'):
                N = 20000
                M = 20000
            elif(difficulty_mode == 'medium'):
                N = 200000
                M = 200000
            else:
                N = 2000000
                M = 2000000 
            n, m, k, site, value, constraint, constraint_type, coefficient = generate_SC(N, M)
            with open('./example/data' + str(i) + '.pickle', 'wb') as f:
                pickle.dump(['minimize', n, m, k, site, value, constraint, constraint_type, coefficient], f)
        
        # Randomly generate instances of the combinatorial auction problem and package the output.
        if(problem_type == 'CAT'):
            if(difficulty_mode == 'easy'):
                N = 10000
                M = 10000
            elif(difficulty_mode == 'medium'):
                N = 100000
                M = 100000
            else:
                N = 1000000
                M = 1000000 
            n, m, k, site, value, constraint, constraint_type, coefficient = generate_CAT(N, M)
            with open('./example/data' + str(i) + '.pickle', 'wb') as f:
                pickle.dump(['maximize', n, m, k, site, value, constraint, constraint_type, coefficient], f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_type", choices = ['IS', 'MVC', 'SC',  'CAT'], default = 'SC', help = "Problem type selection")
    parser.add_argument("--difficulty_mode", choices = ['easy', 'medium', 'hard'], default = 'easy', help = "Difficulty level.")
    parser.add_argument('--seed', type = int, default = 0, help = 'Random generator seed.')
    parser.add_argument("--number", type = int, default = 10, help = 'The number of instances.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    #print(vars(args))
    generate_samples(**vars(args))