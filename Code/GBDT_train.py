import numpy as np
import argparse
import pickle
import random
import os
from model.gbdt_regressor import GradientBoostingRegressor

def train_GBDT(number:int):
    '''
    Function Description:
    Train GBDT based on the given problem instances and the neural encoding results of the decision variables.

    Parameters:
    - number: Number of problem instances.

    Return: 
    The trained GBDT is generated and packaged as data.pickle. The function does not have a return value.
    '''
    print("Tesing the performance of GBDT regressor...")
    # Load data
    data = []
    label = []
    max_num = 200000
    now_num = 0
    for num in range(number):
        # Check if data.pickle exists and read it if it exists.
        if(os.path.exists('./example/node' + str(num) + '.pickle') == False):
            print("No problem file!")
            return 
        with open('./example/node' + str(num) + '.pickle', "rb") as f:
            node = pickle.load(f)
        now_data = node[0]
        now_label = node[1]
        p = max_num / (len(node[0]) * number)
        for i in range(len(now_data)):
            if(random.random() <= p):
                data.append(now_data[i])
                label.append(now_label[i])
                now_num += 1
    # Train model
    print(now_num)
    reg = GradientBoostingRegressor()
    #print(data)
    reg.fit(data=np.array(data), label=np.array(label), n_estimators=30, learning_rate=0.1, max_depth=5, min_samples_split=2)
    # Model evaluation
    with open('./GBDT.pickle', 'wb') as f:
        pickle.dump([reg], f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--number", type = int, default = 10, help = 'The number of instances.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    #print(vars(args))
    train_GBDT(**vars(args))