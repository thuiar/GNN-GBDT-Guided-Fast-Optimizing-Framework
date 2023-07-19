import numpy as np
from numpy import ndarray


class GradientBoostingRegressor():
    '''
    Class Description:
    GBDT class, which stores the trained GBDT.
    '''
    def __init__(self):
        '''
        Function Description:
        Initialize the GBDT.
        '''
        raise NotImplementedError('GradientBoostingRegressor __init__ method should be implemented')

    def fit(self, data: ndarray, 
            label: ndarray, 
            n_estimators: int, 
            learning_rate: float,
            max_depth: int, 
            min_samples_split: int, 
            subsample=None):
        '''
        Function Description:
        Train the GBDT based on the given decision variable neural encoding and optimal solution values.

        Parameters:
        - data: Neural encoding results of the decision variables.
        - label: Values of the decision variables in the optimal solution.
        - n_estimators: Number of decision trees.
        - learning_rate: Learning rate.
        - max_depth: Maximum depth of the decision trees.
        - min_samples_split: Minimum number of samples required to split a leaf node.
        - subsample: Subsample rate without replacement.

        Return: 
        The training results are stored in the class. There is no return value.
        '''
        raise NotImplementedError('GradientBoostingRegressor fit method should be implemented')

    def predict(self, data: ndarray) -> ndarray:
        '''
        Function Description:
        Use the trained GBDT to predict the initial solution based on the given decision variable neural encoding, and return the predicted initial solution.

        Parameters:
        - data: Neural encoding results of the decision variables.

        Return: 
        The predicted initial solution.
        '''
        raise NotImplementedError('GradientBoostingRegressor predict method should be implemented')
    
    def calc(self, data: ndarray) -> ndarray:
        '''
        Function Description:
        Use the trained GBDT to predict the initial solution based on the given decision variable neural encoding, and return the prediction loss.

        Parameters:
        - data: Neural encoding results of the decision variables.

        Return: 
        The prediction loss generated when predicting the initial solution for each decision variable.
        '''
        raise NotImplementedError('GradientBoostingRegressor calc method should be implemented')
