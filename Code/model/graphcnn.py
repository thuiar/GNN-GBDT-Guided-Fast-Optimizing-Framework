import numpy as np
import torch
import torch_geometric

__all__ = ["GNNPolicy"]

class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    Class Description:
    Based on graph convolution, define the bipartite graph semi-convolution process.
    """

    def __init__(self):
        '''
        Function Description:
        Define the size of the encoding space, and implement the semi-convolution layer and output layer.
        '''
        raise NotImplementedError('BipartiteGraphConvolution __init__ method should be implemented')

    def forward(self, left_features, edge_indices, edge_features, right_features):
        '''
        Function Description:
        Based on the given node and edge features, output the result of forward propagation after semi-convolution.

        Parameters:
        - left_features: Features of the nodes on the left side of the bipartite graph.
        - edge_indices: Edge information.
        - edge_features: Edge features.
        - right_features: Features of the nodes on the right side of the bipartite graph.

        Return: The result after forward propagation.
        '''
        raise NotImplementedError('BipartiteGraphConvolution forward method should be implemented')

    def message(self, node_features_i, node_features_j, edge_features):
        '''
        Function Description:
        This method sends the messages, computed in the message method.
        
        Parameters:
        - node_features_i: Features of the nodes on the left side of the bipartite graph.
        - node_features_j: Features of the nodes on the right side of the bipartite graph.
        - edge_features: Edge features.

        Return: The result after the message passing in the semi-convolution.
        '''
        raise NotImplementedError('BipartiteGraphConvolution message method should be implemented')


class GNNPolicy(torch.nn.Module):
    """
    Class Description:
    Based on the semi-convolutional layer, define the entire GNN network structure.
    """
    def __init__(self):
        '''
        Function Description:
        Define the size of the encoding space, and define the layers for decision variable encoding, edge feature encoding, and constraint feature encoding.
        Define two semi-convolutional layers and the final output layer.
        '''
        raise NotImplementedError('GNNPolicy __init__ method should be implemented')

        

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        '''
        Function Description:
        Based on the given constraint, edge, and variable features, output the result of forward propagation after GNN.

        Parameters:
        - constraint_features: Features of the constraint points.
        - edge_indices: Edge information.
        - edge_features: Edge features.
        - variable_features: Features of the variable points.

        Return: The result after forward propagation.
        '''
        raise NotImplementedError('GNNPolicy forward method should be implemented')

