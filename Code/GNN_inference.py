import argparse
import pickle
from pathlib import Path
from typing import Union

import os
import torch
import torch.nn.functional as F
import torch_geometric
from pytorch_metric_learning import losses

from model.graphcnn import GNNPolicy

class BipartiteNodeData(torch_geometric.data.Data):
    """
    Class Description:
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """

    def __init__(
        self,
        constraint_features,
        edge_indices,
        edge_features,
        variable_features,
        assignment1,
        assignment2
    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.assignment1 = assignment1
        self.assignment2 = assignment2

    def __inc__(self, key, value, store, *args, **kwargs):
        """
        Function Description:
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == "edge_index":
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]]
            )
        elif key == "candidates":
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class GraphDataset(torch_geometric.data.Dataset):
    """
    Class Description:
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        """
        Function Description:
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        with open(self.sample_files[index], "rb") as f:
            [variable_features, constraint_features, edge_indices, edge_features, solution1, solution2] = pickle.load(f)

        graph = BipartiteNodeData(
            torch.FloatTensor(constraint_features),
            torch.LongTensor(edge_indices),
            torch.FloatTensor(edge_features),
            torch.FloatTensor(variable_features),
            torch.LongTensor(solution1),
            torch.LongTensor(solution2),
        )

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = len(constraint_features) + len(variable_features)
        graph.cons_nodes = len(constraint_features)
        graph.vars_nodes = len(variable_features)

        return graph

def make(number: int,
         model_path : str,
         device : torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Function Description:
    Obtain the encoding information of the decision variables based on the input problem data and package the output.
    """
    policy = GNNPolicy().to(device)
    policy.load_state_dict(torch.load(model_path, policy.state_dict()))
    File = []
    for num in range(number):
        if(os.path.exists('./example/pair' + str(num) + '.pickle') == False):
            print("No input file!")
            return 
        File.append('example/pair' + str(num) + '.pickle')
    
    data = GraphDataset(File)
    loader = torch_geometric.loader.DataLoader(data, batch_size = 1)

    now_site = 0
    for batch in loader:
        batch = batch.to(device)
        # Compute the logits (i.e. pre-softmax activations) according to the policy on the concatenated graphs
        logits = policy(
            batch.constraint_features,
            batch.edge_index,
            batch.edge_attr,
            batch.variable_features,
        )
        print(logits)
        with open('./example/sample' + str(now_site) + '.pickle', "rb") as f:
            solution = pickle.load(f)
        with open('./example/node' + str(now_site) + '.pickle', 'wb') as f:
            pickle.dump([logits.tolist(), solution[4]], f)
            print(now_site)
            now_site += 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--number", type=int, default=30)
    parser.add_argument("--model_path", type=str, default="trained_model.pkl")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    make(**vars(args))