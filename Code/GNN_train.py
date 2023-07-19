import argparse
import pickle
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
import torch_geometric
from pytorch_metric_learning import losses

from model.graphcnn import GNNPolicy

__all__ = ["train"]

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
        Overload the pytorch geometric method that tells how to increment indices when concatenating graphs for those entries (edge index, candidates) for which this is not obvious.
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


def pad_tensor(input_, pad_sizes, pad_value=-1e8):
    """
    Function Description:
    This utility function splits a tensor and pads each split to make them all the same size, then stacks them.
    """
    max_pad_size = pad_sizes.max()
    output = input_.split(pad_sizes.cpu().numpy().tolist())
    output = torch.stack(
        [
            F.pad(slice_, (0, max_pad_size - slice_.size(0)), "constant", pad_value)
            for slice_ in output
        ],
        dim=0,
    )
    return output

def process(policy, data_loader, device, optimizer=None):
    """
    Function Description:
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """
    mean_loss = 0
    mean_acc = 0

    n_samples_processed = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in data_loader:
            #print("QwQ")
            batch = batch.to(device)
            # Compute the logits (i.e. pre-softmax activations) according to the policy on the concatenated graphs
            logits = policy(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
            )
            # Graph partitioning related metric functions, where num_classes represents the number of partitions in the graph.
            loss_funcA = losses.ProxyAnchorLoss(num_classes = 10, embedding_size = 16)
            # Metric functions related to the optimal solution. In general integer programming problems, clustering the solution values and modifying num_classes can be done.
            loss_funcB = losses.ProxyAnchorLoss(num_classes = 2, embedding_size = 16)
            loss = loss_funcA(logits, batch.assignment1.to(torch.int64)) + loss_funcB(logits, batch.assignment2.to(torch.int64))
            
            loss_optimizerA = torch.optim.SGD(loss_funcA.parameters(), lr = 0.01)
            loss_optimizerB = torch.optim.SGD(loss_funcB.parameters(), lr = 0.01)
            
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_optimizerA.step()
                loss_optimizerB.step()

            mean_loss += loss.item() * batch.num_graphs
            n_samples_processed += batch.num_graphs

    mean_loss /= n_samples_processed
    return mean_loss

def train(
    model_save_path: Union[str, Path],
    batch_size: int = 1,
    learning_rate: float = 1e-3,
    num_epochs: int = 20,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    """
    Function Description:
    This function trains a GNN policy on training data. 

    Parameters:
    - data_path: Path to the data directory.
    - model_save_path: Path to save the model.
    - batch_size: Batch size for training.
    - learning_rate: Learning rate for the optimizer.
    - num_epochs: Number of epochs to train for.
    - device: Device to use for training.
    """
    # load samples from data_path and divide them
    sample_files = [str(path) for path in Path('./example').glob("pair*.pickle")]
    print(sample_files)
    train_files = sample_files[: int(0.6 * len(sample_files))]
    valid_files = sample_files[int(0.9 * len(sample_files)) :]

    train_data = GraphDataset(train_files)
    train_loader = torch_geometric.loader.DataLoader(train_data, batch_size=batch_size, shuffle = False)
    valid_data = GraphDataset(valid_files)
    valid_loader = torch_geometric.loader.DataLoader(valid_data, batch_size=batch_size, shuffle = False)

    policy = GNNPolicy().to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        train_loss = process(policy, train_loader, device, optimizer)
        #valid_loss = process(policy, valid_loader, device, None)
        valid_loss = 0
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:0.3f}, Valid Loss: {valid_loss:0.3f}")

    torch.save(policy.state_dict(), model_save_path)
    print(f"Trained parameters saved to {model_save_path}")

def parse_args():
    """
    Function Description:
    This function parses the command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_save_path", type=str, default="trained_model.pkl", help="Path to save the model.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train for.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(**vars(args))
