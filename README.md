## GNN&GBDT-Guided Fast Optimizing Framework for Large-scale Integer Programming

### Contact

Welcome to academic and business collaborations with funding support. For more details, please contact us via email at [xuhua@tsinghua.edu.cn](mailto:xuhua@tsinghua.edu.cn).

### Overview

This release contains the key processes of the GNN&GBDT-Guided Fast Optimizing Framework, as described in the paper. The provided code implements the main components of the approach, covering data generation, training, and inference. We also provide interfaces that are left to be implemented by the user so that the code can be flexibly used in different contexts.

The following gives a brief overview of the contents; more detailed documentation is available within each file:

*   __Code/data_generation.py__: Generating integer programming problems for training and testing.
*   __Code/data_solution.py__: Generate optimal solutions to integer programming problems for training.
*   __Code/data_partition.py__: Generate graph partition results for bipartite graph representations of integer programming problems used for training.
*   __Code/GNN_train.py__: Use prepared training data to train GNN to generate neural embeddings of decision variables.
*   __Code/GNN_inference.py__: Using a trained GNN to generate neural embeddings of decision variables for training data.
*   __Code/GBDT_train.py__: Train GBDT using the training data and the resulting neural embeddings results.
*   __Code/test.py__: Run test data to get optimized results.
*   __Code/model/graphcnn.py__: GNN model.
*   __Code/model/gbdt_regressor.py__: GBDT model.
*   __Result/CA__: Running results of the very large-scale version of the Combinatorial Auction problem.
*   __Result/MIS__: Running results of the very large-scale version of the Maximum Independent Set problem.
*   __Result/MVC__: Running results of the very large-scale version of the Minimum Vertex Covering problem.
*   __Result/SC__: Running results of the very large-scale version of the Set Covering problem.
*   __Paper/paper.pdf__: PDF version of the paper.

### Requirements

The required environment is shown in GNN_GBDT.yml.

### Usage

1. Implement the interfaces respectively.

3. Perform training according to the following code running order:

   ```
   Code/data_generation.py
   Code/data_solution.py
   Code/data_partition.py
   Code/GNN_train.py
   Code/GNN_inference.py
   Code/GBDT_train.py
   ```

3. Run tests with test.py.

### Citing this work

Paper: [GNN&GBDT-Guided Fast Optimizing Framework for Large-scale Integer Programming](https://openreview.net/pdf?id=tX7ajV69wt)

If you use the code here please cite this paper:

    @inproceedings{ye2023gnn,
      title={GNN\&GBDT-Guided Fast Optimizing Framework for Large-scale Integer Programming},
      author={Ye, Huigen and Xu, Hua and Wang, Hongyan and Wang, Chengming and Jiang, Yu},
      booktitle={ICML},
      year={2023}
    }


