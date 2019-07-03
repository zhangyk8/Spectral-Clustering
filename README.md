# Spectral-Clustering
Python 3 Implementations of normalized and unnormalized spectral clustering algorithms

## Requirements
- Python >= 3.6 (Earlier version might be applicable.)
- [NumPy](http://www.numpy.org/), [Matplotlib](https://matplotlib.org/), [scikit-learn](https://scikit-learn.org/stable/index.html) (Used for KMeans clustering and generating "moon" data), [SciPy](https://www.scipy.org/) (Only the function [scipy.spatial.distance.pdist](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.distance.pdist.html) is envoked to compute the pairwise distance between points.)

## Descriptions
We implement three different versions of Spectral Clustering based on the paper ["A Tutorial on Spectral Clustering"](https://arxiv.org/abs/0711.0189) written by _Ulrike von Luxburg_. The dataset or adjacency matrix is stored in a [NumPy](http://www.numpy.org/) array. To use the function,
```bash
from spectral_clustering import Spectral_Clustering
```
Depending on the RAM of the computer, this naive implementation of Spectral Clustering may not be scalable to a dataset with more than 5000 instances. However, we also furnish a pyspark implementation of [Power Iteration Clustering](http://www.cs.cmu.edu/~frank/papers/icml2010-pic-final.pdf), which is assumed to be scalable to the graph with millions of nodes and edges.  To use the function for Power Iteration Clustering,
```bash
from PIC import Power_Iteration_Clustering
```

## Disclaimer
If you have some questions or detect some bugs, please feel free to email me. Thank you in advance!
