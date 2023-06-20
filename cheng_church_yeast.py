import numpy as np
from SecBiclib.algorithms import ChengChurchAlgorithm
from SecBiclib.datasets import load_yeast_tavazoie

# load yeast data used in the original Cheng and Church's paper
data = load_yeast_tavazoie().values

# missing value imputation suggested by Cheng and Church
missing = np.where(data < 0.0)
data[missing] = np.random.randint(low=0, high=800, size=len(missing[0]))

# creating an instance of the ChengChurchAlgorithm class and running with the parameters of the original study
cca = ChengChurchAlgorithm(num_biclusters=5, msr_threshold=966.0, multiple_node_deletion_threshold=1.2)
biclustering = cca.run(data)

print(biclustering)