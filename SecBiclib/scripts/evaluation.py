from SecBiclib.algorithms import ChengChurchAlgorithm
from SecBiclib.algorithms import SecuredChengChurchAlgorithm
from SecBiclib.evaluation import clustering_error
from SecBiclib.datasets import load_yeast_tavazoie

# load yeast data used in the original Cheng and Church's paper
data = load_yeast_tavazoie().values
num_rows, num_cols = data.shape


# creating an instance of the ChengChurchAlgorithm, SecuredChengChurchAlgorithm classes
# and running with the parameters of the original study
cca = ChengChurchAlgorithm(num_biclusters=5, msr_threshold=300.0, multiple_node_deletion_threshold=1.2)
secca = SecuredChengChurchAlgorithm(num_biclusters=5, msr_threshold=300.0, multiple_node_deletion_threshold=1.2)
biclustering_ref = cca.run(data)
biclustering_pre = secca.run(data)

# creating an instance of the clustering error class
# and running with reference and predicted algorithms
ce_eval = round(clustering_error(biclustering_pre, biclustering_ref, num_rows, num_cols),5)

print(ce_eval)


