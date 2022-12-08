import time
from SecBiclib.algorithms import ChengChurchAlgorithm
from SecBiclib.algorithms import SecuredChengChurchAlgorithm
from SecBiclib.algorithms import SecuredChengChurchAlgorithmType1
from SecBiclib.algorithms import SecuredChengChurchAlgorithmType2
from SecBiclib.algorithms import SecuredChengChurchAlgorithmType3
from SecBiclib.algorithms import SecuredChengChurchAlgorithmType4
from SecBiclib.evaluation import clustering_error
from SecBiclib.datasets import load_yeast_tavazoie
from SecBiclib.datasets import synthetic
import numpy as np

m0 = time.perf_counter()

# load yeast data used in the original Cheng and Church's paper
# data = load_yeast_tavazoie().values
num_rows, num_cols = 10, 5
n_elements = num_rows*num_cols
np.random.seed(42)                                          # Fixed seed for reproducibility
data = np.random.randint(0, 5, size=(num_rows, num_cols))
# data, predicted = synthetic.make_const_data()
# data, predicted = synthetic.make_scale_data()

# data = np.random.randint(low=0, high=255, size=(100, 200))
# data, predicted = synthetic.make_const_data()
# data, predicted = synthetic.make_const_data()
# num_rows, num_cols = data.shape

# print(type(predicted))
# missing value imputation suggested by Cheng and Church
# missing = np.where(data == 999 or data != )
# missing = np.where(data < 0.0)
# data[missing] = np.random.randint(low=0, high=800, size=len(missing[0]))

# data[missing] = np.random.randint(low=-800, high=800, size=len(missing[0]))
# missing = np.where(data < 0.0)
# data[missing] = np.random.randint(low=0, high=800, size=len(missing[0]))

# creating an instance of the ChengChurchAlgorithm class and running with the parameters
cca = ChengChurchAlgorithm()
secca = SecuredChengChurchAlgorithmType1()
# secca = SecuredChengChurchAlgorithmType4()
# secca = SecuredChengChurchAlgorithmType1(num_biclusters=5, msr_threshold=300.0, multiple_node_deletion_threshold=1.2)
# print(predicted)
predict = cca.run(data)
# print(predict)
print(predict)
# biclustering = secca.run(data)
# print(biclustering)
# print(type(biclustering))
# biclustering_ref = cca.run(predicted)
# ce = clustering_error(biclustering, predict,  num_rows, num_cols)

# print("ce:{}".format(ce))
# print(biclustering)
# with open('resBiclustersCCATotal', 'w') as saveFile:
#     saveFile.write(biclustering.__str__())
#     saveFile.write("\n")

# print(biclustering)

m1 = time.perf_counter()
# print("Time Performance in Original Algorithm: ", round(m1 - m0, 5), "Seconds")


# f = open("SecBiclib.txt", "r")


