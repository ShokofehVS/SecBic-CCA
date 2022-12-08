from SecBiclib.algorithms import ChengChurchAlgorithm
from SecBiclib.algorithms import SecuredChengChurchAlgorithm
from SecBiclib.algorithms import SecuredChengChurchAlgorithmType1
from SecBiclib.algorithms import SecuredChengChurchAlgorithmType2
from SecBiclib.algorithms import SecuredChengChurchAlgorithmType3
from SecBiclib.algorithms import SecuredChengChurchAlgorithmType4
from SecBiclib.evaluation import clustering_error
from SecBiclib.datasets import load_yeast_tavazoie
import matplotlib.pyplot as plt


data = load_yeast_tavazoie().values
num_rows, num_cols = data.shape

cca = ChengChurchAlgorithm(num_biclusters=5, msr_threshold=300.0, multiple_node_deletion_threshold=1.2)
secca = SecuredChengChurchAlgorithm(num_biclusters=5, msr_threshold=300.0, multiple_node_deletion_threshold=1.2)
type1 = SecuredChengChurchAlgorithmType1(num_biclusters=5, msr_threshold=300.0, multiple_node_deletion_threshold=1.2)
type2 = SecuredChengChurchAlgorithmType2(num_biclusters=5, msr_threshold=300.0, multiple_node_deletion_threshold=1.2)
type3 = SecuredChengChurchAlgorithmType3(num_biclusters=5, msr_threshold=300.0, multiple_node_deletion_threshold=1.2)
type4 = SecuredChengChurchAlgorithmType4(num_biclusters=5, msr_threshold=300.0, multiple_node_deletion_threshold=1.2)

# secca_alog = [type1, type2, type3, type4, secca]
secca_alog = [type3]
biclustering_ref = cca.run(data)

ce_eval = []

for i in range(len(secca_alog)):
    biclustering_pre = secca_alog[i].run(data)
    ce_eval.append(round(clustering_error(biclustering_pre, biclustering_ref, num_rows, num_cols),5))

print(ce_eval)
# secured_alg = ["Type1", "Type2", "Type3", "Type4", "Total"]
#
# plt.bar(secured_alg, ce_eval, color='blue')
# plt.title('Comparison of SeCCA with CCA')
# plt.xlabel('Types of SeCCA')
# plt.ylabel('CE External Evaluation Measure ')
# plt.savefig('CE_final.png')
# plt.show()

