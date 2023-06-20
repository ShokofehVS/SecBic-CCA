# SecBic-CCA

**SecBic-CCA**: **Sec**ured **Bic**lusterings - **C**heng and **C**hurch **A**lgorithm (Phase 1): privacy-preserving gene expression data analysis by biclustering algorithm -- Cheng and Church algorithm -- over gene expression data (i.e., yeast cell cycle) with Homomorphic Encryption operations such as sum, or matrix multiplication in Python under the MIT license.

We apply [Pyfhel](https://pyfhel.readthedocs.io/en/latest/) as a python wrapper for the Microsoft SEAL library and [biclustlib](https://github.com/padilha/biclustlib), a library for biclustering algorithms.   

## Installation
First you need to ensure that all packages have been installed.
+ See `requirements.txt`
+ numpy>=1.23.1
+ setuptools>=65.5.0
+ pandas>=1.5.0
+ scikit-learn>=1.1.1
+ Pyfhel>=3.3.1
+ matplotlib>=3.5.2
+ scipy>=1.9.0
+ munkres>=1.1.4

You can clone this repository:

	   > git clone https://github.com/ShokofehVS/SecBic-CCA.git

If you miss something you can simply type:

	   > pip install -r requirements.txt

If you have all dependencies installed:

	   > pip3 install .

To install Pyfhel, on Linux,`gcc6` for Python (`3.5+`) should be installed. (more information regarding [installation of Pyfhel ](https://github.com/ibarrond/Pyfhel))

	   > apt install gcc 

## Biclustering Algorithm
Biclustering or simultaneous clustering of both genes and conditions as a new paradigm was introduced by [Cheng and Church's Algorithm (CCA)](https://www.researchgate.net/profile/George_Church/publication/2329589_Biclustering_of_Expression_Data/links/550c04030cf2063799394f5e.pdf). The concept of bicluster refers to a subset of
genes and a subset of conditions with a high similarity score, which measures the coherence of the genes and conditions in the bicluster. It also returns the list of biclusters for the given data set. 

## Gene Expression Data Set
Our input data is *yeast Saccharomyces cerevisiae cell cycle* taken from [Tavazoie et al. (1999)](https://pubmed.ncbi.nlm.nih.gov/10391217/) which was used in the orginal study by [Cheng and Church](https://www.researchgate.net/profile/George_Church/publication/2329589_Biclustering_of_Expression_Data/links/550c04030cf2063799394f5e.pdf);

## External Evaluation Measure
To measure the similarity of encrypted biclusters with non-encrypted version, we use Clustering Error (CE) as an external evaluation measure that was proposed by [Patrikainen and Meila (2006)](http://ieeexplore.ieee.org/abstract/document/1637417/);

## Project Contents
- `SecBiclib/` software library contains algorithms, datasets, evaluation and sample scripts 
  1. `algorithms/` original Cheng and Church algorithm and its encrypted counterpart
  2. `datasets/` real (yeast cell cycle) and synthetic data generation 
  3. `evaluation/` external evaluation measures including CE
  4. `scripts` small programs to showcase functionalities of SecBic library

## Example of Cheng and Church Algorithm (CCA)

To run the sample implementation of Cheng and Church algorithm:

	   > python3 SecBiclib/scripts/cheng_church_yeast.py

```python

import time
from SecBiclib.algorithms import ChengChurchAlgorithm
from SecBiclib.datasets import load_yeast_tavazoie
import numpy as np

m0 = time.perf_counter()

# load yeast data used in the original Cheng and Church's paper
data = load_yeast_tavazoie().values

# missing value imputation suggested by Cheng and Church
missing = np.where(data < 0.0)
data[missing] = np.random.randint(low=0, high=800, size=len(missing[0]))

# creating an instance of the ChengChurchAlgorithm class and running with the parameters
cca = ChengChurchAlgorithm(num_biclusters=5, msr_threshold=996.0, multiple_node_deletion_threshold=1.2)
biclustering = cca.run(data)
print(biclustering)

m1 = time.perf_counter()
print("Time Performance in Original Algorithm: ", round(m1 - m0, 5), "Seconds")
```

## Example of Secured Cheng and Church Algorithm (SeCCA)

To run the sample implementation of Secured Cheng and Church algorithm:

	   > python3 SecBiclib/scripts/secured_cheng_church_yeast.py

```python
import time
from SecBiclib.algorithms import SecuredChengChurchAlgorithm
from SecBiclib.datasets import load_yeast_tavazoie
import numpy as np

m0 = time.perf_counter()

# load yeast data used in the original Cheng and Church's paper
data = load_yeast_tavazoie().values

# missing value imputation suggested by Cheng and Church
missing = np.where(data < 0.0)
data[missing] = np.random.randint(low=0, high=800, size=len(missing[0]))

# creating an instance of the SecuredChengChurchAlgorithm class and running with the parameters
secca = SecuredChengChurchAlgorithm(num_biclusters=5, msr_threshold=996.0, multiple_node_deletion_threshold=1.2)
biclustering = secca.run(data)
print(biclustering)

m1 = time.perf_counter()
print("Time Performance in Calculating Homomorphically: ", round(m1 - m0, 5), "Seconds")
```

## Example of CE Evaluation 

To run the sample implementation of external evaluation measure (i.e., CE) on original and encrypted one:

	   > python3 SecBiclib/scripts/evaluation.py


```python
from SecBiclib.algorithms import ChengChurchAlgorithm
from SecBiclib.algorithms import SecuredChengChurchAlgorithm
from SecBiclib.evaluation import clustering_error
from SecBiclib.datasets import load_yeast_tavazoie

# load yeast data used in the original Cheng and Church's paper
data = load_yeast_tavazoie().values
num_rows, num_cols = data.shape


# creating an instance of the ChengChurchAlgorithm, SecuredChengChurchAlgorithm classes
# and running with the parameters of the original study
cca = ChengChurchAlgorithm(num_biclusters=5, msr_threshold=996.0, multiple_node_deletion_threshold=1.2)
secca = SecuredChengChurchAlgorithm(num_biclusters=5, msr_threshold=966.0, multiple_node_deletion_threshold=1.2)
biclustering_ref = cca.run(data)
biclustering_pre = secca.run(data)

# creating an instance of the clustering error class
# and running with reference and predicted algorithms
ce_eval = round(clustering_error(biclustering_pre, biclustering_ref, num_rows, num_cols),5)

print(ce_eval)
```
