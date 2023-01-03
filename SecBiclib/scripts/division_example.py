import numpy as np
from Pyfhel import Pyfhel

HE = Pyfhel()
ckks_params = {
    'scheme': 'CKKS',
    'n': 2**14,
    'scale': 2**30,
    'qi_sizes': [60, 30, 30, 30, 60]
}
HE.contextGen(**ckks_params)  # Generate context for bfv scheme
HE.keyGen()             # Key Generation: generates a pair of public/secret keys
HE.rotateKeyGen()

# Some example data:
data = np.array([[0.1, 0.2, -0.3, 0.4],
                  [-0.5, 0.6,0.7,-0.8],
                  [0.9,-1.,1.1,1.2]], dtype=np.float64)    # Always use type float64!
# Flatten the data:
flat_data=data.flatten()
data_size=len(flat_data)

# Check whats happening on plain numpy array:
print("Flattened Data Array:")
print(flat_data)
print()
print("Divided by 2:")
print(flat_data/2)
print()
# Encrypt data:
c_data = HE.encryptFrac(flat_data)

# Do division by different methods:
c_half_byint=c_data.copy()/2                        # copy needed because otherwise c_data is affected for further operations?!
c_half_bylist=c_data/[2 for i in range(data_size)]  # division by ist of twos instead of single integer

# Check results:
print("Cipher-Array / 2:")
print(HE.decryptFrac(c_half_byint)[:data_size])     # printing only relevant entries of array (size of the initial data)
print()
print("Cipher-Array / [2, ..., 2]:")
print(HE.decryptFrac(c_half_bylist)[:data_size])
