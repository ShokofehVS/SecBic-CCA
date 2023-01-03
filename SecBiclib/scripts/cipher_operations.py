import numpy as np
from Pyfhel import Pyfhel

def reshape(array, shape):
    sub_array=array[:(shape[0]*shape[1])]
    return sub_array.reshape(shape)
def shift(array,by):
    if not isinstance(by,int):
        raise TypeError('Shift distance has to be integer but was given as: '+str(type(by)))
    elif (abs(by) >= array.size):
        raise ValueError('Shift distance has to be smaller than size of array')
    if by==0:
        return array
    else:
        a_shape=array.shape
        flat=array.flatten()
        flat_shifted=flat.copy()
        flat_shifted[:by]=flat[-by:]
        flat_shifted[by:]=flat[:-by]
        shifted=flat_shifted.reshape(a_shape)
        return shifted

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

data = np.array([[0.1, 0.2, -0.3, 0.4],
                  [-0.5, 0.6,0.7,-0.8],
                  [0.9,-1.,1.1,1.2]], dtype=np.float64)    # Always use type float64!

N_rows,N_cols=np.shape(data)

print("Data-Array:")
print(data)

#Make bigger array with all rows, cols needed for shifting:
sup_data=np.array([[data[i,j] for j in range(-N_cols,N_cols)] for i in range(-N_rows,N_rows)])
real_N_rows=2*N_rows
real_N_cols=2*N_cols
#print("With added rows, cols:")
#print(sup_array)
#print("Converted to 1d:")
flat_sup_data=sup_data.flatten()
#print(flat_sup_array)
c_data = HE.encryptFrac(flat_sup_data) # Encrypts the plaintext ptxt_x and returns a PyCtxt

c_col_sum=c_data.copy()
c_row_sum=c_data.copy()
col_sum=sup_data.copy()
row_sum=sup_data.copy()
print("Shift and sum:")
print(c_data)

# testing shifts:
#c_sum=c_data.copy()
#test_shift=c_data << 1*real_N_cols
#test_shift=c_data << 2*real_N_cols
#c_sum+=test_shift

# col-wise sum:
for i in range(1,N_rows):
    c_col_sum+=c_data << real_N_cols*i
    col_sum+=shift(sup_data,real_N_cols*i)
print("COLUMN-WISE SUM:")
#result=reshape(HE.decryptFrac(c_sum),(real_N_rows,real_N_cols))
result=reshape(HE.decryptFrac(c_col_sum),(real_N_rows,real_N_cols))
print("SHOULD BE:\n",col_sum,"\nRESULT:\n",result)

# row-wise sum:
print("ROW-WISE SUM:")
for i in range(1,N_cols):
    c_row_sum+=c_data << i
    shifted=shift(sup_data,-i)
    row_sum+=shifted
#result=reshape(HE.decryptFrac(c_sum),(real_N_rows,real_N_cols))
result=reshape(HE.decryptFrac(c_row_sum),(real_N_rows,real_N_cols))
print("SHOULD BE:\n",row_sum,"\nRESULT:\n",result)
print("ERRORS:\n",result-row_sum)

#print([result[i] for i in range(len(result))])

##  1. Mean
#c_mean = (ctxt_x + ctxt_y) / 2
##  2. MSE
#c_mse_1 = ~((ctxt_x - c_mean)**2)
#c_mse_2 = (~(ctxt_y - c_mean)**2)
#c_mse = (c_mse_1 + c_mse_2)/ 3
##  3. Cumulative sum
#c_mse += (c_mse << 1)
#c_mse += (c_mse << 2)  # element 0 contains the result
#print("->\tMean: ", c_mean)
#print("->\tMSE_1: ", c_mse_1)
#print("->\tMSE_2: ", c_mse_2)
#print("->\tMSE: ", c_mse)
#