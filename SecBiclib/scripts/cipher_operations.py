import numpy as np
from Pyfhel import Pyfhel

#####################################################################
# Array operations for testing and evaluation:

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

def array_col_mean(array,data_size):
    N_rows=data_size[0][0]
    real_N_cols=data_size[1][1]
    col_sum=array.copy()
    for i in range(1,N_rows):
        col_sum+=shift(array,real_N_cols*i)
    col_mean=col_sum/N_rows
    return col_mean

def array_row_mean(array,data_size):
    N_cols=data_size[0][1]
    row_sum=array.copy()
    for i in range(1,N_cols):
        row_sum+=shift(array,-i)
    row_mean=row_sum/N_cols
    return row_mean

#####################################################################
# Ciphertext operations:

def col_sum(cipher_data, data_size):
    N_rows=data_size[0][0]
    real_N_cols=data_size[1][1]
    c_col_sum=cipher_data.copy()
    for i in range(1,N_rows):
        c_col_sum+=cipher_data << real_N_cols*i
    return c_col_sum

def row_sum(cipher_data, data_size):
    N_cols=data_size[0][1]
    c_row_sum=cipher_data.copy()
    for i in range(1,N_cols):
        c_row_sum+=cipher_data << i
    return c_row_sum

def col_mean(cipher_data, data_size):
    N_rows=data_size[0][0]
    mean=col_sum(cipher_data, data_size)/[N_rows for i in range(data_size[1][0]*data_size[1][0])]
    return mean

def row_mean(cipher_data, data_size):
    N_cols=data_size[0][1]
    mean=row_sum(cipher_data, data_size)/[N_cols for i in range(data_size[1][0]*data_size[1][0])]
    return mean

def data_mean(cipher_data, data_size):
    mean=col_mean(row_mean(cipher_data, data_size), data_size)
    # col(row()) or row(col()) ? Performance ?
    return mean

#####################################################################
# Testing:

if __name__=="__main__":

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
    print("Array / 2:")
    print(data/2)

    #Make bigger array with all rows, cols needed for shifting:
    sup_data=np.array([[data[i,j] for j in range(-N_cols,N_cols)] for i in range(-N_rows,N_rows)])
    real_N_rows=2*N_rows
    real_N_cols=2*N_cols
    data_size=((N_rows, N_cols), (real_N_rows, real_N_cols))
    print("Data-Size:")
    print(data_size)
    #print("With added rows, cols:")
    #print(sup_array)
    #print("Converted to 1d:")
    flat_sup_data=sup_data.flatten()
    #print(flat_sup_array)
    c_data = HE.encryptFrac(flat_sup_data) # Encrypts the plaintext ptxt_x and returns a PyCtxt
    #c_half=c_data/[2 for i in range(real_N_rows*real_N_cols)]
    #print("Cipher-Array / 2:")
    #print(reshape(HE.decryptFrac(c_half), data_size[1]))


    c_col_sum=c_data.copy()
    c_row_sum=c_data.copy()
    n_col_sum=sup_data.copy()
    n_row_sum=sup_data.copy()
    print("Shift and sum:")
    print(c_data)

    # testing shifts:
    #c_sum=c_data.copy()
    #test_shift=c_data << 1*real_N_cols
    #test_shift=c_data << 2*real_N_cols
    #c_sum+=test_shift

    # col-wise sum,mean:
    for i in range(1,N_rows):
        c_col_sum+=c_data << real_N_cols*i
        n_col_sum+=shift(sup_data,real_N_cols*i)
    n_col_mean=n_col_sum/N_rows
    print("COLUMN-WISE SUM:")
    #result=reshape(HE.decryptFrac(c_sum),(real_N_rows,real_N_cols))
    result=reshape(HE.decryptFrac(col_sum(c_data, data_size)),(real_N_rows,real_N_cols))
    print("SHOULD BE:\n",n_col_sum,"\nRESULT:\n",result)

    print("COLUMN-WISE MEAN:")
    #result=reshape(HE.decryptFrac(c_sum),(real_N_rows,real_N_cols))
    result=reshape(HE.decryptFrac(col_mean(c_data, data_size)),(real_N_rows,real_N_cols))
    print("SHOULD BE:\n",n_col_mean,"\nRESULT:\n",result)

    # row-wise sum:
    for i in range(1,N_cols):
        c_row_sum+=c_data << i
        shifted=shift(sup_data,-i)
        n_row_sum+=shifted
    n_row_mean=n_row_sum/N_cols
    print("ROW-WISE SUM:")
    #result=reshape(HE.decryptFrac(c_sum),(real_N_rows,real_N_cols))
    result=reshape(HE.decryptFrac(row_sum(c_data, data_size)),(real_N_rows,real_N_cols))
    print("SHOULD BE:\n",n_row_sum,"\nRESULT:\n",result)
    print("ERRORS:\n",result-n_row_sum)

    print("ROW-WISE MEAN:")
    #result=reshape(HE.decryptFrac(c_sum),(real_N_rows,real_N_cols))
    result=reshape(HE.decryptFrac(row_mean(c_data, data_size)),(real_N_rows,real_N_cols))
    print("SHOULD BE:\n",n_row_mean,"\nRESULT:\n",result)
    print("ERRORS:\n",result-n_row_mean)
    #print([result[i] for i in range(len(result))])

    print("DATA MEAN:")
    result=reshape(HE.decryptFrac(data_mean(c_data, data_size)),(real_N_rows,real_N_cols))
    print("SHOULD BE:\n",array_row_mean(array_col_mean(sup_data, data_size), data_size),"\nRESULT:\n",result)

    print(data)

#    print("\nOLD ROW MEAN\n",n_row_mean,"\nNEW ROW MEAN\n",array_row_mean(sup_data,data_size))
#    print("\nOLD COL MEAN\n",n_col_mean,"\nNEW COL MEAN\n",array_col_mean(sup_data,data_size))

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
