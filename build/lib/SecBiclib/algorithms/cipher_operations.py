import Pyfhel
import numpy as np
from numpy import random as rd
#from Pyfhel import Pyfhel
from SecBiclib.algorithms import cca
import inspect
src = inspect.getsource(Pyfhel)
print(src)
############################################################################################
# Array operations for testing and evaluation:


def enlarge(array):
    N_rows,N_cols=np.shape(array)
    #Make bigger array with all rows, cols needed for shifting:
    sup_data=np.array([[array[i,j] for j in range(-N_cols,N_cols)] for i in range(-N_rows,N_rows)])
    real_N_rows=2*N_rows
    real_N_cols=2*N_cols
    data_size=((N_rows, N_cols), (real_N_rows, real_N_cols))
    return sup_data,data_size

def reshape(array, shape):
    sub_array=array[:(shape[0]*shape[1])]
    return sub_array.reshape(shape)

def array_shift(array,by):
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
        col_sum+=array_shift(array,real_N_cols*i)
    col_mean=col_sum/N_rows
    return col_mean

def array_row_mean(array,data_size):
    N_cols=data_size[0][1]
    row_sum=array.copy()
    for i in range(1,N_cols):
        row_sum+=array_shift(array,-i)
    row_mean=row_sum/N_cols
    return row_mean

############################################################################################
# Ciphertext operations:

def _col_sum(HE,cipher_data, data_size):
    #length=data_size[1][0]*data_size[1][1]
    N_rows=data_size[0][0]
    real_N_cols=data_size[1][1]
    c_col_sum=cipher_data.copy()
    rescale=False
    for i in range(1,N_rows):
        if isinstance(cipher_data,list):
            for j in range(len(cipher_data)):
                shifted=shift(HE,cipher_data,int(real_N_cols/len(cipher_data))*i,data_size)
            # ^ after that do shift by mod(real_N_cols,len(cipher_data)) ?!
            for k in range(len(cipher_data)):
                c_col_sum[k]+=~shifted[k]
            rescale=True
        else:
            c_col_sum+=shift(HE,cipher_data,real_N_cols*i,data_size)
    if rescale:
        for i in range(len(cipher_data)):
            pass
            #HE.rescale_to_next(c_col_sum[i])
    return c_col_sum

def _row_sum(HE,cipher_data, data_size):
    #length=data_size[1][0]*data_size[1][1]
    N_cols=data_size[0][1]
    c_row_sum=cipher_data.copy()
    for i in range(1,N_cols):
        if isinstance(cipher_data,list):
            shifted=shift(HE,cipher_data,i,data_size)
            for i in range(len(cipher_data)):
                c_row_sum[i]+=~shifted[i]
        else:
            c_row_sum+=shift(HE,cipher_data,i,data_size)
    return c_row_sum

def col_mean(HE,cipher_data, data_size):
    N_rows=data_size[0][0]
    if isinstance(cipher_data,list):
        c_col_sum=_col_sum(HE,cipher_data, data_size)
        mean=[c_col_sum[j]/[N_rows for i in range(data_size[1][0]*data_size[1][0])] for j in range(len(cipher_data))]
    else:
        mean=_col_sum(HE,cipher_data, data_size)/[N_rows for i in range(data_size[1][0]*data_size[1][0])]
    return mean

def row_mean(HE,cipher_data, data_size):
    N_cols=data_size[0][1]
    if isinstance(cipher_data,list):
        c_row_sum=_row_sum(HE,cipher_data, data_size)
        mean=[c_row_sum[j]/[N_cols for i in range(data_size[1][0]*data_size[1][0])] for j in range(len(cipher_data))]
    else:
        mean=_row_sum(HE,cipher_data, data_size)/[N_cols for i in range(data_size[1][0]*data_size[1][0])]
    return mean

def data_mean(HE,cipher_data, data_size):
    mean=col_mean(HE,row_mean(HE,cipher_data, data_size), data_size)
    # col(row()) or row(col()) ? Performance ?
    return mean

def calculate_msr(HE,cipher_data, data_size, refill=False):
    cipher_row_mean=row_mean(HE,cipher_data, data_size)
    cipher_col_mean=col_mean(HE,cipher_data, data_size)
    cipher_data_mean=data_mean(HE,cipher_data, data_size)
    # Rescaling:
    if isinstance(cipher_data, list):
        print("Rescaling list")
        print("before",get_scale(cipher_data))
        for i in range(len(cipher_data)):
            HE.rescale_to_next(cipher_row_mean[i])
            HE.rescale_to_next(cipher_col_mean[i])
            HE.rescale_to_next(cipher_data_mean[i])
            HE.rescale_to_next(cipher_data[i])
        print("after",get_scale(cipher_data))

    else:
        print("Rescaling single")
        HE.rescale_to_next(cipher_row_mean)
        HE.rescale_to_next(cipher_col_mean)
        HE.rescale_to_next(cipher_data_mean)
    # MSR-Calculation:
    if isinstance(cipher_data, list):
        cipher_residue,cipher_square_residue,cipher_msr=[],[],[]
        for i in range(len(cipher_data)):
            cipher_residue.append(cipher_data[i]-cipher_row_mean[i]-cipher_col_mean[i]+cipher_data_mean[i])
            cipher_square_residue.append(cipher_residue[i]**2)
            HE.rescale_to_next(cipher_square_residue[i])
            cipher_msr.append(data_mean(HE,~cipher_square_residue[i], data_size))
            #HE.rescale_to_next(cipher_msr[i])
    else:
        cipher_residue=cipher_data-cipher_row_mean-cipher_col_mean+cipher_data_mean
        cipher_square_residue=cipher_residue**2
        HE.rescale_to_next(cipher_square_residue)
        cipher_msr=data_mean(HE,~cipher_square_residue, data_size)
        HE.rescale_to_next(cipher_msr)
    if refill:
        raise NotImplementedError("Not yet implemented: Refilling array with MSR value")
#    Trying to fill the array with reasonable data again:
#    if refill:
#        cipher_msr=cipher_msr/[1 if i==1 else 2 for i in range(data_size[0][1]*data_size[0][0])]
#        cipher_msr_shift=cipher_msr.copy()
#        for i in range(data_size[0][1]*data_size[0][0]):    # OR data_size[1] for whole data?
#            cipher_msr_shift=cipher_msr_shift>>1
#            cipher_msr+=cipher_msr_shift
    #print(cipher_msr)
    return cipher_msr

def _ones(start, end, length):
    if not start:
        if end:
            return [1 if i in range(end) else 0 for i in range(length)]
        else:
            raise ValueError("Start and end arguments cannot both be None")
    elif not end:
        return [1 if i in range(start,length) else 0 for i in range(length)]
    else:
        raise NotImplementedError("Not yet implemented returning ones array between both start and end value")

def _cipher_ones(HE, start, end, length):
    return HE.encrypt(_ones(start,end,length))

def _list_shift(HE,cipher_list, by, sub_len):
    # Check if 'by' is realistic:
    if by==0:
        return cipher_list
    elif by > sub_len:
        raise ValueError("Value of shift distance Argument 'by' cannot be larger than the size of sub-ciphertexts")

    copy_first=cipher_list[0].copy()                # Make a copy of the first ciphertext

    c_ones_array=_cipher_ones(HE,sub_len-by,None,sub_len)

    for i in range(len(cipher_list)-1):             # For all ciphertexts in list except last one

        cipher_list[i]=~cipher_list[i]<<by           # Do the shift of the single ciphertext

        remaining=cipher_list[i]*_cipher_ones(HE,None,sub_len-by,sub_len)    # Force the shifted single ciphertext entries

        from_outside=cipher_list[i+1].copy()                                # Make a copy of the next single ciphertext

        from_outside=(~from_outside>>(sub_len-by))*c_ones_array # Shift this copy so that the
                                                                                          # values which will appear in
        cipher_list[i]=remaining+from_outside       # Add the values from next ciphertext
                                                    # to the first
    # Do the same as above manually for the last ciphertext in the list, with expansion values coming from the first ciphertext in the list:
    cipher_list[len(cipher_list)-1]=~cipher_list[len(cipher_list)-1]<<by           # Do the shift of the single ciphertext
    cipher_list[len(cipher_list)-1]=cipher_list[len(cipher_list)-1]*_cipher_ones(HE,None,sub_len-by,sub_len)+(~copy_first>>(sub_len-by))*_cipher_ones(HE,sub_len-by,None,sub_len)
    return cipher_list  # DONE

def shift(HE,cipher_data,by,data_size):

    if isinstance(cipher_data,list):
        length=data_size[0][0]*data_size[0][1]
        shifted_data=_list_shift(HE,cipher_data,by,length)

    else:
        shifted_data=cipher_data << by

    return shifted_data

def get_scale(cipher_data):
    if not isinstance(cipher_data,list):
        cipher_data=[cipher_data]
    scales=[]
    for ciphertext in cipher_data:
        repr_str=repr(ciphertext)
        where=repr_str.find('scale_bits=')
        scales.append(int(repr_str[where+11:where+14].replace(",","")))
    return scales
############################################################################################
# Testing:

if __name__=="__main__":

    HE = Pyfhel.Pyfhel()
    HE2 =Pyfhel.Pyfhel()
    print(Pyfhel.__file__)
    ckks_params = {
        'scheme': 'CKKS',
        'n': 2**15,
        'scale': 2**30,
        'qi_sizes': [60]+15*[30]+[60]
    }
    HE.contextGen(**ckks_params)  # Generate context for bfv scheme
    HE.keyGen()             # Key Generation: generates a pair of public/secret keys
    HE.rotateKeyGen()
    HE.relinKeyGen()
    HE2.contextGen(**ckks_params)  # Generate context for bfv scheme
    HE2.keyGen()             # Key Generation: generates a pair of public/secret keys
    HE2.rotateKeyGen()
    HE2.relinKeyGen()
#    data = np.array([[0.1, 0.2, -0.3, 0.4],
#                      [-0.5, 0.6,0.7,-0.8],
#                      [0.9,-1.,1.1,1.2]], dtype=np.float64)    # Always use type float64!
#
#    N_rows,N_cols=np.shape(data)
#
#    print("Data-Array:")
#    print(data)
#    print("Array / 2:")
#    print(data/2)

    #Make bigger array with all rows, cols needed for shifting:
#    sup_data=np.array([[data[i,j] for j in range(-N_cols,N_cols)] for i in range(-N_rows,N_rows)])
#    real_N_rows=2*N_rows
#    real_N_cols=2*N_cols
#    data_size=((N_rows, N_cols), (real_N_rows, real_N_cols))
#    print("Data-Size:")
#    print(data_size)
#    print("With added rows, cols:")
#    print(sup_data)
#    #print("Converted to 1d:")
#    flat_sup_data=sup_data.flatten()
#    #print(flat_sup_array)
#    c_data = HE.encryptFrac(flat_sup_data) # Encrypts the plaintext ptxt_x and returns a PyCtxt
#    #c_half=c_data/[2 for i in range(real_N_rows*real_N_cols)]
#    #print("Cipher-Array / 2:")
#    #print(reshape(HE.decryptFrac(c_half), data_size[1]))
#
#
#    c_col_sum=c_data.copy()
#    c_row_sum=c_data.copy()
#    n_col_sum=sup_data.copy()
#    n_row_sum=sup_data.copy()
##    print("Shift and sum:")
##    print(c_data)
#
#    # testing shifts:
#    #c_sum=c_data.copy()
#    #test_shift=c_data << 1*real_N_cols
#    #test_shift=c_data << 2*real_N_cols
#    #c_sum+=test_shift
#
#    # col-wise sum,mean:
#    for i in range(1,N_rows):
#        c_col_sum+=c_data << real_N_cols*i
#        n_col_sum+=array_shift(sup_data,real_N_cols*i)
#    n_col_mean=n_col_sum/N_rows
#    print("##################################################################\nCOLUMN-WISE SUM:")
#    #result=reshape(HE.decryptFrac(c_sum),(real_N_rows,real_N_cols))
#    result=reshape(HE.decryptFrac(_col_sum(c_data, data_size)),(real_N_rows,real_N_cols))
#    print("SHOULD BE:\n",n_col_sum,"\nRESULT:\n",result)
#
#    print("##################################################################\nCOLUMN-WISE MEAN:")
#    #result=reshape(HE.decryptFrac(c_sum),(real_N_rows,real_N_cols))
#    result=reshape(HE.decryptFrac(col_mean(c_data, data_size)),(real_N_rows,real_N_cols))
#    print("SHOULD BE:\n",n_col_mean,"\nRESULT:\n",result)
#
#    # row-wise sum:
#    for i in range(1,N_cols):
#        c_row_sum+=c_data << i
#        shifted=array_shift(sup_data,-i)
#        n_row_sum+=shifted
#    n_row_mean=n_row_sum/N_cols
#
#    # msr:
#    inner_msr=sup_data-n_row_mean-n_col_mean+array_row_mean(array_col_mean(sup_data, data_size), data_size)
#    print(array_col_mean(inner_msr**2,data_size))
#    print(array_row_mean(inner_msr**2,data_size))
#    ref_cca=cca.ChengChurchAlgorithm()
##    #print(sup_data)
#    max_row,max_col=data_size[0]
#    ref_msr=ref_cca._calculate_msr(sup_data,list(range(max_row)), list(range(max_col)))
#
#    print("##################################################################\nROW-WISE SUM:")
#    #result=reshape(HE.decryptFrac(c_sum),(real_N_rows,real_N_cols))
#    result=reshape(HE.decryptFrac(_row_sum(c_data, data_size)),(real_N_rows,real_N_cols))
#    print("SHOULD BE:\n",n_row_sum,"\nRESULT:\n",result)
#    print("ERRORS:\n",result-n_row_sum)
#
#    print("##################################################################\nROW-WISE MEAN:")
#    #result=reshape(HE.decryptFrac(c_sum),(real_N_rows,real_N_cols))
#    result=reshape(HE.decryptFrac(row_mean(c_data, data_size)),(real_N_rows,real_N_cols))
#    print("SHOULD BE:\n",n_row_mean,"\nRESULT:\n",result)
#    print("ERRORS:\n",result-n_row_mean)
#    #print([result[i] for i in range(len(result))])
#
#    print("##################################################################\nDATA MEAN:")
#    result=reshape(HE.decryptFrac(data_mean(c_data, data_size)),(real_N_rows,real_N_cols))
#    print("SHOULD BE:\n",array_row_mean(array_col_mean(sup_data, data_size), data_size),"\nRESULT:\n",result)

#    print("##################################################################\nMSR:")
#    #result=reshape(HE.decryptFrac(c_sum),(real_N_rows,real_N_cols))
#    result=reshape(HE.decryptFrac(calculate_msr(HE,c_data, data_size)),(real_N_rows,real_N_cols))
#    print("SHOULD BE:\n",ref_msr[0],"\nRESULT:\n",result)
#    print("ERRORS:\n",result-ref_msr[0])

#
#
#    #TESTING ONES:
#    print("##################################################################\nONES:")
#    print("len8 ones{2:}:",_ones(2,None,8))
#    print("len10 ones{:5}:",_ones(None,5,10))
#
    #TESTING LIST SHIFT:
#    no_ciphertexts=3
#    length=4
#    by_shift=1
#    plain_list=[np.array([rd.random() for i in range(length)]) for j in range(no_ciphertexts)]
#    list_data=[HE.encrypt(plain_sub) for plain_sub in plain_list]
#    shifted_by_sub=[sub_data<<1 for sub_data in list_data]
#    shifted_in_total=_list_shift(HE,list_data, by_shift, length)
#
#    out_plain=[["%0.2f" % plain_sub[i] for i in range(length)] for plain_sub in plain_list]
#    out_shift_sub=[["%0.2f" % HE.decrypt(shifted_sub)[i] for i in range(length)] for shifted_sub in shifted_by_sub]
#    print(shifted_in_total)
#    out_shift_total=[["%0.2f" % HE.decrypt(shifted_sub)[i] for i in range(length)] for shifted_sub in shifted_in_total]
#
#    print()
#    print("##################################################################\nLIST SHIFT:")
#    print("Example random data:               ",out_plain)
#    #print(shifted_by_sub)
#    print("Shifting of sub_arrays only:       ",out_shift_sub)
#    print("Shifting with boundary correction: ",out_shift_total)
#
    #TESTING METHODS WITH LISTS:
    print("##################################################################\nMULTIPLE CIPHERTEXTS:")
    no_ciphertexts=2
    size=(2,3)
    by_shift=1
    plain_list=[reshape(np.array([rd.random() for i in range(size[0]*size[1])]),size) for j in range(no_ciphertexts)]
    print(plain_list)
    sup_plain_list,data_sizes=zip(*[enlarge(plain_sub) for plain_sub in plain_list])
    data_size=data_sizes[0]
    #list_data=[HE.encrypt(plain_sub.flatten()) for plain_sub in sup_plain_list]
    sup_list_data=[HE.encrypt(plain_sub.flatten()) for plain_sub in sup_plain_list]
    sup_data=sup_list_data[0].copy()
    #print(calculate_msr(HE,sup_list_data,data_size))
    print("calculate MSR now")
    print([HE.decrypt(cypher) for cypher in calculate_msr(HE,sup_list_data,data_size)])

