import Pyfhel
import numpy as np
import inspect
src = inspect.getsource(Pyfhel)

def opt_col_data_mean(HE, ciphertext, data_size):
    # sum and mean of columns and data
    n_elements = data_size[0] * data_size[1]
    col_sum = np.array([HE.cumul_add(c, in_new_ctxt=True) for c in ciphertext])
    copy_col_sum = col_sum.copy()
    data_sum = copy_col_sum.sum()
    col_mean = np.array(
        [copy_col_sum[j] / [data_size[0] for i in range(data_size[1])] for j in range(len(copy_col_sum))])
    data_mean = data_sum / n_elements

    return data_mean, col_mean

def opt_row_mean(HE, cipher_data, data_size):
    # sum and mean of rows
    row_sum = sum(cipher_data)
    copy_row_sum = row_sum.copy()
    row_mean = copy_row_sum / data_size[1]

    return row_mean

def calculate_opt_msr_row_addition(HE, cipher_data, cipher_data_cols):

    data_size = cipher_data.shape
    data_cols_size = cipher_data_cols.shape

    # START of ENCRYPTION
    tol = 1e-2
    decr_round = lambda c: np.round(HE.decrypt(c), int(np.log10(1 / tol)))

    ciphertext = np.array([HE.encrypt(cipher_data[:, i]) for i in range(data_size[1])])
    ciphertext_cols = np.array([HE.encrypt(cipher_data_cols[:, i]) for i in range(data_cols_size[1])])

    # Sum and mean of rows for ciphertext
    row_mean = opt_row_mean(HE, ciphertext_cols, data_cols_size)

    # Sum and mean of cols and whole matrix for ciphertext
    data_mean, col_mean = opt_col_data_mean(HE, ciphertext, data_size)

    # Rescaling of row_mean, data_mean and col_row_mean
    HE.rescale_to_next(data_mean)
    HE.rescale_to_next(row_mean)
    for i in range(len(ciphertext)):
        HE.rescale_to_next(col_mean[i])

    # MSR-Calculation for node addition
    cipher_residue = ciphertext_cols - row_mean - col_mean + data_mean
    cipher_square_residue = cipher_residue ** 2
    for i in range(len(ciphertext)):
        HE.rescale_to_next(cipher_square_residue[i])
        HE.relinearize(cipher_square_residue[i])

    cipher_residue_inverse = -ciphertext_cols + row_mean - col_mean + data_mean
    cipher_square_residue_inverse = cipher_residue_inverse ** 2
    for i in range(len(ciphertext)):
        HE.rescale_to_next(cipher_square_residue_inverse[i])
        HE.relinearize(cipher_square_residue_inverse[i])

    row_msr = opt_row_mean(HE, cipher_square_residue, data_cols_size)
    row_inverse_msr = opt_row_mean(HE, cipher_square_residue_inverse, data_cols_size)

    dec_row_msr = decr_round(row_msr)[:data_cols_size[0]]
    dec_row_inverse_msr = decr_round(row_inverse_msr)[:data_cols_size[0]]

    return dec_row_msr, dec_row_inverse_msr



