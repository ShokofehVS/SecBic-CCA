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
        [copy_col_sum[j] / [data_size[0] for i in range(data_size[0] * data_size[1])] for j in range(len(copy_col_sum))])
    data_mean = data_sum / n_elements

    return data_mean, col_mean


def opt_row_mean(HE, cipher_data, data_size):
    # sum and mean of rows
    row_sum = sum(cipher_data)
    copy_row_sum = row_sum.copy()
    row_mean = copy_row_sum / data_size[1]

    return row_mean


def calculate_opt_msr_col_addition(HE, cipher_data, cipher_data_rows):
    data_size = cipher_data.shape
    data_rows_size = cipher_data_rows.shape

    #START of ENCRYPTION
    tol = 1e-2
    decr_round = lambda c: np.round(HE.decrypt(c), int(np.log10(1 / tol)))
    decr_cols = lambda c_cols: np.round(
        np.asarray([HE.decrypt(c) for c in c_cols]), int(np.log10(1 / tol))).T
    ciphertext = np.array([HE.encrypt(cipher_data[:, i]) for i in range(data_size[1])])
    ciphertext_rows = np.array([HE.encrypt(cipher_data_rows[:, i]) for i in range(data_rows_size[1])])

    # Sum and mean of rows for ciphertext
    row_mean = opt_row_mean(HE, ciphertext, data_size)

    # Sum and mean of cols and whole matrix for ciphertext
    data_mean, col_mean = opt_col_data_mean(HE, ciphertext, data_size)

    # Sum and mean of cols and whole matrix for ciphertext_rows
    data_row_mean, col_row_mean = opt_col_data_mean(HE, ciphertext_rows, data_rows_size)

    # Rescaling of row_mean, data_mean and col_row_mean
    HE.rescale_to_next(data_mean)
    HE.rescale_to_next(row_mean)
    for i in range(len(ciphertext_rows)):
        HE.rescale_to_next(col_row_mean[i])

    # MSR-Calculation for node addition
    cipher_residue = ciphertext_rows - col_row_mean
    for i in range(len(ciphertext_rows)):
        cipher_residue[i] = cipher_residue[i] - row_mean + data_mean

    cipher_square_residue = cipher_residue ** 2
    for i in range(len(ciphertext_rows)):
        HE.rescale_to_next(cipher_square_residue[i])
        HE.relinearize(cipher_square_residue[i])

    msr, col_msr = opt_col_data_mean(HE, cipher_square_residue, data_rows_size)

    for i in range(len(ciphertext_rows)):
        HE.rescale_to_next(col_msr[i])

    dec_col_msr = decr_cols(col_msr)[:data_rows_size[1]]

    return dec_col_msr


