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


def calculate_opt_msr(HE, cipher_data):
    data_size = cipher_data.shape

    # START of ENCRYPTION
    tol = 1e-2
    decr_round = lambda c: np.round(HE.decrypt(c), int(np.log10(1 / tol)))
    decr_cols = lambda c_cols: np.round(
        np.asarray([HE.decrypt(c) for c in c_cols]), int(np.log10(1 / tol))).T
    ciphertext = np.array([HE.encrypt(cipher_data[:, i]) for i in range(data_size[1])])

    # Sum and mean of rows for ciphertext
    row_mean = opt_row_mean(HE, ciphertext, data_size)

    # Sum and mean of cols and whole matrix for ciphertext
    data_mean, col_mean = opt_col_data_mean(HE, ciphertext, data_size)

    # Rescaling of row_mean, data_mean and col_row_mean
    HE.rescale_to_next(data_mean)
    HE.rescale_to_next(row_mean)
    for i in range(len(ciphertext)):
        HE.rescale_to_next(col_mean[i])

    # MSR-Calculation for node deletion
    cipher_residue = ciphertext - col_mean
    for i in range(len(ciphertext)):
        cipher_residue[i] = cipher_residue[i] - row_mean + data_mean
    cipher_square_residue = cipher_residue ** 2
    for i in range(len(ciphertext)):
        HE.rescale_to_next(cipher_square_residue[i])
        HE.relinearize(cipher_square_residue[i])

    row_msr = opt_row_mean(HE, cipher_square_residue, data_size)
    msr, col_msr = opt_col_data_mean(HE, cipher_square_residue, data_size)

    HE.rescale_to_next(msr)
    HE.rescale_to_next(row_msr)
    for i in range(len(ciphertext)):
        HE.rescale_to_next(col_msr[i])

    dec_msr = decr_round(msr)[0]
    dec_row_msr = decr_round(row_msr)[:data_size[0]]
    dec_col_msr = decr_cols(col_msr)[:data_size[1]]

    return dec_msr, dec_row_msr, dec_col_msr





