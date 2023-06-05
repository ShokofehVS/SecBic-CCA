import Pyfhel
import numpy as np
import inspect
src = inspect.getsource(Pyfhel)


def calculate_opt_msr(HE, cipher_data):

    data_size = cipher_data.shape
    n_elements = data_size[0] * data_size[1]
    tol = 1e-2
    decr_round = lambda c: np.round(HE.decrypt(c), int(np.log10(1 / tol)))
    decr_cols = lambda c_cols: np.round(
        np.asarray([HE.decrypt(c) for c in c_cols]), int(np.log10(1 / tol))).T

    # encrypt data --> One column per ciphertext
    c_data = np.array([HE.encrypt(cipher_data[:, i]) for i in range(data_size[1])])

    # sum for each row --> sum all ciphertexts
    c_row_sum = sum(c_data)

    # sum for each column --> sum all elements on each ciphertext
    c_col_sum = np.array([HE.cumul_add(c, in_new_ctxt=True) for c in c_data])

    # sum for all elements --> sum all row/col sums
    c_all_sum = c_col_sum.sum()

    # Square residues (non-normalized)
    res = c_data * n_elements - c_row_sum * data_size[0] - c_col_sum * data_size[1] + c_all_sum
    sq_res = res ** 2
    for i in range(len(sq_res)):
        HE.rescale_to_next(sq_res[i])
        HE.relinearize(sq_res[i])

    c_msr_row_p1 = sum(sq_res) / data_size[1]
    c_msr_row_p2 = c_msr_row_p1 / n_elements
    c_msr_row = c_msr_row_p2 / n_elements
    HE.rescale_to_next(c_msr_row)
    dec_row_msr = decr_round(c_msr_row)[:data_size[0]]

    c_msr_sum = np.array([HE.cumul_add(c, in_new_ctxt=True) for c in sq_res])
    c_msr_col_p1 = c_msr_sum / [data_size[0]]
    c_msr_col_p2 = c_msr_col_p1 / [n_elements]
    c_msr_col = c_msr_col_p2 / [n_elements]
    for i in range(len(c_msr_col)):
        HE.rescale_to_next(c_msr_col[i])
    dec_col_msr = (decr_cols(c_msr_col)[0][:data_size[1]]).T

    c_msr_p1 = c_msr_sum.sum() / (n_elements)
    c_msr_p2 = c_msr_p1 / (n_elements)
    c_msr = c_msr_p2 / (n_elements)
    HE.rescale_to_next(c_msr)
    dec_msr = decr_round(c_msr)[0]

    return dec_msr, dec_row_msr, dec_col_msr




