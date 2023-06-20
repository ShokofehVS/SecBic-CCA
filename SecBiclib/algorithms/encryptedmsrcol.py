import math

import Pyfhel
import numpy as np
import inspect
from SecBiclib.algorithms import optencryptedmsrcol
src = inspect.getsource(Pyfhel)


def enlarge(array):
    """Make larger array with all rows, cols needed for shifting"""
    n_rows, n_cols = np.shape(array)
    sub_data = np.array([[array[i, j] for j in range(-n_cols, n_cols)] for i in range(-n_rows, n_rows)])
    real_n_rows = 2 * n_rows
    real_n_cols = 2 * n_cols
    data_size = ((n_rows, n_cols), (real_n_rows, real_n_cols))

    return sub_data, data_size


def reshape(array, shape):
    """Change the shape of the array"""
    sub_array = array[:(shape[0] * shape[1])]

    return sub_array.reshape(shape)


def array_shift(array, by):
    """Shift the array based on by measure"""
    if not isinstance(by, int):
        raise TypeError('Shift distance has to be integer but was given as: ' + str(type(by)))
    elif (abs(by) >= array.size):
        raise ValueError('Shift distance has to be smaller than size of array')
    if by == 0:
        return array
    else:
        a_shape = array.shape
        flat = array.flatten()
        flat_shifted = flat.copy()
        flat_shifted[:by] = flat[-by:]
        flat_shifted[by:] = flat[:-by]
        shifted = flat_shifted.reshape(a_shape)

        return shifted


def shift(HE, cipher_data, by, data_size):
    """Shift the ciphertexts based on by measure"""
    shifted_cipher = cipher_data.copy()
    shifted_data = cipher_data << by

    return shifted_data


def _list_shift(HE, cipher_list, by, sub_len):
    """Shift the list of ciphertexts based on by measure"""
    if by == 0:
        return cipher_list
    elif by > sub_len:
        raise ValueError("Value of shift distance Argument 'by' cannot be larger than the size of sub-ciphertexts")

    for i in range(len(cipher_list)):
        # Shift of the single ciphertext
        cipher_list[i] = cipher_list[i] << 1

    return cipher_list


def _cipher_ones(HE, start, end, length):
    """Create ciphertext of ones"""
    return HE.encrypt(_ones(start, end, length))


def _ones(start, end, length):
    if not start:
        if end:
            return [1 if i in range(end) else 0 for i in range(length)]
        else:
            raise ValueError("Start and end arguments cannot both be None")
    elif not end:
        return [1 if i in range(start, length) else 0 for i in range(length)]
    else:
        raise NotImplementedError("Not yet implemented returning ones array between both start and end value")


def get_scale(cipher_data):
    """Get scale of ciphertext"""
    scales = []
    for ciphertext in cipher_data:
        repr_str = repr(ciphertext)
        where = repr_str.find('scale_bits=')
        scales.append(int(repr_str[where + 11:where + 14].replace(",", "")))

    return scales


def _col_sum(HE, cipher_data, data_size):
    """Sum of columns in the ciphertext"""
    n_rows = data_size[0]
    n_cols = data_size[1]
    c_col_sum = cipher_data.copy()

    for i in range(1, n_rows):
        c_col_sum += shift(HE, cipher_data, n_cols * i, data_size)

    return c_col_sum


def _row_sum(HE, cipher_data, data_size):
    """Sum of rows in the ciphertext"""
    n_cols = data_size[1]
    c_row_sum = cipher_data.copy()

    for i in range(1, n_cols):
        c_row_sum += cipher_data << i

    return c_row_sum


def col_mean(HE, cipher_data, data_size):
    """Mean of cols in the ciphertext"""
    N_rows = data_size[0]
    mean = _col_sum(HE, cipher_data, data_size) / [N_rows for i in range(data_size[1])]

    # For finding residues, repeated values are added
    rotated_mean = mean.copy()
    for j in range(N_rows - 1):
        rotated_mean = mean + HE.rotate(rotated_mean, -data_size[1], True)

    return rotated_mean


def row_mean(HE, cipher_data, data_size):
    """Mean of rows in the ciphertext"""
    N_cols = data_size[1]
    mean = _row_sum(HE, cipher_data, data_size) / [N_cols for i in range(data_size[0] * data_size[1])]

    # For finding residues, repeated values are added
    plain = [1 if i%N_cols == 0 else 0 for i in range(data_size[0] * data_size[1])]
    mean = mean * plain
    rotated_mean = mean.copy()
    for j in range(N_cols - 1):
        rotated_mean = mean + HE.rotate(rotated_mean, -1, True)

    return rotated_mean


def data_mean(HE, cipher_data, data_size):
    """Mean of data in the ciphertext"""
    n_elements = data_size[0] * data_size[1]
    sum_data = HE.cumul_add(cipher_data, True)
    mean = sum_data / [n_elements for i in range(data_size[0] * data_size[1])]

    return mean


def calculate_msr_col_addition(HE, cipher_data, cipher_data_rows):
    """Calculate the mean squared residues of the columns for the node addition step homomorphically"""
    # Get the size of data, and data_cols
    data_size = cipher_data.shape
    data_rows_size = cipher_data_rows.shape
    n_elements = data_size[0] * data_size[1]
    no_ciphertexts = math.ceil(len(cipher_data.flatten()) / HE.get_nSlots())
    tol = 1e-2

    # Check if storing an input data in lists is needed
    # 1. first conditions when both data and data_col's sizes are above number of ciphertext's slot
    if (len(cipher_data.flatten()) > (HE.get_nSlots()) or len(cipher_data_rows.flatten()) > (HE.get_nSlots())):
        dec_col_msr = optencryptedmsrcol.calculate_opt_msr_col_addition(HE, cipher_data, cipher_data_rows)

        return dec_col_msr
    else:
        data_size_actual = data_size
        data_rows_size_actual = data_rows_size
        ciphertext = HE.encrypt(cipher_data.flatten())
        ciphertext_rows = HE.encrypt(cipher_data_rows.flatten())

        # Mean value calculation
        cipher_row_mean = row_mean(HE, ciphertext, data_size_actual)
        cipher_col_mean = col_mean(HE, ciphertext_rows, data_rows_size_actual)
        cipher_data_mean = data_mean(HE, ciphertext, data_size_actual)

        HE.rescale_to_next(cipher_row_mean)
        HE.rescale_to_next(cipher_col_mean)
        HE.rescale_to_next(cipher_data_mean)

        # Residue and Squared Residue calculation
        cipher_col_residue = ciphertext_rows - cipher_row_mean - cipher_col_mean + cipher_data_mean
        cipher_col_square_residue = cipher_col_residue ** 2
        HE.rescale_to_next(cipher_col_square_residue)
        cipher_col_msr = col_mean(HE, ~cipher_col_square_residue, data_rows_size_actual)
        HE.rescale_to_next(cipher_col_msr)

        # Decrypt the results
        decrypted_col_msr = HE.decrypt(cipher_col_msr)[:data_size[1]]

        return decrypted_col_msr