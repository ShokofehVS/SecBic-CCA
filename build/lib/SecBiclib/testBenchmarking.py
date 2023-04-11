import math
import Pyfhel
import numpy as np
import inspect
src = inspect.getsource(Pyfhel)


def shift(HE, cipher_data, by, data_size):
    """Shift the ciphertexts based on by measure"""

    shifted_data = cipher_data << by

    return shifted_data

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
    n_element = data_size[0] * data_size[1]
    for i in range(1, n_cols):
        c_row_sum += shift(HE, cipher_data, i, data_size)
        rotated_sum = c_row_sum.copy()
        for j in range(n_cols - 1):
            rotated_sum = c_row_sum + HE.rotate(rotated_sum, -(n_element+1))

    return rotated_sum

def col_mean(HE, cipher_data, data_size):
    """Mean of cols in the ciphertext"""
    N_rows = data_size[0]
    mean = _col_sum(HE, cipher_data, data_size) / [N_rows for i in range(data_size[1])]
    rotated_mean = mean + HE.rotate(mean, -data_size[1], True)

    return rotated_mean

def row_mean(HE, cipher_data, data_size):
    """Mean of rows in the ciphertext"""
    N_cols = data_size[1]
    mean = _row_sum(HE, cipher_data, data_size) / [N_cols for i in range(data_size[0] * data_size[1])]

    return mean

def data_mean(HE, row_mean, cipher_data, data_size):
    """Mean of data in the ciphertext"""
    mean = col_mean(HE, row_mean, data_size)

    n_elements = data_size[0]*data_size[1]
    for i in range(n_elements - 1):
        mean += HE.rotate(mean, -1)

    return mean


def encrypted_calculate_msr(HE, cipher_data):
    """Calculate the mean squar ed residues of the rows, of the columns and of the full data matrix
    by homomorphic encryption"""
    data_size_actual = cipher_data.shape
    ciphertext = HE.encrypt(cipher_data.flatten())

    # Mean value calculation
    cipher_row_mean = row_mean(HE, ciphertext, data_size_actual)
    cipher_col_mean = col_mean(HE, ciphertext, data_size_actual)
    cipher_data_mean = data_mean(HE, cipher_row_mean, ciphertext, data_size_actual)

    # Rescaling:
    HE.rescale_to_next(cipher_row_mean)
    HE.rescale_to_next(cipher_col_mean)
    HE.rescale_to_next(cipher_data_mean)

    # MSR-Calculation:
    cipher_residue = ciphertext - cipher_row_mean - cipher_col_mean + cipher_data_mean
    cipher_square_residue = cipher_residue ** 2
    HE.rescale_to_next(cipher_square_residue)
    cipher_row_msr = row_mean(HE, ~cipher_square_residue, data_size_actual)
    cipher_col_msr = col_mean(HE, ~cipher_square_residue, data_size_actual)
    cipher_msr = data_mean(HE, cipher_row_msr, ~cipher_square_residue, data_size_actual)

    HE.rescale_to_next(cipher_msr)
    HE.rescale_to_next(cipher_row_msr)
    HE.rescale_to_next(cipher_col_msr)

    # For test
    decrypted_msr = HE.decrypt(cipher_msr)[0]
    decrypted_msr_row = HE.decrypt(cipher_row_msr)[:data_size_actual[0]]
    decrypted_msr_col = HE.decrypt(cipher_col_msr)[:data_size_actual[1]]

    return decrypted_msr, decrypted_msr_row, decrypted_msr_col

def original_calculate_msr(data, rows, cols):
    sub_data = data[rows][:, cols]

    data_mean = np.mean(sub_data)
    row_means = np.mean(sub_data, axis=1)
    col_means = np.mean(sub_data, axis=0)

    residues = sub_data - row_means[:, np.newaxis] - col_means + data_mean
    squared_residues = residues * residues

    msr = np.mean(squared_residues)
    row_msr = np.mean(squared_residues, axis=1)
    col_msr = np.mean(squared_residues, axis=0)

    return msr, row_msr, col_msr

if __name__=="__main__":

    HE = Pyfhel.Pyfhel()
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

    num_rows, num_cols = 2, 3
    n_elements = num_rows * num_cols
    np.random.seed(42)  # Fixed seed for reproducibility
    cipher_data = np.random.randint(0, 5, size=(num_rows, num_cols))  # Generate sample data
    rows = np.ones(num_rows, dtype=bool)
    cols = np.ones(num_cols, dtype=bool)

    dec_msr, dec_msr_row, dec_msr_col = encrypted_calculate_msr(HE, cipher_data)
    msr, msr_row, msr_col = original_calculate_msr(cipher_data, rows, cols)

    print(np.allclose(msr, dec_msr))
    print(np.allclose(msr_row, dec_msr_row))
    print(np.allclose(msr_col, dec_msr_col))