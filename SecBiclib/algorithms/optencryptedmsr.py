import itertools
import math
import Pyfhel
import numpy as np
import inspect
src = inspect.getsource(Pyfhel)


class ClacOptEncMSR:
    def col_mean(self, HE, cipher_data, data_size):
        # sum and mean of columns
        col_sum = sum(cipher_data)
        copy_col_sum = col_sum.copy()
        col_mean = copy_col_sum / [data_size[0] for i in range(data_size[1])]

        # For finding residues, repeated values are added
        rotated_mean = col_mean.copy()
        for j in range(data_size[0] - 1):
            rotated_mean = col_mean + HE.rotate(rotated_mean, -data_size[1], True)

        return rotated_mean

    def row_mean(self, HE, cipher_data, data_size):
        # sum and mean of rows
        row_sum = sum(cipher_data)
        copy_row_sum = row_sum.copy()
        row_mean = copy_row_sum / [data_size[1] for i in range(data_size[0])]

        # For finding residues, repeated values are added
        plain = [1 if i % data_size[1] == 0 else 0 for i in range(data_size[0] * data_size[1])]
        row_mean = row_mean * plain
        rotated_mean = row_mean.copy()
        for j in range(data_size[1] - 1):
            rotated_mean = row_mean + HE.rotate(rotated_mean, -1, True)

        return rotated_mean

    def data_mean(self, HE, cipher_data, data_size):
        n_elements = data_size[0] * data_size[1]

        sum_data = [HE.cumul_add(cipher_data[i], True) for i in range(len(cipher_data))]
        mean = [sum_data[i] / [n_elements for j in range(data_size[0] * data_size[1])]
                for i in range(len(cipher_data))]

        return mean

    def calculate_msr(self, HE, cipher_data):
        n_rows, n_cols = 2, 3
        n_elements = n_rows * n_cols
        np.random.seed(42)  # Fixed seed for reproducibility
        data = np.random.randint(0, 5, size=(n_rows, n_cols))
        data_size = cipher_data.shape
        n_elements = data_size[0] * data_size[1]

        enc_cipher_col = [HE.encrypt(cipher_data[:, i]) for i in range(data_size[1])]
        enc_cipher_row = [HE.encrypt(cipher_data[i,:]) for i in range(data_size[0])]
        # Finding the maximum no_ciphertexts (*To be tested more*)
        no_ciphertexts = math.ceil(len(cipher_data.flatten()) / HE.get_nSlots())

        # Chunking data according to no_ciphertexts
        chunk_col = math.ceil(data_size[0] / no_ciphertexts)
        plaintext_inList = [cipher_data[j * chunk_col:(j + 1) * chunk_col, :] for j in range(no_ciphertexts)]

        # Actual size of data according to splitting into no_ciphertexts
        data_size_actual = (chunk_col, data_size[1])
        ciphertext = [HE.encrypt(plain_sub.flatten()) for plain_sub in plaintext_inList]

        cipher_row_mean = self.row_mean(HE, enc_cipher_col, data_size)
        cipher_col_mean = self.col_mean(HE, enc_cipher_row, data_size)
        cipher_data_mean = self.data_mean(HE, ciphertext, data_size_actual)

        # Rescaling for list of ciphertexts or single ciphertext
        for i in range(len(ciphertext)):
            HE.rescale_to_next(cipher_data_mean[i])
        HE.rescale_to_next(cipher_row_mean)
        HE.rescale_to_next(cipher_col_mean)

        # MSR-Calculation for list of ciphertexts or single ciphertext
        for i in range(len(ciphertext)):
            cipher_residue, cipher_square_residue, cipher_msr, cipher_row_msr, cipher_col_msr = [], [], [], [], []
            cipher_residue.append(ciphertext[i] - cipher_row_mean - cipher_col_mean + cipher_data_mean[i])
            cipher_square_residue.append(~(~cipher_residue[i] ** 2))
            HE.rescale_to_next(cipher_square_residue[i])
            cipher_row_msr.append(self.row_mean(HE, ~cipher_square_residue[i], data_size_actual))
            cipher_col_msr.append(self.col_mean(HE, ~cipher_square_residue[i], data_size_actual))
            cipher_msr.append(self.data_mean(HE, ~cipher_square_residue[i], data_size_actual))

        # For MPC Connection (decrypting results)
        list_msr = [HE.decrypt(cipher_msr[i]) for i in range(len(ciphertext))]
        decrypted_msr = [sum(msr) for msr in zip(*list_msr)][0] / no_ciphertexts
        decrypted_msr_row = HE.decrypt(cipher_row_msr)[:n_elements:data_size[1]]
        decrypted_msr_col = HE.decrypt(cipher_col_msr)[:data_size[1]]

        return decrypted_msr, decrypted_msr_row, decrypted_msr_col







