import math

import Pyfhel
import numpy as np
import inspect

src = inspect.getsource(Pyfhel)


class ClacEncMSRow:

    def enlarge(self, array):
        """Make larger array with all rows, cols needed for shifting"""
        n_rows, n_cols = np.shape(array)
        sub_data = np.array([[array[i, j] for j in range(-n_cols, n_cols)] for i in range(-n_rows, n_rows)])
        real_n_rows = 2 * n_rows
        real_n_cols = 2 * n_cols
        data_size = ((n_rows, n_cols), (real_n_rows, real_n_cols))

        return sub_data, data_size

    def reshape(self, array, shape):
        """Change the shape of the array"""
        sub_array = array[:(shape[0] * shape[1])]

        return sub_array.reshape(shape)

    def array_shift(self, array, by):
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

    def shift(self, HE, cipher_data, by, data_size):
        """Shift the ciphertexts based on by measure"""
        shifted_cipher = cipher_data.copy()
        if isinstance(cipher_data, list):
            length = data_size[0] * data_size[1]
            shifted_data = self._list_shift(HE, shifted_cipher, by, length)

        else:
            shifted_data = cipher_data << by

        return shifted_data

    def _list_shift(self, HE, cipher_list, by, sub_len):
        """Shift the list of ciphertexts based on by measure"""
        if by == 0:
            return cipher_list
        elif by > sub_len:
            raise ValueError("Value of shift distance Argument 'by' cannot be larger than the size of sub-ciphertexts")

        for i in range(len(cipher_list)):
            # Shift of the single ciphertext
            cipher_list[i] = cipher_list[i] << 1

        return cipher_list

    def _cipher_ones(self, HE, start, end, length):
        """Create ciphertext of ones"""
        return HE.encrypt(self._ones(start, end, length))

    def _ones(self, start, end, length):
        if not start:
            if end:
                return [1 if i in range(end) else 0 for i in range(length)]
            else:
                raise ValueError("Start and end arguments cannot both be None")
        elif not end:
            return [1 if i in range(start, length) else 0 for i in range(length)]
        else:
            raise NotImplementedError("Not yet implemented returning ones array between both start and end value")

    def get_scale(self, cipher_data):
        """Get scale of ciphertext"""

        # Check if storing an input data in lists is needed
        if not isinstance(cipher_data, list):
            cipher_data = [cipher_data]
        scales = []
        for ciphertext in cipher_data:
            repr_str = repr(ciphertext)
            where = repr_str.find('scale_bits=')
            scales.append(int(repr_str[where + 11:where + 14].replace(",", "")))

        return scales

    def _col_sum(self, HE, cipher_data, data_size):
        """Sum of columns in the ciphertext"""
        n_rows = data_size[0]
        n_cols = data_size[1]
        c_col_sum = cipher_data.copy()

        # Check if storing an input data in lists is needed
        if isinstance(cipher_data, list):
            rotation = [HE.rotate(cipher_data[j], n_cols, True) for j in range(len(cipher_data))]
            for i in range(1, n_rows):
                if i == 1:
                    c_col_sum = [c_col_sum[j] + rotation[j] for j in range(len(cipher_data))]

                else:
                    c_col_sum = [c_col_sum[j] + HE.rotate(rotation[j], n_cols) for j in range(len(cipher_data))]

        else:
            for i in range(1, n_rows):
                c_col_sum += self.shift(HE, cipher_data, n_cols * i, data_size)

        return c_col_sum

    def _row_sum(self, HE, cipher_data, data_size):
        """Sum of rows in the ciphertext"""
        n_cols = data_size[1]
        c_row_sum = cipher_data.copy()

        # Check if storing an input data in lists is needed
        if isinstance(cipher_data, list):
            rotation = [HE.rotate(cipher_data[j], 1, True) for j in range(len(cipher_data))]
            for i in range(1, n_cols):
                if i == 1:
                    c_row_sum = [c_row_sum[j] + rotation[j] for j in range(len(cipher_data))]

                else:
                    c_row_sum = [c_row_sum[j] + HE.rotate(rotation[j], 1) for j in range(len(cipher_data))]

        else:
            for i in range(1, n_cols):
                c_row_sum += cipher_data << i

        return c_row_sum

    def col_mean(self, HE, cipher_data, data_size):
        """Mean of cols in the ciphertext"""
        N_rows = data_size[0]

        # Check if storing an input data in lists is needed
        if isinstance(cipher_data, list):
            c_col_sum = self._col_sum(HE, cipher_data, data_size)
            mean = [c_col_sum[j] / [N_rows for i in range(data_size[1])] for j in
                         range(len(cipher_data))]

            # For finding residues, repeated values are added
            rotated_mean = mean.copy()
            for j in range(N_rows - 1):
                rotated_mean = [mean[i] + HE.rotate(rotated_mean[i], -data_size[1], True) for i in range(len(cipher_data))]

        else:
            mean = self._col_sum(HE, cipher_data, data_size) / [N_rows for i in range(data_size[1])]

            # For finding residues, repeated values are added
            rotated_mean = mean.copy()
            for j in range(N_rows - 1):
                rotated_mean = mean + HE.rotate(rotated_mean, -data_size[1], True)

        return rotated_mean

    def row_mean(self, HE, cipher_data, data_size):
        """Mean of rows in the ciphertext"""
        N_cols = data_size[1]

        # Check if storing an input data in lists is needed
        if isinstance(cipher_data, list):
            c_row_sum = self._row_sum(HE, cipher_data, data_size)
            mean = [c_row_sum[j] / [N_cols for i in range(data_size[0] * data_size[1])] for j in
                            range(len(cipher_data))]

            # For finding residues, repeated values are added
            plain = [1 if i % N_cols == 0 else 0 for i in range(data_size[0] * data_size[1])]
            mean = [mean[i] * plain for i in range(len(cipher_data))]
            rotated_mean = mean.copy()
            for j in range(N_cols - 1):
                rotated_mean = [mean[i] + HE.rotate(rotated_mean[i], -1, True) for i in range(len(cipher_data))]

        else:
            mean = self._row_sum(HE, cipher_data, data_size) / [N_cols for i in range(data_size[0] * data_size[1])]

            # For finding residues, repeated values are added
            plain = [1 if i%N_cols == 0 else 0 for i in range(data_size[0] * data_size[1])]
            mean = mean * plain
            rotated_mean = mean.copy()
            for j in range(N_cols - 1):
                rotated_mean = mean + HE.rotate(rotated_mean, -1, True)

        return rotated_mean

    def data_mean(self, HE, cipher_data, data_size):
        """Mean of data in the ciphertext"""
        n_elements = data_size[0] * data_size[1]

        # Check if storing an input data in lists is needed
        if isinstance(cipher_data, list):
            sum_data = [HE.cumul_add(cipher_data[i], True) for i in range(len(cipher_data))]
            mean = [sum_data[i] / [n_elements for j in range(data_size[0] * data_size[1])]
                    for i in range(len(cipher_data))]

        else:
            sum_data = HE.cumul_add(cipher_data, True)
            mean = sum_data / [n_elements for i in range(data_size[0] * data_size[1])]

        return mean

    def calculate_msr_row_addition(self, HE, cipher_data, cipher_data_cols, no_ciphertexts):
        """Calculate the mean squared residues of the rows and of the inverse of the rows
        for the node addition step homomorphically"""
        print("calculate_msr_row_addition")

        # Get the size of data, and data_cols
        data_size = cipher_data.shape
        data_cols_size = cipher_data_cols.shape
        n_elements = data_size[0] * data_size[1]

        # Find the size of each chunk according to the no_ciphertexts
        chunk_col = math.ceil(data_size[0] / no_ciphertexts)
        chunk_col4rows = math.ceil(data_cols_size[0] / no_ciphertexts)

        # Check if storing an input data in lists is needed
        # 1. first conditions when both data and data_col's sizes are above number of ciphertext's slot
        if (len(cipher_data.flatten()) < (HE.get_nSlots()) and len(cipher_data_cols.flatten()) < (HE.get_nSlots())):
            print("List Ciphertexts")
            # Chunk the data and data_col according to no_ciphertexts and chunk sizes
            plaintext_inList = [cipher_data[j * chunk_col:(j + 1) * chunk_col, ]
                                for j in range(no_ciphertexts)]
            plaintext_cols_inList = [cipher_data_cols[j * chunk_col4rows:(j + 1) * chunk_col4rows, ]
                                     for j in range(no_ciphertexts)]

            # Check if input data or data_col is bigger(to synchronize the sizes)
            if len(cipher_data) > len(cipher_data_cols):
                for i in range(len(plaintext_cols_inList)):
                    plaintext_cols_inList[i] = np.append \
                            (plaintext_cols_inList[i],
                             np.zeros((plaintext_inList[i].shape[0] - plaintext_cols_inList[i].shape[0], data_size[1])),
                             axis=0)

            if len(cipher_data_cols) > len(cipher_data):
                for i in range(len(plaintext_inList)):
                    plaintext_inList[i] = np.append \
                            (plaintext_inList[i],
                             np.zeros((plaintext_cols_inList[i].shape[0] - plaintext_inList[i].shape[0], data_size[1])),
                             axis=0)

            # Add zeros to the end of data to have balance sizes in rows
            for i in range(len(plaintext_inList)):
                if plaintext_inList[i].shape[0] < chunk_col:
                    plaintext_inList[i] = np.append \
                        (plaintext_inList[i], np.zeros((chunk_col - plaintext_inList[i].shape[0], data_size[1])),
                         axis=0)
                else:
                    pass

            # Actual size of data according to splitting into no_ciphertexts
            data_size_actual = (chunk_col, data_size[1])
            ciphertext = [HE.encrypt(plain_sub.flatten()) for plain_sub in plaintext_inList]

            # Add zeros to the end of data_col to have balance sizes in rows
            for i in range(len(plaintext_cols_inList)):
                if plaintext_cols_inList[i].shape[0] < chunk_col4rows:
                    plaintext_cols_inList[i] = np.append \
                        (plaintext_cols_inList[i],
                         np.zeros((chunk_col4rows - plaintext_cols_inList[i].shape[0], data_cols_size[1])),
                         axis=0)
                else:
                    pass

            # Actual size of data_col according to splitting into no_ciphertexts
            data_cols_size_actual = (chunk_col, data_cols_size[1])
            ciphertext_cols = [HE.encrypt(plain_sub_rows.flatten()) for plain_sub_rows in plaintext_cols_inList]

        # 2. second conditions when data's size is above number of ciphertext's slot (inner computations are similar)
        elif len(cipher_data.flatten()) > (HE.get_nSlots()):
            print("List Ciphertexts")
            plaintext_inList = [cipher_data[j * chunk_col:(j + 1) * chunk_col, ]
                                for j in range(no_ciphertexts)]

            for i in range(len(plaintext_inList)):
                if plaintext_inList[i].shape[0] < chunk_col:
                    plaintext_inList[i] = np.append \
                        (plaintext_inList[i], np.zeros((chunk_col - plaintext_inList[i].shape[0], data_size[1])),
                         axis=0)
                else:
                    pass

            data_size_actual = (chunk_col, data_size[1])
            ciphertext = [HE.encrypt(plain_sub.flatten()) for plain_sub in plaintext_inList]

            data_cols_size_actual = data_cols_size
            ciphertext_cols = HE.encrypt(cipher_data_cols.flatten())

        # 3. third conditions when data_col's size is above number of ciphertext's slot (inner computations are similar)
        elif len(cipher_data_cols.flatten()) > (HE.get_nSlots()):
            print("List Ciphertexts")
            plaintext_cols_inList = [cipher_data_cols[j * chunk_col4rows:(j + 1) * chunk_col4rows, ]
                                     for j in range(no_ciphertexts)]

            for i in range(len(plaintext_cols_inList)):
                if plaintext_cols_inList[i].shape[0] < chunk_col4rows:
                    plaintext_cols_inList[i] = np.append \
                        (plaintext_cols_inList[i],
                         np.zeros((chunk_col4rows - plaintext_cols_inList[i].shape[0], data_cols_size[1])),
                         axis=0)
                else:
                    pass

            data_cols_size_actual = (chunk_col, data_cols_size[1])
            ciphertext_cols = [HE.encrypt(plain_sub_cols.flatten()) for plain_sub_cols in plaintext_cols_inList]

            data_size_actual = data_size
            ciphertext = HE.encrypt(cipher_data.flatten())

        # 4. fourth conditions when non of them has size above number of ciphertext's slot
        # (inner computations are similar)
        else:
            print("Single ciphertext")
            data_size_actual = data_size
            ciphertext = HE.encrypt(cipher_data.flatten())
            data_cols_size_actual = data_cols_size
            ciphertext_cols = HE.encrypt(cipher_data_cols.flatten())

        # Mean value calculation
        cipher_row_mean = self.row_mean(HE, ciphertext_cols, data_cols_size)
        cipher_col_mean = self.col_mean(HE, ciphertext, data_size_actual)
        cipher_data_mean = self.data_mean(HE, ciphertext, data_size_actual)

        # Check which of data or/and data_col are in list to rescale means
        if isinstance(ciphertext, list) and isinstance(ciphertext_cols, list):
            for i in range(len(ciphertext)):
                HE.rescale_to_next(cipher_row_mean[i])
                HE.rescale_to_next(cipher_col_mean[i])
                HE.rescale_to_next(cipher_data_mean[i])

        elif isinstance(ciphertext, list):
            for i in range(len(ciphertext)):
                HE.rescale_to_next(cipher_row_mean[i])
                HE.rescale_to_next(cipher_data_mean[i])
            HE.rescale_to_next(cipher_col_mean)

        elif isinstance(ciphertext_cols, list):
            for i in range(len(ciphertext)):
                HE.rescale_to_next(cipher_col_mean[i])
            HE.rescale_to_next(cipher_row_mean)
            HE.rescale_to_next(cipher_data_mean)

        else:
            HE.rescale_to_next(cipher_row_mean)
            HE.rescale_to_next(cipher_col_mean)
            HE.rescale_to_next(cipher_data_mean)

        # MSR-Calculation:
        # 1. first conditions when both data and data_col's sizes are above number of ciphertext's slot
        if isinstance(ciphertext, list) and isinstance(ciphertext_cols, list):
            cipher_row_residue, cipher_row_square_residue, cipher_row_msr = [], [], []
            cipher_row_inverse_residue, cipher_row_inverse_square_residue, cipher_row_inverse_msr = [], [], []
            for i in range(len(ciphertext_cols)):
                cipher_row_residue.append(ciphertext_cols[i] - cipher_row_mean[i] - cipher_col_mean[i]
                                          + cipher_data_mean[i])
                cipher_row_square_residue.append(cipher_row_residue[i] ** 2)
                HE.rescale_to_next(cipher_row_square_residue[i])
                cipher_row_msr.append(self.row_mean(HE, ~cipher_row_square_residue[i], data_cols_size))

                cipher_row_inverse_residue.append(-ciphertext_cols[i] + cipher_row_mean[i] - cipher_col_mean[i]
                                                  + cipher_data_mean[i])
                cipher_row_inverse_square_residue.append(cipher_row_inverse_residue[i] ** 2)
                HE.rescale_to_next(cipher_row_inverse_square_residue[i])
                cipher_row_inverse_msr.append(
                    self.row_mean(HE, ~cipher_row_inverse_square_residue[i], data_cols_size))

        # 2. second conditions when data's size is above number of ciphertext's slot (inner computations are similar)
        elif isinstance(ciphertext, list):
            cipher_row_residue, cipher_row_square_residue, cipher_row_msr = [], [], []
            cipher_row_inverse_residue, cipher_row_inverse_square_residue, cipher_row_inverse_msr = [], [], []
            for i in range(len(ciphertext)):
                cipher_row_residue.append(ciphertext_cols - cipher_row_mean - cipher_col_mean[i]
                                          + cipher_data_mean[i])
                cipher_row_square_residue.append(cipher_row_residue[i] ** 2)
                HE.rescale_to_next(cipher_row_square_residue[i])
                cipher_row_msr.append(self.row_mean(HE, ~cipher_row_square_residue[i], data_cols_size))

                cipher_row_inverse_residue.append(-ciphertext_cols + cipher_row_mean - cipher_col_mean[i]
                                                  + cipher_data_mean[i])
                cipher_row_inverse_square_residue.append(cipher_row_inverse_residue[i] ** 2)
                HE.rescale_to_next(cipher_row_inverse_square_residue[i])
                cipher_row_inverse_msr.append(
                    self.row_mean(HE, ~cipher_row_inverse_square_residue[i], data_cols_size))

        # 3. third conditions when data_col's size is above number of ciphertext's slot (inner computations are similar)
        elif isinstance(ciphertext_cols, list):
            cipher_row_residue, cipher_row_square_residue, cipher_row_msr = [], [], []
            cipher_row_inverse_residue, cipher_row_inverse_square_residue, cipher_row_inverse_msr = [], [], []
            for i in range(len(ciphertext)):
                cipher_row_residue.append(ciphertext_cols[i] - cipher_row_mean[i] - cipher_col_mean
                                          + cipher_data_mean)
                cipher_row_square_residue.append(cipher_row_residue[i] ** 2)
                HE.rescale_to_next(cipher_row_square_residue[i])
                cipher_row_msr.append(self.row_mean(HE, ~cipher_row_square_residue[i], data_cols_size))

                cipher_row_inverse_residue.append(-ciphertext_cols[i] + cipher_row_mean[i] - cipher_col_mean
                                                  + cipher_data_mean)
                cipher_row_inverse_square_residue.append(cipher_row_inverse_residue[i] ** 2)
                HE.rescale_to_next(cipher_row_inverse_square_residue[i])
                cipher_row_inverse_msr.append(
                    self.row_mean(HE, ~cipher_row_inverse_square_residue[i], data_cols_size))

        # 4. fourth conditions when non of them has size above number of ciphertext's slot
        # (inner computations are similar)
        else:
            cipher_row_residue = ciphertext_cols - cipher_row_mean - cipher_col_mean + cipher_data_mean
            cipher_row_square_residue = cipher_row_residue ** 2
            HE.rescale_to_next(cipher_row_square_residue)
            cipher_row_msr = self.row_mean(HE, ~cipher_row_square_residue, data_cols_size)
            HE.rescale_to_next(cipher_row_msr)

            cipher_row_inverse_residue = -ciphertext_cols + cipher_row_mean - cipher_col_mean + cipher_data_mean
            cipher_row_inverse_square_residue = cipher_row_inverse_residue ** 2
            HE.rescale_to_next(cipher_row_inverse_square_residue)
            cipher_row_inverse_msr = self.row_mean(HE, ~cipher_row_inverse_square_residue, data_cols_size)
            HE.rescale_to_next(cipher_row_inverse_msr)

        # For MPC Connection (decrypting results)
        # 1. first conditions when both data and/or data_col's sizes are above number of ciphertext's slot
        if isinstance(ciphertext, list) or isinstance(ciphertext_cols, list):
            list_msr_row = [
                HE.decrypt(cipher_row_msr[i])[:data_cols_size[0] * data_cols_size[1]:data_cols_size[1]]
                for i in range(len(ciphertext_cols))]
            decrypted_row_msr = np.concatenate(([list_msr_row[i]
                                                 for i in range(len(ciphertext_cols))]))[:data_cols_size[0]]

            list_msr_row_inverse = [
                HE.decrypt(cipher_row_inverse_msr[i])[:data_size_actual[0] * data_size_actual[1]:data_size_actual[1]]
                for i in range(len(ciphertext))]
            decrypted_row_inverse_msr = np.concatenate(([list_msr_row_inverse[i]
                                                         for i in range(len(ciphertext))]))[:data_size[0]]

        # 4. fourth conditions when non of them has size above number of ciphertext's slot
        # (inner computations are similar)
        else:
            decrypted_row_msr = HE.decrypt(cipher_row_msr)[:n_elements:data_size[1]]
            decrypted_row_inverse_msr = HE.decrypt(cipher_row_inverse_msr)[:n_elements:data_size[1]]

        return decrypted_row_msr, decrypted_row_inverse_msr
