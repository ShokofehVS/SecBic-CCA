import math
import Pyfhel
import numpy as np
import inspect
src = inspect.getsource(Pyfhel)


class ClacEncMSR:

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
        if isinstance(cipher_data, list):
            length = data_size[0] * data_size[1]
            shifted_data = self._list_shift(HE, cipher_data, by, length)

        else:
            shifted_data = cipher_data << by

        return shifted_data

    def _list_shift(self, HE, cipher_list, by, sub_len):
        """Shift the list of ciphertexts based on by measure"""
        if by == 0:
            return cipher_list
        elif by > sub_len:
            raise ValueError("Value of shift distance Argument 'by' cannot be larger than the size of sub-ciphertexts")
        # Make a copy of the first ciphertext
        copy_first = cipher_list[0].copy()
        c_ones_array = self._cipher_ones(HE, sub_len - by, None, sub_len)

        for i in range(len(cipher_list)-1):
            # Shift of the single ciphertext
            cipher_list[i] = ~cipher_list[i] << by
            # Force the shifted single ciphertext entries
            remaining = ~(cipher_list[i] * self._cipher_ones(HE, None, sub_len - by, sub_len))
            # Make a copy of the next single ciphertext
            from_outside = cipher_list[i + 1].copy()
            # Shift this copy so that the values which will appear in
            from_outside = ~((~from_outside >> (sub_len - by)) * c_ones_array)
            # Add the values from next ciphertext to the first
            cipher_list[i] = remaining + from_outside
        # Do the same as above manually for the last ciphertext in the list,
        # with expansion values coming from the first ciphertext in the list:
        cipher_list[len(cipher_list) - 1] = ~cipher_list[len(cipher_list) - 1] << by
        cipher_list[len(cipher_list) - 1] = \
            ~(cipher_list[len(cipher_list) - 1] * self._cipher_ones(HE, None, sub_len - by, sub_len)) \
            + ~((~copy_first >> (sub_len - by)) * self._cipher_ones(HE, sub_len - by, None, sub_len))

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
        for i in range(1, n_rows):
            if isinstance(cipher_data, list):
                for j in range(len(cipher_data)):
                    shifted = self.shift(HE, cipher_data, n_cols * i, data_size)
                    c_col_sum[i] += ~shifted[i]
            else:
                c_col_sum += self.shift(HE, cipher_data, n_cols * i, data_size)

        return c_col_sum

    def _row_sum(self, HE, cipher_data, data_size):
        """Sum of rows in the ciphertext"""
        n_cols = data_size[1]
        c_row_sum = cipher_data.copy()
        n_element = data_size[0] * data_size[1]
        for i in range(1, n_cols):
            if isinstance(cipher_data, list):
                shifted = self.shift(HE, cipher_data, i, data_size)
                for j in range(len(cipher_data)):
                    c_row_sum[j] += ~shifted[j]
            else:
                c_row_sum += cipher_data << i

        return c_row_sum

    def col_mean(self, HE, cipher_data, data_size):
        """Mean of cols in the ciphertext"""
        N_rows = data_size[0]
        if isinstance(cipher_data, list):
            c_col_sum = self._col_sum(HE, cipher_data, data_size)
            mean = [c_col_sum[j] / [N_rows for i in range(data_size[0] * data_size[1])] for j in
                    range(len(cipher_data))]
        else:
            mean = self._col_sum(HE, cipher_data, data_size) / [N_rows for i in range(data_size[1])]
            rotated_mean = mean.copy()
            for j in range(N_rows - 1):
                rotated_mean = mean + HE.rotate(rotated_mean, -data_size[1], True)

        return rotated_mean

    def row_mean(self, HE, cipher_data, data_size):
        """Mean of rows in the ciphertext"""
        N_cols = data_size[1]
        if isinstance(cipher_data, list):
            c_row_sum = self._row_sum(HE, cipher_data, data_size)
            mean = [c_row_sum[j] / [N_cols for i in range(data_size[0] * data_size[1])] for j in
                    range(len(cipher_data))]
        else:
            mean = self._row_sum(HE, cipher_data, data_size) / [N_cols for i in range(data_size[0] * data_size[1])]
            plain = [1 if i%N_cols == 0 else 0 for i in range(data_size[0] * data_size[1])]
            mean = mean * plain
            rotated_mean = mean.copy()
            for j in range(N_cols - 1):
                rotated_mean = mean + HE.rotate(rotated_mean, -1, True)

        return rotated_mean

    def data_mean(self, HE, row_mean, cipher_data, data_size):
        """Mean of data in the ciphertext"""
        if isinstance(cipher_data, list):
            mean = self.col_mean(HE, row_mean, data_size)
        else:
            n_elements = data_size[0] * data_size[1]
            sum_data = HE.cumul_add(cipher_data, True)
            mean = sum_data / [n_elements for i in range(data_size[0] * data_size[1])]

        return mean

    def calculate_msr(self, HE, cipher_data, no_ciphertexts):
        """Calculate the mean square ed residues of the rows, of the columns and of the full data matrix
        by homomorphic encryption"""
        data_size = cipher_data.shape
        n_elements = data_size[0] * data_size[1]

        if len(cipher_data.flatten()) > (HE.get_nSlots()):
            print("List Ciphertexts")
            chunk_col = math.ceil(data_size[1] / no_ciphertexts)
            plaintext_inList = [cipher_data[i:i + 1, j * chunk_col:(j + 1) * chunk_col]
                                for j in range(no_ciphertexts) for i in range(data_size[0])]
            enlarged_plaintext, data_sizes = zip(*[self.enlarge(plain_sub) for plain_sub in plaintext_inList])
            data_size_actual = data_sizes[0]
            ciphertext = [HE.encrypt(plain_sub.flatten()) for plain_sub in plaintext_inList]

        else:
            print("Single ciphertext")
            data_size_actual = data_size
            ciphertext = HE.encrypt(cipher_data.flatten())

        # Mean value calculation
        cipher_row_mean = self.row_mean(HE, ciphertext, data_size_actual)
        cipher_col_mean = self.col_mean(HE, ciphertext, data_size_actual)
        cipher_data_mean = self.data_mean(HE, cipher_row_mean, ciphertext, data_size_actual)

        # Rescaling:
        if isinstance(ciphertext, list):
            print("Rescaling list")
            print("before", self.get_scale(ciphertext))
            for i in range(len(ciphertext)):
                HE.rescale_to_next(cipher_row_mean[i])
                HE.rescale_to_next(cipher_col_mean[i])
                HE.rescale_to_next(cipher_data_mean[i])
                HE.rescale_to_next(ciphertext[i])
            print("after", self.get_scale(ciphertext))

        else:
            print("Rescaling single")
            HE.rescale_to_next(cipher_row_mean)
            HE.rescale_to_next(cipher_col_mean)
            HE.rescale_to_next(cipher_data_mean)

        # MSR-Calculation:
        if isinstance(ciphertext, list):
            cipher_residue, cipher_square_residue, cipher_msr, cipher_row_msr, cipher_col_msr = [], [], [], [], []
            for i in range(len(ciphertext)):
                cipher_residue.append(ciphertext[i] - cipher_row_mean[i] - cipher_col_mean[i] + cipher_data_mean[i])
                cipher_square_residue.append(~(~cipher_residue[i] ** 2))

                HE.rescale_to_next(cipher_square_residue[i])
                cipher_row_msr.append(self.row_mean(HE, ~cipher_square_residue[i], data_size_actual))
                cipher_col_msr.append(self.col_mean(HE, ~cipher_square_residue[i], data_size_actual))
                cipher_msr.append(self.data_mean(HE, ~cipher_square_residue[i], ciphertext, data_size_actual))

        else:
            cipher_residue = ciphertext - cipher_row_mean - cipher_col_mean + cipher_data_mean
            cipher_square_residue = cipher_residue ** 2
            HE.rescale_to_next(cipher_square_residue)
            cipher_row_msr = self.row_mean(HE, ~cipher_square_residue, data_size_actual)
            cipher_col_msr = self.col_mean(HE, ~cipher_square_residue, data_size_actual)
            cipher_msr = self.data_mean(HE, cipher_row_msr, ~cipher_square_residue, data_size_actual)

            HE.rescale_to_next(cipher_msr)
            HE.rescale_to_next(cipher_row_msr)
            HE.rescale_to_next(cipher_col_msr)

        # For test
        if isinstance(ciphertext, list):
            decrypted_msr = [HE.decrypt(cipher_msr[i]) for i in range(len(ciphertext))][0][0]
            decrypted_msr_row = [HE.decrypt(cipher_row_msr[i]) for i in range(len(ciphertext))][0][0]
            decrypted_msr_col = [HE.decrypt(cipher_col_msr[i]) for i in range(len(ciphertext))][0][0]

        else:
            decrypted_msr = HE.decrypt(cipher_msr)[0]
            decrypted_msr_row = HE.decrypt(cipher_row_msr)[:n_elements:data_size[1]]
            decrypted_msr_col = HE.decrypt(cipher_col_msr)[:data_size[1]]

        return decrypted_msr, decrypted_msr_row, decrypted_msr_col


