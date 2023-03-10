"""
    SecBic-CCA: A Python library of privacy-preserving biclustering algorithm (Cheng and Church) with Homomorphic Encryption

    Copyright (C) 2023  Shokofeh VahidianSadegh

    This file is part of SecBic-CCA.

"""

from ._base import BaseBiclusteringAlgorithm
from ..models import Bicluster, Biclustering
from sklearn.utils.validation import check_array
from Pyfhel import Pyfhel, PyCtxt
import numpy as np
from SecBiclib.algorithms import encryptedmsr, encryptedmsrcol, encryptedmsrow


class SecuredChengChurchAlgorithm(BaseBiclusteringAlgorithm):
    """Secured Cheng and Church's Algorithm

     Secured Cheng and Church's Algorithm searches for maximal submatrices with a Mean Squared Residue value below a pre-defined threshold
        by Homomorphic Encryption operations

    Parameters
    ----------
    num_biclusters : int, default: 100
        Number of biclusters to be found.

    msr_threshold : float or str, default: 300
        Maximum mean squared residue accepted (delta parameter in the original paper).

    multiple_node_deletion_threshold : float, default: 1.2
        Scaling factor to remove multiple rows or columns (alpha parameter in the original paper).

    data_min_cols : int, default: 100
        Minimum number of dataset columns required to perform multiple column deletion.
    """

    def __init__(self, num_biclusters=100, msr_threshold=300, multiple_node_deletion_threshold=1.2,
                 data_min_cols=100, no_ciphertexts=2):
        self.num_biclusters = num_biclusters
        self.msr_threshold = msr_threshold
        self.multiple_node_deletion_threshold = multiple_node_deletion_threshold
        self.data_min_cols = data_min_cols
        self.no_ciphertexts = no_ciphertexts

    def run(self, data):
        """Compute biclustering.

        Parameters
        ----------
        data : numpy.ndarray
        """
        # Creating encrypted msr object
        enc_msr = encryptedmsr.ClacEncMSR()

        # Creating encrypted msr_cols object in node addition
        enc_msr_col = encryptedmsrcol.ClacEncMSRCol()

        # Creating encrypted msr_rows object in node addition
        enc_msr_row = encryptedmsrow.ClacEncMSRow()

        # Creating empty Pyfhel object
        HE = Pyfhel()
        ckks_params = {
            'scheme': 'CKKS',
            'n': 2 ** 15,
            'scale': 2 ** 30,
            'qi_sizes': [60] + 15 * [30] + [60]
        }
        HE.contextGen(**ckks_params)  # Generate context for ckks scheme
        HE.keyGen()  # Key Generation: generates a pair of public/secret keys
        HE.rotateKeyGen()  # Rotation values in the vector
        HE.relinKeyGen()  # Relinearization key generation

        data = check_array(data, dtype=np.double, copy=True)
        self._validate_parameters()

        num_rows, num_cols = data.shape
        min_value = np.min(data)
        max_value = np.max(data)

        msr_thr = self.msr_threshold

        biclusters = []
        for i in range(self.num_biclusters):
            rows = np.ones(num_rows, dtype=np.bool)
            cols = np.ones(num_cols, dtype=np.bool)

            self._multiple_node_deletion(data, rows, cols, msr_thr, HE, enc_msr, self.no_ciphertexts)
            self._single_node_deletion(data, rows, cols, msr_thr, HE, enc_msr, self.no_ciphertexts)
            self._node_addition(data, rows, cols, HE, enc_msr, enc_msr_col, enc_msr_row, self.no_ciphertexts)

            row_indices = np.nonzero(rows)[0]
            col_indices = np.nonzero(cols)[0]

            if len(row_indices) == 0 or len(col_indices) == 0:
                break

            # masking matrix values
            if i < self.num_biclusters - 1:
                bicluster_shape = (len(row_indices), len(col_indices))
                data[row_indices[:, np.newaxis], col_indices] = np.random.uniform(low=min_value, high=max_value,
                                                                                  size=bicluster_shape)

            biclusters.append(Bicluster(row_indices, col_indices))

        return Biclustering(biclusters)

    def _single_node_deletion(self, data, rows, cols, msr_thr, HE, enc_msr, no_ciphertexts):
        """Performs the single row/column deletion step (this is a direct implementation of the Algorithm 1 described in
        the original paper)"""
        msr, row_msr, col_msr = enc_msr.calculate_msr(HE, data[rows][:, cols], no_ciphertexts)

        while msr > msr_thr:
            self._single_deletion(data, rows, cols, row_msr, col_msr)
            msr, row_msr, col_msr = enc_msr.calculate_msr(HE, data[rows][:, cols], no_ciphertexts)

    def _single_deletion(self, data, rows, cols, row_msr, col_msr):
        """Deletes a row or column from the bicluster being computed."""
        row_indices = np.nonzero(rows)[0]
        col_indices = np.nonzero(cols)[0]

        row_max_msr = np.argmax(row_msr)
        col_max_msr = np.argmax(col_msr)

        if row_msr[row_max_msr] >= col_msr[col_max_msr]:
            row2remove = row_indices[row_max_msr]
            rows[row2remove] = False
        else:
            col2remove = col_indices[col_max_msr]
            cols[col2remove] = False

    def _multiple_node_deletion(self, data, rows, cols, msr_thr, HE, enc_msr, no_ciphertexts):
        """Performs the multiple row/column deletion step (this is a direct implementation of the Algorithm 2 described in
        the original paper)"""
        msr, row_msr, col_msr = enc_msr.calculate_msr(HE, data[rows][:, cols], no_ciphertexts)

        stop = True if msr <= msr_thr else False

        while not stop:
            cols_old = np.copy(cols)
            rows_old = np.copy(rows)

            row_indices = np.nonzero(rows)[0]
            rows2remove = row_indices[np.where(row_msr > self.multiple_node_deletion_threshold * msr)]
            rows[rows2remove] = False

            if len(cols) >= self.data_min_cols:
                msr, row_msr, col_msr = enc_msr.calculate_msr(HE, data[rows][:, cols], no_ciphertexts)
                col_indices = np.nonzero(cols)[0]
                cols2remove = col_indices[np.where(col_msr > self.multiple_node_deletion_threshold * msr)]
                cols[cols2remove] = False

            msr, row_msr, col_msr = enc_msr.calculate_msr(HE, data[rows][:, cols], no_ciphertexts)

            # Tests if the new MSR value is smaller than the acceptable MSR threshold.
            # Tests if no rows and no columns were removed during this iteration.
            # If one of the conditions is true the loop must stop, otherwise it will become an infinite loop.
            if msr <= msr_thr or (np.all(rows == rows_old) and np.all(cols == cols_old)):
                stop = True

    def _node_addition(self, data, rows, cols, HE, enc_msr, enc_msr_col, enc_msr_row, no_ciphertexts):
        """Performs the row/column addition step (this is a direct implementation of the Algorithm 3 described in
        the original paper)"""
        stop = False
        while not stop:
            cols_old = np.copy(cols)
            rows_old = np.copy(rows)

            msr, _, _ = enc_msr.calculate_msr(HE, data[rows][:, cols], no_ciphertexts)
            col_msr = enc_msr_col.calculate_msr_col_addition(HE, data[rows][:, cols], data[rows], (len(rows), len(cols)),
                                                             no_ciphertexts)
            cols2add = np.where(col_msr.all() <= msr.all())[0]
            cols[cols2add] = True

            msr, _, _ = enc_msr.calculate_msr(HE, data[rows][:, cols], no_ciphertexts)
            row_msr, row_inverse_msr = enc_msr_row.calculate_msr_row_addition(HE, data[rows][:, cols], data[:, cols],
                                                                              (len(rows), len(cols)), no_ciphertexts)
            rows2add = np.where(np.logical_or(row_msr.all() <= msr.all(), row_inverse_msr.all() <= msr.all()))[0]
            rows[rows2add] = True

            if np.all(rows == rows_old) and np.all(cols == cols_old):
                stop = True

    def _validate_parameters(self):
        if self.num_biclusters <= 0:
            raise ValueError("num_biclusters must be > 0, got {}".format(self.num_biclusters))

        if self.msr_threshold != 'estimate' and self.msr_threshold < 0.0:
            raise ValueError("msr_threshold must be equal to 'estimate' or a numeric value >= 0.0, got {}".format(
                self.msr_threshold))

        if self.multiple_node_deletion_threshold < 1.0:
            raise ValueError(
                "multiple_node_deletion_threshold must be >= 1.0, got {}".format(self.multiple_node_deletion_threshold))

        if self.data_min_cols < 100:
            raise ValueError("data_min_cols must be >= 100, got {}".format(self.data_min_cols))