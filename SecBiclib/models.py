"""
    SecBic-CCA: A Python library of privacy-preserving biclustering algorithm (Cheng and Church) with Homomorphic Encryption

    Copyright (C) 2022  Shokofeh VahidianSadegh

    This file is part of SecBic-CCA.

"""

import numpy as np


class Bicluster:
    """This class models a bicluster.

    Parameters
    ----------
    rows : numpy.array
        Rows of the bicluster (assumes that row indexing starts at 0).

    cols : numpy.array
        Columns of the bicluster (assumes that column indexing starts at 0).

    data : numpy.ndarray
        bla
    """

    def __init__(self, rows, cols, data=None):
        if isinstance(rows, np.ndarray) and rows.dtype == bool and cols.dtype == bool:
            self.rows = np.nonzero(rows)[0]
            self.cols = np.nonzero(cols)[0]
        elif isinstance(cols, np.ndarray) and rows.dtype == int and cols.dtype == int:
            self.rows = rows
            self.cols = cols
        else:
            raise ValueError("rows and cols must be bool or int numpy.arrays")

        if data is not None:
            n, m = len(self.rows), len(self.cols)

            if isinstance(data, np.ndarray) and (data.shape == (n, m) or (len(data) == 0 and n == 0)):
                self.data = data
            else:
                raise ValueError("")

    def intersection(self, other):
        """Returns a bicluster that represents the area of overlap between two biclusters."""
        rows_intersec = np.intersect1d(self.rows, other.rows)
        cols_intersec = np.intersect1d(self.cols, other.cols)
        return Bicluster(rows_intersec, cols_intersec)

    def union(self, other):
        rows_union = np.union1d(self.rows, other.rows)
        cols_union = np.union1d(self.cols, other.cols)
        return Bicluster(rows_union, cols_union)

    def overlap(self, other):
        min_area = min(self.area, other.area)
        return self.intersection(other).area / min_area

    @property
    def area(self):
        """Calculates the number of matrix elements of the bicluster."""
        return len(self.rows) * len(self.cols)

    def sort(self):
        """Sorts the array of row and the array of column indices of the bicluster."""
        self.rows.sort()
        self.cols.sort()

    def __str__(self):
        return 'Bicluster(rows={0}, cols={1})'.format(self.rows, self.cols)


class Biclustering:
    """This class models a biclustering.

    Parameters
    ----------
    biclusters : list
        A list of instances from the Bicluster class.
    """

    def __init__(self, biclusters):
        if all(isinstance(b, Bicluster) for b in biclusters):
            self.biclusters = biclusters
        else:
            raise ValueError("biclusters list contains an element that is not a Bicluster instance")

    def __str__(self):
        return '\n'.join(str(b) for b in self.biclusters)
