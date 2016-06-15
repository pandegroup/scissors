"""
SCISSORS Calculates Interpolated Shape Signatures Over Rapid Overlay of
Chemical Structures (ROCS) Space.

Rewrites of SCISSORS methods originally written by Imran Haque, with some
modifications and additions by Steven Kearnes.

Sources:
* J. Chem. Inf. Model. 2010, 50, 1075-1088
* J. Chem. Inf. Model. 2011, 51, 2248-2253
* J. Chem. Inf. Model. 2014, 54, 5-15

Steven Kearnes
Pande Lab, Stanford University
"""

# Copyright (C) 2012 Stanford University
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np


class SCISSORS(object):
    """
    Calculate SCISSORS vectors for molecules.

    Parameters
    ----------
    bb_ip : array_like
        Basis vs. basis inner products.
    allow_imaginary : bool
        Whether to include dimensions corresponding to negative
        eigenvalues.
    center : bool
        Whether to center the kernel matrix in feature space.
    """
    def __init__(self, bb_ip, allow_imaginary=False, center=False):
        assert bb_ip.shape[0] == bb_ip.shape[1]
        if center:
            bb_ip = self._center_kernel_matrix(bb_ip)
        self.bb_ip = bb_ip
        self.allow_imaginary = allow_imaginary
        self.e_vals = None
        self.e_vecs = None
        self.projection_matrix = None
        self.inverse_projection_matrix = None

    def get_projection_matrix(self):
        """Calculate SCISSORS projection matrix."""
        # spectral decomposition
        e_vals, e_vecs = np.linalg.eigh(self.bb_ip)

        # only keep nonzero eigenvalues
        # allow 'imaginary' dimensions (sort on eigenvalue magnitude)
        if self.allow_imaginary:
            sort = np.argsort(np.fabs(e_vals))[::-1]
            dim = np.count_nonzero(e_vals != 0)

        # only keep dim largest positive eigenvalues
        else:
            sort = np.argsort(e_vals)[::-1]
            dim = np.count_nonzero(e_vals > 0)

        # reduce to maximum allowed dimensionality
        e_vals = e_vals[sort][:dim]
        e_vecs = e_vecs[:, sort][:, :dim]

        # get diagonal matrix of (eigenvalues)^(-0.5)
        # use abs in case of imaginary dimensions
        D = np.diag(np.reciprocal(np.sqrt(np.fabs(e_vals))))
        D_inv = np.diag(np.sqrt(np.fabs(e_vals)))

        # projection matrix is D^(-1/2)V^T (d x b)
        p = np.asarray(np.asmatrix(D) * np.asmatrix(e_vecs).T)
        p_inv = np.asarray(np.asmatrix(e_vecs) * np.asmatrix(D_inv))

        self.e_vals = e_vals
        self.e_vecs = e_vecs
        self.projection_matrix = p
        self.inverse_projection_matrix = p_inv

        return self.projection_matrix

    def get_max_dim(self):
        """Return maximum dimensionality of SCISSORS vectors."""
        if self.projection_matrix is None:
            self.get_projection_matrix()
        return self.projection_matrix.shape[0]

    def get_eigenvalues(self):
        """Return eigenvalues of retained eigenvectors."""
        if self.e_vals is None:
            self.get_projection_matrix()
        return self.e_vals

    def get_vectors(self, ip, max_dim=None):
        """
        Generate SCISSORS vectors for library molecules.

        Parameters
        ----------
        ip : array_like
            Inner products for library molecules vs. the basis set, with
            library molecules in rows.
        max_dim : int or None
            Maximum dimensionality of SCISSORS vectors. If None, all
            available dimensions will be used.

        Returns
        -------
        vectors : array_like
            SCISSORS vectors for library molecules. Molecules are in rows.
        """
        assert ip.shape[1] == self.bb_ip.shape[0], (ip.shape, self.bb_ip.shape)
        if self.projection_matrix is None:
            self.get_projection_matrix()
        vectors = np.asarray(np.asmatrix(self.projection_matrix) *
                             np.asmatrix(ip).T)
        if max_dim is not None:
            assert isinstance(max_dim, int)
            vectors = vectors[:max_dim]
        return vectors.T

    def get_tanimotos(self, ip, self_overlap=None, max_dim=None):
        """
        Get all vs. all SCISSORS Tanimoto approximations for library
        molecules.

        Parameters
        ----------
        ip : ip : array_like
            Inner products between library molecules (rwos) and basis set
            molecules.
        self_overlap : array_like or None
            Self-overlap values for molecules. If None, squared vector
            norms of SCISSORS vectors will be used.
        max_dim : int or None
            Maximum dimensionality of SCISSORS vectors. If None, all
            available dimensions will be used.
        """
        vectors = self.get_vectors(ip, max_dim)
        return self.vector_tanimotos(vectors, vectors, self_overlap,
                                     self_overlap)

    @staticmethod
    def get_inner_products_from_tanimotos(tanimotos):
        """
        Compute inner products from Tanimotos under the parsimonious
        assumption that inner products between molecules are unity.

        Parameters
        ----------
        tanimotos : array_like
            Tanimotos between library molecules and basis set molecules.
        """
        return (2 * tanimotos) / (1 + tanimotos)

    @staticmethod
    def vector_tanimotos(a, b, a_overlap=None, b_overlap=None):
        """
        Calculate Tanimoto similarity coefficients for vectors.

        Parameters
        ----------
        a, b : array_like
            Vectors between which to calculate similarity (in rows).
        a_overlap, b_overlap : array_like or None
            Self-overlap values for vectors. If None, inner products
            between vectors will be used.

        Returns
        -------
        tanimotos : array_like
            Vector tanimotos (a x b).
        """
        def _get_squared_magnitudes(x):
            """
            Compute squared vector magnitudes.

            Parameters
            ----------
            x : array_like
                Vectors (in rows).
            """
            mags2 = np.diag(np.inner(x, x))
            return mags2

        if a_overlap is None:
            a_overlap = _get_squared_magnitudes(a)
        if b_overlap is None:
            b_overlap = _get_squared_magnitudes(b)

        assert a.shape[1] == b.shape[1]
        assert a.shape[0] == a_overlap.size, (a.shape, a_overlap.size)
        assert b.shape[0] == b_overlap.size

        b_overlap, a_overlap = np.meshgrid(b_overlap, a_overlap)
        ab_overlap = np.asmatrix(a) * np.asmatrix(b).T

        # check for invalid values
        assert not np.any(np.ma.masked_invalid(ab_overlap).mask)
        assert not np.any(np.ma.masked_invalid(a_overlap).mask)
        assert not np.any(np.ma.masked_invalid(b_overlap).mask)

        # masked divide and fill masked values with zero
        tanimotos = np.ma.true_divide(ab_overlap,
                                      a_overlap + b_overlap - ab_overlap)
        tanimotos = np.ma.filled(tanimotos, 0)
        return np.asarray(tanimotos)

    @staticmethod
    def _center_kernel_matrix(m):
        """Center a kernel matrix in feature space.

        Sources:
        * Graph Classification And Clustering Based On Vector Space
            Embedding, Chapter 5: Kernel Methods, Series in Machine
            Perception and Artificial Intelligence, Volume 77, 2010.
        * J. Chemometrics 2011; 25: 92-99

        Parameters
        ----------
        m : array_like
            Kernel matrix to center.
        """
        norm = np.asmatrix(np.ones_like(m, dtype=float) / m.shape[0])
        m = np.asmatrix(m)
        return m - (norm * m) - (m * norm) + (norm * m * norm)
