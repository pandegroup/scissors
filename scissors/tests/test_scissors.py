"""
SCISSORS tests.

Steven Kearnes
Pande Lab, Stanford University

Copyright (c) 2012 Stanford University.
"""
import numpy as np
import os.path
import unittest

from scissors import SCISSORS


def assert_scissors(a, b):
    """Allow RMS error up to 25%."""
    rmse = np.sqrt(np.square(a-b).sum()/float(a.size))
    assert rmse <= 0.25, rmse


class TestSCISSORS(unittest.TestCase):
    def setUp(self):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.data = np.load(os.path.join(data_dir, 'pc3dfp-rocs.npz'))
        self.n_mols = self.data['ab_overlap'].shape[0]

    def test_scissors_tanimotos(self):
        """Test default Tanimoto approximation."""
        basis = np.random.randint(self.n_mols, size=200)
        s = SCISSORS(self.data['ab_overlap'][basis][:, basis])
        tanimotos = s.get_tanimotos(self.data['ab_overlap'][:, basis])
        import IPython
        IPython.embed()
        assert_scissors(tanimotos, self.data['tanimotos'])

    def test_scissors_tanimotos_with_overlaps(self):
        """
        Test Tanimoto approximation using precalculated self-overlap
        values.
        """
        basis = np.random.randint(self.n_mols, size=200)
        s = SCISSORS(self.data['ab_overlap'][basis][:, basis])
        tanimotos = s.get_tanimotos(self.data['ab_overlap'][:, basis],
                                    self_overlap=self.data['a_overlap'])
        assert_scissors(tanimotos, self.data['tanimotos'])
