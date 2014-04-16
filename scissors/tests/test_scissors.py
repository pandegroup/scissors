"""
SCISSORS tests.

Steven Kearnes
Pande Lab, Stanford University

Copyright (c) 2012 Stanford University.
"""
import numpy as np
import os
import unittest

from scissors import SCISSORS


def assert_scissors(a, b):
    """Allow up to 10% RMS error."""
    rmse = np.sqrt(np.square(a-b).sum()/float(a.size))
    assert rmse <= 0.1, rmse


class TestSCISSORS(unittest.TestCase):
    def setUp(self):
        """Load test data."""
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.data = np.load(os.path.join(data_dir, 'pc3dfp-rocs.npz'))
        self.n_mols = self.data['ab_overlap'].shape[0]

    def test_scissors_tanimotos(self):
        """Test default Tanimoto approximation."""
        basis = np.random.randint(self.n_mols, size=200)
        bb_ip = self.data['ab_overlap'][basis][:, basis]
        lb_ip = self.data['ab_overlap'][:, basis]
        s = SCISSORS(bb_ip)
        tanimotos = s.get_tanimotos(lb_ip, max_dim=100)
        assert_scissors(tanimotos, self.data['tanimotos'])

    def test_scissors_tanimotos_with_overlaps(self):
        """
        Test Tanimoto approximation using precalculated self-overlap
        values.
        """
        basis = np.random.randint(self.n_mols, size=200)
        bb_ip = self.data['ab_overlap'][basis][:, basis]
        lb_ip = self.data['ab_overlap'][:, basis]
        s = SCISSORS(bb_ip)
        tanimotos = s.get_tanimotos(lb_ip, self_overlap=self.data['a_overlap'],
                                    max_dim=100)
        assert_scissors(tanimotos, self.data['tanimotos'])

    def test_parsimonious_scissors_tanimotos(self):
        """
        Test default Tanimoto approximation using parsimonious overlap
        values.
        """
        basis = np.random.randint(self.n_mols, size=200)
        bb_ip = self.data['tanimotos'][basis][:, basis]
        bb_ip = SCISSORS.get_inner_products_from_tanimotos(bb_ip)
        lb_ip = self.data['tanimotos'][:, basis]
        lb_ip = SCISSORS.get_inner_products_from_tanimotos(lb_ip)
        s = SCISSORS(bb_ip)
        tanimotos = s.get_tanimotos(lb_ip, max_dim=100)
        assert_scissors(tanimotos, self.data['tanimotos'])

    def test_parsimonious_scissors_tanimotos_with_overlaps(self):
        """
        Test Tanimoto approximation using parsimonious overlap values and
        precalculated self-overlap values.
        """
        basis = np.random.randint(self.n_mols, size=200)
        bb_ip = self.data['tanimotos'][basis][:, basis]
        bb_ip = SCISSORS.get_inner_products_from_tanimotos(bb_ip)
        lb_ip = self.data['tanimotos'][:, basis]
        lb_ip = SCISSORS.get_inner_products_from_tanimotos(lb_ip)
        s = SCISSORS(bb_ip)
        tanimotos = s.get_tanimotos(lb_ip, self_overlap=np.ones(lb_ip.shape[0],
                                                                dtype=float),
                                    max_dim=100)
        assert_scissors(tanimotos, self.data['tanimotos'])
