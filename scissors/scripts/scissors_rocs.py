#!/usr/bin/env python
"""
scissors.py

Generate SCISSORS vectors from precalculated ROCS similarities.

Steven Kearnes
Pande Lab, Stanford University

Copyright (c) 2014 Stanford University.
"""
import argparse
import h5py
from scissors import SCISSORS


parser = argparse.ArgumentParser(description="")
parser.add_argument("--bb", help="Basis vs. basis ROCS comparisons.")
parser.add_argument("--lb", help="Library vs. basis ROCS comparisons.")
parser.add_argument("-o", "--output", required=True, help="Output filename.")
parser.add_argument("-d", "--dim", type=int, default=None, required=False,
                    help="Maximum dimensionality of SCISSORS vectors.")
parser.add_argument("--shape-dim", type=int,
                    help="Maximum dimensionality of SCISSORS shape vectors.")
parser.add_argument("--color-dim", type=int,
                    help="Maximum dimensionality of SCISSORS color vectors.")
parser.add_argument("--overlap", action="store_true",
                    help="Use actual pairwise overlaps as inner products " +
                         "instead of calculating from Tanimotos.")
args = parser.parse_args()


def load(filename, overlap):
    """
    Load ROCS data from HDF5.

    Parameters
    ----------
    filename : str
        File containing ROCS overlay results.
    overlap : bool
        Whether to use actual pairwise overlaps as inner products. If
        False, use overlaps calculated under the parsimonious assumption
        of unity molecular self-overlap values.
    """
    with h5py.File(filename) as f:
        if overlap:
            shape_ip = f['shape_overlap'][:]
            color_ip = f['color_overlap'][:]
        else:
            shape_ip = SCISSORS.get_inner_products_from_tanimotos(
                f['shape_tanimoto'][:])
            color_ip = SCISSORS.get_inner_products_from_tanimotos(
                f['color_tanimoto'][:])
    return shape_ip, color_ip


def save(data, filename, options=None):
    """
    Write data to HDF5.

    Parameters
    ----------
    data : dict
        Datasets to save.
    filename : str
        Output filename.
    options : dict or None
        HDF5 options.
    """
    if options is None:
        options = {'chunks': True, 'fletcher32': True, 'shuffle': True,
                   'compression': 'gzip', 'compression_opts': 1}
    with h5py.File(filename, 'w') as f:
        for key, val in data.items():
            f.create_dataset(key, data=val, **options)


def main():

    # load input data
    shape_bb_ip, color_bb_ip = load(args.bb)
    shape_lb_ip, color_lb_ip = load(args.lb)

    # setup dimensionality
    shape_dim = None
    color_dim = None
    if args.dim:
        shape_dim = args.dim
        color_dim = args.dim
    if args.shape_dim:
        shape_dim = args.shape_dim
    if args.color_dim:
        color_dim = args.color_dim

    # generate SCISSORS vectors
    shape_s = SCISSORS(shape_bb_ip)
    shape_vectors = shape_s.get_vectors(shape_lb_ip, shape_dim)
    color_s = SCISSORS(color_bb_ip)
    color_vectors = color_s.get_vectors(color_lb_ip, color_dim)

    data = {'shape_vectors': shape_vectors,
            'shape_projection_matrix': shape_s.get_projection_matrix(),
            'color_vectors': color_vectors,
            'color_projection_matrix': color_s.get_projection_matrix()}
    save(data, args.output)

if __name__ == "__main__":
    main()
