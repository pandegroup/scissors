#!/usr/bin/env python
"""
choose_basis.py

Choose a SCISSORS basis set by sampling from a molecule dataset.

Steven Kearnes
Pande Lab, Stanford University

Copyright (c) 2014 Stanford University.
"""
import argparse
from sklearn.cross_validation import train_test_split

from scissors import openeye as oe

parser = argparse.ArgumentParser(description="")
parser.add_argument("-i", "--input", required=True, help="Molecule dataset.")
parser.add_argument("-o", "--output", required=True, help="Output filename.")
parser.add_argument("-s", "--size", required=True, type=float,
                    help="Basis set size. If between 0 and 1, interpreted as " +
                         "a proportion of the full dataset.")
parser.add_argument("--seed", type=int,
                    help="Seed for pseudo-random number generator used to " +
                         "select basis molecules.")
args = parser.parse_args()


def main():
    mols = oe.read(args.input)
    if args.size > 1:
        args.size = int(args.size)
    _, basis = train_test_split(mols, test_size=args.size,
                                random_state=args.seed)
    oe.write(basis, args.output)

if __name__ == '__main__':
    main()
