"""
OpenEye functions for handling molecules.
"""
import numpy as np
from openeye.oechem import *


def read(filename):
    """
    Read molecules from a file.

    Parameters
    ----------
    filename : str
        Input filename.
    """
    mols = np.asarray(list(read_generator(filename)))
    return mols


def read_generator(filename):
    """
    Generator for reading molecules that automatically checks for
    multiconformer molecules.

    Parameters
    ----------
    filename : str
        Input filename.
    """
    assert filename is not None
    ifs = oemolistream()
    ifs.SetConfTest(OEOmegaConfTest(False))
    if not ifs.open(filename):
        raise IOError("Cannot read '{}'.".format(filename))
    for mol in ifs.GetOEMols():
        mol = OEMol(mol)
        yield mol
    ifs.close()


def write(mols, filename):
    """
    Write molecules to a file.

    Parameters
    ----------
    mols : array_like or OEMol
        Molecule(s) to save.
    filename : str
        Output filename.
    """
    ofs = oemolostream()
    if not ofs.open(filename):
        raise Exception("Cannot open '{}'.".format(filename))
    for mol in np.atleast_1d(mols):
        OEWriteMolecule(ofs, mol)
    ofs.close()
