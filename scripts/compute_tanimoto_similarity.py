"""Computes fingerprint Tanimoto coefficients between the ligands in the PDBbind dataset.

Usage:
    compute_ligand_similarity.py [-h] <pdb_list_file> <pdbbind_dir> <output_file>

Arguments:
    pdb_list_file   file containing pdb codes of complexes to use
    pdbbind_dir     top-level directory of the PDBbind data set
    output_file     file to save the computed features to

Options:
    -h --help       show this message and exit

Computes Tanimoto coefficient between Morgan fingerprints (radius 2; 2048 bits) of
the ligands of the specified complexes from the PDBbind data set
and saves the results to the specified file in .csv format.

"""
import os

import pandas as pd

from docopt import docopt
from rdkit import Chem
from rdkit import DataStructs
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint

def main():

    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    # parse command line arguments
    args = docopt(__doc__)
    pdb_list_file = args['<pdb_list_file>']
    pdbbind_dir = args['<pdbbind_dir>']
    output_file = args['<output_file>']

    with open(pdb_list_file, 'r') as f:
        pdbs = [l.strip() for l in f]

    # load ligands and compute features
    fingerprints = {}
    for pdb in pdbs:
        # prefer to use the .sdf provided by PDBbind
        sdf = os.path.join(pdbbind_dir, pdb, f'{pdb}_ligand.sdf')
        mol = next(Chem.SDMolSupplier(sdf, removeHs=False))

        # but we'll try the .mol2 if RDKit can't parse the .sdf
        if mol is None:
            mol2 = os.path.join(pdbbind_dir, pdb, f'{pdb}_ligand.mol2')
            mol = Chem.MolFromMol2File(mol2, removeHs=False)

        # skip the ligand if RDKit can't parse the .mol2
        if mol is None:
            continue

        try:
            fingerprints[pdb] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        except ValueError as e:
            print(e)
            continue

    tc = {pdb1: {pdb2: DataStructs.FingerprintSimilarity(fingerprints[pdb1], fingerprints[pdb2]) for pdb2 in fingerprints} for pdb1 in fingerprints}

    tc = pd.DataFrame(tc)
    tc.to_csv(output_file)


if __name__=='__main__':
    main()
