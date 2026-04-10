from typing import List, Any, Dict

import os
import os.path as osp
from tqdm import tqdm
import numpy as np

import torch
from torch_geometric.data import (InMemoryDataset, Data)

from rdkit import Chem

import pandas as pd

HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])

types = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
    'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
    'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
    'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
    'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109,
    'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
}

x_map: Dict[str, List[Any]] = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'CHI_TETRAHEDRAL',
        'CHI_ALLENE',
        'CHI_SQUAREPLANAR',
        'CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map: Dict[str, List[Any]] = {
    'bond_type': [
        'UNSPECIFIED',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'QUADRUPLE',
        'QUINTUPLE',
        'HEXTUPLE',
        'ONEANDAHALF',
        'TWOANDAHALF',
        'THREEANDAHALF',
        'FOURANDAHALF',
        'FIVEANDAHALF',
        'AROMATIC',
        'IONIC',
        'HYDROGEN',
        'THREECENTER',
        'DATIVEONE',
        'DATIVE',
        'DATIVEL',
        'DATIVER',
        'OTHER',
        'ZERO',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}

allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

# Organic molecules dataset class, elements must in [CHONPSClBrI]
class PP_smiles_2d(InMemoryDataset):
    def __init__(self, root, split, property_name, ratio=0.8, radius=8.0, update_atomrefs=False, defined=False, classification=False):
        assert split in ["train", "valid", "test"]
        self.split = split
        self.property_name = property_name
        self.root = osp.abspath(root)
        self.ratio = ratio
        self.radius = radius
        self.update_atomrefs = update_atomrefs
        self.exceed_ele = None
        self.fail_mole = None
        self.defined = defined
        self.classification = classification
        self.class_num = 0
        super().__init__(self.root)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.class_num = len(torch.unique(self.data.y))

    def mean(self) -> float:
        return float(self.data.y.mean())

    def std(self) -> float:
        return float(self.data.y.std())

    def cumpute_avg(self) -> float:
        return len(self.data.x) / len(self.data.y)

    @property
    def raw_file_names(self) -> List[str]:
        if self.defined:
            return ['{}_{}.csv'.format(self.property_name, self.split)]
        else:
            return ['{}.csv'.format(self.property_name)]

    @property
    def processed_file_names(self) -> str:
        if not self.update_atomrefs:
            return self.property_name + '_' + self.split + '_2d.pt'
        else:
            return self.property_name + '_' + self.split + '_atomref2d.pt'

    def process(self):
        if not self.defined:
            suppl = pd.read_csv(self.raw_paths[0])

            Nmols = len(suppl['SMILES'])
            Ntrain = int(self.ratio * Nmols)

            np.random.seed(0)
            data_perm = np.random.permutation(Nmols)

            train, valid = np.split(data_perm, [Ntrain])
            indices = {"train": train, "valid": valid}

            np.savez(os.path.join(self.root, 'splits.npz'), idx_train=train, idx_valid=valid)

            if self.classification:
                label_list = []
                for i in range(Nmols):
                    label_list.append(suppl['value'][i])
                label_dict = unique_strings_to_int(label_list)
                self.class_num = len(label_dict.keys())

            j = 0
            fail = 0
            allow_ele = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}
            data_list = []
            for i in tqdm(range(Nmols)):
                if j not in indices[self.split]:
                    j += 1
                    continue
                j += 1
                mol = suppl['SMILES'][i]
                mol_rdkit = Chem.MolFromSmiles(mol)
                if mol_rdkit is None:
                    fail += 1
                    continue
                try:
                    smiles_dict, _ = from_smiles(mol, with_hydrogen=True)
                    x, edge_index, edge_attr, edge_index_all = smiles_dict
                    atom_types = x
                    edge_index = edge_index
                    edge_attr = edge_attr
                    edge_index_all = edge_index_all

                    if not contains_only_set(atom_types[:, 0].tolist(), allow_ele):
                        self.exceed_ele = 'Unseen elements, please check dataset.'
                        continue
                    
                except Exception as e:
                    fail += 1
                    continue

                node_attr = torch.tensor(atom_types, dtype=torch.long)
                edge_index = torch.tensor(edge_index, dtype=torch.long)
                edge_index_all = torch.tensor(edge_index_all, dtype=torch.long)
                edge_attr = torch.tensor(edge_attr, dtype=torch.long)

                if self.classification:
                    y = torch.tensor(suppl['value'][i], dtype=torch.long)
                else:
                    y = torch.tensor(suppl['value'][i], dtype=torch.float)
                data = Data(x=node_attr, y=y,
                    edge_index=edge_index, edge_attr=edge_attr, edge_index_all=edge_index_all)
                data_list.append(data)
            print(f'The size of {self.split} dataset: {len(data_list)}')
            print(f'Failed to process {fail} molecules:')
            self.fail_mole = fail
            torch.save(self.collate(data_list), self.processed_paths[0])
        else:
            suppl = pd.read_csv(self.raw_paths[0])

            fail = 0
            allow_ele = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}
            data_list = []
            for i in tqdm(range(len(suppl))):
                mol = suppl['SMILES'][i]
                mol_rdkit = Chem.MolFromSmiles(mol)
                if mol_rdkit is None:
                    fail += 1
                    continue
                try:
                    smiles_dict, _ = from_smiles(mol, with_hydrogen=True)
                    x, edge_index, edge_attr, edge_index_all = smiles_dict
                    atom_types = x
                    edge_index = edge_index
                    edge_attr = edge_attr
                    edge_index_all = edge_index_all

                    if not contains_only_set(atom_types[:, 0].tolist(), allow_ele):
                        self.exceed_ele = 'Unseen elements, please check dataset.'
                        continue
                    
                except Exception as e:
                    fail += 1
                    continue

                node_attr = torch.tensor(atom_types, dtype=torch.long)
                edge_index = torch.tensor(edge_index, dtype=torch.long)
                edge_index_all = torch.tensor(edge_index_all, dtype=torch.long)
                edge_attr = torch.tensor(edge_attr, dtype=torch.long)
                
                if self.classification:
                    y = torch.tensor(suppl['value'][i], dtype=torch.long)
                else:
                    y = torch.tensor(suppl['value'][i], dtype=torch.float)
                data = Data(x=node_attr, y=y,
                    edge_index=edge_index, edge_attr=edge_attr, edge_index_all=edge_index_all, smiles=mol)
                data_list.append(data)
            print(f'The size of {self.split} dataset: {len(data_list)}')
            print(f'Failed to process {fail} molecules:')
            self.fail_mole = fail
            torch.save(self.collate(data_list), self.processed_paths[0])

def contains_only_set(arr, allowed_elements):
    return set(arr).issubset(allowed_elements)

def unique_strings_to_int(strings):
    unique_strings = sorted(set(strings))
    
    string_to_int = {string: index for index, string in enumerate(unique_strings)}
    
    return string_to_int


def from_rdmol(mol) -> 'torch_geometric.data.Data':
    r"""Converts a :class:`rdkit.Chem.Mol` instance to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        mol (rdkit.Chem.Mol): The :class:`rdkit` molecule.
    """
    from rdkit import Chem

    from torch_geometric.data import Data

    assert isinstance(mol, Chem.Mol)

    xs: List[List[int]] = []
    for atom in mol.GetAtoms():  # type: ignore
        row: List[int] = []
        row.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
        row.append(x_map['chirality'].index(str(atom.GetChiralTag())))
        # row.append(x_map['degree'].index(atom.GetTotalDegree()))
        row.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
        # row.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
        # row.append(x_map['num_radical_electrons'].index(
        #     atom.GetNumRadicalElectrons()))
        # row.append(x_map['hybridization'].index(str(atom.GetHybridization())))
        row.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
        row.append(x_map['is_in_ring'].index(atom.IsInRing()))
        xs.append(row)

    x = torch.tensor(xs, dtype=torch.long).view(-1, 5)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():  # type: ignore
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map['bond_type'].index(str(bond.GetBondType())))
        e.append(e_map['stereo'].index(str(bond.GetStereo())))
        e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]
        
    # full-connected graph
    nodes = torch.arange(len(xs))
    i, j = torch.combinations(nodes, 2).T
    edge_index_tmp = torch.stack([i, j], dim=0)
    edge_index_reversed = torch.stack([j, i], dim=0)
    edge_index_all = torch.cat([edge_index_tmp, edge_index_reversed], dim=1)


    return x, edge_index, edge_attr, edge_index_all

def from_smiles(
    smiles: str,
    with_hydrogen: bool = False,
    kekulize: bool = False,
) -> 'torch_geometric.data.Data':
    r"""Converts a SMILES string to a :class:`torch_geometric.data.Data`
    instance.

    Args:
        smiles (str): The SMILES string.
        with_hydrogen (bool, optional): If set to :obj:`True`, will store
            hydrogens in the molecule graph. (default: :obj:`False`)
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog('rdApp.*')  # type: ignore

    mol = Chem.MolFromSmiles(smiles)
    mol_flag = True
    
    if mol is None:
        mol = Chem.MolFromSmiles('')
        mol_flag = False
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)
        
    return from_rdmol(mol), mol_flag


if __name__ == "__main__":
    pass
