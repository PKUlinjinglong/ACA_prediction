import os
from collections import OrderedDict
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem
from rdkit.Chem.Descriptors import rdMolDescriptors
import os
from mordred import Calculator, descriptors,is_missing
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from sklearn.cluster import KMeans
import tqdm
import pandas as pd
from tqdm import tqdm
import warnings
import torch
import numpy as np

DAY_LIGHT_FG_SMARTS_LIST = [
        # C
        "[CX4]",
        "[$([CX2](=C)=C)]",
        "[$([CX3]=[CX3])]",
        "[$([CX2]#C)]",
        # C & O
        "[CX3]=[OX1]",
        "[$([CX3]=[OX1]),$([CX3+]-[OX1-])]",
        "[CX3](=[OX1])C",
        "[OX1]=CN",
        "[CX3](=[OX1])O",
        "[CX3](=[OX1])[F,Cl,Br,I]",
        "[CX3H1](=O)[#6]",
        "[CX3](=[OX1])[OX2][CX3](=[OX1])",
        "[NX3][CX3](=[OX1])[#6]",
        "[NX3][CX3]=[NX3+]",
        "[NX3,NX4+][CX3](=[OX1])[OX2,OX1-]",
        "[NX3][CX3](=[OX1])[OX2H0]",
        "[NX3,NX4+][CX3](=[OX1])[OX2H,OX1-]",
        "[CX3](=O)[O-]",
        "[CX3](=[OX1])(O)O",
        "[CX3](=[OX1])([OX2])[OX2H,OX1H0-1]",
        "C[OX2][CX3](=[OX1])[OX2]C",
        "[CX3](=O)[OX2H1]",
        "[CX3](=O)[OX1H0-,OX2H1]",
        "[NX3][CX2]#[NX1]",
        "[#6][CX3](=O)[OX2H0][#6]",
        "[#6][CX3](=O)[#6]",
        "[OD2]([#6])[#6]",
        # H
        "[H]",
        "[!#1]",
        "[H+]",
        "[+H]",
        "[!H]",
        # N
        "[NX3;H2,H1;!$(NC=O)]",
        "[NX3][CX3]=[CX3]",
        "[NX3;H2;!$(NC=[!#6]);!$(NC#[!#6])][#6]",
        "[NX3;H2,H1;!$(NC=O)].[NX3;H2,H1;!$(NC=O)]",
        "[NX3][$(C=C),$(cc)]",
        "[NX3,NX4+][CX4H]([*])[CX3](=[OX1])[O,N]",
        "[NX3H2,NH3X4+][CX4H]([*])[CX3](=[OX1])[NX3,NX4+][CX4H]([*])[CX3](=[OX1])[OX2H,OX1-]",
        "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([*])[CX3](=[OX1])[OX2H,OX1-,N]",
        "[CH3X4]",
        "[CH2X4][CH2X4][CH2X4][NHX3][CH0X3](=[NH2X3+,NHX2+0])[NH2X3]",
        "[CH2X4][CX3](=[OX1])[NX3H2]",
        "[CH2X4][CX3](=[OX1])[OH0-,OH]",
        "[CH2X4][SX2H,SX1H0-]",
        "[CH2X4][CH2X4][CX3](=[OX1])[OH0-,OH]",
        "[$([$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H2][CX3](=[OX1])[OX2H,OX1-,N])]",
        "[CH2X4][#6X3]1:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]:\
[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]1",
        "[CHX4]([CH3X4])[CH2X4][CH3X4]",
        "[CH2X4][CHX4]([CH3X4])[CH3X4]",
        "[CH2X4][CH2X4][CH2X4][CH2X4][NX4+,NX3+0]",
        "[CH2X4][CH2X4][SX2][CH3X4]",
        "[CH2X4][cX3]1[cX3H][cX3H][cX3H][cX3H][cX3H]1",
        "[$([NX3H,NX4H2+]),$([NX3](C)(C)(C))]1[CX4H]([CH2][CH2][CH2]1)[CX3](=[OX1])[OX2H,OX1-,N]",
        "[CH2X4][OX2H]",
        "[NX3][CX3]=[SX1]",
        "[CHX4]([CH3X4])[OX2H]",
        "[CH2X4][cX3]1[cX3H][nX3H][cX3]2[cX3H][cX3H][cX3H][cX3H][cX3]12",
        "[CH2X4][cX3]1[cX3H][cX3H][cX3]([OHX2,OH0X1-])[cX3H][cX3H]1",
        "[CHX4]([CH3X4])[CH3X4]",
        "N[CX4H2][CX3](=[OX1])[O,N]",
        "N1[CX4H]([CH2][CH2][CH2]1)[CX3](=[OX1])[O,N]",
        "[$(*-[NX2-]-[NX2+]#[NX1]),$(*-[NX2]=[NX2+]=[NX1-])]",
        "[$([NX1-]=[NX2+]=[NX1-]),$([NX1]#[NX2+]-[NX1-2])]",
        "[#7]",
        "[NX2]=N",
        "[NX2]=[NX2]",
        "[$([NX2]=[NX3+]([O-])[#6]),$([NX2]=[NX3+0](=[O])[#6])]",
        "[$([#6]=[N+]=[N-]),$([#6-]-[N+]#[N])]",
        "[$([nr5]:[nr5,or5,sr5]),$([nr5]:[cr5]:[nr5,or5,sr5])]",
        "[NX3][NX3]",
        "[NX3][NX2]=[*]",
        "[CX3;$([C]([#6])[#6]),$([CH][#6])]=[NX2][#6]",
        "[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]",
        "[NX3+]=[CX3]",
        "[CX3](=[OX1])[NX3H][CX3](=[OX1])",
        "[CX3](=[OX1])[NX3H0]([#6])[CX3](=[OX1])",
        "[CX3](=[OX1])[NX3H0]([NX3H0]([CX3](=[OX1]))[CX3](=[OX1]))[CX3](=[OX1])",
        "[$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)]",
        "[$([OX1]=[NX3](=[OX1])[OX1-]),$([OX1]=[NX3+]([OX1-])[OX1-])]",
        "[NX1]#[CX2]",
        "[CX1-]#[NX2+]",
        "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
        "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8].[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
        "[NX2]=[OX1]",
        "[$([#7+][OX1-]),$([#7v5]=[OX1]);!$([#7](~[O])~[O]);!$([#7]=[#7])]",
        # O
        "[OX2H]",
        "[#6][OX2H]",
        "[OX2H][CX3]=[OX1]",
        "[OX2H]P",
        "[OX2H][#6X3]=[#6]",
        "[OX2H][cX3]:[c]",
        "[OX2H][$(C=C),$(cc)]",
        "[$([OH]-*=[!#6])]",
        "[OX2,OX1-][OX2,OX1-]",
        # P
        "[$(P(=[OX1])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),\
$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)]),$([P+]([OX1-])([$([OX2H]),$([OX1-])\
,$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)])]",
        "[$(P(=[OX1])([OX2][#6])([$([OX2H]),$([OX1-]),$([OX2][#6])])[$([OX2H]),\
$([OX1-]),$([OX2][#6]),$([OX2]P)]),$([P+]([OX1-])([OX2][#6])([$([OX2H]),$([OX1-]),\
$([OX2][#6])])[$([OX2H]),$([OX1-]),$([OX2][#6]),$([OX2]P)])]",
        # S
        "[S-][CX3](=S)[#6]",
        "[#6X3](=[SX1])([!N])[!N]",
        "[SX2]",
        "[#16X2H]",
        "[#16!H0]",
        "[#16X2H0]",
        "[#16X2H0][!#16]",
        "[#16X2H0][#16X2H0]",
        "[#16X2H0][!#16].[#16X2H0][!#16]",
        "[$([#16X3](=[OX1])[OX2H0]),$([#16X3+]([OX1-])[OX2H0])]",
        "[$([#16X3](=[OX1])[OX2H,OX1H0-]),$([#16X3+]([OX1-])[OX2H,OX1H0-])]",
        "[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]",
        "[$([#16X4](=[OX1])(=[OX1])([#6])[#6]),$([#16X4+2]([OX1-])([OX1-])([#6])[#6])]",
        "[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H,OX1H0-]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H,OX1H0-])]",
        "[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H0]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H0])]",
        "[$([#16X4]([NX3])(=[OX1])(=[OX1])[#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[#6])]",
        "[SX4](C)(C)(=O)=N",
        "[$([SX4](=[OX1])(=[OX1])([!O])[NX3]),$([SX4+2]([OX1-])([OX1-])([!O])[NX3])]",
        "[$([#16X3]=[OX1]),$([#16X3+][OX1-])]",
        "[$([#16X3](=[OX1])([#6])[#6]),$([#16X3+]([OX1-])([#6])[#6])]",
        "[$([#16X4](=[OX1])(=[OX1])([OX2H,OX1H0-])[OX2][#6]),$([#16X4+2]([OX1-])([OX1-])([OX2H,OX1H0-])[OX2][#6])]",
        "[$([SX4](=O)(=O)(O)O),$([SX4+2]([O-])([O-])(O)O)]",
        "[$([#16X4](=[OX1])(=[OX1])([OX2][#6])[OX2][#6]),$([#16X4](=[OX1])(=[OX1])([OX2][#6])[OX2][#6])]",
        "[$([#16X4]([NX3])(=[OX1])(=[OX1])[OX2][#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[OX2][#6])]",
        "[$([#16X4]([NX3])(=[OX1])(=[OX1])[OX2H,OX1H0-]),$([#16X4+2]([NX3])([OX1-])([OX1-])[OX2H,OX1H0-])]",
        "[#16X2][OX2H,OX1H0-]",
        "[#16X2][OX2H0]",
        # X
        "[#6][F,Cl,Br,I]",
        "[F,Cl,Br,I]",
        "[F,Cl,Br,I].[F,Cl,Br,I].[F,Cl,Br,I]",
    ]


def get_gasteiger_partial_charges(mol, n_iter=12):
    """
    Gasteiger偏分数，也被称为Gasteiger-Marsili偏分数，是用于估计分子中各原子的偏分数的一种方法。
    它基于经验的方式来估算原子上的电荷分布

    分子建模和药物设计：了解分子中的电荷分布可以帮助我们理解分子的结构、稳定性和反应活性。这对于新药的设计和发现是非常重要的。
    预测分子相互作用：偏分数可以用于预测分子间的非共价相互作用，如氢键、范德华力等。这对于理解蛋白质-配体相互作用、蛋白质-蛋白质相互作用非常重要。
    QSAR/QSPR研究：在量化结构-活性关系（QSAR）和量化结构-性质关系
    反应机制研究：在研究化学反应的机制时，了解分子（QSPR）研究中，偏分数可以作为描述符来预测分子的生物活性或其他性质。
    电荷分布可以帮助我们更好地理解反应路径和过渡状态的性质。
    物质的物理化学性质：它也可以用于预测和解释物质的物理化学性质，如溶解度、熔点和沸点。

    代码计算部分：
    使用Gasteiger算法来计算传入的rdkit分子对象中的每一个原子的偏分量。
    ● nIter: Gasteiger算法的迭代次数，默认为12。
    ● throwOnParamFailure: 一个布尔值，指示当参数失败时是否抛出异常
    第一行是负责计算每个原子的偏分数
    第二行是负责访问每个原子的偏分数，如果没有第一行的话，是无法访问.mol对象的
    """
    Chem.rdPartialCharges.ComputeGasteigerCharges(mol, nIter=n_iter,throwOnParamFailure=True)
    partial_charges = [float(a.GetProp('_GasteigerCharge')) for a in mol.GetAtoms()]
    return partial_charges


def create_standardized_mol_id(smiles):
    """
    Args:
        smiles: smiles sequence.
    Returns:
        inchi.
    InChI是IUPAC国际化学命名规则的一种标准化化学结构标识符，用于更准确地表示和存储化学分子结构的信息
    """
    if check_smiles_validity(smiles):
        # remove stereochemistry 去除立体化学信息
        smiles = AllChem.MolToSmiles(AllChem.MolFromSmiles(smiles),isomericSmiles=False)
        mol = AllChem.MolFromSmiles(smiles)
        if not mol is None:  # to catch weird issue with O=C1O[al]2oc(=O)c3ccc(cn3)c3ccccc3c3cccc(c3)c3ccccc3c3cc(C(F)(F)F)c(cc3o2)-c2ccccc2-c2cccc(c2)-c2ccccc2-c2cccnc21
            if '.' in smiles:  # if multiple species, pick largest molecule
                mol_species_list = split_rdkit_mol_obj(mol)
                largest_mol = get_largest_mol(mol_species_list)
                inchi = AllChem.MolToInchi(largest_mol)
            else:
                inchi = AllChem.MolToInchi(mol)
            return inchi
        else:
            return
    else:
        return

def check_smiles_validity(smiles):
    """
    Check whether the smile can't be converted to rdkit mol object.
    """
    try:
        m = Chem.MolFromSmiles(smiles)
        if m:
            return True
        else:
            return False
    except Exception as e:
        return False


def split_rdkit_mol_obj(mol):
    """
    Split rdkit mol object containing multiple species or one species into a
    list of mol objects or a list containing a single object respectively.
    Args:
        mol: rdkit mol object.
    """
    smiles = AllChem.MolToSmiles(mol, isomericSmiles=True)
    smiles_list = smiles.split('.')
    mol_species_list = []
    for s in smiles_list:
        if check_smiles_validity(s):
            mol_species_list.append(AllChem.MolFromSmiles(s))
    return mol_species_list


def get_largest_mol(mol_list):
    """
    Given a list of rdkit mol objects, returns mol object containing the
    largest num of atoms. If multiple containing largest num of atoms,
    picks the first one.
    Args:
        mol_list(list): a list of rdkit mol object.
    Returns:
        the largest mol.
    """
    num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
    largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
    return mol_list[largest_mol_idx]


def rdchem_enum_to_list(values):
    """values = {0: rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            1: rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            2: rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            3: rdkit.Chem.rdchem.ChiralType.CHI_OTHER}
    该函数接受一个字典 values 作为参数并返回一个由字典中的值组成的列表。
    """
    return [values[i] for i in range(len(values))]


def safe_index(alist, elem):
    """
    Return index of element e in list l. If e is not present, return the last index
    该函数接收两个参数：一个列表alist和一个元素elem。它试图找到elem在alist中的索引，并在找不到时返回列表的最后一个索引。
    """
    try:
        return alist.index(elem)    #寻找列表当中某元素的索引
    except ValueError:
        return len(alist) - 1


def get_atom_feature_dims(list_acquired_feature_names):
    """ tbd
    下面有更好理解的版本
    # 假设 CompoundKit.atom_vocab_dict 是这样的:
    # {
    #     "feature1": [1, 2, 3],
    #     "feature2": [1, 2, 3, 4, 5]
    # }
    # output 将会是: [3, 5]
    """
    # feature_dims = []
    # for name in list_acquired_feature_names:
    #     feature = CompoundKit.atom_vocab_dict[name]
    #     feature_dim = len(feature)
    #     feature_dims.append(feature_dim)
    # return feature_dims
    return list(map(len, [CompoundKit.atom_vocab_dict[name] for name in list_acquired_feature_names]))


def get_bond_feature_dims(list_acquired_feature_names):
    """ tbd
    """
    # list_bond_feat_dim = []
    # for name in list_acquired_feature_names:
    #     feature = CompoundKit.bond_vocab_dict[name]
    #     feature_dim = len(feature)
    #     list_bond_feat_dim.append(feature_dim)
    # # +1 for self loop edges
    # for i in range(len(list_bond_feat_dim)):
    #     list_bond_feat_dim[i] += 1
    # return list_bond_feat_dim
    list_bond_feat_dim = list(map(len, [CompoundKit.bond_vocab_dict[name] for name in list_acquired_feature_names]))
    # +1 for self loop edges
    return [_l + 1 for _l in list_bond_feat_dim]
    #每个特征维度都增加了1来考虑所谓的“自循环边”。在图论和网络分析中，自循环边是指从一个节点出发并返回到同一个节点的边。
    #在一些图神经网络的实现中，自循环边可以用来确保每个节点可以“看到”其自身的特征，这样它就不仅仅依赖于它的邻居的特征。


class CompoundKit(object):
    """
    CompoundKit
    """
    #是一个字典，其中包含多种原子特性的可能值的列表。每个键表示一个特性名称，每个值表示该特性的所有可能值的列表。这些特性用于描述化学结构中的原子。
    atom_vocab_dict = {
        "atomic_num": list(range(1, 119)) + ['misc'],
        "chiral_tag": rdchem_enum_to_list(rdchem.ChiralType.values),
        "degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
        "explicit_valence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'misc'],
        "formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
        "hybridization": rdchem_enum_to_list(rdchem.HybridizationType.values),
        "implicit_valence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'misc'],
        "is_aromatic": [0, 1],
        "total_numHs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'num_radical_e': [0, 1, 2, 3, 4, 'misc'],
        'atom_is_in_ring': [0, 1],
        'valence_out_shell': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size3': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size4': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size5': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size6': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size7': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size8': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    }
    #键的特征
    bond_vocab_dict = {
        "bond_dir": rdchem_enum_to_list(rdchem.BondDir.values),
        "bond_type": rdchem_enum_to_list(rdchem.BondType.values),
        "is_in_ring": [0, 1],
        'bond_stereo': rdchem_enum_to_list(rdchem.BondStereo.values),
        'is_conjugated': [0, 1],
    }
    # float features
    # 这行定义了一个列表，包含原子的一些浮点数特性的名称。
    atom_float_names = ["van_der_waals_radis", "partial_charge", 'mass']
    # bond_float_feats= ["bond_length", "bond_angle"]     # optional

    ### functional groups
    day_light_fg_smarts_list = DAY_LIGHT_FG_SMARTS_LIST
    day_light_fg_mo_list = [Chem.MolFromSmarts(smarts) for smarts in day_light_fg_smarts_list]

    morgan_fp_N = 200
    morgan2048_fp_N = 2048
    maccs_fp_N = 167

    period_table = Chem.GetPeriodicTable()
    # 这行获取RDKit库中的周期表对象，它提供了一个方便的方式来访问周期表元素的属性和信息。
    ### atom

    @staticmethod
    def get_atom_value(atom, name):
        """get atom values"""
        if name == 'atomic_num':
            return atom.GetAtomicNum()
        elif name == 'chiral_tag':
            return atom.GetChiralTag()
        elif name == 'degree':
            return atom.GetDegree()
        elif name == 'explicit_valence':
            return atom.GetExplicitValence()
        elif name == 'formal_charge':
            return atom.GetFormalCharge()
        elif name == 'hybridization':
            return atom.GetHybridization()
        elif name == 'implicit_valence':
            return atom.GetImplicitValence()
        elif name == 'is_aromatic':
            return int(atom.GetIsAromatic())
        elif name == 'mass':
            return int(atom.GetMass())
        elif name == 'total_numHs':
            return atom.GetTotalNumHs()
        elif name == 'num_radical_e':
            return atom.GetNumRadicalElectrons()
        elif name == 'atom_is_in_ring':
            return int(atom.IsInRing())
        elif name == 'valence_out_shell':
            return CompoundKit.period_table.GetNOuterElecs(atom.GetAtomicNum())
        else:
            raise ValueError(name)

    @staticmethod
    def get_atom_feature_id(atom, name):
        """get atom features id"""
        assert name in CompoundKit.atom_vocab_dict, "%s not found in atom_vocab_dict" % name
        return safe_index(CompoundKit.atom_vocab_dict[name], CompoundKit.get_atom_value(atom, name))

    @staticmethod
    def get_atom_feature_size(name):
        """get atom features size"""
        assert name in CompoundKit.atom_vocab_dict, "%s not found in atom_vocab_dict" % name
        return len(CompoundKit.atom_vocab_dict[name])

    ### bond

    @staticmethod
    def get_bond_value(bond, name):
        """get bond values"""
        if name == 'bond_dir':
            return bond.GetBondDir()
        elif name == 'bond_type':
            return bond.GetBondType()
        elif name == 'is_in_ring':
            return int(bond.IsInRing())
        elif name == 'is_conjugated':
            return int(bond.GetIsConjugated())
        elif name == 'bond_stereo':
            return bond.GetStereo()
        else:
            raise ValueError(name)

    @staticmethod
    def get_bond_feature_id(bond, name):
        """get bond features id"""
        assert name in CompoundKit.bond_vocab_dict, "%s not found in bond_vocab_dict" % name
        return safe_index(CompoundKit.bond_vocab_dict[name], CompoundKit.get_bond_value(bond, name))

    @staticmethod
    def get_bond_feature_size(name):
        """get bond features size"""
        assert name in CompoundKit.bond_vocab_dict, "%s not found in bond_vocab_dict" % name
        return len(CompoundKit.bond_vocab_dict[name])

    ### fingerprint

    @staticmethod
    def get_morgan_fingerprint(mol, radius=2):
        """get morgan fingerprint"""
        nBits = CompoundKit.morgan_fp_N
        mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return [int(b) for b in mfp.ToBitString()]

    @staticmethod
    def get_morgan2048_fingerprint(mol, radius=2):
        """get morgan2048 fingerprint"""
        nBits = CompoundKit.morgan2048_fp_N
        mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return [int(b) for b in mfp.ToBitString()]

    @staticmethod
    #静态方法，这意味着这个方法可以不创建类的实例而直接用类名来调用。
    def get_maccs_fingerprint(mol):
        """get maccs fingerprint"""
        fp = AllChem.GetMACCSKeysFingerprint(mol)
        return [int(b) for b in fp.ToBitString()]

    ### functional groups

    @staticmethod
    def get_daylight_functional_group_counts(mol):
        """get daylight functional group counts
        该方法计算了一个分子中每个在CompoundKit.day_light_fg_mo_list列表中定义的功能群的数量。
        它使用Chem.Mol.GetSubstructMatches方法来找到所有的匹配，然后返回一个包含所有计数的列表。
        """
        fg_counts = []
        for fg_mol in CompoundKit.day_light_fg_mo_list:
            sub_structs = Chem.Mol.GetSubstructMatches(mol, fg_mol, uniquify=True)
            fg_counts.append(len(sub_structs))
        return fg_counts

    @staticmethod
    def get_ring_size(mol):
        """return (N,6) list
        一个嵌套列表：
        列表的第一个元素对应于分子中的第一个原子，第二个元素对应于第二个原子，依此类推。
        每个内部列表包含6个元素，分别表示该原子出现在3, 4, 5, 6, 7, 和 8成员环中的次数。
        """
        rings = mol.GetRingInfo()
        rings_info = []
        for r in rings.AtomRings():
            rings_info.append(r)
        ring_list = []
        for atom in mol.GetAtoms():
            atom_result = []
            for ringsize in range(3, 9):
                num_of_ring_at_ringsize = 0
                for r in rings_info:
                    if len(r) == ringsize and atom.GetIdx() in r:
                        num_of_ring_at_ringsize += 1
                if num_of_ring_at_ringsize > 8:
                    num_of_ring_at_ringsize = 9
                atom_result.append(num_of_ring_at_ringsize)
            ring_list.append(atom_result)
        return ring_list

    @staticmethod
    def atom_to_feat_vector(atom):
        """ tbd """
        atom_names = {
            "atomic_num": safe_index(CompoundKit.atom_vocab_dict["atomic_num"], atom.GetAtomicNum()),
            "chiral_tag": safe_index(CompoundKit.atom_vocab_dict["chiral_tag"], atom.GetChiralTag()),
            "degree": safe_index(CompoundKit.atom_vocab_dict["degree"], atom.GetTotalDegree()),
            "explicit_valence": safe_index(CompoundKit.atom_vocab_dict["explicit_valence"], atom.GetExplicitValence()),
            "formal_charge": safe_index(CompoundKit.atom_vocab_dict["formal_charge"], atom.GetFormalCharge()),
            "hybridization": safe_index(CompoundKit.atom_vocab_dict["hybridization"], atom.GetHybridization()),
            "implicit_valence": safe_index(CompoundKit.atom_vocab_dict["implicit_valence"], atom.GetImplicitValence()),
            "is_aromatic": safe_index(CompoundKit.atom_vocab_dict["is_aromatic"], int(atom.GetIsAromatic())),
            "total_numHs": safe_index(CompoundKit.atom_vocab_dict["total_numHs"], atom.GetTotalNumHs()),
            'num_radical_e': safe_index(CompoundKit.atom_vocab_dict['num_radical_e'], atom.GetNumRadicalElectrons()),
            'atom_is_in_ring': safe_index(CompoundKit.atom_vocab_dict['atom_is_in_ring'], int(atom.IsInRing())),
            'valence_out_shell': safe_index(CompoundKit.atom_vocab_dict['valence_out_shell'],CompoundKit.period_table.GetNOuterElecs(atom.GetAtomicNum())),
            'van_der_waals_radis': CompoundKit.period_table.GetRvdw(atom.GetAtomicNum()),
            'partial_charge': CompoundKit.check_partial_charge(atom),
            'mass': atom.GetMass(),
        }
        return atom_names

    @staticmethod
    def get_atom_names(mol):
        """get atom name list
        TODO: to be remove in the future
        """
        atom_features_dicts = []
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        for i, atom in enumerate(mol.GetAtoms()):
            atom_features_dicts.append(CompoundKit.atom_to_feat_vector(atom))

        ring_list = CompoundKit.get_ring_size(mol)
        for i, atom in enumerate(mol.GetAtoms()):
            atom_features_dicts[i]['in_num_ring_with_size3'] = safe_index(
                CompoundKit.atom_vocab_dict['in_num_ring_with_size3'], ring_list[i][0])
            atom_features_dicts[i]['in_num_ring_with_size4'] = safe_index(
                CompoundKit.atom_vocab_dict['in_num_ring_with_size4'], ring_list[i][1])
            atom_features_dicts[i]['in_num_ring_with_size5'] = safe_index(
                CompoundKit.atom_vocab_dict['in_num_ring_with_size5'], ring_list[i][2])
            atom_features_dicts[i]['in_num_ring_with_size6'] = safe_index(
                CompoundKit.atom_vocab_dict['in_num_ring_with_size6'], ring_list[i][3])
            atom_features_dicts[i]['in_num_ring_with_size7'] = safe_index(
                CompoundKit.atom_vocab_dict['in_num_ring_with_size7'], ring_list[i][4])
            atom_features_dicts[i]['in_num_ring_with_size8'] = safe_index(
                CompoundKit.atom_vocab_dict['in_num_ring_with_size8'], ring_list[i][5])

        return atom_features_dicts

    @staticmethod
    def check_partial_charge(atom):
        """tbd"""
        pc = atom.GetDoubleProp('_GasteigerCharge')
        if pc != pc:
            # unsupported atom, replace nan with 0
            pc = 0
        if pc == float('inf'):
            # max 4 for other atoms, set to 10 here if inf is get
            pc = 10
        return pc


class Compound3DKit(object):
    """the 3Dkit of Compound"""

    @staticmethod
    def get_atom_poses(mol, conf):
        """tbd"""
        atom_poses = []
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomicNum() == 0:
                return [[0.0, 0.0, 0.0]] * len(mol.GetAtoms())
            pos = conf.GetAtomPosition(i)
            atom_poses.append([pos.x, pos.y, pos.z])
        return atom_poses

    @staticmethod
    def get_MMFF_atom_poses(mol, numConfs=None, return_energy=False):
        """the atoms of mol will be changed in some cases."""
        conf = mol.GetConformer()
        atom_poses = Compound3DKit.get_atom_poses(mol, conf)
        return mol,atom_poses
        # try:
        #     new_mol = Chem.AddHs(mol)
        #     res = AllChem.EmbedMultipleConfs(new_mol, numConfs=numConfs)
        #     ### MMFF generates multiple conformations
        #     res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
        #     new_mol = Chem.RemoveHs(new_mol)
        #     index = np.argmin([x[1] for x in res])
        #     energy = res[index][1]
        #     conf = new_mol.GetConformer(id=int(index))
        # except:
        #     new_mol = mol
        #     AllChem.Compute2DCoords(new_mol)
        #     energy = 0
        #     conf = new_mol.GetConformer()
        #
        # atom_poses = Compound3DKit.get_atom_poses(new_mol, conf)
        # if return_energy:
        #     return new_mol, atom_poses, energy
        # else:
        #     return new_mol, atom_poses

    @staticmethod
    def get_2d_atom_poses(mol):
        """get 2d atom poses"""
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        atom_poses = Compound3DKit.get_atom_poses(mol, conf)
        return atom_poses

    @staticmethod
    def get_bond_lengths(edges, atom_poses):
        """get bond lengths"""
        bond_lengths = []
        for src_node_i, tar_node_j in edges:
            bond_lengths.append(np.linalg.norm(atom_poses[tar_node_j] - atom_poses[src_node_i]))
        bond_lengths = np.array(bond_lengths, 'float32')
        return bond_lengths

    @staticmethod
    def get_superedge_angles(edges, atom_poses, dir_type='HT'):
        """get superedge angles"""

        def _get_vec(atom_poses, edge):
            return atom_poses[edge[1]] - atom_poses[edge[0]]

        def _get_angle(vec1, vec2):
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0
            vec1 = vec1 / (norm1 + 1e-5)  # 1e-5: prevent numerical errors
            vec2 = vec2 / (norm2 + 1e-5)
            angle = np.arccos(np.dot(vec1, vec2))
            return angle

        E = len(edges)
        edge_indices = np.arange(E)
        super_edges = []
        bond_angles = []
        bond_angle_dirs = []
        for tar_edge_i in range(E):
            tar_edge = edges[tar_edge_i]
            if dir_type == 'HT':
                src_edge_indices = edge_indices[edges[:, 1] == tar_edge[0]]
            elif dir_type == 'HH':
                src_edge_indices = edge_indices[edges[:, 1] == tar_edge[1]]
            else:
                raise ValueError(dir_type)
            for src_edge_i in src_edge_indices:
                if src_edge_i == tar_edge_i:
                    continue
                src_edge = edges[src_edge_i]
                src_vec = _get_vec(atom_poses, src_edge)
                tar_vec = _get_vec(atom_poses, tar_edge)
                super_edges.append([src_edge_i, tar_edge_i])
                angle = _get_angle(src_vec, tar_vec)
                bond_angles.append(angle)
                bond_angle_dirs.append(src_edge[1] == tar_edge[0])  # H -> H or H -> T

        if len(super_edges) == 0:
            super_edges = np.zeros([0, 2], 'int64')
            bond_angles = np.zeros([0, ], 'float32')
        else:
            super_edges = np.array(super_edges, 'int64')
            bond_angles = np.array(bond_angles, 'float32')
        return super_edges, bond_angles, bond_angle_dirs


def new_smiles_to_graph_data(smiles, **kwargs):
    """
    Convert smiles to graph data.
    """
    mol = AllChem.MolFromSmiles(smiles)
    if mol is None:
        return None
    data = new_mol_to_graph_data(mol)
    return data


def new_mol_to_graph_data(mol):
    """
    mol_to_graph_data
    Args:
        atom_features: Atom features.
        edge_features: Edge features.
        morgan_fingerprint: Morgan fingerprint.
        functional_groups: Functional groups.
    """
    if len(mol.GetAtoms()) == 0:
        return None

    atom_id_names = list(CompoundKit.atom_vocab_dict.keys()) + CompoundKit.atom_float_names
    bond_id_names = list(CompoundKit.bond_vocab_dict.keys())

    data = {}

    ### atom features
    data = {name: [] for name in atom_id_names}

    raw_atom_feat_dicts = CompoundKit.get_atom_names(mol)
    for atom_feat in raw_atom_feat_dicts:
        for name in atom_id_names:
            data[name].append(atom_feat[name])

    ### bond and bond features
    for name in bond_id_names:
        data[name] = []
    data['edges'] = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # i->j and j->i
        data['edges'] += [(i, j), (j, i)]
        for name in bond_id_names:
            bond_feature_id = CompoundKit.get_bond_feature_id(bond, name)
            data[name] += [bond_feature_id] * 2

    #### self loop
    N = len(data[atom_id_names[0]])
    for i in range(N):
        data['edges'] += [(i, i)]
    for name in bond_id_names:
        bond_feature_id = get_bond_feature_dims([name])[0] - 1  # self loop: value = len - 1
        data[name] += [bond_feature_id] * N

    ### make ndarray and check length
    for name in list(CompoundKit.atom_vocab_dict.keys()):
        data[name] = np.array(data[name], 'int64')
    for name in CompoundKit.atom_float_names:
        data[name] = np.array(data[name], 'float32')
    for name in bond_id_names:
        data[name] = np.array(data[name], 'int64')
    data['edges'] = np.array(data['edges'], 'int64')

    ### morgan fingerprint
    data['morgan_fp'] = np.array(CompoundKit.get_morgan_fingerprint(mol), 'int64')
    # data['morgan2048_fp'] = np.array(CompoundKit.get_morgan2048_fingerprint(mol), 'int64')
    data['maccs_fp'] = np.array(CompoundKit.get_maccs_fingerprint(mol), 'int64')
    data['daylight_fg_counts'] = np.array(CompoundKit.get_daylight_functional_group_counts(mol), 'int64')
    return data


def mol_to_graph_data(mol):
    """
    mol_to_graph_data
    Args:
        atom_features: Atom features.
        edge_features: Edge features.
        morgan_fingerprint: Morgan fingerprint.
        functional_groups: Functional groups.
    """
    if len(mol.GetAtoms()) == 0:
        return None

    atom_id_names = [
        "atomic_num", "chiral_tag", "degree", "explicit_valence",
        "formal_charge", "hybridization", "implicit_valence",
        "is_aromatic", "total_numHs",
    ]
    bond_id_names = [
        "bond_dir", "bond_type", "is_in_ring",
    ]

    data = {}
    for name in atom_id_names:
        data[name] = []
    data['mass'] = []
    for name in bond_id_names:
        data[name] = []
    data['edges'] = []

    ### atom features
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 0:
            return None
        for name in atom_id_names:
            data[name].append(CompoundKit.get_atom_feature_id(atom, name) + 1)  # 0: OOV
        data['mass'].append(CompoundKit.get_atom_value(atom, 'mass') * 0.01)

    ### bond features
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # i->j and j->i
        data['edges'] += [(i, j), (j, i)]
        for name in bond_id_names:
            bond_feature_id = CompoundKit.get_bond_feature_id(bond, name) + 1  # 0: OOV
            data[name] += [bond_feature_id] * 2

    ### self loop (+2)
    N = len(data[atom_id_names[0]])
    for i in range(N):
        data['edges'] += [(i, i)]
    for name in bond_id_names:
        bond_feature_id = CompoundKit.get_bond_feature_size(name) + 2  # N + 2: self loop
        data[name] += [bond_feature_id] * N

    ### check whether edge exists
    if len(data['edges']) == 0:  # mol has no bonds
        for name in bond_id_names:
            data[name] = np.zeros((0,), dtype="int64")
        data['edges'] = np.zeros((0, 2), dtype="int64")

    ### make ndarray and check length
    for name in atom_id_names:
        data[name] = np.array(data[name], 'int64')
    data['mass'] = np.array(data['mass'], 'float32')
    for name in bond_id_names:
        data[name] = np.array(data[name], 'int64')
    data['edges'] = np.array(data['edges'], 'int64')

    ### morgan fingerprint
    data['morgan_fp'] = np.array(CompoundKit.get_morgan_fingerprint(mol), 'int64')
    # data['morgan2048_fp'] = np.array(CompoundKit.get_morgan2048_fingerprint(mol), 'int64')
    data['maccs_fp'] = np.array(CompoundKit.get_maccs_fingerprint(mol), 'int64')
    data['daylight_fg_counts'] = np.array(CompoundKit.get_daylight_functional_group_counts(mol), 'int64')
    return data


def mol_to_geognn_graph_data(mol, atom_poses, dir_type):
    """
    mol: rdkit molecule
    dir_type: direction type for bond_angle grpah
    """
    if len(mol.GetAtoms()) == 0:
        return None

    data = mol_to_graph_data(mol)

    data['atom_pos'] = np.array(atom_poses, 'float32')
    data['bond_length'] = Compound3DKit.get_bond_lengths(data['edges'], data['atom_pos'])
    BondAngleGraph_edges, bond_angles, bond_angle_dirs = \
        Compound3DKit.get_superedge_angles(data['edges'], data['atom_pos'])
    data['BondAngleGraph_edges'] = BondAngleGraph_edges
    data['bond_angle'] = np.array(bond_angles, 'float32')
    return data


def mol_to_geognn_graph_data_MMFF3d(mol):
    """tbd"""
    if len(mol.GetAtoms()) <= 400:
        mol, atom_poses = Compound3DKit.get_MMFF_atom_poses(mol, numConfs=10)
    else:
        atom_poses = Compound3DKit.get_2d_atom_poses(mol)
    return mol_to_geognn_graph_data(mol, atom_poses, dir_type='HT')


def mol_to_geognn_graph_data_raw3d(mol):
    """tbd"""
    atom_poses = Compound3DKit.get_atom_poses(mol, mol.GetConformer())
    return mol_to_geognn_graph_data(mol, atom_poses, dir_type='HT')

def obtain_3D_mol(smiles,name):
    mol = AllChem.MolFromSmiles(smiles)
    new_mol = Chem.AddHs(mol)
    res = AllChem.EmbedMultipleConfs(new_mol)
    ### MMFF generates multiple conformations
    res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
    new_mol = Chem.RemoveHs(new_mol)
    Chem.MolToMolFile(new_mol, name+'.mol')
    return new_mol

def mord(mol, nBits=1826, errors_as_zeros=True):
    try:
        result = calc(mol)
        desc_list = [r if not is_missing(r) else 0 for r in result]
        np_arr = np.array(desc_list)
        return np_arr
    except:
        return np.NaN if not errors_as_zeros else np.zeros((nBits,), dtype=np.float32)

def save_3D_mol(all_smile,mol_save_dir):
    """
    该函数尝试将一个包含SMILES字符串的列表转换为一系列的3D分子结构文件，并将这些文件保存到指定的目录。
    如果某些SMILES字符串不能成功转换，它们的索引将被记录和返回。
    """
    index=0
    bad_conformer=[]
    pbar=tqdm(all_smile)
    try:
        os.makedirs(f'{mol_save_dir}')
    except OSError:
        pass
    for smiles in pbar:
        try:
            obtain_3D_mol(smiles,f'{mol_save_dir}/3D_mol_{index}')
        except ValueError:
            bad_conformer.append(index)
            index += 1
            continue
        index += 1
    return bad_conformer   #保存不好的构象的index

def save_dataset(charity_smile,mol_save_dir,charity_name,moder_name,bad_conformer):
    dataset=[]
    dataset_mord=[]
    pbar = tqdm(charity_smile)
    index=0
    for smile in pbar:
        if index in bad_conformer:
            index+=1
            continue
        mol=Chem.MolFromMolFile(f"{mol_save_dir}/3D_mol_{index}.mol")
        #mol = AllChem.MolFromSmiles(smile)
        descriptor=mord(mol)
        data = mol_to_geognn_graph_data_MMFF3d(mol)
        dataset.append(data)
        dataset_mord.append(descriptor)
        index+=1
    dataset_mord=np.array(dataset_mord)
    np.save(f"{charity_name}.npy",dataset,allow_pickle=True)  #保留图数据
    np.save(f'{moder_name}.npy',dataset_mord)   #保留描述符数据


if __name__ == "__main__":
    #smiles = "OCc1ccccc1CN"
    smiles = r"[H]/[NH+]=C(\N)C1=CC(=O)/C(=C\C=c2ccc(=C(N)[NH3+])cc2)C=C1"
    #smiles='CC'
    mol = AllChem.MolFromSmiles(smiles)
    data = mol_to_geognn_graph_data_MMFF3d(mol)
    # for key, value in data.items():
    #     print(key, value.shape)